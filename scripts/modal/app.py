"""Modal app for the microbubbles 3D ULM pipeline (bead mb-crr.1 / MODAL).

Substrate decisions (see docs/roadmap.md):
  * No local GPU + 87GB local disk -> all heavy compute (download / beamform / track /
    volume generation) runs here; only small artifacts are pulled local for browser viewing.
  * The raw ~98GB sample and the beamformed output live on the shared `research` volume
    under our tenant subdir /root/data/microbubbles/ . They never come local.

Cheapest-first probing (respects the spend gate):
  * `inspect`  - CPU, ~free. Lazily reads ONLY /config + acq 0 over HTTP range (fsspec),
                 computes the exact beamformed grid analytically (no GPU, no mach), and
                 reports the full-run download + beamformed storage estimate.
  * `probe`    - GPU. Additionally times beamforming one acquisition and verifies the `mach`
                 kernel imports, so we can estimate full-run GPU cost before committing.
  * `download` - CPU. Pull the full 98GB to the volume (resumable). Only after approval.

SPEND GATE: every function here bills Rome's account. Do not run any of them without his
explicit go-ahead, and verify teardown (`modal app list` / `modal app stop`) after each run.
"""

import modal

PROJECT = "microbubbles"
DATA_ROOT = f"/root/data/{PROJECT}"
SAMPLE_URL = (
    "https://pub-9c1be6312b2441eb8732660783d9ee81.r2.dev/"
    "sanitized_neutral_ultratrace.h5"
)

vol = modal.Volume.from_name("research")

# CPU image: lazy remote HDF5 reads (fsspec range) + the pipeline's own pure-numpy core.
# Build steps BEFORE any add_local_dir (Modal invariant).
cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "h5py==3.11.0",
        "numpy<2.0",
        "scipy==1.13.1",
        "fsspec==2024.6.1",
        "aiohttp==3.9.5",
        "requests==2.32.3",
        "tqdm==4.66.4",
    )
    .add_local_dir("ultratrace_ulm", remote_path="/workspace/ultratrace_ulm")
)

# GPU image: CUDA devel base (mach may JIT CUDA kernels -> needs nvcc), pinned torch + cupy.
gpu_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install(
        "h5py==3.11.0",
        "numpy<2.0",
        "scipy==1.13.1",
        "fsspec==2024.6.1",
        "aiohttp==3.9.5",
        "requests==2.32.3",
        "tqdm==4.66.4",
        "torch==2.4.1",
        "cupy-cuda12x==13.3.0",
        "mach-beamform",
    )
    .add_local_dir("ultratrace_ulm", remote_path="/workspace/ultratrace_ulm")
)

app = modal.App(f"{PROJECT}-pipeline")


# --------------------------------------------------------------------------- #
# Tenant guard (Modal invariant 3): every write must resolve inside our subdir.
# --------------------------------------------------------------------------- #
def _guard(path: str) -> str:
    import os

    resolved = os.path.realpath(path)
    root = os.path.realpath(DATA_ROOT)
    if not (resolved == root or resolved.startswith(root + os.sep)):
        raise RuntimeError(f"Refusing to write outside tenant subdir: {resolved} !~ {root}")
    return resolved


def _ensure_root():
    import os

    os.makedirs(DATA_ROOT, exist_ok=True)
    marker = os.path.join(DATA_ROOT, ".tenant")
    if not os.path.exists(marker):
        with open(marker, "w") as fh:
            fh.write("owner: microbubbles (cajal, rig microbubbles)\n")


def _open_remote_h5(url: str):
    """Lazily open a remote HDF5 over HTTP range reads (only touched chunks transfer)."""
    import fsspec
    import h5py

    fs = fsspec.filesystem("http")
    # block_size caching keeps random h5 chunk reads from thrashing tiny requests.
    fobj = fs.open(url, mode="rb", block_size=8 * 1024 * 1024)
    return h5py.File(fobj, "r"), fobj


def _grid_dims(config) -> tuple[int, int, int]:
    """Exact beamformed (z, elev, x) grid from build_grid math, no GPU needed."""
    import sys

    sys.path.insert(0, "/workspace")
    from ultratrace_ulm.beamform_core import build_grid

    g = build_grid(config)
    return int(g.depth_pixels), int(g.height_pixels), int(g.width_pixels)


# --------------------------------------------------------------------------- #
# inspect — CPU, cheapest. Reads /config + acq 0 over HTTP range; sizes the run.
# --------------------------------------------------------------------------- #
@app.function(image=cpu_image, timeout=1800, volumes={"/root/data": vol})
def inspect(url: str = SAMPLE_URL, elev_planes: int = 25) -> dict:
    import json
    import sys
    import time

    import numpy as np

    sys.path.insert(0, "/workspace")
    from ultratrace_ulm.beamform_core import NeutralConfig

    _ensure_root()
    t0 = time.time()
    h5, fobj = _open_remote_h5(url)
    report: dict = {"url": url}
    try:
        acq_ids = sorted(int(k) for k in h5["acquisitions"].keys() if str(k).isdigit())
        attrs = dict(h5["config"].attrs)
        attrs["num_elev_planes"] = elev_planes
        # Defaults mirror beamform_mach._load_neutral_config knobs.
        attrs.setdefault("z_coarseness", 0.5)
        attrs.setdefault("x_coarseness", 0.5)
        attrs.setdefault("large_fov", True)
        attrs.setdefault("xlarge_fov", False)
        config = NeutralConfig.from_h5_attrs(attrs)

        g = h5[f"acquisitions/{acq_ids[0]}"]
        iq = g["iq_frames"]
        iq_shape = tuple(int(s) for s in iq.shape)
        iq_dtype = str(iq.dtype)
        z, elev, x = _grid_dims(config)
        # preprocess keeps loops-num_noise_loops frames (frames ~ loops here).
        frames = iq_shape[0] - int(getattr(config, "num_noise_loops", 0) or 0)

        per_acq_bf_bytes = frames * elev * z * x * 8  # complex64
        n_acq = len(acq_ids)
        report.update(
            {
                "n_acquisitions": n_acq,
                "acq0_iq_shape": iq_shape,
                "acq0_iq_dtype": iq_dtype,
                "config": {k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v))
                           for k, v in dict(h5["config"].attrs).items()},
                "beamformed_grid_zelevx": [z, elev, x],
                "frames_per_acq_est": int(frames),
                "per_acq_beamformed_bytes": int(per_acq_bf_bytes),
                "per_acq_beamformed_GiB": round(per_acq_bf_bytes / 2**30, 3),
                "full_beamformed_uncompressed_GiB": round(per_acq_bf_bytes * n_acq / 2**30, 2),
                "inspect_seconds": round(time.time() - t0, 1),
            }
        )
    finally:
        h5.close()
        fobj.close()

    out = _guard(f"{DATA_ROOT}/inspect_report.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    vol.commit()
    print(json.dumps(report, indent=2))
    return report


# --------------------------------------------------------------------------- #
# probe — GPU. Times one-acquisition beamform + verifies the mach kernel imports.
# --------------------------------------------------------------------------- #
@app.function(image=gpu_image, gpu="A10G", timeout=1800, memory=98304,
              volumes={"/root/data": vol})
def probe(url: str = SAMPLE_URL, elev_planes: int = 25, frame_rate_hz: float = 222.0) -> dict:
    """Full cost-model probe: verify mach + time one fused acq (beamform + SVD +
    detect) + measure lzf compression, so we can size the full run vs the budget."""
    import json
    import os
    import sys
    import time

    import numpy as np

    sys.path.insert(0, "/workspace")
    from ultratrace_ulm.beamform_core import MACH_AVAILABLE, NeutralConfig, beamform_iq

    _ensure_root()
    report: dict = {"url": url, "mach_available": bool(MACH_AVAILABLE)}
    if not MACH_AVAILABLE:
        report["error"] = "mach kernel not importable in gpu_image; beamforming blocked"
        with open(_guard(f"{DATA_ROOT}/probe_report.json"), "w") as fh:
            json.dump(report, fh, indent=2)
        vol.commit()
        print(json.dumps(report, indent=2))
        return report

    from ultratrace_ulm.svd import filtered_magnitude
    from ultratrace_ulm.tracking import detect_batch

    h5, fobj = _open_remote_h5(url)
    try:
        acq_ids = sorted(int(k) for k in h5["acquisitions"].keys() if str(k).isdigit())
        attrs = dict(h5["config"].attrs)
        attrs["num_elev_planes"] = elev_planes
        attrs.setdefault("z_coarseness", 0.5)
        attrs.setdefault("x_coarseness", 0.5)
        attrs.setdefault("large_fov", True)
        attrs.setdefault("xlarge_fov", False)
        config = NeutralConfig.from_h5_attrs(attrs)

        aid = acq_ids[0]
        g = h5[f"acquisitions/{aid}"]
        iq = np.asarray(g["iq_frames"], dtype=np.complex64)
        txd = np.asarray(g["tx_delays"], dtype=np.float64)
        txd_elev = np.asarray(g["tx_delays_elev"], dtype=np.float64)

        t0 = time.time()
        compound, grid = beamform_iq(iq, txd, txd_elev, config, stream_accumulate=True)
        t_beamform = time.time() - t0

        t0 = time.time()
        mag = filtered_magnitude(
            compound, low_cutoff=0.1, method="adaptive", temporal_sigma=0.0,
            frame_rate_hz=frame_rate_hz, tissue_freq_hz=100.0,
        )
        t_filter = time.time() - t0

        t0 = time.time()
        batch = detect_batch(mag, sigma_threshold=2.0, min_distance=2, smoothing_sigma=1.0)
        t_detect = time.time() - t0
        n_det = int(sum(len(p) for p, _, _ in batch))

        # lzf compression ratio on one acq's compound.
        import h5py

        tmp = "/tmp/probe_shard.h5"
        with h5py.File(tmp, "w") as out:
            out.create_dataset("c", data=compound, chunks=(1,) + tuple(compound.shape[1:]),
                               compression="lzf")
        comp_bytes = os.path.getsize(tmp)
        raw_bytes = int(compound.size * 8)
        os.remove(tmp)

        per_acq_total = t_beamform + t_filter + t_detect
        report.update({
            "compound_shape": [int(s) for s in compound.shape],
            "per_acq_beamform_s": round(t_beamform, 2),
            "per_acq_filter_svd_s": round(t_filter, 2),
            "per_acq_detect_s": round(t_detect, 2),
            "per_acq_total_s": round(per_acq_total, 2),
            "detections_in_acq": n_det,
            "lzf_ratio": round(raw_bytes / comp_bytes, 3),
            "compressed_per_acq_GiB": round(comp_bytes / 2**30, 3),
            "full_223_fused_gpu_hours_est": round(per_acq_total * 223 / 3600, 2),
            "full_223_beamform_only_gpu_hours_est": round(t_beamform * 223 / 3600, 2),
        })
    finally:
        h5.close()
        fobj.close()

    with open(_guard(f"{DATA_ROOT}/probe_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    vol.commit()
    print(json.dumps(report, indent=2))
    return report


# --------------------------------------------------------------------------- #
# download — CPU. Pull the full 98GB to the volume (resumable). Only after approval.
# --------------------------------------------------------------------------- #
@app.function(image=cpu_image, timeout=6 * 3600, volumes={"/root/data": vol})
def download(url: str = SAMPLE_URL) -> dict:
    import os
    import sys
    import time

    sys.path.insert(0, "/workspace")
    from ultratrace_ulm.download import download_sample

    _ensure_root()
    dest = _guard(f"{DATA_ROOT}/sanitized_neutral_ultratrace.h5")
    t0 = time.time()
    # Commit periodically so an eviction mid-download keeps the resumable partial.
    download_sample(url, dest)
    vol.commit()
    size = os.path.getsize(dest)
    return {"dest": dest, "bytes": size, "GiB": round(size / 2**30, 2), "seconds": round(time.time() - t0, 1)}


def _build_config(h5, elev_planes: int):
    import sys

    sys.path.insert(0, "/workspace")
    from ultratrace_ulm.beamform_core import NeutralConfig

    attrs = dict(h5["config"].attrs)
    attrs["num_elev_planes"] = elev_planes
    attrs.setdefault("z_coarseness", 0.5)
    attrs.setdefault("x_coarseness", 0.5)
    attrs.setdefault("large_fov", True)
    attrs.setdefault("xlarge_fov", False)
    return NeutralConfig.from_h5_attrs(attrs)


# --------------------------------------------------------------------------- #
# beamform_all — GPU, STREAMING. Lazily reads each acq from the remote (no 98GB
# raw stored on the volume), two-pass global TGC, writes one COMPRESSED+CHUNKED
# shard per acquisition with per-shard commit + resume. This is the STREAM-BF
# (mb-crr.2) design producing the BASELINE beamformed output.
# --------------------------------------------------------------------------- #
@app.function(image=gpu_image, gpu="A10G", timeout=12 * 3600, memory=32768,
              volumes={"/root/data": vol})
def beamform_all(url: str = SAMPLE_URL, elev_planes: int = 25, spatial_tgc: bool = True,
                 tgc_acqs: int = 12, tgc_sigma_lambda: float = 9.0, tgc_svd_cut: float = 0.05,
                 acq_start: int = 0, num_acqs: int = 0, acq_step: int = 1,
                 compression: str = "lzf") -> dict:
    import json
    import os
    import sys
    import time

    import h5py
    import numpy as np

    sys.path.insert(0, "/workspace")
    from ultratrace_ulm.beamform_core import beamform_iq

    _ensure_root()
    shard_dir = _guard(f"{DATA_ROOT}/beamformed/shards")
    os.makedirs(shard_dir, exist_ok=True)

    h5, fobj = _open_remote_h5(url)
    try:
        all_ids = sorted(int(k) for k in h5["acquisitions"].keys() if str(k).isdigit())
        sel = all_ids[acq_start::acq_step]
        if num_acqs and num_acqs > 0:
            sel = sel[:num_acqs]
        config = _build_config(h5, elev_planes)

        def _read_acq(aid):
            g = h5[f"acquisitions/{aid}"]
            return (
                np.asarray(g["iq_frames"], dtype=np.complex64),
                np.asarray(g["tx_delays"], dtype=np.float64),
                np.asarray(g["tx_delays_elev"], dtype=np.float64),
            )

        # ---- Pass 1: global spatial TGC from a sampled subset, accumulated one
        # acq at a time (the shipped compute_global_tgc takes a list of all
        # samples = ~142GB here). Recomputed on resume; deterministic. ----
        inv_sqrt = None
        if spatial_tgc:
            from scipy.ndimage import gaussian_filter

            from ultratrace_ulm.svd import filter_svd_3d

            if tgc_acqs and tgc_acqs < len(sel):
                idx = np.unique(np.linspace(0, len(sel) - 1, tgc_acqs).round().astype(int))
                tgc_sel = [sel[i] for i in idx]
            else:
                tgc_sel = list(sel)
            print(f"[tgc] computing global TGC from {len(tgc_sel)} acq(s)", flush=True)
            pd_sum, ref_grid = None, None
            for aid in tgc_sel:
                iq, txd, txde = _read_acq(aid)
                comp, ref_grid = beamform_iq(iq, txd, txde, config, stream_accumulate=True)
                out = filter_svd_3d(comp, low_cutoff=tgc_svd_cut, method="full")
                pd = (np.abs(out) ** 2).mean(0)  # (elev, z, x)
                pd_sum = pd if pd_sum is None else pd_sum + pd
                del comp, out, pd
            pd_mean = pd_sum / len(tgc_sel)
            z_ax, y_ax, x_ax = ref_grid.z[:, 0, 0], ref_grid.y[0, :, 0], ref_grid.x[0, 0, :]
            wavelength = 1540.0 / config.tx_freq_hz
            sigma_m = tgc_sigma_lambda * wavelength
            dz = abs(z_ax[1] - z_ax[0]) if len(z_ax) > 1 else None
            dx = abs(x_ax[1] - x_ax[0]) if len(x_ax) > 1 else None
            dy = abs(y_ax[1] - y_ax[0]) if len(y_ax) > 1 else None
            sig = (sigma_m / dy if (dy and dy > 0) else 0.0,
                   sigma_m / dz if dz else 0.0,
                   sigma_m / dx if dx else 0.0)
            tgc = gaussian_filter(pd_mean, sigma=sig).astype(np.float32)
            inv_sqrt = (1.0 / np.sqrt(np.maximum(tgc, np.finfo(np.float32).eps)))[None]

        # ---- Pass 2: beamform each acq, apply TGC, write compressed shard. ----
        manifest, times = [], []
        for order, aid in enumerate(sel):
            shard = os.path.join(shard_dir, f"acq_{order:04d}.h5")
            if os.path.exists(shard):
                try:
                    with h5py.File(shard, "r") as t:
                        if f"acquisitions/{order}/meta/compound_image" in t:
                            manifest.append({"order": order, "src_acq_id": int(aid)})
                            print(f"[beamform] order={order} acq={aid} resume-skip", flush=True)
                            continue
                except Exception:
                    os.remove(shard)
            iq, txd, txde = _read_acq(aid)
            t0 = time.time()
            comp, grid = beamform_iq(iq, txd, txde, config, stream_accumulate=True)
            if inv_sqrt is not None:
                comp = (comp * inv_sqrt).astype(np.complex64)
            dt = time.time() - t0
            times.append(dt)
            tmp = shard + ".tmp"
            with h5py.File(tmp, "w") as out:
                meta = out.require_group(f"acquisitions/{order}/meta")
                meta.create_dataset(
                    "compound_image", data=comp.astype(np.complex64),
                    chunks=(1,) + tuple(comp.shape[1:]), compression=compression,
                )
                gg = meta.require_group("grid")
                for axis, arr in (("x", grid.x), ("y", grid.y), ("z", grid.z)):
                    gg.create_dataset(axis, data=arr.astype(np.float64), compression=compression)
                out.attrs["src_acq_id"] = int(aid)
            os.replace(tmp, shard)
            vol.commit()
            manifest.append({"order": order, "src_acq_id": int(aid), "shape": list(comp.shape)})
            print(f"[beamform] order={order} acq={aid} {tuple(comp.shape)} {dt:.1f}s", flush=True)

        report = {
            "n_shards": len(manifest), "shard_dir": shard_dir, "spatial_tgc": spatial_tgc,
            "compression": compression,
            "median_beamform_s": round(float(np.median(times)), 2) if times else None,
            "acqs": manifest,
        }
        with open(_guard(f"{DATA_ROOT}/beamformed/manifest.json"), "w") as fh:
            json.dump(report, fh, indent=2)
        vol.commit()
    finally:
        h5.close()
        fobj.close()
    print(f"DONE beamform_all: {len(manifest)} shards")
    return {k: v for k, v in report.items() if k != "acqs"}


# --------------------------------------------------------------------------- #
# consolidate — CPU. Zero-copy single-file view via HDF5 ExternalLinks so the
# existing tracking reader sees one `baseline.h5` with all acquisitions.
# --------------------------------------------------------------------------- #
@app.function(image=cpu_image, timeout=1800, volumes={"/root/data": vol})
def consolidate(out_name: str = "baseline.h5") -> dict:
    import os

    import h5py

    vol.reload()
    shard_dir = f"{DATA_ROOT}/beamformed/shards"
    shards = sorted(f for f in os.listdir(shard_dir) if f.endswith(".h5"))
    out_path = _guard(f"{DATA_ROOT}/beamformed/{out_name}")
    with h5py.File(out_path, "w") as out:
        grp = out.require_group("acquisitions")
        for i, sh in enumerate(shards):
            with h5py.File(os.path.join(shard_dir, sh), "r") as t:
                inner = sorted(t["acquisitions"].keys(), key=int)[0]
            grp[str(i)] = h5py.ExternalLink(f"shards/{sh}", f"/acquisitions/{inner}")
    vol.commit()
    print(f"consolidated {len(shards)} shards -> {out_path}")
    return {"out": out_path, "n": len(shards)}


# --------------------------------------------------------------------------- #
# track — CPU. Runs the shipped tracking with the README baseline recipe to
# produce the BASELINE tracks (+ .bin exports we pull local for viewing).
# --------------------------------------------------------------------------- #
@app.function(image=cpu_image, timeout=6 * 3600, memory=49152, volumes={"/root/data": vol})
def track(beamformed: str = "baseline.h5", frame_rate_hz: float = 222.0,
          svd_method: str = "adaptive", min_track_length: int = 5,
          tag: str = "baseline") -> dict:
    import sys
    from pathlib import Path

    sys.path.insert(0, "/workspace")
    from ultratrace_ulm.tracking import TrackingOptions, run_tracking_outputs

    vol.reload()
    _ensure_root()
    out_dir = Path(_guard(f"{DATA_ROOT}/tracks/{tag}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    opts = TrackingOptions(
        beamformed_path=Path(f"{DATA_ROOT}/beamformed/{beamformed}"),
        tracks_path=out_dir / "tracks.pkl",
        svd_method=svd_method, knee_filter=True, tissue_freq_hz=100.0,
        temporal_sigma=0.0, filter_method="svd", svd_low_cutoff=0.1,
        sigma_threshold=2.0, min_distance=2, smoothing_sigma=1.0,
        subpixel="centroid", window_size=5, tracking="kalman",
        frame_rate_hz=frame_rate_hz, max_gap=3, min_track_length=min_track_length,
        reversal_penalty=10.0, max_cost=10.0, smooth_sigma=2.0,
        smooth_method="gaussian", export_dir=out_dir, export_stem="tracks",
        export_min_lengths=(5, 20, 50),
    )
    smoothed = run_tracking_outputs(opts)
    vol.commit()
    print(f"DONE track -> {out_dir}")
    return {"tracks_dir": str(out_dir), "smoothed": str(smoothed)}


# --------------------------------------------------------------------------- #
# baseline — GPU, FUSED beamform->SVD->detect->localize->track. Beamforms each
# acq, feeds it straight to the streamed tracker, and DISCARDS the 11.9GB volume.
# Persists only: tracks (tiny) + compressed shards for a few reference acqs (for
# the 3D volume viewer + diffs). This is the storage-sane baseline producer.
# --------------------------------------------------------------------------- #
@app.function(image=gpu_image, gpu="A10G", timeout=12 * 3600, memory=98304,
              volumes={"/root/data": vol})
def baseline(url: str = SAMPLE_URL, elev_planes: int = 25, frame_rate_hz: float = 222.0,
             spatial_tgc: bool = True, tgc_acqs: int = 12, tgc_sigma_lambda: float = 9.0,
             tgc_svd_cut: float = 0.05, acq_start: int = 0, num_acqs: int = 0,
             acq_step: int = 1, keep_orders: str = "0,74,148,222",
             min_track_length: int = 5, svd_method: str = "adaptive",
             use_gpu_svd: bool = True, tag: str = "baseline") -> dict:
    import json
    import os
    import sys
    import time
    from pathlib import Path

    import h5py
    import numpy as np

    sys.path.insert(0, "/workspace")
    from ultratrace_ulm.beamform_core import beamform_iq
    from ultratrace_ulm.tracking import TrackingOptions, run_tracking_outputs_streamed

    _ensure_root()
    out_dir = Path(_guard(f"{DATA_ROOT}/tracks/{tag}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_dir = _guard(f"{DATA_ROOT}/beamformed/{tag}_refs")
    os.makedirs(ref_dir, exist_ok=True)
    keep = {int(x) for x in keep_orders.split(",") if x.strip() != ""}

    h5, fobj = _open_remote_h5(url)
    all_ids = sorted(int(k) for k in h5["acquisitions"].keys() if str(k).isdigit())
    sel = all_ids[acq_start::acq_step]
    if num_acqs and num_acqs > 0:
        sel = sel[:num_acqs]
    config = _build_config(h5, elev_planes)

    def _read_acq(aid):
        g = h5[f"acquisitions/{aid}"]
        return (
            np.asarray(g["iq_frames"], dtype=np.complex64),
            np.asarray(g["tx_delays"], dtype=np.float64),
            np.asarray(g["tx_delays_elev"], dtype=np.float64),
        )

    # ---- TGC pass (incremental power map; see beamform_all). ----
    inv_sqrt = None
    if spatial_tgc:
        from scipy.ndimage import gaussian_filter

        if use_gpu_svd:
            from ultratrace_ulm.gpu_svd import filter_svd_3d_gpu as _tgc_svd
        else:
            from ultratrace_ulm.svd import filter_svd_3d as _tgc_svd

        if tgc_acqs and tgc_acqs < len(sel):
            idx = np.unique(np.linspace(0, len(sel) - 1, tgc_acqs).round().astype(int))
            tgc_sel = [sel[i] for i in idx]
        else:
            tgc_sel = list(sel)
        print(f"[tgc] {len(tgc_sel)} acq(s)", flush=True)
        pd_sum, ref_grid = None, None
        for aid in tgc_sel:
            iq, txd, txde = _read_acq(aid)
            comp, ref_grid = beamform_iq(iq, txd, txde, config, stream_accumulate=True)
            # GPU path uses the fast cov-projection for the TGC power map (the
            # full-SVD power map is ~identical for a smooth normalisation map).
            out = _tgc_svd(comp, low_cutoff=tgc_svd_cut, method="fast" if use_gpu_svd else "full")
            pd = (np.abs(out) ** 2).mean(0)
            pd_sum = pd if pd_sum is None else pd_sum + pd
            del comp, out, pd
        pd_mean = pd_sum / len(tgc_sel)
        z_ax, y_ax, x_ax = ref_grid.z[:, 0, 0], ref_grid.y[0, :, 0], ref_grid.x[0, 0, :]
        sigma_m = tgc_sigma_lambda * (1540.0 / config.tx_freq_hz)
        dz = abs(z_ax[1] - z_ax[0]) if len(z_ax) > 1 else None
        dx = abs(x_ax[1] - x_ax[0]) if len(x_ax) > 1 else None
        dy = abs(y_ax[1] - y_ax[0]) if len(y_ax) > 1 else None
        sig = (sigma_m / dy if (dy and dy > 0) else 0.0,
               sigma_m / dz if dz else 0.0, sigma_m / dx if dx else 0.0)
        tgc = gaussian_filter(pd_mean, sigma=sig).astype(np.float32)
        inv_sqrt = (1.0 / np.sqrt(np.maximum(tgc, np.finfo(np.float32).eps)))[None]

    times = []

    def _provider():
        for order, aid in enumerate(sel):
            iq, txd, txde = _read_acq(aid)
            t0 = time.time()
            comp, grid = beamform_iq(iq, txd, txde, config, stream_accumulate=True)
            if inv_sqrt is not None:
                comp = (comp * inv_sqrt).astype(np.complex64)
            times.append(time.time() - t0)
            if order in keep:
                shard = os.path.join(ref_dir, f"acq_{order:04d}.h5")
                tmp = shard + ".tmp"
                with h5py.File(tmp, "w") as o:
                    meta = o.require_group(f"acquisitions/{order}/meta")
                    meta.create_dataset("compound_image", data=comp,
                                        chunks=(1,) + tuple(comp.shape[1:]), compression="lzf")
                    gg = meta.require_group("grid")
                    for axis, arr in (("x", grid.x), ("y", grid.y), ("z", grid.z)):
                        gg.create_dataset(axis, data=arr.astype(np.float64), compression="lzf")
                    o.attrs["src_acq_id"] = int(aid)
                os.replace(tmp, shard)
                vol.commit()
            # Grid (z,elev,x) meters -> (elev,z,x) mm, matching h5_io.grid_arrays.
            gx = (np.transpose(grid.x, (1, 0, 2)) * 1000.0).astype(np.float32)
            gy = (np.transpose(grid.y, (1, 0, 2)) * 1000.0).astype(np.float32)
            gz = (np.transpose(grid.z, (1, 0, 2)) * 1000.0).astype(np.float32)
            print(f"[fused] order={order} acq={aid} {tuple(comp.shape)} "
                  f"{times[-1]:.1f}s", flush=True)
            yield int(aid), comp, gx, gy, gz

    opts = TrackingOptions(
        beamformed_path=Path(f"{DATA_ROOT}/none"), tracks_path=out_dir / "tracks.pkl",
        svd_method=svd_method, knee_filter=True, tissue_freq_hz=100.0, temporal_sigma=0.0,
        filter_method="svd", svd_low_cutoff=0.1, sigma_threshold=2.0, min_distance=2,
        smoothing_sigma=1.0, subpixel="centroid", window_size=5, tracking="kalman",
        frame_rate_hz=frame_rate_hz, max_gap=3, min_track_length=min_track_length,
        reversal_penalty=10.0, max_cost=10.0, smooth_sigma=2.0, smooth_method="gaussian",
        export_dir=out_dir, export_stem="tracks", export_min_lengths=(5, 20, 50),
    )
    gpu_filter = None
    if use_gpu_svd:
        from ultratrace_ulm.gpu_svd import filtered_magnitude_gpu

        def gpu_filter(compound, o):
            return filtered_magnitude_gpu(
                compound, low_cutoff=o.svd_low_cutoff, high_cutoff=o.svd_high_cutoff,
                method=o.svd_method, temporal_sigma=o.temporal_sigma,
                n_components=o.svd_n_components, frame_rate_hz=o.frame_rate_hz,
                tissue_freq_hz=o.tissue_freq_hz,
            )

    try:
        smoothed = run_tracking_outputs_streamed(
            opts, [int(a) for a in sel], _provider(), filter_fn=gpu_filter,
        )
    finally:
        h5.close()
        fobj.close()
    vol.commit()
    report = {
        "tag": tag, "n_acqs": len(sel), "tracks_dir": str(out_dir),
        "ref_shards": sorted(os.listdir(ref_dir)),
        "median_beamform_s": round(float(np.median(times)), 2) if times else None,
        "smoothed": str(smoothed),
    }
    with open(_guard(f"{DATA_ROOT}/tracks/{tag}/baseline_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    vol.commit()
    print(json.dumps(report, indent=2))
    return report


@app.local_entrypoint()
def main(fn: str = "inspect"):
    """Convenience: `modal run scripts/modal/app.py` runs inspect by default.

    Prefer explicit `modal run scripts/modal/app.py::<fn>` for inspect / probe /
    download / beamform_all / consolidate / track / baseline.
    """
    table = {
        "inspect": inspect, "probe": probe, "download": download,
        "beamform_all": beamform_all, "consolidate": consolidate, "track": track,
        "baseline": baseline,
    }
    if fn not in table:
        raise SystemExit(f"unknown fn {fn!r}; choose from {sorted(table)}")
    print(table[fn].remote())
