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
def download(url: str = SAMPLE_URL, max_attempts: int = 80) -> dict:
    """Resume-until-VERIFIED-COMPLETE download. R2's public endpoint silently
    drops large transfers (urllib read() returns empty EOF without error), so a
    single pass truncates. Loop download_sample (each call resumes via HTTP Range)
    and commit until the file size matches the remote Content-Length."""
    import os
    import sys
    import time

    sys.path.insert(0, "/workspace")
    from ultratrace_ulm.download import _remote_size, download_sample

    _ensure_root()
    dest = _guard(f"{DATA_ROOT}/sanitized_neutral_ultratrace.h5")
    total = _remote_size(url)
    t0 = time.time()
    size = os.path.getsize(dest) if os.path.exists(dest) else 0
    for attempt in range(max_attempts):
        if total and size >= total:
            break
        try:
            download_sample(url, dest)  # resumes from current size
        except Exception as e:  # noqa: BLE001 - network drop; retry
            print(f"[download] attempt {attempt} dropped: {e}", flush=True)
        vol.commit()
        new = os.path.getsize(dest)
        print(f"[download] attempt {attempt}: {new/2**30:.1f}/{(total or 0)/2**30:.1f} GiB", flush=True)
        if new == size and (not total or new < total):
            time.sleep(2)  # no progress; brief backoff before retry
        size = new
    complete = bool(total and size >= total)
    return {"dest": dest, "bytes": size, "GiB": round(size / 2**30, 2), "total_GiB": round((total or 0) / 2**30, 2),
            "complete": complete, "attempts": attempt + 1, "seconds": round(time.time() - t0, 1)}


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
             use_gpu_svd: bool = True, motion: bool = False,
             filter_variant: str = "global", gate_on_prediction: bool = False,
             n_z_blocks: int = 3, n_x_blocks: int = 3, tag: str = "baseline") -> dict:
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
            if motion:
                from ultratrace_ulm.gpu_motion import correct_motion_gpu
                comp, _shifts = correct_motion_gpu(comp)
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
        reversal_penalty=10.0, max_cost=10.0, gate_on_prediction=gate_on_prediction,
        smooth_sigma=2.0, smooth_method="gaussian",
        export_dir=out_dir, export_stem="tracks", export_min_lengths=(5, 20, 50),
    )
    gpu_filter = gpu_detect = None
    if use_gpu_svd:
        from ultratrace_ulm.gpu_detect import detect_batch_gpu
        from ultratrace_ulm.gpu_svd import filtered_magnitude_gpu

        def gpu_filter(compound, o):
            if filter_variant == "region":
                from ultratrace_ulm.gpu_svd_region import filtered_magnitude_region_gpu
                return filtered_magnitude_region_gpu(
                    compound, low_cutoff=o.svd_low_cutoff, high_cutoff=o.svd_high_cutoff,
                    method=o.svd_method, temporal_sigma=o.temporal_sigma,
                    n_components=o.svd_n_components, frame_rate_hz=o.frame_rate_hz,
                    tissue_freq_hz=o.tissue_freq_hz, n_z_blocks=n_z_blocks, n_x_blocks=n_x_blocks,
                )
            return filtered_magnitude_gpu(
                compound, low_cutoff=o.svd_low_cutoff, high_cutoff=o.svd_high_cutoff,
                method=o.svd_method, temporal_sigma=o.temporal_sigma,
                n_components=o.svd_n_components, frame_rate_hz=o.frame_rate_hz,
                tissue_freq_hz=o.tissue_freq_hz,
            )

        gpu_detect = detect_batch_gpu

    try:
        smoothed = run_tracking_outputs_streamed(
            opts, [int(a) for a in sel], _provider(), filter_fn=gpu_filter, detect_fn=gpu_detect,
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


def _open_local_or_remote_h5(url: str):
    """Prefer the downloaded volume copy (reliable local reads); fall back to
    lazy HTTP. Returns (h5, fobj_or_None, source)."""
    import os

    import h5py

    local = f"{DATA_ROOT}/sanitized_neutral_ultratrace.h5"
    if os.path.exists(local):
        return h5py.File(local, "r"), None, "local"
    h5, fobj = _open_remote_h5(url)
    return h5, fobj, "remote"


# --------------------------------------------------------------------------- #
# detect_acqs (Phase A) — GPU, CHECKPOINTED + ADDITIVE + INSTRUMENTED. Per acq:
# read -> beamform(+motion) -> filter -> detect -> localize -> save a tiny npz of
# localizations. Skips acqs already done (resume); new acqs append (additive). A
# timeout/stall only loses the current acq. Logs per-stage timings so we can SEE
# where time goes (the 12h-timeout post-mortem). Prefers the local downloaded h5.
# --------------------------------------------------------------------------- #
@app.function(image=gpu_image, gpu="A10G", timeout=12 * 3600, memory=98304,
              volumes={"/root/data": vol})
def detect_acqs(url: str = SAMPLE_URL, elev_planes: int = 25, frame_rate_hz: float = 222.0,
                spatial_tgc: bool = True, tgc_acqs: int = 12, tgc_sigma_lambda: float = 9.0,
                tgc_svd_cut: float = 0.05, acq_start: int = 0, num_acqs: int = 0, acq_step: int = 1,
                svd_method: str = "adaptive", motion: bool = False, filter_variant: str = "global",
                n_z_blocks: int = 3, n_x_blocks: int = 3, keep_orders: str = "", tag: str = "baseline",
                detector: str = "zscore", svd_rank: bool = False, rank_delta: float = 2.0,
                nms_elev: int = 0, elev_debias: bool = False, low_conf: bool = False) -> dict:
    import json
    import os
    import sys
    import time
    from pathlib import Path

    import h5py
    import numpy as np

    sys.path.insert(0, "/workspace")
    from ultratrace_ulm.beamform_core import beamform_iq
    from ultratrace_ulm.gpu_detect import detect_batch_gpu
    from ultratrace_ulm.gpu_svd import filtered_magnitude_gpu
    from ultratrace_ulm.tracking import TrackingOptions, _grid_spacing, detect_localize_acq

    _ensure_root()
    det_dir = _guard(f"{DATA_ROOT}/detections/{tag}")
    ref_dir = _guard(f"{DATA_ROOT}/beamformed/{tag}_refs")
    os.makedirs(det_dir, exist_ok=True)
    os.makedirs(ref_dir, exist_ok=True)
    keep = {int(x) for x in keep_orders.split(",") if x.strip() != ""}

    h5, fobj, source = _open_local_or_remote_h5(url)
    print(f"[detect_acqs] source={source} tag={tag}", flush=True)
    all_ids = sorted(int(k) for k in h5["acquisitions"].keys() if str(k).isdigit())
    sel = all_ids[acq_start::acq_step]
    if num_acqs and num_acqs > 0:
        sel = sel[:num_acqs]
    config = _build_config(h5, elev_planes)
    detector = str(detector).lower()
    if detector not in {"zscore", "cfar", "psf"}:
        raise ValueError("detector must be one of: zscore, cfar, psf")
    if isinstance(svd_rank, str):
        rank_arg = svd_rank.strip().lower()
        use_svd_rank = rank_arg not in {"", "0", "false", "no", "off", "none"}
        rank_method = rank_arg if rank_arg in {"elbow", "energy"} else "elbow"
    else:
        use_svd_rank = bool(svd_rank)
        rank_method = "elbow"

    def _read(aid):
        g = h5[f"acquisitions/{aid}"]
        return (np.asarray(g["iq_frames"], dtype=np.complex64),
                np.asarray(g["tx_delays"], dtype=np.float64),
                np.asarray(g["tx_delays_elev"], dtype=np.float64))

    def gpu_filter(comp, o):
        if use_svd_rank:
            from ultratrace_ulm.svd_rank import filter_svd_3d_region_ranked_gpu

            filtered, _kmap, _raw = filter_svd_3d_region_ranked_gpu(
                comp,
                n_z_blocks=n_z_blocks,
                n_x_blocks=n_x_blocks,
                rank_method=rank_method,
                delta=float(rank_delta),
                high_cutoff=o.svd_high_cutoff,
            )
            mag = np.abs(filtered).astype(np.float32, copy=False)
            if o.temporal_sigma > 0:
                from scipy.ndimage import gaussian_filter1d

                mag = gaussian_filter1d(mag, sigma=o.temporal_sigma, axis=0)
            return mag
        if filter_variant == "region":
            from ultratrace_ulm.gpu_svd_region import filtered_magnitude_region_gpu
            return filtered_magnitude_region_gpu(
                comp, low_cutoff=o.svd_low_cutoff, method=o.svd_method,
                frame_rate_hz=o.frame_rate_hz, tissue_freq_hz=o.tissue_freq_hz,
                n_z_blocks=n_z_blocks, n_x_blocks=n_x_blocks)
        return filtered_magnitude_gpu(comp, low_cutoff=o.svd_low_cutoff, method=o.svd_method,
                                      frame_rate_hz=o.frame_rate_hz, tissue_freq_hz=o.tissue_freq_hz)

    opts = TrackingOptions(
        beamformed_path=Path(f"{DATA_ROOT}/none"), tracks_path=Path(det_dir) / "x.pkl",
        svd_method=svd_method, knee_filter=True, tissue_freq_hz=100.0, temporal_sigma=0.0,
        filter_method="svd", svd_low_cutoff=0.1, sigma_threshold=2.0, min_distance=2,
        smoothing_sigma=1.0, subpixel="centroid", window_size=5, tracking="kalman",
        frame_rate_hz=frame_rate_hz)

    def _with_confidence(batch, high_threshold: float):
        out = []
        for pixels, intensities, zscores in batch:
            conf = (np.asarray(zscores) >= float(high_threshold)).astype(np.uint8, copy=False)
            out.append((pixels, intensities, zscores, conf))
        return out

    def _detect_zscore(magnitude, sigma_threshold, min_distance, smoothing_sigma):
        threshold = max(0.0, float(sigma_threshold) - 1.0) if low_conf else float(sigma_threshold)
        batch = detect_batch_gpu(magnitude, threshold, min_distance, smoothing_sigma)
        return _with_confidence(batch, sigma_threshold) if low_conf else batch

    def _nms_radius(min_distance: int):
        if int(nms_elev) > 0:
            return (int(nms_elev), int(min_distance), int(min_distance))
        return (int(min_distance), int(min_distance), int(min_distance))

    def _detect_cfar(magnitude, sigma_threshold, min_distance, smoothing_sigma):
        from ultratrace_ulm.detect_cfar import detect_batch_cfar_gpu

        threshold = max(0.0, float(sigma_threshold) - 1.0) if low_conf else float(sigma_threshold)
        batch = detect_batch_cfar_gpu(
            magnitude,
            sigma_threshold=threshold,
            min_distance=min_distance,
            smoothing_sigma=smoothing_sigma,
            nms_radius=_nms_radius(min_distance),
            debias=bool(elev_debias),
        )
        return _with_confidence(batch, sigma_threshold) if low_conf else batch

    def _detect_psf(magnitude, sigma_threshold, min_distance, smoothing_sigma):
        from ultratrace_ulm.psf import detect_batch_matched_filter_gpu, estimate_empirical_psf

        pilot = detect_batch_gpu(
            magnitude,
            max(float(sigma_threshold), 5.0),
            min_distance,
            smoothing_sigma,
        )
        try:
            psf = estimate_empirical_psf(
                magnitude,
                pilot,
                min_zscore=max(float(sigma_threshold), 5.0),
            )
        except ValueError:
            threshold = max(0.0, float(sigma_threshold) - 1.0) if low_conf else float(sigma_threshold)
            batch = detect_batch_gpu(magnitude, threshold, min_distance, smoothing_sigma)
        else:
            threshold = max(0.0, float(sigma_threshold) - 1.0) if low_conf else float(sigma_threshold)
            batch = detect_batch_matched_filter_gpu(
                magnitude,
                sigma_threshold=threshold,
                min_distance=min_distance,
                smoothing_sigma=smoothing_sigma,
                psf=psf,
                nms_radius=_nms_radius(min_distance),
                debias=bool(elev_debias),
            )
        return _with_confidence(batch, sigma_threshold) if low_conf else batch

    detect_fn = {
        "zscore": _detect_zscore,
        "cfar": _detect_cfar,
        "psf": _detect_psf,
    }[detector]

    # ---- TGC: compute once, cache inv_sqrt.npy on the volume (resumable). ----
    inv_path = os.path.join(det_dir, "inv_sqrt.npy")
    inv_sqrt = None
    if spatial_tgc:
        if os.path.exists(inv_path):
            inv_sqrt = np.load(inv_path)
            print("[tgc] loaded cached inv_sqrt", flush=True)
        else:
            from scipy.ndimage import gaussian_filter

            from ultratrace_ulm.gpu_svd import filter_svd_3d_gpu
            idx = np.unique(np.linspace(0, len(sel) - 1, min(tgc_acqs, len(sel))).round().astype(int))
            pd_sum, ref_grid = None, None
            for aid in [sel[i] for i in idx]:
                iq, txd, txde = _read(aid)
                comp, ref_grid = beamform_iq(iq, txd, txde, config, stream_accumulate=True)
                out = filter_svd_3d_gpu(comp, low_cutoff=tgc_svd_cut, method="fast")
                pd = (np.abs(out) ** 2).mean(0)
                pd_sum = pd if pd_sum is None else pd_sum + pd
                del comp, out
            pd_mean = pd_sum / len(idx)
            z_ax, y_ax, x_ax = ref_grid.z[:, 0, 0], ref_grid.y[0, :, 0], ref_grid.x[0, 0, :]
            sm = tgc_sigma_lambda * (1540.0 / config.tx_freq_hz)
            dz = abs(z_ax[1] - z_ax[0]) if len(z_ax) > 1 else None
            dx = abs(x_ax[1] - x_ax[0]) if len(x_ax) > 1 else None
            dy = abs(y_ax[1] - y_ax[0]) if len(y_ax) > 1 else None
            sig = (sm / dy if (dy and dy > 0) else 0.0, sm / dz if dz else 0.0, sm / dx if dx else 0.0)
            tgc = gaussian_filter(pd_mean, sigma=sig).astype(np.float32)
            inv_sqrt = (1.0 / np.sqrt(np.maximum(tgc, np.finfo(np.float32).eps)))[None]
            np.save(inv_path, inv_sqrt)
            vol.commit()

    timings = []
    try:
        for order, aid in enumerate(sel):
            out_npz = os.path.join(det_dir, f"acq_{order:04d}.npz")
            if os.path.exists(out_npz):
                continue
            t0 = time.time(); iq, txd, txde = _read(aid); t_read = time.time() - t0
            t0 = time.time()
            comp, grid = beamform_iq(iq, txd, txde, config, stream_accumulate=True)
            if inv_sqrt is not None:
                comp = (comp * inv_sqrt).astype(np.complex64)
            if motion:
                from ultratrace_ulm.gpu_motion import correct_motion_gpu
                comp, _ = correct_motion_gpu(comp)
            t_bf = time.time() - t0
            gx = (np.transpose(grid.x, (1, 0, 2)) * 1000.0).astype(np.float32)
            gy = (np.transpose(grid.y, (1, 0, 2)) * 1000.0).astype(np.float32)
            gz = (np.transpose(grid.z, (1, 0, 2)) * 1000.0).astype(np.float32)
            t0 = time.time()
            d = detect_localize_acq(comp, gx, gy, gz, opts, filter_fn=gpu_filter, detect_fn=detect_fn)
            t_det = time.time() - t0
            if not os.path.exists(os.path.join(det_dir, "meta.json")):
                meta = {"spacing": _grid_spacing(gx, gy, gz), "frames_per_acq": int(d["n_frames"]),
                        "frame_rate_hz": frame_rate_hz, "motion": motion,
                        "filter_variant": filter_variant, "svd_method": svd_method}
                if (
                    detector != "zscore" or use_svd_rank or int(nms_elev) > 0
                    or bool(elev_debias) or bool(low_conf)
                ):
                    meta.update({
                        "detector": detector,
                        "svd_rank": bool(use_svd_rank),
                        "rank_method": rank_method if use_svd_rank else "",
                        "rank_delta": float(rank_delta),
                        "nms_elev": int(nms_elev),
                        "elev_debias": bool(elev_debias),
                        "low_conf": bool(low_conf),
                    })
                with open(os.path.join(det_dir, "meta.json"), "w") as fh:
                    json.dump(meta, fh)
            tmp = out_npz + ".tmp"
            arrays = {
                "positions_mm": d["positions_mm"],
                "intensities": d["intensities"],
                "frame_in_acq": d["frame_in_acq"],
                "n_frames": d["n_frames"],
                "src_acq_id": int(aid),
            }
            if low_conf:
                arrays["confidence"] = d.get("confidence", np.empty(0, dtype=np.uint8))
            with open(tmp, "wb") as fh:
                np.savez(fh, **arrays)
            os.replace(tmp, out_npz)
            if order in keep:
                with h5py.File(os.path.join(ref_dir, f"acq_{order:04d}.h5"), "w") as o:
                    m = o.require_group(f"acquisitions/{order}/meta")
                    m.create_dataset("compound_image", data=comp, chunks=(1,) + tuple(comp.shape[1:]), compression="lzf")
                    gg = m.require_group("grid")
                    for ax, arr in (("x", grid.x), ("y", grid.y), ("z", grid.z)):
                        gg.create_dataset(ax, data=arr.astype(np.float64), compression="lzf")
            vol.commit()
            timings.append([round(t_read, 1), round(t_bf, 1), round(t_det, 1)])
            print(f"[detect] order={order} acq={aid} read={t_read:.1f}s beamform+motion={t_bf:.1f}s "
                  f"filter+detect={t_det:.1f}s ndet={len(d['positions_mm'])}", flush=True)
    finally:
        h5.close()
        if fobj is not None:
            fobj.close()
    done = len([f for f in os.listdir(det_dir) if f.startswith("acq_") and f.endswith(".npz")])
    tarr = np.array(timings) if timings else np.zeros((1, 3))
    report = {"tag": tag, "source": source, "selected": len(sel), "checkpoints_total": done,
              "this_run_processed": len(timings),
              "median_s": {"read": float(np.median(tarr[:, 0])), "beamform_motion": float(np.median(tarr[:, 1])),
                           "filter_detect": float(np.median(tarr[:, 2]))}}
    if detector != "zscore" or use_svd_rank or int(nms_elev) > 0 or bool(elev_debias) or bool(low_conf):
        report.update({"detector": detector, "svd_rank": bool(use_svd_rank), "rank_delta": float(rank_delta),
                       "nms_elev": int(nms_elev), "elev_debias": bool(elev_debias),
                       "low_conf": bool(low_conf)})
    print(json.dumps(report, indent=2))
    return report


# --------------------------------------------------------------------------- #
# track_acqs (Phase B) — CPU, cheap. Track from ALL per-acq detection
# checkpoints for a tag (additive: re-run after adding more acqs). Produces the
# same tracks/bins as the fused baseline, but the expensive part can never be
# lost to a timeout.
# --------------------------------------------------------------------------- #
@app.function(image=cpu_image, timeout=4 * 3600, memory=49152, volumes={"/root/data": vol})
def track_acqs(tag: str = "baseline", min_track_length: int = 5,
               gate_on_prediction: bool = False, out_tag: str = "") -> dict:
    import json
    import os
    import sys
    from pathlib import Path

    import numpy as np

    sys.path.insert(0, "/workspace")
    from ultratrace_ulm.tracking import (
        TrackingOptions, _smooth_and_export, track_from_acq_detections,
    )

    vol.reload()
    det_dir = _guard(f"{DATA_ROOT}/detections/{tag}")
    meta = json.load(open(os.path.join(det_dir, "meta.json")))
    files = sorted(f for f in os.listdir(det_dir) if f.startswith("acq_") and f.endswith(".npz"))
    per_acq = []
    for f in files:
        z = np.load(os.path.join(det_dir, f))
        item = {k: z[k] for k in ("positions_mm", "intensities", "frame_in_acq", "n_frames")}
        if "confidence" in z:
            item["confidence"] = z["confidence"]
        per_acq.append(item)
    out_tag = out_tag or tag
    out_dir = Path(_guard(f"{DATA_ROOT}/tracks/{out_tag}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    opts = TrackingOptions(
        beamformed_path=Path(f"{DATA_ROOT}/none"), tracks_path=out_dir / "tracks.pkl",
        tracking="kalman", frame_rate_hz=meta["frame_rate_hz"], max_gap=3,
        min_track_length=min_track_length, reversal_penalty=10.0, max_cost=10.0,
        gate_on_prediction=gate_on_prediction, smooth_sigma=2.0, smooth_method="gaussian",
        export_dir=out_dir, export_stem="tracks", export_min_lengths=(5, 20, 50))
    tracks_pkl = track_from_acq_detections(per_acq, opts, meta["spacing"], opts.tracks_path,
                                           extra={"from_detections": tag, **meta})
    smoothed = _smooth_and_export(opts, tracks_pkl)
    vol.commit()
    n = len(per_acq)
    print(f"DONE track_acqs {tag}: {n} acqs -> {out_dir}")
    return {"out_tag": out_tag, "n_acqs": n, "smoothed": str(smoothed)}


# --------------------------------------------------------------------------- #
# retrack — CPU, ~free. Re-run tracking on a baseline run's STORED detections
# with an insight's tracking-stage change (e.g. predicted-state Kalman gate),
# without re-beamforming. The cheap-insight engine for all tracking-stage diffs.
# --------------------------------------------------------------------------- #
@app.function(image=cpu_image, timeout=3600, memory=49152, volumes={"/root/data": vol})
def retrack(baseline_tag: str = "baseline", out_tag: str = "kalman_pred",
            gate_on_prediction: bool = True, min_track_length: int = 5) -> dict:
    import sys
    from pathlib import Path

    sys.path.insert(0, "/workspace")
    from ultratrace_ulm.runtime import dump_pickle, load_pickle
    from ultratrace_ulm.tracking import (
        TrackingOptions, _track_detections, export_tracks_bin, smooth_tracks_pickle,
    )

    vol.reload()
    src = Path(f"{DATA_ROOT}/tracks/{baseline_tag}/tracks.pkl")
    data = load_pickle(src)
    det = data.get("detections_by_frame")
    inten = data.get("intensities_by_frame")
    if det is None:
        raise SystemExit(f"{src} has no detections_by_frame (re-run baseline keeps them)")
    spacing = data.get("spacing") or {}
    p = data.get("params", {})
    md = p.get("max_distance_mm")
    opts = TrackingOptions(
        beamformed_path=Path(f"{DATA_ROOT}/none"),
        tracks_path=Path(f"{DATA_ROOT}/tracks/{out_tag}/tracks.pkl"),
        tracking=p.get("tracking_method", "kalman"),
        max_dist=tuple(float(v) for v in md) if md else None,
        frame_rate_hz=p.get("frame_rate_hz"), max_gap=int(p.get("max_gap", 3)),
        min_track_length=min_track_length, reversal_penalty=10.0,
        max_cost=float(p.get("max_cost", 10.0)), gate_on_prediction=gate_on_prediction,
        smooth_sigma=2.0, smooth_method="gaussian", export_min_lengths=(5, 20, 50),
    )
    out_dir = Path(_guard(f"{DATA_ROOT}/tracks/{out_tag}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    tracks = _track_detections(det, inten, opts, spacing)
    out = dict(data)
    out["tracks"] = tracks
    out.setdefault("params", {})
    out["params"] = {**p, "retrack_from": baseline_tag, "gate_on_prediction": gate_on_prediction}
    dump_pickle(out, opts.tracks_path)
    smoothed = smooth_tracks_pickle(opts.tracks_path, None, sigma=2.0, method="gaussian")
    for ml in (5, 20, 50):
        export_tracks_bin(smoothed, out_dir / f"tracks_min{ml}.bin", min_length=ml)
    vol.commit()
    print(f"DONE retrack {baseline_tag}->{out_tag}: {len(tracks)} tracks")
    return {"out_tag": out_tag, "n_tracks": len(tracks), "gate_on_prediction": gate_on_prediction}


# --------------------------------------------------------------------------- #
# stitch — CPU, cheap. Post-hoc stitch a saved track pickle and export viewer
# bins so stitched vs baseline can be compared without re-running detection.
# --------------------------------------------------------------------------- #
@app.function(image=cpu_image, timeout=1800, memory=16384, volumes={"/root/data": vol})
def stitch(tag: str = "baseline", out_tag: str = "", max_gap: int = 12,
           tol_lateral_mm: float = 0.6, tol_elev_mm: float = 2.0,
           vel_cos_min: float = 0.3, intensity_log_tol: float = 1.5,
           max_cost: float = 1.0) -> dict:
    import sys
    from pathlib import Path

    sys.path.insert(0, "/workspace")
    from ultratrace_ulm.runtime import dump_pickle, load_pickle
    from ultratrace_ulm.track_stitch import stitch_pickle_data
    from ultratrace_ulm.tracking import export_tracks_bin

    vol.reload()
    _ensure_root()
    out_tag = out_tag or f"{tag}_stitched"
    src = Path(_guard(f"{DATA_ROOT}/tracks/{tag}/tracks.pkl"))
    out_dir = Path(_guard(f"{DATA_ROOT}/tracks/{out_tag}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tracks.pkl"
    data = load_pickle(src)
    stitched = stitch_pickle_data(
        data,
        max_gap=max_gap,
        tol_lateral_mm=tol_lateral_mm,
        tol_elev_mm=tol_elev_mm,
        vel_cos_min=vel_cos_min,
        intensity_log_tol=intensity_log_tol,
        max_cost=max_cost,
    )
    dump_pickle(stitched, out_path)
    exports = []
    for min_length in (5, 20, 50):
        exported = export_tracks_bin(out_path, out_dir / f"tracks_min{min_length}.bin", min_length=min_length)
        if exported is not None:
            exports.append(str(exported))
    vol.commit()
    n_in = len(data.get("tracks_smoothed") or data.get("tracks") or [])
    n_out = len(stitched.get("tracks_smoothed") or stitched.get("tracks") or [])
    print(f"DONE stitch {tag}->{out_tag}: {n_in} -> {n_out} tracks")
    return {"tag": tag, "out_tag": out_tag, "tracks_path": str(out_path),
            "n_tracks_in": n_in, "n_tracks_out": n_out, "exports": exports}


# --------------------------------------------------------------------------- #
# volume3d — GPU. Reduce reference beamformed shards to small 3D SVD power
# volumes (time-mean |filtered|^2), ~8.5MB each, that CAN be pulled local for
# browser/MIP viewing and baseline-vs-insight diffing. The DIFFVIZ backbone.
# --------------------------------------------------------------------------- #
@app.function(image=gpu_image, gpu="A10G", timeout=3600, memory=65536,
              volumes={"/root/data": vol})
def volume3d(refs_dir: str = "baseline_refs", svd_method: str = "adaptive",
             frame_rate_hz: float = 222.0, low_cutoff: float = 0.1,
             filter_mode: str = "global", motion: bool = False,
             n_z_blocks: int = 3, n_x_blocks: int = 3, tag: str = "baseline") -> dict:
    import os
    import sys

    import h5py
    import numpy as np

    sys.path.insert(0, "/workspace")
    import cupy as cp

    from ultratrace_ulm.gpu_svd import filter_svd_3d_gpu

    vol.reload()
    src = f"{DATA_ROOT}/beamformed/{refs_dir}"
    out = _guard(f"{DATA_ROOT}/viz/{tag}")
    os.makedirs(out, exist_ok=True)
    arts = []
    for fn in sorted(f for f in os.listdir(src) if f.endswith(".h5")):
        with h5py.File(os.path.join(src, fn), "r") as f:
            order = sorted(f["acquisitions"].keys(), key=int)[0]
            comp = np.asarray(f[f"acquisitions/{order}/meta/compound_image"], dtype=np.complex64)
            gx = np.asarray(f[f"acquisitions/{order}/meta/grid/x"])  # (z, elev, x) meters
            gy = np.asarray(f[f"acquisitions/{order}/meta/grid/y"])
            gz = np.asarray(f[f"acquisitions/{order}/meta/grid/z"])
        if motion:
            from ultratrace_ulm.gpu_motion import correct_motion_gpu
            comp, _shifts = correct_motion_gpu(comp)
        if filter_mode == "region":
            from ultratrace_ulm.gpu_svd_region import filter_svd_3d_region_gpu
            filt = filter_svd_3d_region_gpu(
                comp, n_z_blocks=n_z_blocks, n_x_blocks=n_x_blocks, low_cutoff=low_cutoff,
                method=svd_method, frame_rate_hz=frame_rate_hz, tissue_freq_hz=100.0)
        else:
            filt = filter_svd_3d_gpu(comp, low_cutoff=low_cutoff, method=svd_method,
                                     frame_rate_hz=frame_rate_hz, tissue_freq_hz=100.0)
        power = (np.abs(filt) ** 2).mean(0).astype(np.float32)  # (elev, z, x), ~8.5MB
        base = fn[:-3]
        np.save(os.path.join(out, f"{base}_power.npy"), power)
        np.savez(os.path.join(out, f"{base}_axes.npz"),
                 z=(gz[:, 0, 0] * 1000).astype(np.float32),
                 elev=(gy[0, :, 0] * 1000).astype(np.float32),
                 x=(gx[0, 0, :] * 1000).astype(np.float32))
        arts.append(f"{base}_power.npy")
        vol.commit()
        del filt, power
        cp.get_default_memory_pool().free_all_blocks()
        print(f"[volume3d] {base}: power saved", flush=True)
    print(f"DONE volume3d: {len(arts)} volumes -> viz/{tag}")
    return {"out": f"{DATA_ROOT}/viz/{tag}", "svd_method": svd_method, "artifacts": arts}


@app.function(image=gpu_image, gpu="A10G", timeout=1800, memory=98304,
              volumes={"/root/data": vol})
def validate_svd(url: str = SAMPLE_URL, elev_planes: int = 25, frame_rate_hz: float = 222.0,
                 acq_index: int = 0, knee_high: bool = False) -> dict:
    """SVD cutoff probe on one real acq (mb-3k4): reports the cutoff each rule
    selects (legacy adaptive-centroid vs data-driven knee vs the fixed 10% floor),
    the singular-value spectrum head, and the detection count adaptive vs knee --
    the cheap logged validation the bake-off plan calls for.

    GPU-only (the CPU<->GPU adaptive equivalence was validated in mb-crr.2; the CPU
    full-SVD path is ~5 min and adds nothing to the knee question). Reads the acq
    from the local volume copy (HTTP range reads of ~1 GB IQ can be R2-truncated)."""
    import json
    import sys

    import numpy as np

    sys.path.insert(0, "/workspace")
    import cupy as cp

    from ultratrace_ulm.beamform_core import beamform_iq
    from ultratrace_ulm.gpu_svd import _spectral_centroid_cutoff_gpu, filtered_magnitude_gpu
    from ultratrace_ulm.svd import _component_count
    from ultratrace_ulm.svd_knee import select_svd_cutoffs, singular_value_knee
    from ultratrace_ulm.tracking import detect_batch

    _ensure_root()
    h5, fobj, source = _open_local_or_remote_h5(url)
    ids = sorted(int(k) for k in h5["acquisitions"].keys() if str(k).isdigit())
    config = _build_config(h5, elev_planes)
    g = h5[f"acquisitions/{ids[int(acq_index)]}"]
    comp, _ = beamform_iq(
        np.asarray(g["iq_frames"], dtype=np.complex64),
        np.asarray(g["tx_delays"], dtype=np.float64),
        np.asarray(g["tx_delays_elev"], dtype=np.float64),
        config, stream_accumulate=True,
    )
    h5.close()
    if fobj is not None:
        fobj.close()

    F = comp.shape[0]
    mat = cp.asarray(comp, dtype=cp.complex64).reshape(F, -1)
    n_vox = int(mat.shape[1])
    # raw Gram (the projection basis) for the spectrum + knee; mean-subtracted Gc
    # for the legacy adaptive-centroid cutoff -- both in one voxel-chunk pass.
    G = cp.zeros((F, F), dtype=cp.complex128)
    Gc = cp.zeros((F, F), dtype=cp.complex64)
    mc = xc = None
    for s0 in range(0, n_vox, 300_000):
        mc = mat[:, s0:s0 + 300_000]
        G += (mc @ mc.conj().T).astype(cp.complex128)
        xc = mc - mc.mean(axis=0, keepdims=True)
        Gc += xc @ xc.conj().T
    adaptive_low = int(_spectral_centroid_cutoff_gpu(Gc, F, frame_rate_hz, 100.0))
    evals = cp.asnumpy(cp.linalg.eigvalsh(G).real)
    del mat, G, Gc, mc, xc
    cp.get_default_memory_pool().free_all_blocks()

    svals = np.sqrt(np.maximum(np.sort(evals)[::-1], 0.0))
    ceiling = _component_count(0.1, F)  # the old floor, now used as a guard ceiling
    knee_low_guarded, knee_high_remove = select_svd_cutoffs(
        evals, F, n_vox, low_min=1, low_max=ceiling, high=bool(knee_high))
    knee_low_unguarded = singular_value_knee(svals, min_rank=1, max_rank=F - 1)

    adaptive_mag = filtered_magnitude_gpu(comp, method="adaptive", frame_rate_hz=frame_rate_hz,
                                          tissue_freq_hz=100.0)
    knee_mag = filtered_magnitude_gpu(comp, method="knee", low_cutoff=0.1, knee_high=bool(knee_high))
    adaptive_det = int(sum(len(p) for p, _, _ in detect_batch(adaptive_mag, 2.0, 2, 1.0)))
    knee_det = int(sum(len(p) for p, _, _ in detect_batch(knee_mag, 2.0, 2, 1.0)))
    report = {
        "source": source, "acq_index": int(acq_index), "n_frames": F, "n_voxels": n_vox,
        "adaptive_low": adaptive_low, "fixed_floor_low": ceiling,
        "knee_low": int(knee_low_guarded), "knee_low_unguarded": int(knee_low_unguarded),
        "knee_high_remove": int(knee_high_remove),
        "svals_head": [float(v) for v in svals[:25]],
        "svals_tail": [float(v) for v in svals[-5:]],
        "adaptive_detections": adaptive_det, "knee_detections": knee_det,
        "det_pct_change": round(100.0 * (knee_det - adaptive_det) / max(adaptive_det, 1), 1),
    }
    print(json.dumps(report, indent=2))
    return report


@app.local_entrypoint()
def main(fn: str = "inspect"):
    """Convenience: `modal run scripts/modal/app.py` runs inspect by default.

    Prefer explicit `modal run scripts/modal/app.py::<fn>` for inspect / probe /
    download / beamform_all / consolidate / track / baseline / detect_acqs /
    track_acqs / stitch.
    """
    table = {
        "inspect": inspect, "probe": probe, "download": download,
        "beamform_all": beamform_all, "consolidate": consolidate, "track": track,
        "baseline": baseline, "validate_svd": validate_svd, "volume3d": volume3d,
        "retrack": retrack, "detect_acqs": detect_acqs, "track_acqs": track_acqs,
        "stitch": stitch,
    }
    if fn not in table:
        raise SystemExit(f"unknown fn {fn!r}; choose from {sorted(table)}")
    print(table[fn].remote())
