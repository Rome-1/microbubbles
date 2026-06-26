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
@app.function(image=gpu_image, gpu="A10G", timeout=1800, volumes={"/root/data": vol})
def probe(url: str = SAMPLE_URL, elev_planes: int = 25, n_acqs: int = 1) -> dict:
    import json
    import sys
    import time

    sys.path.insert(0, "/workspace")
    from ultratrace_ulm.beamform_core import MACH_AVAILABLE, NeutralConfig, beamform_iq
    import numpy as np

    _ensure_root()
    report: dict = {"url": url, "mach_available": bool(MACH_AVAILABLE)}
    if not MACH_AVAILABLE:
        report["error"] = "mach kernel not importable in gpu_image; beamforming blocked"
        out = _guard(f"{DATA_ROOT}/probe_report.json")
        with open(out, "w") as fh:
            json.dump(report, fh, indent=2)
        vol.commit()
        print(json.dumps(report, indent=2))
        return report

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

        times = []
        shape = None
        for aid in acq_ids[:n_acqs]:
            g = h5[f"acquisitions/{aid}"]
            iq = np.asarray(g["iq_frames"], dtype=np.complex64)
            txd = np.asarray(g["tx_delays"], dtype=np.float64)
            txd_elev = np.asarray(g["tx_delays_elev"], dtype=np.float64)
            t0 = time.time()
            compound, grid = beamform_iq(iq, txd, txd_elev, config)
            times.append(time.time() - t0)
            shape = tuple(int(s) for s in compound.shape)
        per_acq_s = float(np.median(times))
        report.update(
            {
                "n_acqs_timed": len(times),
                "compound_shape": shape,
                "per_acq_beamform_seconds": round(per_acq_s, 2),
                "full_223_acq_gpu_hours_est": round(per_acq_s * 223 / 3600, 2),
            }
        )
    finally:
        h5.close()
        fobj.close()

    out = _guard(f"{DATA_ROOT}/probe_report.json")
    with open(out, "w") as fh:
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


@app.local_entrypoint()
def main(fn: str = "inspect"):
    """Convenience: `modal run scripts/modal/app.py` runs inspect by default.

    Prefer explicit `modal run scripts/modal/app.py::inspect` / `::probe` / `::download`.
    """
    if fn == "inspect":
        print(inspect.remote())
    elif fn == "probe":
        print(probe.remote())
    elif fn == "download":
        print(download.remote())
    else:
        raise SystemExit(f"unknown fn {fn!r}")
