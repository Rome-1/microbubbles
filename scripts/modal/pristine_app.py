"""Run the PRISTINE upstream braindump pipeline verbatim on Modal (no fork code).

Installs `ultratrace-ulm-pipeline` straight from github and shells out to its own
CLI (`ultratrace-ulm run` / `beamform` / `track`), reading the sample h5 already
on the `research` volume. This is the "run exactly what they shipped" test.

Memory note: their `beamform_mach` holds EVERY selected acq's ~11.9 GB compound in
RAM before writing (beamform_mach.py:119-158), so num_acqs is RAM-bound: ~num*11.9
+ ~120 GB peak per beamform_iq. 8 acqs ~= 215 GB -> needs a big-RAM instance.
"""
import subprocess
import modal

PROJECT = "microbubbles"
DATA_ROOT = f"/root/data/{PROJECT}"
LOCAL_H5 = f"{DATA_ROOT}/sanitized_neutral_ultratrace.h5"

vol = modal.Volume.from_name("research")

# Pinned deps that match our working fork image, then the PRISTINE package with
# --no-deps so it doesn't unpin numpy/scipy.
pristine_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .pip_install(
        "h5py==3.11.0", "numpy<2.0", "scipy==1.13.1", "tqdm==4.66.4",
        "torch==2.4.1", "cupy-cuda12x==13.3.0", "mach-beamform",
    )
    .pip_install(
        "ultratrace-ulm-pipeline @ git+https://github.com/alephneuro/braindump.git",
        extra_options="--no-deps",
    )
)

app = modal.App(f"{PROJECT}-pristine")


def _run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


@app.function(image=pristine_image, gpu="A100-80GB", timeout=4 * 3600,
              memory=262144, cpu=16.0, volumes={"/root/data": vol})
def pristine_run(num_acqs: int = 8, spatial_tgc: bool = True, frame_rate: float = 222.0,
                 min_length: int = 5, work: str = "pristine_out") -> dict:
    """Verbatim `ultratrace-ulm run`: beamform -> track -> track-viewer, their code."""
    import os
    vol.reload()
    workdir = f"{DATA_ROOT}/{work}"
    os.makedirs(workdir, exist_ok=True)
    cmd = ["ultratrace-ulm", "run",
           "--input", LOCAL_H5,
           "--work-dir", workdir,
           "--num-acqs", str(num_acqs),
           "--frame-rate", str(frame_rate),
           "--svd-method", "adaptive",
           "--min-length", str(min_length)]
    if spatial_tgc:
        cmd.append("--spatial-tgc")
    _run(cmd)
    vol.commit()
    arts = []
    for root, _dirs, files in os.walk(workdir):
        for f in files:
            arts.append(os.path.relpath(os.path.join(root, f), workdir))
    print("ARTIFACTS:", arts, flush=True)
    return {"work": workdir, "num_acqs": num_acqs, "artifacts": arts}


@app.function(image=pristine_image, gpu="A100-80GB", timeout=2 * 3600,
              memory=262144, cpu=16.0, volumes={"/root/data": vol})
def pristine_beamform(num_acqs: int = 8, spatial_tgc: bool = True,
                      out: str = "pristine_bf.h5") -> dict:
    """Verbatim `ultratrace-ulm beamform` only (for beamform-exactness compare)."""
    import os
    vol.reload()
    outp = f"{DATA_ROOT}/{out}"
    cmd = ["ultratrace-ulm", "beamform", "--input", LOCAL_H5, "--output", outp,
           "--num-acqs", str(num_acqs)]
    if spatial_tgc:
        cmd.append("--spatial-tgc")
    _run(cmd)
    vol.commit()
    return {"out": outp, "num_acqs": num_acqs}


@app.function(image=pristine_image, timeout=2 * 3600, memory=65536, cpu=16.0,
              volumes={"/root/data": vol})
def compare_beamform(pristine_h5: str = "pristine_out/beamformed.h5",
                     fork_shard: str = "beamformed/shards/acq_0000.h5", order: int = 0) -> dict:
    """Element-wise compare pristine vs fork beamformed compound for one acq."""
    import h5py, numpy as np
    vol.reload()
    with h5py.File(f"{DATA_ROOT}/{pristine_h5}", "r") as p:
        pc = np.asarray(p[f"acquisitions/{order}/meta/compound_image"], dtype=np.complex64)
    with h5py.File(f"{DATA_ROOT}/{fork_shard}", "r") as f:
        k = sorted(f["acquisitions"].keys(), key=int)[0]
        fc = np.asarray(f[f"acquisitions/{k}/meta/compound_image"], dtype=np.complex64)
    rep = {"pristine_shape": list(pc.shape), "fork_shape": list(fc.shape)}
    if pc.shape != fc.shape:
        rep["note"] = "shape mismatch"; print(rep); return rep
    diff = np.abs(pc - fc)
    denom = np.abs(pc).mean() + 1e-20
    rep.update({
        "max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean()),
        "rel_mean_diff": float(diff.mean() / denom),
        "pristine_mean_abs": float(np.abs(pc).mean()), "fork_mean_abs": float(np.abs(fc).mean()),
        "allclose_rtol1e-4": bool(np.allclose(pc, fc, rtol=1e-4, atol=1e-3)),
        "bit_identical": bool(np.array_equal(pc, fc)),
    })
    print(rep); return rep


@app.function(image=pristine_image, timeout=4 * 3600, memory=98304, cpu=16.0,
              volumes={"/root/data": vol})
def pristine_track(beamformed: str = "pristine_out/beamformed.h5", frame_rate: float = 222.0,
                   min_length: int = 5, work: str = "pristine_out") -> dict:
    """Verbatim pristine `track` (CPU c64) + `track-viewer` on an existing beamformed h5."""
    import os
    vol.reload()
    workdir = f"{DATA_ROOT}/{work}"
    tracks = f"{workdir}/tracks.pkl"
    _run(["ultratrace-ulm", "track", "--beamformed", f"{DATA_ROOT}/{beamformed}",
          "--tracks", tracks, "--svd-method", "adaptive", "--frame-rate", str(frame_rate),
          "--knee-filter", "--temporal-sigma", "0", "--sigma-threshold", "2.0",
          "--svd-low-cutoff", "0.1", "--min-distance", "2", "--smoothing-sigma", "1.0",
          "--tracking", "kalman", "--max-gap", "3", "--min-track-length", "5",
          "--max-cost", "10", "--export-dir", workdir, "--min-lengths", "5", "20", "50"])
    vol.commit()
    _run(["ultratrace-ulm", "track-viewer", "--tracks", f"{workdir}/tracks_smoothed.pkl",
          "--output-dir", f"{workdir}/viewer", "--min-length", str(min_length)])
    vol.commit()
    arts = sorted(os.listdir(workdir))
    print("ARTIFACTS:", arts); return {"work": workdir, "artifacts": arts}


@app.local_entrypoint()
def main(fn: str = "pristine_run", num_acqs: int = 8):
    table = {"pristine_run": pristine_run, "pristine_beamform": pristine_beamform,
             "compare_beamform": compare_beamform, "pristine_track": pristine_track}
    print(table[fn].remote(num_acqs=num_acqs) if fn in ("pristine_run", "pristine_beamform")
          else table[fn].remote())
