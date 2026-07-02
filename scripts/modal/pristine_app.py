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


@app.local_entrypoint()
def main(fn: str = "pristine_run", num_acqs: int = 8):
    table = {"pristine_run": pristine_run, "pristine_beamform": pristine_beamform}
    print(table[fn].remote(num_acqs=num_acqs))
