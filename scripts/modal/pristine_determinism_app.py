"""End-to-end determinism check of the PRISTINE upstream CLI (mb-ivt / issue #2).

Runs `ultratrace-ulm track` twice, in two fresh subprocesses, on the SAME pristine
beamformed acquisition already on the volume (`pristine_bf_1.h5`), with the recipe from
the bug report:

    --svd-method adaptive --frame-rate 222 --sigma-threshold 2.0 --knee-filter
    --temporal-sigma 0

and compares the two runs' raw detection sets. Two pinned upstream revisions:

  prefix   6ed4116d  -- last commit before PR #4 (complex64 eigendecomposition)
  postfix  1939006   -- HEAD, PR #4 merged (complex128 eigendecomposition)

The second run is deliberately given a different BLAS thread count, which is the
in-practice source of the cross-run reduction-order jitter the issue describes.

CPU-only (beamforming is already done and is deterministic), so this is cheap.
"""

import subprocess

import modal

PROJECT = "microbubbles"
DATA_ROOT = f"/root/data/{PROJECT}"
PREFIX_SHA = "6ed4116d7b852dd9a20eb387f50e9516a4523046"
POSTFIX_SHA = "193900631ad7eabdce1e39b7f3ecae7d7d1c88ad"

vol = modal.Volume.from_name("research")


def _image(ref: str) -> modal.Image:
    return (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("git")
        .pip_install("h5py==3.11.0", "numpy<2.0", "scipy==1.13.1", "tqdm==4.66.4")
        .pip_install(
            f"ultratrace-ulm-pipeline @ git+https://github.com/alephneuro/microbubbles.git@{ref}",
            extra_options="--no-deps",
        )
    )


app = modal.App(f"{PROJECT}-pristine-determinism")

COMPARE = r'''
import pickle
import sys
import numpy as np

def load(path):
    with open(path, "rb") as fh:
        d = pickle.load(fh)
    det = d["detections"]
    idx = np.asarray(det["indices"])
    fr = np.asarray(det["frame_indices"]).astype(np.int64)
    pos = np.asarray(det["positions_mm"])
    return idx, fr, pos, d

def main(a_path, b_path):
    ia, fa, pa, da = load(a_path)
    ib, fb, pb, db = load(b_path)
    # voxel-resolution identity: (frame, rounded voxel index)
    sa = {(int(f),) + tuple(np.rint(r).astype(int)) for f, r in zip(fa, ia)}
    sb = {(int(f),) + tuple(np.rint(r).astype(int)) for f, r in zip(fb, ib)}
    inter = len(sa & sb)
    # median nearest-neighbour distance (mm), run B -> run A, within the same frame
    by_frame = {}
    for f, p in zip(fa, pa):
        by_frame.setdefault(int(f), []).append(p)
    by_frame = {k: np.asarray(v, np.float64) for k, v in by_frame.items()}
    nn = []
    for f, p in zip(fb, pb):
        ref = by_frame.get(int(f))
        if ref is None:
            continue
        nn.append(float(np.sqrt(((ref - p) ** 2).sum(axis=1)).min()))
    out = {
        "n_a": int(len(ia)), "n_b": int(len(ib)),
        "overlap_voxel_exact": inter / max(len(sa), len(sb)),
        "median_nn_mm": float(np.median(nn)) if nn else None,
        "p90_nn_mm": float(np.percentile(nn, 90)) if nn else None,
        "n_tracks_a": len(da.get("tracks", [])), "n_tracks_b": len(db.get("tracks", [])),
    }
    print("COMPARE " + repr(out), flush=True)

main(sys.argv[1], sys.argv[2])
'''


def _track(beamformed: str, out: str, threads: int) -> None:
    import os

    env = dict(os.environ)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        env[var] = str(threads)
    cmd = ["ultratrace-ulm", "track",
           "--beamformed", beamformed, "--tracks", out,
           "--svd-method", "adaptive", "--frame-rate", "222",
           "--sigma-threshold", "2.0", "--knee-filter", "--temporal-sigma", "0"]
    print(f"+ OMP_NUM_THREADS={threads} " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def _twice(beamformed: str, tag: str) -> dict:
    import shutil

    vol.reload()
    bf = f"{DATA_ROOT}/{beamformed}"
    a, b = f"/tmp/{tag}_run_a.pkl", f"/tmp/{tag}_run_b.pkl"
    _track(bf, a, threads=16)
    _track(bf, b, threads=1)
    script = "/tmp/compare.py"
    with open(script, "w") as fh:
        fh.write(COMPARE)
    res = subprocess.run(["python", script, a, b], check=True, capture_output=True, text=True)
    print(res.stdout, flush=True)
    line = [l for l in res.stdout.splitlines() if l.startswith("COMPARE ")][-1]
    rep = eval(line[len("COMPARE "):])
    rep["revision"] = tag
    for src, dst in ((a, f"{DATA_ROOT}/determinism_{tag}_a.pkl"),
                     (b, f"{DATA_ROOT}/determinism_{tag}_b.pkl")):
        shutil.copy2(src, dst)
    vol.commit()
    print(rep, flush=True)
    return rep


gpu_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .pip_install("h5py==3.11.0", "numpy<2.0", "scipy==1.13.1", "tqdm==4.66.4",
                 "torch==2.4.1", "cupy-cuda12x==13.3.0", "mach-beamform")
    .pip_install(
        f"ultratrace-ulm-pipeline @ git+https://github.com/alephneuro/microbubbles.git@{POSTFIX_SHA}",
        extra_options="--no-deps",
    )
)


@app.function(image=gpu_image, gpu="A100-80GB", timeout=3 * 3600, memory=262144, cpu=16.0,
              volumes={"/root/data": vol})
def beamform_twice(num_acqs: int = 1) -> dict:
    """Is the beamforming stage itself reproducible? The bug report assumed it was
    ('identical beamformed compounds across runs'), and the SVD-stage measurements can
    only account for ~6% of the reported disagreement -- so this checks the assumption
    directly: same command, same input, two runs, element-wise compare."""
    import h5py
    import numpy as np

    vol.reload()
    outs = []
    for tag in ("a", "b"):
        out = f"{DATA_ROOT}/determinism_bf_{tag}.h5"
        cmd = ["ultratrace-ulm", "beamform",
               "--input", f"{DATA_ROOT}/sanitized_neutral_ultratrace.h5",
               "--output", out, "--num-acqs", str(num_acqs), "--spatial-tgc"]
        print("+ " + " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        outs.append(out)
    vol.commit()

    with h5py.File(outs[0], "r") as fa, h5py.File(outs[1], "r") as fb:
        ka = sorted(fa["acquisitions"].keys(), key=int)[0]
        kb = sorted(fb["acquisitions"].keys(), key=int)[0]
        a = np.asarray(fa[f"acquisitions/{ka}/meta/compound_image"], dtype=np.complex64)
        b = np.asarray(fb[f"acquisitions/{kb}/meta/compound_image"], dtype=np.complex64)
    diff = np.abs(a - b)
    rep = {"shape": list(a.shape),
           "bit_identical": bool(np.array_equal(a, b)),
           "max_abs_diff": float(diff.max()),
           "mean_abs_diff": float(diff.mean()),
           "rel_mean_diff": float(diff.mean() / (np.abs(a).mean() + 1e-20))}
    print("BEAMFORM_COMPARE " + repr(rep), flush=True)
    return rep


@app.function(image=_image(PREFIX_SHA), timeout=3 * 3600, memory=131072, cpu=16.0,
              volumes={"/root/data": vol})
def prefix_twice(beamformed: str = "pristine_bf_1.h5") -> dict:
    return _twice(beamformed, "prefix")


@app.function(image=_image(POSTFIX_SHA), timeout=3 * 3600, memory=131072, cpu=16.0,
              volumes={"/root/data": vol})
def postfix_twice(beamformed: str = "pristine_bf_1.h5") -> dict:
    return _twice(beamformed, "postfix")
