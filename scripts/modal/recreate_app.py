"""Recreate the reference result from the CORRECTED sample, with upstream's own code (mb-ivt).

Everything the project has processed came from the export upstream retracted on 2026-07-11
(commit 23c49e57): a single-row acquisition with no elevation extent, 223 acqs x ~700 frames,
whose "SVD detections would not link into 3D tracks" -- upstream's words for the exact symptom
recorded in bead mb-bfl. The corrected 8-row re-export is 216 acqs x 240 frames, matching the
reference artifact's own `n_acquisitions: 216` / `frames_per_acq: 240`, and is now on the
volume as `sanitized_neutral_ultratrace_216.h5` (95.71 GiB, verified).

This runs the PRISTINE upstream pipeline (HEAD, post-#4) on the corrected data with the recipe
from the bug report, then scores the detections it produces against the reference's released
acq-0 detections -- the same scoring that returned chance on the retracted export:

    same-frame recall   1.2%   (chance ~0.5-1%)
    frame-agnostic     70.1%   (chance   70.0%)

If the diagnosis is right, these move decisively above chance here. If they do not, the
released artifact is not reachable from the public inputs by the public code, and every
"reproduces the reference" claim in the repo needs restating.

Chance is computed explicitly for each comparison (1 - exp(-n*v/V)) rather than eyeballed --
on the retracted data the 70% figure looked like agreement and was pure coincidence.
"""

import subprocess

import modal

PROJECT = "microbubbles"
DATA_ROOT = f"/root/data/{PROJECT}"
H5 = f"{DATA_ROOT}/sanitized_neutral_ultratrace_216.h5"
POSTFIX_SHA = "193900631ad7eabdce1e39b7f3ecae7d7d1c88ad"
SCRATCH = ("/home/rome/tmp/claude-1000/-home-rome-gt-microbubbles-crew-cajal/"
           "d06d6857-7ce7-4afc-9f8d-3f4a28ac48e3/scratchpad/refdet")

vol = modal.Volume.from_name("research")

gpu_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .pip_install("h5py==3.11.0", "numpy<2.0", "scipy==1.13.1", "tqdm==4.66.4",
                 "torch==2.4.1", "cupy-cuda12x==13.3.0", "mach-beamform")
    .pip_install(
        f"ultratrace-ulm-pipeline @ git+https://github.com/alephneuro/microbubbles.git@{POSTFIX_SHA}",
        extra_options="--no-deps",
    )
    .add_local_dir(SCRATCH, remote_path="/workspace/refdet")
)

app = modal.App(f"{PROJECT}-recreate")


def _run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


@app.function(image=gpu_image, gpu="A100-80GB", timeout=6 * 3600, memory=262144, cpu=16.0,
              volumes={"/root/data": vol})
def recreate(num_acqs: int = 1, spatial_tgc: bool = True, frame_rate: float = 222.43,
             match_mm: float = 0.3, bf_out: str = "corrected_bf_1.h5",
             tracks_out: str = "corrected_tracks_1.pkl") -> dict:
    import json
    import pickle

    import h5py
    import numpy as np
    from scipy.spatial import cKDTree

    vol.reload()
    bf = f"{DATA_ROOT}/{bf_out}"
    tracks = f"{DATA_ROOT}/{tracks_out}"

    _run(["ultratrace-ulm", "beamform", "--input", H5, "--output", bf,
          "--num-acqs", str(num_acqs)] + (["--spatial-tgc"] if spatial_tgc else []))
    vol.commit()

    with h5py.File(bf, "r") as h5:
        key = sorted(h5["acquisitions"].keys(), key=int)[0]
        grp = h5[f"acquisitions/{key}"]
        comp = grp["meta/compound_image"] if "meta" in grp else None
        shape = list(comp.shape) if comp is not None else None
    print(f"[beamform] acq {key} compound {shape}", flush=True)

    _run(["ultratrace-ulm", "track", "--beamformed", bf, "--tracks", tracks,
          "--svd-method", "adaptive", "--frame-rate", str(frame_rate),
          "--sigma-threshold", "2.0", "--knee-filter", "--temporal-sigma", "0"])
    vol.commit()

    with open(tracks, "rb") as fh:
        got = pickle.load(fh)
    det = got["detections"]
    P = np.asarray(det["positions_mm"], dtype=np.float64)
    F = np.asarray(det["frame_indices"], dtype=np.int64)
    F = F - F.min()
    n_frames = int(F.max()) + 1

    ref = np.load("/workspace/refdet/acq0_reference_detections.npz")
    R = ref["positions_mm"].astype(np.float64)
    RF = ref["frame_indices"].astype(np.int64)
    RF = RF - RF.min()

    bbox = float(np.prod(R.max(0) - R.min(0)))
    sphere = 4.0 / 3.0 * np.pi * match_mm ** 3
    chance = lambda n: float(1.0 - np.exp(-n * sphere / bbox))

    tree = cKDTree(P)
    d_any, _ = tree.query(R)
    same_hit = np.zeros(len(R), bool)
    ours_hit = np.zeros(len(P), bool)
    per_frame_n = []
    for fr in range(min(n_frames, int(RF.max()) + 1)):
        rm, om = RF == fr, F == fr
        if not rm.any() or not om.any():
            continue
        per_frame_n.append(int(om.sum()))
        d, _ = cKDTree(P[om]).query(R[rm])
        same_hit[np.where(rm)[0]] = d <= match_mm
        d2, _ = cKDTree(R[rm]).query(P[om])
        ours_hit[np.where(om)[0]] = d2 <= match_mm

    counts = np.bincount(F, minlength=n_frames)
    out = {
        "n_acqs": num_acqs, "compound_shape": shape,
        "grid_shape": [int(v) for v in np.asarray(got["grid_x"]).shape],
        "spacing": {k: round(float(v), 4) for k, v in got["spacing"].items()},
        "n_ours": int(len(P)), "n_reference": int(len(R)),
        "our_frames": n_frames, "reference_frames": int(RF.max()) + 1,
        "our_dets_per_frame": round(float(counts.mean()), 1),
        "our_occupancy": round(float((counts > 0).mean()), 3),
        "reference_dets_per_frame": round(float(len(R) / (RF.max() + 1)), 1),
        "same_frame_recall": round(float(same_hit.mean()), 3),
        "same_frame_chance": round(chance(float(np.mean(per_frame_n or [0]))), 3),
        "same_frame_precision": round(float(ours_hit.mean()), 3),
        "frame_agnostic_recall": round(float((d_any <= match_mm).mean()), 3),
        "frame_agnostic_chance": round(chance(len(P)), 3),
        "median_nn_mm": round(float(np.median(d_any)), 3),
        "n_tracks": len(got.get("tracks", [])),
    }
    with open(f"{DATA_ROOT}/recreate_report.json", "w") as fh:
        json.dump(out, fh, indent=2)
    vol.commit()
    print("RECREATE " + json.dumps(out, indent=2), flush=True)
    return out
