"""Dataset-scale recreation on the corrected sample, in storage-bounded batches (mb-ivt).

Single-acquisition recreation already lands (`recreate_app.py`): the pristine pipeline on the
corrected data reproduces the reference grid exactly (25, 154, 275 at 0.2004/0.5547/0.2008 mm)
and its acq-0 detections at 57.8% same-frame recall / 63.4% precision, median NN 0.142 mm --
against ~0% chance. This extends that to a sample of acquisitions so the recreation can be
scored per-acq against the reference's own per-acq output, which we hold for all 216
(`outputs/reference/dissolved/`).

Storage discipline: a corrected acquisition beamforms to 1.89 GiB (vs 11.9 GB on the retracted
export), but 216 of them is still ~408 GiB on the volume. So this works in batches -- beamform
a batch, track it, keep only the detections + tracks, delete the beamformed h5, move on. Peak
volume usage stays at one batch.
"""

import os
import subprocess

import modal

PROJECT = "microbubbles"
DATA_ROOT = f"/root/data/{PROJECT}"
H5 = f"{DATA_ROOT}/sanitized_neutral_ultratrace_216.h5"
POSTFIX_SHA = "193900631ad7eabdce1e39b7f3ecae7d7d1c88ad"

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
)

app = modal.App(f"{PROJECT}-recreate-batch")


def _run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


@app.function(image=gpu_image, gpu="A100-80GB", timeout=12 * 3600, memory=262144, cpu=16.0,
              volumes={"/root/data": vol})
def recreate_batch(n_acqs: int = 24, batch: int = 8, acq_start: int = 0,
                   frame_rate: float = 222.43, out_dir: str = "corrected") -> dict:
    import json
    import pickle
    import time

    import numpy as np

    vol.reload()
    root = f"{DATA_ROOT}/{out_dir}"
    os.makedirs(root, exist_ok=True)
    summary = []
    t_start = time.time()

    for b0 in range(acq_start, acq_start + n_acqs, batch):
        n = min(batch, acq_start + n_acqs - b0)
        bf = f"{root}/_bf_{b0:04d}.h5"
        tracks = f"{root}/tracks_{b0:04d}.pkl"
        t0 = time.time()
        _run(["ultratrace-ulm", "beamform", "--input", H5, "--output", bf,
              "--acq-start", str(b0), "--num-acqs", str(n), "--spatial-tgc"])
        t_bf = time.time() - t0
        t0 = time.time()
        _run(["ultratrace-ulm", "track", "--beamformed", bf, "--tracks", tracks,
              "--svd-method", "adaptive", "--frame-rate", str(frame_rate),
              "--sigma-threshold", "2.0", "--knee-filter", "--temporal-sigma", "0"])
        t_tr = time.time() - t0

        with open(tracks, "rb") as fh:
            got = pickle.load(fh)
        det = got["detections"]
        np.savez_compressed(
            f"{root}/detections_{b0:04d}.npz",
            positions_mm=np.asarray(det["positions_mm"], np.float32),
            intensities=np.asarray(det["intensities"], np.float32),
            zscores=np.asarray(det["zscores"], np.float32),
            frame_indices=np.asarray(det["frame_indices"], np.int32),
            acq_indices=np.asarray(det["acq_indices"], np.int32),
        )
        lens = np.array([len(t["positions"]) for t in got.get("tracks", [])] or [0])
        rec = {"acq_start": b0, "n_acqs": n, "n_detections": int(len(det["positions_mm"])),
               "n_tracks": int(len(got.get("tracks", []))),
               "tracks_ge35": int((lens >= 35).sum()), "tracks_ge50": int((lens >= 50).sum()),
               "mean_len": round(float(lens.mean()), 1),
               "beamform_s": round(t_bf, 1), "track_s": round(t_tr, 1)}
        summary.append(rec)
        print(f"[batch {b0}] {rec}", flush=True)

        os.remove(bf)                      # keep peak volume usage at one batch
        for extra in (f"{root}/_bf_{b0:04d}_smoothed.pkl",):
            if os.path.exists(extra):
                os.remove(extra)
        vol.commit()

    tot_tracks = sum(r["n_tracks"] for r in summary)
    tot_35 = sum(r["tracks_ge35"] for r in summary)
    acqs = sum(r["n_acqs"] for r in summary)
    out = {"acqs": acqs, "total_tracks": tot_tracks, "total_ge35": tot_35,
           "tracks_per_acq": round(tot_tracks / max(1, acqs), 1),
           "ge35_per_acq": round(tot_35 / max(1, acqs), 2),
           "reference_ge35_total_dataset": 1421,
           "projected_ge35_at_216": int(round(tot_35 / max(1, acqs) * 216)),
           "wall_s": round(time.time() - t_start, 1), "batches": summary}
    with open(f"{root}/recreate_batch_report.json", "w") as fh:
        json.dump(out, fh, indent=2)
    vol.commit()
    print("BATCH_REPORT " + json.dumps({k: v for k, v in out.items() if k != "batches"},
                                       indent=2), flush=True)
    return out
