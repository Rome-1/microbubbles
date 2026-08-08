"""Full 216-acquisition recreation of the reference result (mb-ivt).

Rome approved $50 for this run. Measured cost basis from the 24-acq batch: 165 s/acq
(145 s beamform + 28 s track) on A100-80GB, so 216 acqs is ~9.9 GPU-hours ~ $37.

Design notes:

  TGC CONSISTENCY  `--spatial-tgc` computes a global gain map from the acquisitions in the
                   invocation, so sharding across parallel workers would give each shard a
                   different normalization and make the halves incomparable. This runs
                   sequentially in ONE container with a fixed `--tgc-acqs 12` per batch, so
                   every batch derives its map the same way. Slower than fanning out; the
                   point of the run is fidelity, not wall time.

  STORAGE          a corrected acquisition beamforms to 1.89 GiB, so all 216 at once would be
                   408 GiB on the shared volume. Each batch's beamformed h5 is deleted after
                   its detections and tracks are saved; peak usage stays at one batch (~23 GiB).

  RESUME           batches whose tracks pickle already exists are skipped, so a timeout or a
                   dropped container costs only the batch in flight. Re-invoke to continue.

  BUDGET GUARD     the run stops cleanly at `max_hours` and reports what it completed rather
                   than running past the approved spend.
"""

import os
import subprocess
import time

import modal

PROJECT = "microbubbles"
DATA_ROOT = f"/root/data/{PROJECT}"
H5 = f"{DATA_ROOT}/sanitized_neutral_ultratrace_216.h5"
POSTFIX_SHA = "193900631ad7eabdce1e39b7f3ecae7d7d1c88ad"   # PR #4 merge; pinned for the record

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

app = modal.App(f"{PROJECT}-recreate-full")


def _run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


@app.function(image=gpu_image, gpu="A100-80GB", timeout=13 * 3600, memory=262144, cpu=16.0,
              volumes={"/root/data": vol})
def recreate_full(n_acqs: int = 216, batch: int = 12, tgc_acqs: int = 12,
                  frame_rate: float = 222.43, max_hours: float = 11.0,
                  out_dir: str = "corrected_full") -> dict:
    import json
    import pickle

    import numpy as np

    vol.reload()
    root = f"{DATA_ROOT}/{out_dir}"
    os.makedirs(root, exist_ok=True)
    t_start = time.time()
    done, skipped = [], []

    for b0 in range(0, n_acqs, batch):
        n = min(batch, n_acqs - b0)
        tracks = f"{root}/tracks_{b0:04d}.pkl"
        if os.path.exists(tracks):
            print(f"[skip] batch {b0} already done", flush=True)
            skipped.append(b0)
            continue
        if (time.time() - t_start) / 3600.0 > max_hours:
            print(f"[budget] stopping before batch {b0}: {max_hours}h reached", flush=True)
            break

        bf = f"{root}/_bf_{b0:04d}.h5"
        t0 = time.time()
        _run(["ultratrace-ulm", "beamform", "--input", H5, "--output", bf,
              "--acq-start", str(b0), "--num-acqs", str(n),
              "--spatial-tgc", "--tgc-acqs", str(tgc_acqs)])
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
        rec = {"acq_start": b0, "n_acqs": n,
               "n_detections": int(len(det["positions_mm"])),
               "n_tracks": int(len(got.get("tracks", []))),
               "tracks_ge35": int((lens >= 35).sum()),
               "tracks_ge50": int((lens >= 50).sum()),
               "mean_len": round(float(lens.mean()), 1),
               "beamform_s": round(t_bf, 1), "track_s": round(t_tr, 1)}
        done.append(rec)
        print(f"[batch {b0}] {rec}", flush=True)

        os.remove(bf)                       # peak volume usage stays at one batch
        vol.commit()

    # ---- roll up every batch present on disk, including earlier invocations ----
    all_batches = []
    for path in sorted(f for f in os.listdir(root) if f.startswith("tracks_")):
        with open(f"{root}/{path}", "rb") as fh:
            got = pickle.load(fh)
        lens = np.array([len(t["positions"]) for t in got.get("tracks", [])] or [0])
        all_batches.append({"file": path, "n_tracks": int(len(got.get("tracks", []))),
                            "tracks_ge35": int((lens >= 35).sum()),
                            "tracks_ge50": int((lens >= 50).sum())})
    acqs_done = (len(skipped) + len(done)) * batch
    tot35 = sum(b["tracks_ge35"] for b in all_batches)
    out = {"acqs_processed": min(acqs_done, n_acqs),
           "batches_this_run": len(done), "batches_skipped": len(skipped),
           "total_tracks": sum(b["n_tracks"] for b in all_batches),
           "total_ge35": tot35,
           "total_ge50": sum(b["tracks_ge50"] for b in all_batches),
           "reference_ge35_dataset": 1421,
           "fraction_of_reference": round(tot35 / 1421.0, 3),
           "gpu_hours": round((time.time() - t_start) / 3600.0, 2),
           "batches": done}
    with open(f"{root}/recreate_full_report.json", "w") as fh:
        json.dump(out, fh, indent=2)
    vol.commit()
    print("FULL_REPORT " + json.dumps({k: v for k, v in out.items() if k != "batches"},
                                      indent=2), flush=True)
    return out
