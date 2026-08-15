"""Apply the settled operating point to all 216 acquisitions and price it honestly.

Two changes survived the J1-J5 program, both measured rather than chosen:

  GATE      the tracker's default box gate is 2 voxels/frame, which in physical units is
            89 mm/s in-plane and 247 mm/s in elevation -- 2.77x looser on the worst-localized
            axis. Per-step velocities across the 216-acq recreation hard-wall at exactly that
            value (89.2 / 89.3 / 246.7 observed vs 89.15 / 89.33 / 246.76 predicted), so the
            speed distribution was censored by our own default. Replaced with a physical,
            anisotropic 130 / 130 mm/s.

  FLOOR     `min_track_length` is applied ONLY at emission (tracking.py:343, :722, :729), so it
            recovers no associations -- it decides what gets published. A purity curve against
            a frame-permutation null puts the knee at 8: purity clears 95% there and flattens,
            while 8 -> 15 costs 3x the yield for 3.4 points. The shipped 15 buys no
            planted-crossing precision (flat 0.91-0.96 across the whole range).

Both are quoted together because they interact: at L=15 only 1.2% of the gate's link gain is
null-reproducible, but 11.8% at L=8. A gain figure without its floor is not meaningful.

Baseline here is our own 216-acq recreation of the reference (1,451 tracks >=35 vs the
reference's 1,421), so the comparison is like-for-like on identical detections.
"""

from __future__ import annotations

import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DET_DIR = REPO / "outputs" / "corrected_full"
OUT_DIR = REPO / "outputs" / "operating_point"
FRAME_RATE = 222.4306816130359
MAX_WORKERS = 4
MAX_COST = 10.0          # production TrackingOptions value, not the function default (1e5)

BASELINE = {"inplane": None, "elev": None, "min_len": 15}      # None => shipped voxel gate
PROPOSED = {"inplane": 130.0, "elev": 247.0, "min_len": 10}  # elevation at the Aleph default; floor re-derived for this gate


# `_tracking_gate` returns (dx, dy, dz) * 2 with dy = elevation, and `max_dist_mms` is
# divided by the frame rate, so the gate tuple is per-frame mm in (x, elevation, z) order.
SPACING = {"dx": 0.2004, "dy": 0.5547, "dz": 0.2008}


def _gate_mm(cfg):
    if cfg["inplane"] is None:
        return (SPACING["dx"] * 2.0, SPACING["dy"] * 2.0, SPACING["dz"] * 2.0)
    return (cfg["inplane"] / FRAME_RATE, cfg["elev"] / FRAME_RATE, cfg["inplane"] / FRAME_RATE)


def _track_one(args):
    """Track one batch under one config. Returns per-track lengths + linked count."""
    path, cfg, permute_seed = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import sys

    import numpy as np

    if str(REPO) not in sys.path:          # spawned workers do not inherit sys.path
        sys.path.insert(0, str(REPO))
    from ultratrace_ulm.tracking import kalman_tracking_3d

    d = np.load(path)
    pos = d["positions_mm"].astype(np.float64)
    frames = d["frame_indices"].astype(np.int64)
    if permute_seed is not None:
        # frame-permutation null: same detections, trajectories destroyed. This is a TRACKER
        # null (useless as a detector null -- it removes trajectories, not echoes).
        rng = np.random.default_rng(permute_seed)
        for a in np.unique(d["acq_indices"]):
            m = d["acq_indices"] == a
            f = frames[m]
            uniq = np.unique(f)
            perm = dict(zip(uniq.tolist(), rng.permutation(uniq).tolist()))
            frames[m] = np.array([perm[int(v)] for v in f], dtype=np.int64)
    order = np.argsort(frames, kind="stable")
    pos, frames = pos[order], frames[order]

    by_frame, ints_by_frame = [], []
    for f in range(int(frames.min()), int(frames.max()) + 1):
        m = frames == f
        by_frame.append(pos[m])
        ints_by_frame.append(np.ones(int(m.sum()), np.float32))

    # PRODUCTION-FAITHFUL: `kalman_tracking_3d`'s own max_cost default is 1e5, but the
    # production TrackingOptions uses 10.0. Calling the function directly with its defaults
    # silently runs a far more permissive assignment than the pipeline ships, which inflated
    # an earlier version of this comparison (baseline 1,890 tracks >=35 against the
    # recreation's 1,451) and made the numbers non-comparable to the reference.
    tracks = kalman_tracking_3d(by_frame, max_distance_mm=_gate_mm(cfg), max_gap=3,
                                min_track_length=int(cfg["min_len"]),
                                intensities=ints_by_frame,
                                max_cost=MAX_COST, reversal_penalty=10.0)
    lens = np.array([len(t["positions"]) for t in tracks] or [0])
    linked = int(sum(len(t["positions"]) for t in tracks))
    return {"n_tracks": int(len(tracks)), "n_ge35": int((lens >= 35).sum()),
            "n_ge50": int((lens >= 50).sum()), "linked": linked,
            "n_det": int(len(pos)), "median_len": float(np.median(lens))}


def run(cfg, permute_seed=None, label=""):
    paths = sorted(DET_DIR.glob("detections_*.npz"))
    jobs = [(str(p), cfg, permute_seed) for p in paths]
    rows = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for r in ex.map(_track_one, jobs):
            rows.append(r)
    agg = {k: int(sum(r[k] for r in rows)) for k in ("n_tracks", "n_ge35", "n_ge50", "linked", "n_det")}
    agg["linked_frac"] = round(agg["linked"] / max(1, agg["n_det"]), 4)
    agg["batches"] = len(rows)
    agg["label"] = label
    print(f"[{label}] {json.dumps(agg)}", flush=True)
    return agg


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = run(BASELINE, label="baseline (shipped gate, min_len 15)")
    prop = run(PROPOSED, label="proposed (130 in-plane / 247 elev mm/s, min_len 10)")
    base_null = run(BASELINE, permute_seed=7, label="baseline NULL")
    prop_null = run(PROPOSED, permute_seed=7, label="proposed NULL")

    def purity(real, null):
        return round(1.0 - null["n_tracks"] / max(1, real["n_tracks"]), 4)

    report = {
        "reference_ge35_dataset": 1421,
        "baseline": base, "proposed": prop,
        "baseline_null": base_null, "proposed_null": prop_null,
        "baseline_purity": purity(base, base_null),
        "proposed_purity": purity(prop, prop_null),
        "ge35_gain": prop["n_ge35"] - base["n_ge35"],
        "ge35_vs_reference": round(prop["n_ge35"] / 1421.0, 3),
        "linked_gain_pp": round(100 * (prop["linked_frac"] - base["linked_frac"]), 2),
    }
    (OUT_DIR / "operating_point.json").write_text(json.dumps(report, indent=2))
    print("REPORT " + json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
