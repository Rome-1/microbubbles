"""Purity vs min_track_length at a given gate, on all 216 acquisitions.

The chain set is independent of the floor (the floor is applied only at emission), so one
tracking pass at min_len=2 yields every floor by counting survivors -- real and null alike.
Rerun whenever the gate changes: a looser gate makes longer chance chains, so the knee moves.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DET = REPO / "outputs" / "corrected_full"
FR = 222.4306816130359
GATE_MMS = (130.0, 247.0, 130.0)      # (x, elevation, z) -- elevation at the Aleph default
MAX_COST = 10.0


def _one(args):
    path, permute = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import numpy as np

    from ultratrace_ulm.tracking import kalman_tracking_3d

    d = np.load(path)
    pos = d["positions_mm"].astype(np.float64)
    fr = d["frame_indices"].astype(np.int64)
    if permute is not None:
        rng = np.random.default_rng(permute)
        for a in np.unique(d["acq_indices"]):
            m = d["acq_indices"] == a
            u = np.unique(fr[m])
            mp = dict(zip(u.tolist(), rng.permutation(u).tolist()))
            fr[m] = np.array([mp[int(v)] for v in fr[m]], np.int64)
    o = np.argsort(fr, kind="stable")
    pos, fr = pos[o], fr[o]
    by = [pos[fr == f] for f in range(int(fr.min()), int(fr.max()) + 1)]
    ints = [np.ones(len(b), np.float32) for b in by]
    tr = kalman_tracking_3d(by, max_distance_mm=tuple(v / FR for v in GATE_MMS), max_gap=3,
                            min_track_length=2, intensities=ints, max_cost=MAX_COST,
                            reversal_penalty=10.0)
    return Counter(len(t["positions"]) for t in tr)


def survivors(permute=None):
    paths = [(str(p), permute) for p in sorted(DET.glob("detections_*.npz"))]
    total = Counter()
    with ProcessPoolExecutor(max_workers=4) as ex:
        for c in ex.map(_one, paths):
            total.update(c)
    return total


if __name__ == "__main__":
    real, null = survivors(), survivors(permute=7)
    rows = []
    for L in list(range(2, 21)) + [25, 35]:
        r = sum(v for k, v in real.items() if k >= L)
        n = sum(v for k, v in null.items() if k >= L)
        rows.append({"L": L, "real": r, "null": n, "purity": round(1 - n / max(r, 1), 4)})
        print(f"L={L:>3}  real {r:>7,}  null {n:>6,}  purity {rows[-1]['purity']:.4f}", flush=True)
    out = REPO / "outputs" / "operating_point" / "purity_curve_130_247.json"
    out.write_text(json.dumps({"gate_mms": GATE_MMS, "rows": rows}, indent=2))
