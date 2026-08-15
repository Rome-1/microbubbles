"""Emit tracks at the settled operating point (130/130 mm/s gate, min_track_length 8).

`apply_operating_point.py` measured the operating point but kept only summary statistics.
The viewer needs the trajectories themselves, so this re-runs the proposed arm and writes
per-batch pickles in the same shape the exporter already understands.

max_cost is passed explicitly: kalman_tracking_3d defaults to 1e5 while production uses 10.0.
"""
from __future__ import annotations

import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DET_DIR = REPO / "outputs" / "corrected_full"
OUT_DIR = REPO / "outputs" / "operating_point" / "tracks"
FRAME_RATE = 222.4306816130359
GATE_MMS = (130.0, 246.76, 130.0)        # (x, elevation, z) -- elevation at the Aleph 2-voxel default
MIN_LEN = 10             # re-derived at this gate (95% purity knee)
MAX_COST = 10.0


def _one(path: str):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import numpy as np

    from ultratrace_ulm.tracking import kalman_tracking_3d

    d = np.load(path)
    pos = d["positions_mm"].astype(np.float64)
    frames = d["frame_indices"].astype(np.int64)
    order = np.argsort(frames, kind="stable")
    pos, frames = pos[order], frames[order]
    by_frame, ints = [], []
    for f in range(int(frames.min()), int(frames.max()) + 1):
        m = frames == f
        by_frame.append(pos[m])
        ints.append(np.ones(int(m.sum()), np.float32))
    gate_mm = tuple(v / FRAME_RATE for v in GATE_MMS)
    tracks = kalman_tracking_3d(by_frame, max_distance_mm=gate_mm, max_gap=3,
                                min_track_length=MIN_LEN, intensities=ints,
                                max_cost=MAX_COST, reversal_penalty=10.0)
    out = OUT_DIR / (Path(path).name.replace("detections_", "tracks_").replace(".npz", ".pkl"))
    with open(out, "wb") as fh:
        pickle.dump({"tracks": tracks}, fh)
    return out.name, len(tracks)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = [str(p) for p in sorted(DET_DIR.glob("detections_*.npz"))]
    with ProcessPoolExecutor(max_workers=4) as ex:
        for name, n in ex.map(_one, paths):
            print(f"{name}: {n} tracks", flush=True)
