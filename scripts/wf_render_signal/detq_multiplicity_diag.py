#!/usr/bin/env python3
"""Detection-quality diagnostics and post-processing for the 3D-ULM acq data."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


ROOT = Path("/home/rome/gt/microbubbles/crew/cajal")
SCRATCH = Path("/tmp/claude-1000/-home-rome-gt-microbubbles-crew-cajal/90d3a697-ad85-4c11-bdd8-ba646ff29fce/scratchpad")
DET_DIR = SCRATCH / "base60" / "base60"
OUT = SCRATCH / "detq"
REF_PICKLE = ROOT / "outputs" / "reference" / "full_tracks_smoothed.pkl"
VBOX = np.array([0.4008175182481752, 1.1093333333333333, 0.4015686274509804], dtype=np.float64)


SAFE = {
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "scalar"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
}


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        for mod in (
            module,
            module.replace("numpy._core", "numpy.core"),
            module.replace("numpy.core", "numpy._core"),
        ):
            if (mod, name) in SAFE:
                return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked pickle global {module}.{name}")


def load_our(acq: int) -> dict[str, np.ndarray]:
    with np.load(DET_DIR / f"acq_{acq:04d}.npz") as npz:
        return {k: npz[k] for k in npz.files}


def load_ref() -> dict:
    with REF_PICKLE.open("rb") as f:
        return SafeUnpickler(f).load()


def z_proxy(intensity: np.ndarray) -> np.ndarray:
    return 2.4687 * intensity.astype(np.float64) - 0.1831


def peak_window(frames: np.ndarray, n_frames_total: int | None, width: int = 240) -> tuple[int, int, np.ndarray]:
    max_frame = int(frames.max()) if len(frames) else 0
    n = max(max_frame + 1, int(n_frames_total or 0))
    counts = np.bincount(frames.astype(np.int64), minlength=n)
    if len(counts) <= width:
        return 0, len(counts), counts
    c = np.concatenate([[0], np.cumsum(counts)])
    sums = c[width:] - c[:-width]
    start = int(np.argmax(sums))
    return start, start + width, counts


def window_arrays(pos: np.ndarray, frame: np.ndarray, z: np.ndarray, start: int, end: int, rebase: bool = True):
    m = (frame >= start) & (frame < end)
    fr = frame[m].astype(np.int64)
    if rebase:
        fr = fr - start
    return pos[m].astype(np.float64), fr, z[m].astype(np.float64), m


def duplicate_stats(pos: np.ndarray, frame: np.ndarray, radii: tuple[float, ...] = (0.08, 0.12, 0.20, 0.30)) -> dict:
    rows = {}
    for r in radii:
        counts = []
        clustered = 0
        pairs = 0
        for f in np.unique(frame):
            p = pos[frame == f]
            if len(p) < 2:
                continue
            tree = cKDTree(p)
            neigh = tree.query_ball_tree(tree, r)
            c = np.array([len(v) - 1 for v in neigh], dtype=np.int64)
            counts.append(c)
            clustered += int(np.sum(c > 0))
            pairs += int(sum(len(v) - 1 for v in neigh) // 2)
        allc = np.concatenate(counts) if counts else np.zeros(0, dtype=np.int64)
        rows[f"r{r:.2f}"] = {
            "frac_with_same_frame_neighbor": float(np.mean(allc > 0)) if len(allc) else 0.0,
            "mean_neighbors": float(np.mean(allc)) if len(allc) else 0.0,
            "max_neighbors": int(allc.max()) if len(allc) else 0,
            "pairs": int(pairs),
            "clustered_detections": int(clustered),
        }
    return rows


def next_frame_stats(pos: np.ndarray, frame: np.ndarray, max_gap: int = 1) -> dict:
    scaled = pos / VBOX
    nn_gate = []
    nn_phys = []
    mult_gate = []
    n_with = 0
    n_total = 0
    for f in range(int(frame.min()), int(frame.max()) + 1 - max_gap):
        a = np.where(frame == f)[0]
        b = np.where(frame == f + max_gap)[0]
        if not len(a) or not len(b):
            continue
        tree = cKDTree(scaled[b])
        near = tree.query_ball_point(scaled[a], r=np.sqrt(3.0))
        for ii, js in enumerate(near):
            n_total += 1
            if not js:
                mult_gate.append(0)
                continue
            d = np.abs(pos[b[js]] - pos[a[ii]])
            ok = np.all(d <= VBOX * max_gap + 1e-9, axis=1)
            js2 = np.array(js, dtype=np.int64)[ok]
            mult_gate.append(int(len(js2)))
            if len(js2):
                n_with += 1
                ds = np.linalg.norm((pos[b[js2]] - pos[a[ii]]) / (VBOX * max_gap), axis=1)
                jbest = int(np.argmin(ds))
                nn_gate.append(float(ds[jbest]))
                nn_phys.append(float(np.linalg.norm(pos[b[js2[jbest]]] - pos[a[ii]])))
    mult = np.array(mult_gate, dtype=np.int64)
    gate = np.array(nn_gate, dtype=np.float64)
    phys = np.array(nn_phys, dtype=np.float64)
    return {
        "detections_with_next_frame": int(n_total),
        "frac_has_gate_candidate": float(n_with / n_total) if n_total else 0.0,
        "candidate_count_mean": float(mult.mean()) if len(mult) else 0.0,
        "candidate_count_p95": float(np.quantile(mult, 0.95)) if len(mult) else 0.0,
        "candidate_count_max": int(mult.max()) if len(mult) else 0,
        "nn_gate_median": float(np.median(gate)) if len(gate) else None,
        "nn_gate_p90": float(np.quantile(gate, 0.90)) if len(gate) else None,
        "nn_phys_median_mm": float(np.median(phys)) if len(phys) else None,
        "nn_phys_p90_mm": float(np.quantile(phys, 0.90)) if len(phys) else None,
    }


def triplet_residual_stats(pos: np.ndarray, frame: np.ndarray) -> dict:
    """Nearest-neighbor 3-frame constant-velocity residuals for cheap jitter evidence."""
    scaled = pos / VBOX
    residuals = []
    step01 = []
    step12 = []
    mult1 = []
    for f in range(int(frame.min()), int(frame.max()) - 1):
        i0 = np.where(frame == f)[0]
        i1 = np.where(frame == f + 1)[0]
        i2 = np.where(frame == f + 2)[0]
        if not len(i0) or not len(i1) or not len(i2):
            continue
        t1 = cKDTree(scaled[i1])
        t2 = cKDTree(scaled[i2])
        cand1 = t1.query_ball_point(scaled[i0], r=np.sqrt(3.0))
        for local0, js1 in enumerate(cand1):
            if not js1:
                continue
            p0 = pos[i0[local0]]
            d1 = pos[i1[js1]] - p0
            box1 = np.all(np.abs(d1) <= VBOX + 1e-9, axis=1)
            js1 = np.array(js1, dtype=np.int64)[box1]
            if len(js1) == 0:
                continue
            mult1.append(int(len(js1)))
            dscore = np.linalg.norm(d1[box1] / VBOX, axis=1)
            j1 = int(js1[np.argmin(dscore)])
            p1 = pos[i1[j1]]
            pred2 = p1 + (p1 - p0)
            js2 = t2.query_ball_point(pred2 / VBOX, r=np.sqrt(3.0))
            if not js2:
                continue
            d2 = np.abs(pos[i2[js2]] - p1)
            ok2 = np.all(d2 <= VBOX + 1e-9, axis=1)
            js2 = np.array(js2, dtype=np.int64)[ok2]
            if len(js2) == 0:
                continue
            res = pos[i2[js2]] - pred2
            scores = np.linalg.norm(res / VBOX, axis=1)
            best = int(np.argmin(scores))
            p2 = pos[i2[js2[best]]]
            residuals.append(float(scores[best]))
            step01.append(float(np.linalg.norm((p1 - p0) / VBOX)))
            step12.append(float(np.linalg.norm((p2 - p1) / VBOX)))
    r = np.array(residuals, dtype=np.float64)
    s01 = np.array(step01, dtype=np.float64)
    s12 = np.array(step12, dtype=np.float64)
    m1 = np.array(mult1, dtype=np.float64)
    return {
        "triplets": int(len(r)),
        "resid_gate_median": float(np.median(r)) if len(r) else None,
        "resid_gate_p90": float(np.quantile(r, 0.90)) if len(r) else None,
        "resid_gate_p99": float(np.quantile(r, 0.99)) if len(r) else None,
        "step01_gate_median": float(np.median(s01)) if len(s01) else None,
        "step12_gate_median": float(np.median(s12)) if len(s12) else None,
        "first_step_multiplicity_mean": float(np.mean(m1)) if len(m1) else None,
        "first_step_multiplicity_p95": float(np.quantile(m1, 0.95)) if len(m1) else None,
    }


def summarize_detection_set(name: str, pos: np.ndarray, frame: np.ndarray, z: np.ndarray) -> dict:
    counts = np.bincount(frame.astype(np.int64), minlength=int(frame.max()) + 1)
    return {
        "name": name,
        "n_detections": int(len(frame)),
        "frame_min": int(frame.min()) if len(frame) else None,
        "frame_max": int(frame.max()) if len(frame) else None,
        "n_frames_spanned": int(frame.max() - frame.min() + 1) if len(frame) else 0,
        "per_frame_mean": float(counts.mean()) if len(counts) else 0.0,
        "per_frame_median": float(np.median(counts)) if len(counts) else 0.0,
        "per_frame_p95": float(np.quantile(counts, 0.95)) if len(counts) else 0.0,
        "z_median": float(np.median(z)) if len(z) else None,
        "z_p90": float(np.quantile(z, 0.90)) if len(z) else None,
        "z_p99": float(np.quantile(z, 0.99)) if len(z) else None,
        "bbox_min": [float(x) for x in pos.min(axis=0)] if len(pos) else None,
        "bbox_max": [float(x) for x in pos.max(axis=0)] if len(pos) else None,
        "duplicate_stats": duplicate_stats(pos, frame),
        "next_frame": next_frame_stats(pos, frame),
        "triplets": triplet_residual_stats(pos, frame),
    }


def cmd_stats(args: argparse.Namespace) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ref_obj = load_ref()
    det = ref_obj["detections"]
    ref_pos = np.asarray(det["positions_mm"], dtype=np.float64)
    ref_frame = np.asarray(det["frame_indices"], dtype=np.int64)
    ref_z = np.asarray(det["zscores"], dtype=np.float64)
    our = load_our(args.acq)
    our_pos = np.asarray(our["positions_mm"], dtype=np.float64)
    our_frame = np.asarray(our["frame_in_acq"], dtype=np.int64)
    our_z = z_proxy(np.asarray(our["intensities"]))
    start, end, counts = peak_window(our_frame, int(np.asarray(our.get("n_frames", [our_frame.max() + 1])).ravel()[0]), args.width)
    our_win = window_arrays(our_pos, our_frame, our_z, start, end)

    result = {
        "created_unix": time.time(),
        "acq": args.acq,
        "bolus_window": {
            "width": args.width,
            "start": int(start),
            "end": int(end),
            "detections": int(counts[start:end].sum()),
            "full_detections": int(len(our_frame)),
            "full_frames": int(len(counts)),
            "window_fraction_of_detections": float(counts[start:end].sum() / len(our_frame)),
            "first_240_detections": int(counts[: args.width].sum()),
        },
        "sets": [
            summarize_detection_set("reference_acq0_all240", ref_pos, ref_frame, ref_z),
            summarize_detection_set("our_acq%d_full" % args.acq, our_pos, our_frame, our_z),
            summarize_detection_set("our_acq%d_bolus%d_%d" % (args.acq, start, end), our_win[0], our_win[1], our_win[2]),
        ],
    }
    out = OUT / f"stats_acq{args.acq:04d}.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["bolus_window"], indent=2, sort_keys=True))
    for s in result["sets"]:
        print(s["name"], "n=", s["n_detections"], "next=", s["next_frame"], "triplets=", s["triplets"])
    print(out)


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    ps = sub.add_parser("stats")
    ps.add_argument("--acq", type=int, default=0)
    ps.add_argument("--width", type=int, default=240)
    ps.set_defaults(func=cmd_stats)
    args = p.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
