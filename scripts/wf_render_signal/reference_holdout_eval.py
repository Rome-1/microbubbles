"""Held-out validation: run the PRODUCTION tracker's frozen `reference-config` operating
point (docs/bubble-tracking-acq0.md; scripts/wf_render_signal/track_production.py) on
reference-linked point clouds for acquisitions the operating point was never tuned on, and
score both trajectory stats and LINK-LEVEL precision/recall against the reference's own
`track_id` labels (see reference_dissolve.py for how those labels were recovered).

*** CAVEAT (repeated here on purpose -- read before citing these numbers) ***
The dissolved cloud fed to the tracker here is the reference's LINKED subset only (~31% of
raw detections on acq-0; we don't have per-acq raw detections for acq 1-215 at all). That
makes association strictly EASIER than real operation: there is no unlinkable noise to
reject, and no discovery of the ~69% of raw detections the reference itself never linked.
Treat the precision/recall below as an OPTIMISTIC ceiling relative to real operation on raw
detections, not a measurement of real-world accuracy.

The `reference-config` operating point (sigma_a=0.05, gate_chi2=9.0, max_step_scale=1.0,
max_gap=3, min_len=5, smoother=gaussian2) is used UNCHANGED -- this is a frozen pre-registered
protocol. No tuning happens in this script.

CPU numpy/scipy only. Fleet-safe: nice -15, single-threaded, no plotting, no multiprocessing
(per-acq clouds here are ~1-3k points -- trivially cheap; not worth spinning up a pool).
"""
from __future__ import annotations
import os, sys, argparse
from collections import defaultdict, deque
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

SEED = 42
N_HOLDOUT = 15


def _init_worker_nice():
    os.nice(15)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = "1"


def pick_holdout_acqs(n=N_HOLDOUT, seed=SEED, n_total=216, exclude=(0,)):
    rng = np.random.default_rng(seed)
    pool = np.array([a for a in range(n_total) if a not in exclude])
    return sorted(int(a) for a in rng.choice(pool, size=n, replace=False))


def build_pos_index(P, F):
    """frame -> {rounded (x,y,z): deque of original point ids}. Used to recover which
    original dissolved detection a tracker-output measurement (frames[i], z[i]) came from --
    the tracker never modifies raw measurements before assignment, so exact-match recovery
    is safe (ties from true float-duplicate positions in one frame are vanishingly rare for
    continuous 3D data and are resolved FIFO)."""
    idx = defaultdict(dict)
    for pid in range(len(P)):
        key = (round(float(P[pid, 0]), 6), round(float(P[pid, 1]), 6), round(float(P[pid, 2]), 6))
        idx[int(F[pid])].setdefault(key, deque()).append(pid)
    return idx


def match_point(idx, f, p):
    key = (round(float(p[0]), 6), round(float(p[1]), 6), round(float(p[2]), 6))
    q = idx.get(int(f), {}).get(key)
    if not q:
        return None
    return q.popleft()


def true_edges(track_id):
    """Reference's own links: adjacent-index pairs sharing track_id. Valid because
    reference_dissolve.py concatenates each reference track as a contiguous, already
    frame-ordered block."""
    return {frozenset((i, i + 1)) for i in range(len(track_id) - 1) if track_id[i] == track_id[i + 1]}


def predicted_edges(recs, idx):
    edges, unmatched, total = set(), 0, 0
    for r in recs:
        pids = []
        for f, p in zip(r["frames"], r["z"]):
            total += 1
            pid = match_point(idx, f, p)
            if pid is None:
                unmatched += 1
            pids.append(pid)
        for i in range(len(pids) - 1):
            if pids[i] is not None and pids[i + 1] is not None:
                edges.add(frozenset((pids[i], pids[i + 1])))
    return edges, unmatched, total


def eval_acq(acq: int, cfg: dict):
    """Load the dissolved cloud for `acq`, run the tracker, score trajectory stats + link
    precision/recall against the reference's track_id labels."""
    from reference_dissolve import load_dissolved
    from track_rts import track_kf_rts
    from track_production import TRACKER_KEYS, _smooth
    from track_bubbles import phys_plausibility, length_stats

    d = load_dissolved(acq)
    P = d["positions_mm"].astype(np.float64)
    F = d["frame_in_acq"].astype(np.int64)
    tid = d["track_id"]
    M = len(P)

    tparams = {k: cfg[k] for k in TRACKER_KEYS}
    recs = track_kf_rts(P, F, **tparams)
    smoothed = _smooth(recs, cfg["smoother"])          # list of (positions, frames)

    st = length_stats(smoothed)
    sp, ang = phys_plausibility(smoothed)
    linked_frac = st["pts"] / M if M else float("nan")

    idx = build_pos_index(P, F)
    t_edges = true_edges(tid)
    p_edges, unmatched, total = predicted_edges(recs, idx)
    inter = p_edges & t_edges
    precision = len(inter) / len(p_edges) if p_edges else float("nan")
    recall = len(inter) / len(t_edges) if t_edges else float("nan")

    return dict(acq=acq, n_points=M, n_ref_tracks=int(len(np.unique(tid))),
                n_out_tracks=st["n"], mean_len=st["mean"], median_len=st["median"],
                max_len=st["mx"], linked_frac=linked_frac,
                turn_med=float(np.median(ang)) if len(ang) else float("nan"),
                speed_med=float(np.median(sp)) if len(sp) else float("nan"),
                precision=precision, recall=recall,
                n_true_edges=len(t_edges), n_pred_edges=len(p_edges),
                unmatched=unmatched, total_matched_pts=total)


def print_table(rows, title):
    print(f"\n{title}")
    hdr = (f"{'acq':>4} {'pts':>5} {'refTr':>5} {'outTr':>5} {'meanL':>6} {'medL':>5} "
           f"{'linked%':>7} {'turn':>5} {'speed':>6} {'prec':>5} {'rec':>5}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['acq']:>4} {r['n_points']:>5} {r['n_ref_tracks']:>5} {r['n_out_tracks']:>5} "
              f"{r['mean_len']:>6.2f} {r['median_len']:>5.1f} {r['linked_frac']*100:>6.1f}% "
              f"{r['turn_med']:>5.1f} {r['speed_med']:>6.1f} {r['precision']:>5.2f} {r['recall']:>5.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=N_HOLDOUT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--mode", default="reference-config")
    a = ap.parse_args()
    _init_worker_nice()

    from track_production import MODES
    cfg = MODES[a.mode]
    print(f"mode={a.mode}: {cfg}")

    acqs = pick_holdout_acqs(a.n, a.seed)
    print(f"held-out acqs (seed={a.seed}, n={a.n}, acq-0 excluded from sampling): {acqs}")

    print("\nscoring acq-0 (dissolved linked cloud, same pipeline, for comparison)...")
    r0 = eval_acq(0, cfg)

    print(f"scoring {len(acqs)} held-out acqs...")
    rows = [eval_acq(a_, cfg) for a_ in acqs]

    print_table([r0], "ACQ-0 (dissolved-cloud baseline, NOT randomly sampled)")
    print_table(rows, f"HELD-OUT SAMPLE (n={len(rows)}, seed={a.seed}, excludes acq-0)")

    def col(k):
        return np.array([r[k] for r in rows])
    print("\nHELD-OUT DISTRIBUTION SUMMARY (vs acq-0):")
    print(f"  {'metric':<12} {'acq-0':>8} {'held-out mean':>14} {'median':>8} {'std':>8} "
          f"{'min':>7} {'max':>7}")
    for k, label in [("linked_frac", "linked%"), ("mean_len", "mean_len"),
                      ("turn_med", "turn(deg)"), ("speed_med", "speed(mm/s)"),
                      ("precision", "precision"), ("recall", "recall")]:
        v = col(k)
        scale = 100.0 if k == "linked_frac" else 1.0
        r0v = r0[k] * scale
        print(f"  {label:<12} {r0v:>8.3f} {np.nanmean(v)*scale:>14.3f} {np.nanmedian(v)*scale:>8.3f} "
              f"{np.nanstd(v)*scale:>8.3f} {np.nanmin(v)*scale:>7.3f} {np.nanmax(v)*scale:>7.3f}")

    dprec = r0["precision"] - np.nanmean(col("precision"))
    drec = r0["recall"] - np.nanmean(col("recall"))
    print(f"\nDEGRADATION vs acq-0: precision drops {dprec*100:+.1f}pp, recall drops {drec*100:+.1f}pp "
          f"on the held-out mean vs the acq-0 baseline (positive = held-out worse).")
    print("\nCAVEAT (repeat): all numbers above are on the reference's LINKED subset, not raw "
          "per-acq detections (unavailable for acq 1-215). This is an easier problem than real "
          "operation -- treat precision/recall as an optimistic ceiling, not a real-world estimate.")
    return r0, rows


if __name__ == "__main__":
    main()
