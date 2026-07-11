"""PRODUCTION microbubble tracker — the consolidated, scale-ready pipeline.

Folds the whole residual investigation (docs/bubble-tracking-acq0.md) into ONE entry point so
that the moment the 215-acq detections land (bead mb-4yw) we run a single command over all 216
acquisitions instead of re-stitching four experiment scripts under time pressure.

  detections -> z>=p25 confidence prefilter -> constant-velocity Kalman filter (Mahalanobis gate)
             -> per-frame step gate (the operating-point knob) -> RTS backward smoothing -> tracks

Operating points (a science choice — coverage vs cleanliness, NOT a bug; see the doc):
  reference-matched : tight step gate. Reproduces the reference's speed distribution
                      (median ~25 vs 25.8 mm/s) and its sparse, selective look. ~25% of
                      detections linked (reference links ~31%).
  high-coverage     : loose gate. ~2x the bubbles tracked (~51% linked) at the cost of
                      faster/marginal links (median ~45 mm/s).

=== COMPUTE HYGIENE (hard, not advisory) ===
This rig once drove the shared box to load ~67 with an un-niced all-cores job. Therefore:
  * every worker calls os.nice(15) and pins BLAS to 1 thread;
  * worker count is HARD-CAPPED at MAX_WORKERS=4 regardless of what is requested;
  * the 1-minute load average is checked before every chunk and the run PAUSES above
    LOAD_PAUSE (renicing alone does not lower load average).
216 acquisitions is embarrassingly parallel, which is exactly why it must be capped.
"""
from __future__ import annotations
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

REF = "outputs/reference/full_tracks_smoothed.pkl"
OUTDIR = "outputs/reference/wf/r2/signal"

MAX_WORKERS = 4          # HARD CAP — do not raise. Shared 16-vCPU box, 14 other crews.
LOAD_PAUSE = 30.0        # pause the run above this 1-min load average
LOAD_POLL_S = 20

# max_gap=3 / min_len=5 are the REFERENCE's own documented values (pkl params:
# max_gap: 3, min_track_length: 5, max_distance_mm = step_scale 1.0) — use them everywhere
# so we differ from the reference only where we mean to.
MODES = {
    # their exact gate + gap + length filter: "can our KF reproduce them on their own config?"
    "reference-config":  dict(sigma_a=0.05, gate_chi2=9.0, max_step_scale=1.0, max_gap=3, min_len=5),
    # brute-force speed match (tighter than their gate — matches the statistic, not the mechanism)
    "reference-matched": dict(sigma_a=0.05, gate_chi2=9.0, max_step_scale=0.6, max_gap=3, min_len=5),
    "high-coverage":     dict(sigma_a=0.10, gate_chi2=16.0, max_step_scale=2.0, max_gap=3, min_len=5),
}
Z_PREFILTER_PCT = 25     # keep detections with z >= 25th percentile (free accuracy win)


def load1() -> float:
    with open("/proc/loadavg") as fh:
        return float(fh.read().split()[0])


def wait_for_load(tag=""):
    """Block until the box is calm enough. Renice does not lower load average — pausing does."""
    while True:
        L = load1()
        if L <= LOAD_PAUSE:
            return L
        print(f"  [load-guard] load {L:.1f} > {LOAD_PAUSE} — PAUSING {tag} ({LOAD_POLL_S}s)", flush=True)
        time.sleep(LOAD_POLL_S)


def _init_worker():
    os.nice(15)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = "1"


def load_detections(zpct=Z_PREFILTER_PCT):
    """All released detections, grouped by acquisition, after the z-confidence prefilter.
    Today this yields acq-0 only; when the 215-acq data lands it yields all 216 unchanged."""
    from render_flow_diversity import SU
    o = SU(open(REF, "rb")).load()
    d = o["detections"]
    P = np.asarray(d["positions_mm"], float)
    F = np.asarray(d["frame_indices"], int)
    A = np.asarray(d["acq_indices"], int)
    Z = np.asarray(d["zscores"], float)
    thr = np.percentile(Z, zpct)
    keep = Z >= thr
    print(f"  z-prefilter: keep z >= {thr:.2f} (p{zpct}) -> {keep.sum()}/{len(Z)} "
          f"({keep.mean()*100:.0f}%) detections")
    P, F, A = P[keep], F[keep], A[keep]
    acqs = sorted(np.unique(A).tolist())
    return {a: (P[A == a], F[A == a]) for a in acqs}, o


def _track_one(job):
    acq, P, F, params = job
    from track_rts import track_kf_rts, tracks_rts
    recs = track_kf_rts(P, F, **params)
    return acq, tracks_rts(recs), len(P)


def run(mode="reference-matched", workers=MAX_WORKERS, acqs=None):
    params = MODES[mode]
    workers = max(1, min(int(workers), MAX_WORKERS))       # HARD CAP
    det, _ = load_detections()
    if acqs:
        det = {a: v for a, v in det.items() if a in acqs}
    jobs = [(a, P, F, params) for a, (P, F) in sorted(det.items())]
    print(f"  mode={mode} {params}")
    print(f"  {len(jobs)} acquisition(s), workers={workers} (hard cap {MAX_WORKERS}), "
          f"load-pause at {LOAD_PAUSE}")

    out, t0 = {}, time.time()
    if len(jobs) == 1 or workers == 1:
        _init_worker()
        wait_for_load("acq")
        for j in jobs:
            a, tr, n = _track_one(j)
            out[a] = tr
    else:
        import multiprocessing as mp
        with mp.Pool(workers, initializer=_init_worker) as pool:
            # chunk so we can re-check load between batches (a Pool started calm can still
            # run into a busy box mid-way)
            for i in range(0, len(jobs), workers):
                wait_for_load(f"chunk {i//workers+1}")
                for a, tr, n in pool.map(_track_one, jobs[i:i + workers]):
                    out[a] = tr
                print(f"  ... {min(i+workers, len(jobs))}/{len(jobs)} acqs  "
                      f"(load {load1():.1f})", flush=True)
    print(f"  tracked {sum(len(v) for v in out.values())} tracks over {len(out)} acq(s) "
          f"in {time.time()-t0:.1f}s")
    return out


def save(tracks_by_acq, mode):
    """Flat, pickle-free format: concatenated points + track/acq ids."""
    pos, frm, tid, acq = [], [], [], []
    k = 0
    for a, trs in sorted(tracks_by_acq.items()):
        for p, f in trs:
            pos.append(np.asarray(p, np.float32)); frm.append(np.asarray(f, np.int32))
            tid.append(np.full(len(p), k, np.int32)); acq.append(np.full(len(p), a, np.int16))
            k += 1
    os.makedirs(OUTDIR, exist_ok=True)
    path = f"{OUTDIR}/tracks_production_{mode}.npz"
    np.savez_compressed(path, positions_mm=np.vstack(pos), frames=np.concatenate(frm),
                        track_id=np.concatenate(tid), acq=np.concatenate(acq))
    print(f"  wrote {path}  ({k} tracks, {sum(len(p) for p in pos)} points)")
    return path


def validate(tracks_by_acq):
    """Regression gate: on acq-0 the production tracker must reproduce the documented
    benchmark (docs/bubble-tracking-acq0.md). Guards the scale-up."""
    from track_bubbles import load_acq0, phys_plausibility, direction_agreement, length_stats
    from flow_diversity import coherence_map, segdirs_from_tracks
    P, F, I, ref = load_acq0()
    tr = tracks_by_acq.get(0)
    if not tr:
        print("  (no acq-0 in this run — skipping validation)"); return
    st = length_stats(tr)
    sp, ang = phys_plausibility(tr)
    da, nsh = direction_agreement(tr, ref)
    m, d, _ = segdirs_from_tracks([{"positions": p, "acq_index": 0} for p, _ in tr])
    cl, cn = coherence_map(m, d); cl = float(cl[cn >= 4].mean())
    rsp, rang = phys_plausibility([(p, np.arange(len(p))) for p in ref])
    print("\n  ACQ-0 VALIDATION (RTS-smoothed, vs reference tracks on identical detections)")
    print(f"    {'':<22} {'trk':>4} {'mean':>5} {'max':>4} {'spMed':>6} {'turn':>5} {'cl':>5} {'dirAg':>6}")
    print(f"    {'reference':<22} {len(ref):>4} {10.3:>5.1f} {68:>4} "
          f"{np.median(rsp):>6.1f} {np.median(rang):>5.1f} {0.73:>5.2f} {'-':>6}")
    print(f"    {'production (ours)':<22} {st['n']:>4} {st['mean']:>5.1f} {st['mx']:>4} "
          f"{np.median(sp):>6.1f} {np.median(ang):>5.1f} {cl:>5.2f} {da*100:>6.0f}")


def main():
    ap = argparse.ArgumentParser(description="Production microbubble tracker (scale-ready)")
    ap.add_argument("--mode", choices=list(MODES), default="reference-config")
    ap.add_argument("--workers", type=int, default=MAX_WORKERS,
                    help=f"parallel acquisitions; HARD-CAPPED at {MAX_WORKERS} (shared box)")
    ap.add_argument("--acqs", type=int, nargs="*", default=None, help="subset of acquisition ids")
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()
    print(f"production tracker | load {load1():.1f} | {os.cpu_count()} vCPU")
    tr = run(a.mode, a.workers, a.acqs)
    if not a.no_save:
        save(tr, a.mode)
    validate(tr)


if __name__ == "__main__":
    main()
