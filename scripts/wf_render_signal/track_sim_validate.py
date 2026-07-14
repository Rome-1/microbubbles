"""Reference-independent accuracy test for the bubble tracker, via a SYNTHETIC benchmark with
known links (planted ground truth). Answers the question the reference comparison cannot: does
the tracker LINK correctly (precision/recall on detection-to-detection edges), or does it merely
reproduce the reference's summary statistics?

Motivated by an adversarial re-check (codex, 2026-07-14): reference-matching is circular
(the reference is not ground truth); direction-agreement and bootstrap stability do not bound
the mis-link rate; Gaussian smoothing can manufacture the matched physiology while leaving bad
associations intact. Only labeled links settle accuracy — so we plant them.

The simulation is matched to acq-0's MEASURED statistics (density, speed, turning, track-length,
dropout, localization noise) so link precision/recall here is representative of the real regime.
It is a simplification (independent smooth trajectories + uniform noise, isotropic motion), not a
vessel-network model — good enough to bound linking accuracy, not to validate morphology.

Metric definitions (edge-level, identity-aware):
  GT link      = ordered pair of consecutive DETECTED positions of the same planted bubble.
  pred link    = ordered pair of consecutive raw measurements within one output track.
  precision    = |pred ∩ GT| / |pred|   (fraction of the tracker's links that are real)
  recall       = |pred ∩ GT| / |GT|     (fraction of real links the tracker recovered)
A link touching a noise detection, or joining two different bubbles, is a false link by construction.

Fleet-safe: nice -15, single-thread, load-gated by caller.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from render_flow_diversity import SU
from track_rts import track_kf_rts

REF = "outputs/reference/full_tracks_smoothed.pkl"
FPS = 222.43
CFG = dict(sigma_a=0.05, gate_chi2=9.0, max_step_scale=1.0, max_gap=3, min_len=5)


def acq0_stats():
    """Pull the acq-0 statistics the simulation must match."""
    o = SU(open(REF, "rb")).load()
    d = o["detections"]
    P = np.asarray(d["positions_mm"], float); F = np.asarray(d["frame_indices"], int)
    ref = [np.asarray(t["positions"], float) for t in o["tracks_smoothed"]
           if int(t.get("acq_index", -1)) == 0]
    speeds = []
    for p in ref:
        speeds.append(np.linalg.norm(np.diff(p, axis=0), axis=1) * FPS)   # mm/s
    speeds = np.concatenate(speeds)
    lens = np.array([len(p) for p in ref])
    return dict(lo=P.min(0), hi=P.max(0), nframes=int(F.max()) + 1,
                dets_per_frame=len(P) / (F.max() + 1),
                linked_per_frame=sum(lens) / (F.max() + 1),
                speed_med=float(np.median(speeds)), speed_lo=float(np.percentile(speeds, 10)),
                speed_hi=float(np.percentile(speeds, 90)),
                len_mean=float(lens.mean()), len_min=int(lens.min()), len_max=int(lens.max()))


def simulate(st, seed=0, p_detect=0.85, loc_noise=0.12):
    """Plant bubbles matched to acq-0 stats. Returns detections P (N,3), frames F (N,), and
    identity ID (N,) where ID>=0 is a bubble id and ID=-1 is spurious noise."""
    rng = np.random.default_rng(seed)
    T = st["nframes"]; lo, hi = st["lo"], st["hi"]
    # bubbles: choose count so DETECTED on-track density matches the reference linked/frame
    n_bub = int(round(st["linked_per_frame"] * T / (st["len_mean"] * p_detect)))
    dt = 1.0 / FPS
    Ps, Fs, Is = [], [], []
    for b in range(n_bub):
        life = int(np.clip(rng.exponential(st["len_mean"]), st["len_min"], st["len_max"]))
        f0 = rng.integers(0, max(1, T - life))
        # constant-ish velocity with small heading noise (~8 deg/frame => turning match)
        spd = np.clip(rng.lognormal(np.log(st["speed_med"]), 0.6), st["speed_lo"], 3 * st["speed_hi"])
        v = rng.normal(size=3); v /= np.linalg.norm(v) + 1e-9; v *= spd * dt      # mm/frame
        pos = lo + (hi - lo) * rng.random(3)
        for k in range(life):
            f = f0 + k
            if rng.random() < p_detect:                 # per-frame dropout
                Ps.append(pos + rng.normal(0, loc_noise, 3)); Fs.append(f); Is.append(b)
            # small random heading change (curvature)
            dv = rng.normal(0, np.linalg.norm(v) * 0.14, 3)
            v = v + dv; pos = pos + v
    # spurious noise detections to hit acq-0's total density
    n_noise = int((st["dets_per_frame"] * T) - len(Ps))
    for _ in range(max(0, n_noise)):
        Ps.append(lo + (hi - lo) * rng.random(3)); Fs.append(rng.integers(0, T)); Is.append(-1)
    P = np.array(Ps); F = np.array(Fs, int); ID = np.array(Is, int)
    order = np.lexsort((F,))                              # frame-sorted (tracker consumes by frame)
    return P[order], F[order], ID[order]


def gt_links(F, ID):
    """Ordered consecutive same-bubble detection pairs (by detection row index)."""
    links = set()
    for b in np.unique(ID[ID >= 0]):
        rows = np.where(ID == b)[0]
        rows = rows[np.argsort(F[rows])]
        links.update(zip(rows[:-1].tolist(), rows[1:].tolist()))
    return links


def pred_links(recs, P):
    """Map each output track's raw measurements back to detection row indices, emit ordered pairs."""
    from scipy.spatial import cKDTree
    tree = cKDTree(P)
    links = set()
    for r in recs:
        _, idx = tree.query(r["z"])              # z are exactly input detections -> exact rows
        idx = idx.tolist()
        links.update(zip(idx[:-1], idx[1:]))
    return links


def heldout_linked_prediction(seed=1):
    """Reference-independent test on the REAL acq-0 data: hold out 15% of the detections that
    form tracks, re-track, and predict their positions from the re-tracked trajectories. Tests
    whether tracks localize a held-out point of a trackable bubble (vs a frame-shuffled null).
    Restricted to LINKED detections — the un-trackable ~68% have no track to predict them."""
    from scipy.spatial import cKDTree
    o = SU(open(REF, "rb")).load(); d = o["detections"]
    P = np.asarray(d["positions_mm"], float); F = np.asarray(d["frame_indices"], int)
    recs = track_kf_rts(P, F, **CFG)
    Zl = np.vstack([r["z"] for r in recs])
    _, idx = cKDTree(P).query(Zl); linked = np.zeros(len(P), bool); linked[idx] = True
    rng = np.random.default_rng(seed)
    lidx = np.where(linked)[0]; hold = rng.choice(lidx, int(0.15 * len(lidx)), replace=False)
    mask = np.ones(len(P), bool); mask[hold] = False
    tr = [(r["xf"][:, :3], r["frames"]) for r in track_kf_rts(P[mask], F[mask], **CFG)]

    def interp(f):
        pts = [[np.interp(f, fr, p[:, k]) for k in range(3)] for p, fr in tr if fr[0] <= f <= fr[-1]]
        return np.array(pts) if pts else np.empty((0, 3))
    frames = np.unique(F[hold]); by_f = {int(f): interp(int(f)) for f in frames}
    shuf = dict(zip(frames.tolist(), rng.permutation(frames).tolist()))
    err, nul = [], []
    for pt, f in zip(P[hold], F[hold]):
        C = by_f[int(f)]
        if len(C):
            err.append(np.min(np.linalg.norm(C - pt, axis=1)))
        Cs = by_f.get(int(shuf[int(f)]), np.empty((0, 3)))
        if len(Cs):
            nul.append(np.min(np.linalg.norm(Cs - pt, axis=1)))
    err, nul = np.array(err), np.array(nul)
    print("HELD-OUT LINKED-DETECTION PREDICTION (real acq-0, reference-independent)")
    print(f"  held out {len(hold)} of {linked.sum()} linked detections; re-tracked; predicted positions")
    print(f"  error mm: median {np.median(err):.3f}  within 0.5mm {np.mean(err<0.5)*100:.0f}%  "
          f"within 1mm {np.mean(err<1.0)*100:.0f}%  | frame-shuffled null median {np.median(nul):.3f} "
          f"({np.median(nul)/np.median(err):.1f}x)\n")


def main():
    st = acq0_stats()
    print("acq-0 stats matched by the sim:")
    print(f"  {st['dets_per_frame']:.1f} det/frame, {st['linked_per_frame']:.1f} linked/frame, "
          f"speed med {st['speed_med']:.1f} mm/s, track len mean {st['len_mean']:.1f} "
          f"[{st['len_min']},{st['len_max']}]\n")
    heldout_linked_prediction()
    print("SYNTHETIC LINK PRECISION / RECALL (planted ground truth, tracker config = reference-config)")
    print(f"  {'seed':>4} {'dets':>6} {'onTrk%':>7} {'GTlinks':>8} {'predL':>6} {'prec':>6} {'rec':>6} {'F1':>6}")
    Ps, Rs = [], []
    for seed in range(5):
        P, F, ID = simulate(st, seed=seed)
        recs = track_kf_rts(P, F, **CFG)
        G = gt_links(F, ID); Pl = pred_links(recs, P)
        tp = len(Pl & G)
        prec = tp / max(1, len(Pl)); rec = tp / max(1, len(G))
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        Ps.append(prec); Rs.append(rec)
        print(f"  {seed:>4} {len(P):>6} {(ID>=0).mean()*100:>6.0f}% {len(G):>8} {len(Pl):>6} "
              f"{prec:>6.2f} {rec:>6.2f} {f1:>6.2f}")
    print(f"\n  mean link precision {np.mean(Ps):.2f} +/- {np.std(Ps):.2f}  |  "
          f"recall {np.mean(Rs):.2f} +/- {np.std(Rs):.2f}")
    print("  precision = fraction of the tracker's links that are correct (bounds the mis-link rate);")
    print("  recall = fraction of true links recovered. Reference-independent — no reference used.")


if __name__ == "__main__":
    main()
