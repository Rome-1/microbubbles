"""Accuracy test for the bubble tracker on links the reference never supplied: a SYNTHETIC
benchmark with planted ground truth. Answers the question the reference comparison cannot: does
the tracker LINK correctly (precision/recall on detection-to-detection edges), or does it merely
reproduce the reference's summary statistics?

Not "reference-independent" — reference-CALIBRATED. The LABELS are planted, so they cannot be
circular; but the REGIME is read out of the reference's own tracks_smoothed (density, speed, track
length, linked fraction, via acq0_stats). If the reference tracker's regime is wrong — if it links
too tightly and its 31% linked fraction understates the real bubble population — the simulation
inherits that error and these precision/recall numbers describe the wrong regime. Only the
held-out-prediction test below touches nothing but raw detections.

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


def _trajectory(st, rng, dt, direction=None):
    """One curved constant-ish-speed trajectory: returns (life, positions (life,3)) centered at
    origin (pos[0]=0). Heading noise ~8 deg/frame reproduces the acq-0 turning distribution."""
    life = int(np.clip(rng.exponential(st["len_mean"]), st["len_min"], st["len_max"]))
    spd = np.clip(rng.lognormal(np.log(st["speed_med"]), 0.6), st["speed_lo"], 3 * st["speed_hi"])
    if direction is None:
        direction = rng.normal(size=3)
    v = direction / (np.linalg.norm(direction) + 1e-9) * spd * dt                 # mm/frame
    pos = np.zeros(3); out = []
    for _ in range(life):
        out.append(pos.copy())
        v = v + rng.normal(0, np.linalg.norm(v) * 0.14, 3); pos = pos + v         # curvature
    return life, np.array(out)


def _emit(traj, f0, bid, p_detect, loc_noise, rng, Ps, Fs, Is):
    """Sample a trajectory into (noisy, dropout-thinned) detections tagged with bubble id `bid`."""
    for k, p in enumerate(traj):
        if rng.random() < p_detect:
            Ps.append(p + rng.normal(0, loc_noise, 3)); Fs.append(f0 + k); Is.append(bid)


def simulate(st, seed=0, p_detect=0.85, loc_noise=0.12, n_cross=0, cross_ang=np.pi / 3):
    """Plant bubbles matched to acq-0 stats. Returns detections P (N,3), frames F (N,), and
    identity ID (N,) where ID>=0 is a bubble id and ID=-1 is spurious noise.

    n_cross>0 additionally plants that many ENGINEERED CROSSING PAIRS: two bubbles routed through
    a shared point at a shared frame, with headings forced at least `cross_ang` apart. These are
    the near-miss confusers absent from the independent-motion baseline — the case where a gating
    tracker can swap identities.

    CLUTTER IS HELD FIXED across the crossing sweep (review, 2026-07-28). The first version set the
    noise count as a residual against a fixed total density, so planting crossings DISPLACED uniform
    clutter: the sweep changed two variables at once and the falling clutter fraction flattered the
    tracker exactly where the crossing load was heaviest. Here the clutter count is computed from the
    n_cross=0 population, so crossings ADD to the scene and only the crossing load varies."""
    rng = np.random.default_rng(seed)
    T = st["nframes"]; lo, hi = st["lo"], st["hi"]
    # bubbles: choose count so DETECTED on-track density matches the reference linked/frame
    n_bub = int(round(st["linked_per_frame"] * T / (st["len_mean"] * p_detect)))
    dt = 1.0 / FPS
    Ps, Fs, Is = [], [], []
    bid = 0
    for _ in range(n_bub):
        life, traj = _trajectory(st, rng, dt)
        f0 = rng.integers(0, max(1, T - life))
        _emit(traj + (lo + (hi - lo) * rng.random(3)), f0, bid, p_detect, loc_noise, rng, Ps, Fs, Is)
        bid += 1
    # clutter budget is fixed by the BASELINE population, before any crossings are planted, so
    # the crossing sweep does not silently thin the clutter it is being scored against
    n_noise = int((st["dets_per_frame"] * T) - len(Ps))
    # engineered crossings: both members pass through meeting point m at meeting frame fm
    for _ in range(n_cross):
        m = lo + (hi - lo) * rng.random(3); fm = rng.integers(0, T)
        d0 = rng.normal(size=3); d0 /= np.linalg.norm(d0) + 1e-9
        perp = rng.normal(size=3); perp -= perp.dot(d0) * d0; perp /= np.linalg.norm(perp) + 1e-9
        d1 = np.cos(cross_ang) * d0 + np.sin(cross_ang) * perp   # forced >= cross_ang from d0
        for direction in (d0, d1):
            life, traj = _trajectory(st, rng, dt, direction=direction)
            k_meet = int(rng.integers(0, life))                  # which vertex sits on m
            f0 = int(np.clip(fm - k_meet, 0, max(0, T - life)))
            _emit(traj + (m - traj[k_meet]), f0, bid, p_detect, loc_noise, rng, Ps, Fs, Is)
            bid += 1
    # spurious noise detections: at n_cross=0 this hits acq-0's total density exactly
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


def _owner_map(recs, Psub, rowmap):
    """master-detection-row -> output-track index, for a tracking run over the rows `rowmap`."""
    from scipy.spatial import cKDTree
    tree = cKDTree(Psub)
    own = {}
    for ti, r in enumerate(recs):
        _, idx = tree.query(r["z"])            # z are exactly input detections -> exact rows
        for j in np.atleast_1d(idx):
            own[int(rowmap[j])] = ti
    return own


def _states(recs):
    return [dict(f=r["frames"], p=r["xf"][:, :3], v=r["xf"][:, 3:6]) for r in recs]


def _predict(t, f):
    """Position of track `t` at frame `f`: interpolated inside its span, constant-velocity
    EXTRAPOLATED outside it (state velocity is mm/frame)."""
    fr = t["f"]
    if fr[0] <= f <= fr[-1]:
        return np.array([np.interp(f, fr, t["p"][:, k]) for k in range(3)])
    if f > fr[-1]:
        return t["p"][-1] + t["v"][-1] * (f - fr[-1])
    return t["p"][0] + t["v"][0] * (f - fr[0])


def heldout_prediction(seed=1, mode="iid", verbose=True):
    """Sim-free test on the REAL acq-0 data: hold out detections that form tracks, re-track the
    survivors, and predict the held-out positions.

    Two things this test gets right that the first version did not (review, 2026-07-28):

    * IDENTITY-RESTRICTED scoring. The first version scored the min distance over EVERY
      re-tracked trajectory whose frame span bracketed the held-out frame, so a completely
      wrong track passing nearby counted as a hit — it measured "is some bubble near here",
      not "does the right track predict this point". Here the held-out detection's identity
      comes from the full-data tracking, and only the re-tracked track(s) claiming its
      immediate surviving NEIGHBOURS may score it. If no such track survives, the point is
      unscored (reported as coverage), not silently rescued by a stranger.
    * A holdout that is not trivial. mode="iid" drops 15% of linked detections independently,
      so nearly every held-out point is bracketed by its own surviving neighbours one frame
      away — that measures 1-frame interpolation, at or below the ~0.12 mm localization noise
      floor. mode="block" instead removes a CONTIGUOUS run of max_gap+1 (=4) frames from each
      track, which exceeds the tracker's coasting budget: the track must break, and the
      prediction is a genuine extrapolation across the hole.

    Competitive baselines (the frame-shuffled null only measures spatial chance):
      persistence  = last surviving RAW detection of the same bubble before the hole.
      raw-interp   = tracker-free linear interpolation between the surrounding RAW detections
                     of the same bubble (same identities, no motion model, no filter).
    The tracker earns its keep only by the margin over those.
    """
    o = SU(open(REF, "rb")).load(); d = o["detections"]
    P = np.asarray(d["positions_mm"], float); F = np.asarray(d["frame_indices"], int)
    recs0 = track_kf_rts(P, F, **CFG)
    own0 = _owner_map(recs0, P, np.arange(len(P)))
    rows_by_trk = {}
    for row, ti in own0.items():
        rows_by_trk.setdefault(ti, []).append(row)
    for ti, r in rows_by_trk.items():
        r = np.array(r); rows_by_trk[ti] = r[np.argsort(F[r])]

    rng = np.random.default_rng(seed)
    block = CFG["max_gap"] + 1
    if mode == "iid":
        lidx = np.array(sorted(own0))
        hold = np.sort(rng.choice(lidx, int(0.15 * len(lidx)), replace=False))
        ntrk_held = len(set(own0[int(h)] for h in hold))
    else:
        held = []
        for ti, rows in rows_by_trk.items():
            fr = F[rows]
            # the hole must leave >= min_len detections before it (so the causal segment can
            # survive as a track at all) and >= 1 detection after it (so raw-interp brackets)
            starts = [k for k in range(CFG["min_len"], len(rows)) if fr[k] + block <= fr[-1]]
            if not starts:
                continue
            f0 = fr[int(rng.choice(starts))]
            held.extend(rows[(fr >= f0) & (fr < f0 + block)].tolist())
        hold = np.array(sorted(set(held)))
        ntrk_held = len(set(own0[int(h)] for h in hold))

    mask = np.ones(len(P), bool); mask[hold] = False
    surv = np.where(mask)[0]
    recs1 = track_kf_rts(P[mask], F[mask], **CFG)
    own1 = _owner_map(recs1, P[mask], surv)
    T1 = _states(recs1)

    frames = np.unique(F[hold])
    any_at = {}
    for fq in frames.tolist():
        C = [_predict(t, fq) for t in T1 if t["f"][0] <= fq <= t["f"][-1]]
        any_at[int(fq)] = np.array(C).reshape(-1, 3)
    shuf = dict(zip(frames.tolist(), rng.permutation(frames).tolist()))

    e_trk, e_any, e_per, e_lin, e_nul, scored = [], [], [], [], [], []
    for h in hold.tolist():
        f = int(F[h]); pt = P[h]
        rows = rows_by_trk[own0[h]]
        srv = rows[mask[rows]]; sf = F[srv]
        prv = srv[sf < f]; nxt = srv[sf > f]
        r_prev = int(prv[-1]) if len(prv) else None
        r_next = int(nxt[0]) if len(nxt) else None
        cands = {own1[r] for r in (r_prev, r_next) if r is not None and r in own1}
        et = min((float(np.linalg.norm(_predict(T1[c], f) - pt)) for c in cands), default=np.nan)
        scored.append(np.isfinite(et))
        ep = float(np.linalg.norm(P[r_prev] - pt)) if r_prev is not None else np.nan
        if r_prev is not None and r_next is not None:
            w = (f - F[r_prev]) / (F[r_next] - F[r_prev])
            el = float(np.linalg.norm(P[r_prev] * (1 - w) + P[r_next] * w - pt))
        else:
            el = np.nan
        C = any_at[f]
        ea = float(np.min(np.linalg.norm(C - pt, axis=1))) if len(C) else np.nan
        Cs = any_at.get(int(shuf[f]), np.empty((0, 3)))
        en = float(np.min(np.linalg.norm(Cs - pt, axis=1))) if len(Cs) else np.nan
        e_trk.append(et); e_any.append(ea); e_per.append(ep); e_lin.append(el); e_nul.append(en)

    A = {k: np.array(v, float) for k, v in
         dict(trk=e_trk, any=e_any, per=e_per, lin=e_lin, nul=e_nul).items()}
    ok = np.isfinite(A["trk"]) & np.isfinite(A["per"]) & np.isfinite(A["lin"])
    C = {k: v[ok] for k, v in A.items()}
    med = {k: float(np.median(v[np.isfinite(v)])) for k, v in C.items()}
    if verbose:
        tag = ("i.i.d. 15% per-detection (1-frame interpolation)" if mode == "iid"
               else f"contiguous {block}-frame blocks, one per track (forces extrapolation)")
        print(f"HELD-OUT PREDICTION on real acq-0 — holdout: {tag}")
        print(f"  held out {len(hold)} detections from {ntrk_held} tracks "
              f"({len(own0)} linked of {len(P)} total); re-tracked the survivors")
        print(f"  identity-restricted coverage: {np.mean(scored)*100:.0f}% of held-out points had a "
              f"surviving track claiming their neighbours (rest unscored)")
        print(f"  median error mm, on the {ok.sum()} points where every method is defined:")
        print(f"    tracker (identity-restricted)   {med['trk']:.3f}   "
              f"within 0.5mm {np.mean(C['trk']<0.5)*100:.0f}%  within 1mm {np.mean(C['trk']<1.0)*100:.0f}%")
        print(f"    tracker-free raw interpolation  {med['lin']:.3f}   "
              f"(tracker margin {med['lin']/med['trk']:.2f}x)")
        print(f"    persistence (last raw det)      {med['per']:.3f}   "
              f"(tracker margin {med['per']/med['trk']:.2f}x)")
        print(f"    frame-shuffled null             {med['nul']:.3f}   "
              f"(tracker margin {med['nul']/med['trk']:.1f}x)  <- spatial chance only")
        print(f"    [old metric: min over ANY bracketing track {med['any']:.3f} — "
              f"{med['trk']/med['any']:.2f}x optimistic vs identity-restricted]\n")
    return med


def main():
    st = acq0_stats()
    print("acq-0 stats matched by the sim:")
    print(f"  {st['dets_per_frame']:.1f} det/frame, {st['linked_per_frame']:.1f} linked/frame, "
          f"speed med {st['speed_med']:.1f} mm/s, track len mean {st['len_mean']:.1f} "
          f"[{st['len_min']},{st['len_max']}]\n")
    heldout_prediction(mode="iid")
    heldout_prediction(mode="block")
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
    print("  recall = fraction of true links recovered. The LABELS are reference-independent (they")
    print("  are planted, not borrowed), but the simulation is reference-CALIBRATED: acq0_stats()")
    print("  takes density, speed, track length and linked-fraction from the reference's own")
    print("  tracks_smoothed. If the reference's regime is wrong, this benchmark is wrong with it.")

    n_bub0 = int(round(st["linked_per_frame"] * st["nframes"] / (st["len_mean"] * 0.85)))
    print("\nVESSEL-CROSSING STRESS (engineered near-miss confusers absent from the baseline above)")
    print("  Plants N crossing pairs (two bubbles through one point/frame, headings >= 60 deg apart).")
    print(f"  Baseline is ~{n_bub0} bubbles, so N=100 pairs (200 bubbles) is a HEAVY crossing load.")
    print("  Precision here bounds the mis-link rate UNDER crossings — the regime that breaks gating.")
    print("  CLUTTER IS HELD FIXED: crossings ADD detections rather than displacing uniform noise,")
    print("  so the sweep varies one thing. (The earlier version kept total density fixed, which")
    print("  thinned the clutter as the crossing load rose — a confound that flattered the tracker.)")
    print(f"  {'crossings':>9} {'dets':>6} {'clutter%':>8} {'prec':>6} {'rec':>6} {'F1':>6}")
    for n_cross in (0, 20, 50, 100):
        Ps, Rs, Ns, Cf = [], [], [], []
        for seed in range(5):
            P, F, ID = simulate(st, seed=seed, n_cross=n_cross)
            recs = track_kf_rts(P, F, **CFG)
            G = gt_links(F, ID); Pl = pred_links(recs, P)
            tp = len(Pl & G)
            Ps.append(tp / max(1, len(Pl))); Rs.append(tp / max(1, len(G)))
            Ns.append(len(P)); Cf.append((ID < 0).mean())
        prec = np.mean(Ps); rec = np.mean(Rs)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        print(f"  {n_cross:>9} {np.mean(Ns):>6.0f} {np.mean(Cf)*100:>7.0f}% "
              f"{prec:>6.2f} {rec:>6.2f} {f1:>6.2f}")
    print("  If precision holds flat across crossing density, gating resists identity swaps at the")
    print("  crossings; a downward slope quantifies how much the crossing-free baseline was luck.")


if __name__ == "__main__":
    main()
