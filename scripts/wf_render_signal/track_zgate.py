"""Detection-quality gating / weighting for the acq-0 Kalman tracker.

LEVER (Rome, 2026-07-09): use the detections' own confidence (z-scores) to stop the
tracker chasing noisy detections. The released acq-0 detections carry a per-detection
`zscores` field (min 4.9, median 6.3, max 53.7 — heavy right tail). The reference tracker
KEEPS only ~31% of detections; if that kept subset is the high-confidence one, then
z-gating is the reference's implicit selection and should pull our tracks toward it
(lower speed/turning, higher direction-agreement + coherence).

Three experiments, all measured GT-free on identical acq-0 data, all vs the reference's
OWN acq-0 tracks (292, mean len 10.3, 31% linked; smoothed speed 26 mm/s, turning 8 deg,
cl 0.72):

  1. HARD z-threshold  : feed only detections with z >= tau; sweep tau over z-percentiles.
                         Report the coverage<->accuracy tradeoff curve.
  2. SOFT weighting    : per-detection KF measurement noise R_i = sigma_z^2 * (z0/z_i)^2
                         (clamped) so low-z detections pull the track LESS. Keeps coverage,
                         down-weights instead of dropping. Compared to hard + baseline.
  3. DIAGNOSTIC        : is the reference's kept ~31% subset high-z? Compare the z of
                         detections lying ON/near a reference acq-0 track vs those far from
                         any. If reference-linked detections are systematically higher-z,
                         that VALIDATES z-gating as the reference's implicit selection.

The tracker itself is track_bubbles.track_kf (constant-velocity Kalman, Mahalanobis gate)
held at the paper's operating point (sigma_a=0.03, gate_chi2=9) so the ONLY thing changing
is the detection set / weighting — isolating the confidence lever. The soft-weighting run
uses a copy of track_kf adapted to accept a per-detection R (kept in THIS file; the
upstream tracker is not edited).

CPU numpy/scipy only. Fleet-safe: nice -15, single-threaded, matplotlib Agg, load-gated
by the caller. Run from the repo root (paths are relative to it, like track_bubbles).
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from render_flow_diversity import SU
from track_bubbles import (track_kf, phys_plausibility, direction_agreement,
                           onestep_consistency, bootstrap_stability, smooth_tracks,
                           length_stats, DT, BIG)

REF = "outputs/reference/full_tracks_smoothed.pkl"

# operating point held FIXED across all detection-quality experiments (the baseline config)
SIGMA_A = 0.03
GATE_CHI2 = 9.0
SIGMA_Z = 0.12          # nominal localization noise (mm), track_kf default

# reference yardstick (its own acq-0 tracks, smoothed) — measured, not assumed
REF_TARGETS = dict(tracks=292, linked=0.31, speed=25.8, turning=8.1, cl=0.72)


# ------------------------------------------------------------------ data loading
def load_acq0_z():
    """Mirror track_bubbles.load_acq0 but ALSO return zscores Z and the reference
    tracks' per-point frames (needed for the frame-aware membership diagnostic)."""
    o = SU(open(REF, "rb")).load()
    det = o["detections"]
    P = np.asarray(det["positions_mm"], float)
    F = np.asarray(det["frame_indices"], int)
    I = np.asarray(det["intensities"], float)
    Z = np.asarray(det["zscores"], float)
    reft = [t for t in o["tracks_smoothed"] if int(t.get("acq_index", -1)) == 0]
    ref = [np.asarray(t["positions"], float) for t in reft]
    ref_frames = [np.asarray(t["frames"], int) for t in reft]
    return P, F, I, Z, ref, ref_frames


# ------------------------------------------------- SOFT weighting: per-detection R KF
def track_kf_weighted(P, F, rvar, sigma_a=SIGMA_A, sigma_z=SIGMA_Z, gate_chi2=GATE_CHI2,
                      v0=0.5, max_gap=2, min_len=4):
    """track_kf, adapted so the measurement noise is PER DETECTION: R_m = rvar[m] * I3.

    Identical constant-velocity Kalman / Mahalanobis-gate / gap-closing machinery as
    track_bubbles.track_kf; the only change is that the innovation covariance
    S_tm = H Pp_t H^T + rvar[m] I3 now depends on the detection's own confidence, so a
    low-confidence (large rvar) detection produces a small Kalman gain and barely moves the
    state — down-weighting noise WITHOUT discarding it. `rvar` is indexed by global
    detection index (len == len(P))."""
    I3 = np.eye(3)
    H = np.zeros((3, 6)); H[:, :3] = I3
    nf = int(F.max()) + 1
    by_frame = [np.where(F == f)[0] for f in range(nf)]      # global det indices per frame

    def A_Q(dt):
        A = np.eye(6); A[:3, 3:] = dt * I3
        q = sigma_a ** 2
        Q = np.zeros((6, 6))
        Q[:3, :3] = q * dt ** 4 / 4 * I3; Q[:3, 3:] = q * dt ** 3 / 2 * I3
        Q[3:, :3] = q * dt ** 3 / 2 * I3; Q[3:, 3:] = q * dt ** 2 * I3
        return A, Q

    X = np.zeros((0, 6)); Pc = np.zeros((0, 6, 6))
    last_f = np.zeros(0, int); missed = np.zeros(0, int)
    pts, frs, done = [], [], []

    def retire(mask):
        nonlocal X, Pc, last_f, missed, pts, frs
        for i in np.where(mask)[0]:
            done.append((np.array(pts[i]), np.array(frs[i])))
        keep = ~mask
        X, Pc, last_f, missed = X[keep], Pc[keep], last_f[keep], missed[keep]
        pts = [pts[i] for i in np.where(keep)[0]]
        frs = [frs[i] for i in np.where(keep)[0]]

    for f in range(nf):
        di = by_frame[f]
        D = P[di]; rv = rvar[di]
        T = len(X)
        if T and len(D):
            Xp = np.zeros_like(X); Pp = np.zeros_like(Pc)
            zpred = np.zeros((T, 3)); HPpH = np.zeros((T, 3, 3))
            for t in range(T):
                A, Q = A_Q(f - last_f[t])
                xp = A @ X[t]; pp = A @ Pc[t] @ A.T + Q
                Xp[t] = xp; Pp[t] = pp
                zpred[t] = H @ xp; HPpH[t] = H @ pp @ H.T
            C = np.full((T, len(D)), BIG)
            for t in range(T):
                S = HPpH[t][None] + rv[:, None, None] * I3[None]     # (M,3,3) per-det R
                Sinv = np.linalg.inv(S)
                y = D - zpred[t]                                      # (M,3)
                C[t] = np.einsum("mi,mij,mj->m", y, Sinv, y)
            C[C > gate_chi2] = BIG
            ri, ci = linear_sum_assignment(C)
            ok = C[ri, ci] < BIG
            ri, ci = ri[ok], ci[ok]
            upd = np.zeros(T, bool)
            for t, m in zip(ri, ci):
                A, Q = A_Q(f - last_f[t])
                xp = A @ X[t]; pp = A @ Pc[t] @ A.T + Q
                S = H @ pp @ H.T + rv[m] * I3
                K = pp @ H.T @ np.linalg.inv(S)
                X[t] = xp + K @ (D[m] - H @ xp)
                Pc[t] = (np.eye(6) - K @ H) @ pp
                last_f[t] = f; missed[t] = 0; upd[t] = True
                pts[t].append(X[t][:3].copy()); frs[t].append(f)
            assigned = np.zeros(len(D), bool); assigned[ci] = True
            missed[~upd] += 1
            retire(missed > max_gap)
        else:
            assigned = np.zeros(len(D), bool)
            if T:
                missed += 1; retire(missed > max_gap)
        for m in np.where(~assigned)[0]:
            x0 = np.concatenate([D[m], np.zeros(3)])
            P0 = np.diag([sigma_z ** 2] * 3 + [v0 ** 2] * 3)
            X = np.vstack([X, x0]); Pc = np.concatenate([Pc, P0[None]])
            last_f = np.append(last_f, f); missed = np.append(missed, 0)
            pts.append([D[m]]); frs.append([f])
    retire(np.ones(len(X), bool))
    return [(p, fr) for p, fr in done if len(p) >= min_len]


# ------------------------------------------------------------------ evaluation
def cl_of(tracks_sm):
    """Within-acq orientation coherence cl over smoothed tracks (reported, NOT targeted)."""
    from flow_diversity import coherence_map, segdirs_from_tracks
    td = [{"positions": p, "acq_index": 0} for p, _ in tracks_sm if len(p) >= 2]
    if not td:
        return float("nan"), 0
    m, d, _ = segdirs_from_tracks(td)
    cl, cn = coherence_map(m, d)
    o = cn >= 4
    return (float(cl[o].mean()) if o.any() else float("nan")), int(o.sum())


def cl_ref(ref_polys):
    from flow_diversity import coherence_map, segdirs_from_tracks
    m, d, _ = segdirs_from_tracks([{"positions": p, "acq_index": 0} for p in ref_polys])
    cl, cn = coherence_map(m, d)
    o = cn >= 4
    return float(cl[o].mean()), int(o.sum())


def full_eval(tr, ref_polys, N_total):
    """Everything vs the reference. linked% is vs the FULL detection count (N_total=9761),
    so it reflects the true coverage cost of dropping detections, not coverage of the
    (already-filtered) input."""
    st = length_stats(tr)
    trs = smooth_tracks(tr)
    sp, ang = phys_plausibility(trs)
    da, nsh = direction_agreement(trs, ref_polys)
    clv, cln = cl_of(trs)
    return dict(n=st["n"], mean=st["mean"], mx=st["mx"], pts=st["pts"],
                linked=st["pts"] / N_total,
                sp_med=float(np.median(sp)) if len(sp) else 0.0,
                turn_med=float(np.median(ang)) if len(ang) else 0.0,
                da=da, nsh=nsh, cl=clv, cln=cln)


# ------------------------------------------------------------------ diagnostic
def ref_membership(P, F, ref_polys, ref_frames, radius):
    """Frame-aware nearest-ref-point distance for every detection; label 'near' a reference
    track if within `radius` mm at the SAME frame. Reference positions are smoothed, so a
    tight radius still captures the on-track detections (calibrated: nearest-dist is bimodal,
    25th pct ~0.29mm vs median ~3.5mm)."""
    rbf = defaultdict(list)
    for poly, frs in zip(ref_polys, ref_frames):
        for p, fr in zip(poly, frs):
            rbf[int(fr)].append(p)
    trees = {fr: cKDTree(np.array(v)) for fr, v in rbf.items()}
    dist = np.full(len(P), np.inf)
    for i in range(len(P)):
        fr = int(F[i])
        if fr in trees:
            dist[i] = trees[fr].query(P[i])[0]
    return dist < radius, dist


# ------------------------------------------------------------------ main
def main():
    P, F, I, Z, ref, ref_frames = load_acq0_z()
    N = len(P)
    print(f"acq-0: {N} detections, {int(F.max())+1} frames")
    print(f"z-score distribution: min {Z.min():.2f}  p25 {np.percentile(Z,25):.2f}  "
          f"median {np.median(Z):.2f}  p75 {np.percentile(Z,75):.2f}  "
          f"p90 {np.percentile(Z,90):.2f}  max {Z.max():.2f}  (heavy right tail)")
    rcl, rcn = cl_ref(ref)
    refL = np.array([len(p) for p in ref])
    rsp, rang = phys_plausibility([(p, np.arange(len(p))) for p in ref])
    print(f"REFERENCE (target): {len(ref)} tracks, mean len {refL.mean():.1f}, "
          f"{refL.sum()/N*100:.0f}% linked | smoothed speed {np.median(rsp):.1f} mm/s | "
          f"turning {np.median(rang):.1f} deg | cl {rcl:.2f} ({rcn} cells)")
    print(f"BASELINE (paper, all detections): 398 tr, 33% linked, speed 40, turn 18, "
          f"dir-agree 70%, cl 0.55\n")

    # ---- baseline: track_kf on ALL detections at the fixed operating point -------------
    base_tr = track_kf(P, F, sigma_a=SIGMA_A, gate_chi2=GATE_CHI2)
    base = full_eval(base_tr, ref, N)

    # =====================================================================================
    # EXPERIMENT 1 — HARD z-threshold sweep
    # =====================================================================================
    print("=" * 92)
    print("EXPERIMENT 1 — HARD z-threshold: feed only detections with z >= tau")
    print("=" * 92)
    pcts = [0, 25, 50, 75, 90]
    taus = [(f"p{q}", float(np.percentile(Z, q))) for q in pcts]
    print(f"  {'tau':>10} {'zval':>6} {'fed%':>6} {'ntr':>5} {'mean':>5} {'max':>4} "
          f"{'link%':>6} {'speed':>6} {'turn':>6} {'dirAg':>6} {'cl':>6}")
    hard_rows = []
    # baseline row (all detections = p0 threshold; z>=min keeps all)
    for name, tau in taus:
        keep = Z >= tau
        fed = keep.mean()
        tr = track_kf(P[keep], F[keep], sigma_a=SIGMA_A, gate_chi2=GATE_CHI2)
        e = full_eval(tr, ref, N)
        e["name"], e["tau"], e["fed"] = name, tau, fed
        hard_rows.append(e)
        print(f"  {name:>10} {tau:6.2f} {fed*100:6.1f} {e['n']:5d} {e['mean']:5.1f} "
              f"{e['mx']:4d} {e['linked']*100:6.1f} {e['sp_med']:6.1f} {e['turn_med']:6.1f} "
              f"{e['da']*100:6.0f} {e['cl']:6.2f}")
    print(f"  {'REFERENCE':>10} {'--':>6} {'31.0':>6} {292:5d} {10.3:5.1f} {68:4d} "
          f"{31.0:6.1f} {REF_TARGETS['speed']:6.1f} {REF_TARGETS['turning']:6.1f} "
          f"{'--':>6} {REF_TARGETS['cl']:6.2f}")
    print("  fed% = fraction of the 9761 detections passed to the tracker (coverage of input)")
    print("  link% = fraction of ALL 9761 detections that end up in a track (true coverage)\n")

    # =====================================================================================
    # EXPERIMENT 2 — SOFT weighting: per-detection R from confidence
    # =====================================================================================
    print("=" * 92)
    print("EXPERIMENT 2 — SOFT weighting: R_i = sigma_z^2 * (z0/z_i)^2  (clamped), all detections")
    print("=" * 92)
    print(f"  {'z0':>10} {'clamp':>10} {'ntr':>5} {'mean':>5} {'max':>4} {'link%':>6} "
          f"{'speed':>6} {'turn':>6} {'dirAg':>6} {'cl':>6}")
    soft_rows = []
    # z0 sets the confidence at which R == nominal sigma_z^2. Median trusts the tail hard;
    # p75 down-weights the low-z bulk more. Clamp the sigma multiplier so R stays sane.
    soft_cfgs = [
        (float(np.median(Z)), (0.4, 2.5)),
        (float(np.percentile(Z, 75)), (0.4, 2.5)),
        (float(np.percentile(Z, 90)), (0.4, 3.0)),
    ]
    for z0, clamp in soft_cfgs:
        mult = np.clip(z0 / Z, clamp[0], clamp[1])          # sigma multiplier per detection
        rvar = (SIGMA_Z * mult) ** 2
        tr = track_kf_weighted(P, F, rvar, sigma_a=SIGMA_A, sigma_z=SIGMA_Z, gate_chi2=GATE_CHI2)
        e = full_eval(tr, ref, N)
        e["z0"], e["clamp"] = z0, clamp
        soft_rows.append(e)
        print(f"  {z0:10.2f} {str(clamp):>10} {e['n']:5d} {e['mean']:5.1f} {e['mx']:4d} "
              f"{e['linked']*100:6.1f} {e['sp_med']:6.1f} {e['turn_med']:6.1f} "
              f"{e['da']*100:6.0f} {e['cl']:6.2f}")
    print(f"  {'BASELINE':>10} {'(R const)':>10} {base['n']:5d} {base['mean']:5.1f} "
          f"{base['mx']:4d} {base['linked']*100:6.1f} {base['sp_med']:6.1f} "
          f"{base['turn_med']:6.1f} {base['da']*100:6.0f} {base['cl']:6.2f}")
    print(f"  {'REFERENCE':>10} {'':>10} {292:5d} {10.3:5.1f} {68:4d} {31.0:6.1f} "
          f"{REF_TARGETS['speed']:6.1f} {REF_TARGETS['turning']:6.1f} {'--':>6} "
          f"{REF_TARGETS['cl']:6.2f}\n")

    # =====================================================================================
    # EXPERIMENT 3 — DIAGNOSTIC: is the reference's kept subset high-z?
    # =====================================================================================
    print("=" * 92)
    print("EXPERIMENT 3 — DIAGNOSTIC: z-scores of detections ON/near a reference track vs FAR")
    print("=" * 92)
    print("  (frame-aware nearest reference-point distance; near = within radius at same frame)")
    print(f"  {'radius':>7} {'near_n':>7} {'near%':>6} | {'z near: med':>11} {'mean':>6} "
          f"{'p75':>6} | {'z far: med':>10} {'mean':>6} {'p75':>6} | {'medΔ':>6}")
    for r in (0.1, 0.15, 0.2, 0.3):
        near, dist = ref_membership(P, F, ref, ref_frames, radius=r)
        zn, zf = Z[near], Z[~near]
        if len(zn) == 0:
            continue
        print(f"  {r:7.2f} {near.sum():7d} {near.mean()*100:6.1f} | "
              f"{np.median(zn):11.2f} {zn.mean():6.2f} {np.percentile(zn,75):6.2f} | "
              f"{np.median(zf):10.2f} {zf.mean():6.2f} {np.percentile(zf,75):6.2f} | "
              f"{np.median(zn)-np.median(zf):6.2f}")
    # primary radius for the fuller readout
    near, dist = ref_membership(P, F, ref, ref_frames, radius=0.2)
    zn, zf = Z[near], Z[~near]
    from scipy.stats import mannwhitneyu, ks_2samp
    U, pu = mannwhitneyu(zn, zf, alternative="greater")
    ks, pks = ks_2samp(zn, zf)
    print(f"\n  primary radius 0.20mm: {near.sum()} near ({near.mean()*100:.0f}%), "
          f"{(~near).sum()} far")
    print(f"    Mann-Whitney U (near z > far z): p = {pu:.2e}  "
          f"(one-sided; tiny p => near detections are higher-z)")
    print(f"    KS 2-sample: D = {ks:.3f}, p = {pks:.2e}")
    # fraction of each z-quartile that lands near a reference track
    print("  fraction of detections landing near a reference track, by z-quartile:")
    qedges = np.percentile(Z, [0, 25, 50, 75, 100])
    for a, b, lab in [(qedges[0], qedges[1], "Q1 (lowest z)"), (qedges[1], qedges[2], "Q2"),
                      (qedges[2], qedges[3], "Q3"), (qedges[3], qedges[4] + 1, "Q4 (highest z)")]:
        m = (Z >= a) & (Z < b)
        print(f"    {lab:16s} z[{a:5.2f},{b:5.2f}): near-ref fraction {near[m].mean()*100:5.1f}%  "
              f"(n={m.sum()})")

    # =====================================================================================
    # VERDICT
    # =====================================================================================
    print("\n" + "=" * 92)
    print("VERDICT — does confidence gating close the residual, and at what coverage cost?")
    print("=" * 92)
    # best hard config by closeness to reference physiology among those keeping >=25% linked
    cov = [e for e in hard_rows if e["linked"] >= 0.25]
    pool = cov if cov else hard_rows
    best_hard = min(pool, key=lambda e: abs(e["sp_med"] - REF_TARGETS["speed"])
                    + abs(e["turn_med"] - REF_TARGETS["turning"]))
    best_soft = min(soft_rows, key=lambda e: abs(e["sp_med"] - REF_TARGETS["speed"])
                    + abs(e["turn_med"] - REF_TARGETS["turning"]))
    print(f"  baseline (all det)      : speed {base['sp_med']:.1f}  turn {base['turn_med']:.1f}  "
          f"cl {base['cl']:.2f}  link {base['linked']*100:.0f}%  dirAg {base['da']*100:.0f}%")
    print(f"  best hard  ({best_hard['name']:>4}, z>={best_hard['tau']:.1f}): "
          f"speed {best_hard['sp_med']:.1f}  turn {best_hard['turn_med']:.1f}  "
          f"cl {best_hard['cl']:.2f}  link {best_hard['linked']*100:.0f}%  "
          f"dirAg {best_hard['da']*100:.0f}%  (fed {best_hard['fed']*100:.0f}%)")
    print(f"  best soft  (z0={best_soft['z0']:.1f})   : speed {best_soft['sp_med']:.1f}  "
          f"turn {best_soft['turn_med']:.1f}  cl {best_soft['cl']:.2f}  "
          f"link {best_soft['linked']*100:.0f}%  dirAg {best_soft['da']*100:.0f}%")
    print(f"  REFERENCE               : speed {REF_TARGETS['speed']:.1f}  "
          f"turn {REF_TARGETS['turning']:.1f}  cl {REF_TARGETS['cl']:.2f}  link 31%")

    # extra rigor on the best hard config: one-step + bootstrap
    keep = Z >= best_hard["tau"]
    bh_tr = track_kf(P[keep], F[keep], sigma_a=SIGMA_A, gate_chi2=GATE_CHI2)
    os_err = onestep_consistency(bh_tr)
    stab = bootstrap_stability(P[keep], F[keep], tracker=track_kf,
                               sigma_a=SIGMA_A, gate_chi2=GATE_CHI2)
    print(f"\n  best-hard extra: one-step {np.median(os_err):.3f} mm (base "
          f"{np.median(onestep_consistency(base_tr)):.3f}) | bootstrap {stab*100:.0f}% recovered")

    try:
        render(hard_rows, soft_rows, base, Z, near, dist)
        print("\n  wrote renders/track_zgate_vs_reference.png")
    except Exception as ex:
        print(f"\n  (render skipped: {ex})")


def render(hard_rows, soft_rows, base, Z, near, dist):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))

    # (a) coverage vs accuracy tradeoff (speed & turning vs link%)
    link = [e["linked"] * 100 for e in hard_rows]
    ax[0].plot(link, [e["sp_med"] for e in hard_rows], "o-", label="our speed (mm/s)", color="tab:red")
    ax[0].plot(link, [e["turn_med"] for e in hard_rows], "s-", label="our turning (deg)", color="tab:orange")
    for e in hard_rows:
        ax[0].annotate(e["name"], (e["linked"] * 100, e["sp_med"]), fontsize=7,
                       xytext=(2, 3), textcoords="offset points")
    ax[0].axhline(REF_TARGETS["speed"], ls="--", color="tab:red", alpha=0.5)
    ax[0].axhline(REF_TARGETS["turning"], ls="--", color="tab:orange", alpha=0.5)
    ax[0].axvline(31, ls=":", color="k", alpha=0.5)
    ax[0].text(31, ax[0].get_ylim()[1] * 0.95, " ref 31% linked", fontsize=8)
    ax[0].set_xlabel("detections linked (% of all 9761)")
    ax[0].set_ylabel("median speed / turning")
    ax[0].set_title("HARD z-threshold: coverage <-> accuracy\n(dashed = reference target)")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

    # (b) cl & dir-agree vs tau
    tau = [e["tau"] for e in hard_rows]
    ax[1].plot(tau, [e["cl"] for e in hard_rows], "o-", label="our cl", color="tab:blue")
    ax[1].plot(tau, [e["da"] for e in hard_rows], "s-", label="dir-agree", color="tab:green")
    ax[1].axhline(REF_TARGETS["cl"], ls="--", color="tab:blue", alpha=0.6, label="ref cl 0.72")
    ax[1].set_xlabel("z threshold tau"); ax[1].set_ylabel("coherence cl / dir-agree frac")
    ax[1].set_title("HARD z-threshold: coherence & direction-agreement")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)

    # (c) diagnostic: z distribution near vs far from reference tracks
    zn, zf = Z[near], Z[~near]
    bins = np.linspace(4.5, 20, 40)
    ax[2].hist(zf, bins=bins, density=True, alpha=0.55, label=f"far from ref (n={len(zf)})", color="0.5")
    ax[2].hist(zn, bins=bins, density=True, alpha=0.65, label=f"near ref track (n={len(zn)})", color="tab:purple")
    ax[2].axvline(np.median(zf), ls="--", color="0.4")
    ax[2].axvline(np.median(zn), ls="--", color="tab:purple")
    ax[2].set_xlabel("detection z-score"); ax[2].set_ylabel("density")
    ax[2].set_title(f"DIAGNOSTIC: reference-linked detections are higher-z\n"
                    f"median {np.median(zn):.2f} (near) vs {np.median(zf):.2f} (far)")
    ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)

    fig.suptitle("Detection-quality gating/weighting on acq-0 vs the reference tracker", fontsize=13)
    os.makedirs("renders", exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("renders/track_zgate_vs_reference.png", dpi=140, bbox_inches="tight")


if __name__ == "__main__":
    main()
