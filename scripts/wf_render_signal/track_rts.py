"""RTS (Rauch-Tung-Striebel) smoother for the constant-velocity Kalman bubble tracker.

The baseline (`track_bubbles.track_kf`) runs a constant-velocity Kalman FILTER (forward
pass only) and then applies a SEPARATE Savitzky-Golay polynomial smoother to the output.
Savgol is model-agnostic: it fits a local polynomial, ignoring the motion model and the
per-step uncertainty the filter already computed. The RTS smoother instead runs the
filter's OWN model backward, blending each filtered estimate with future evidence weighted
by covariance -- the statistically-optimal fixed-interval smoother for a linear-Gaussian
state space. It should straighten trajectories toward the reference's physiological
turning (8 deg) more principledly than Savgol.

This file COPIES/adapts `track_kf` (does NOT edit track_bubbles.py) so it can expose the
per-step KF internals the RTS recursion needs: for every matched step k we store the
predicted state/cov (x_pred, P_pred), the filtered state/cov (x_filt, P_filt), the raw
measurement z, and the transition A. The linking/association is IDENTICAL to track_kf, so
the track SET is the same -- raw, Savgol, and RTS are compared on exactly the same links.

Backward RTS recursion (per track, k = N-2 .. 0):
    C_k        = P_filt[k] A_k^T P_pred[k+1]^-1
    x_smooth[k]= x_filt[k] + C_k (x_smooth[k+1] - x_pred[k+1])
    P_smooth[k]= P_filt[k] + C_k (P_smooth[k+1] - P_pred[k+1]) C_k^T

HONEST CAVEAT (point 4 of the task): smoothing improves the TRAJECTORY ESTIMATE but cannot
fix a WRONG LINK -- a mis-association hidden under a smooth curve is still an accuracy
error. So we separate two things:
  * trajectory-estimate quality  -> turning / speed / one-step on the smoothed states
                                    (RTS helps; but a low value can be MASKING a bad link)
  * link correctness             -> direction agreement with the reference, and one-step
                                    prediction error on the FILTERED (un-smoothed) states.
                                    These are what smoothing can't fake; RTS leaves them
                                    essentially unchanged.

CPU numpy/scipy only. Fleet-safe: nice -15, single-threaded, load-gated by the caller.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy.optimize import linear_sum_assignment
from track_bubbles import (load_acq0, track_kf, phys_plausibility, onestep_consistency,
                           direction_agreement, smooth_tracks, length_stats, DT)

BIG = 1e6


def _A_Q(dt, sigma_a):
    I3 = np.eye(3)
    A = np.eye(6); A[:3, 3:] = dt * I3
    q = sigma_a ** 2
    Q = np.zeros((6, 6))
    Q[:3, :3] = q * dt ** 4 / 4 * I3; Q[:3, 3:] = q * dt ** 3 / 2 * I3
    Q[3:, :3] = q * dt ** 3 / 2 * I3; Q[3:, 3:] = q * dt ** 2 * I3
    return A, Q


def _rts_smooth(xf, Pf, xp, Pp, AA):
    """Backward RTS recursion over one track's stored per-step internals.
    xf/Pf: filtered state/cov ; xp/Pp: predicted (prior) state/cov ; AA: transition into
    each step. Step 0 is the spawn (no prediction; AA[0], xp[0], Pp[0] are placeholders).
    Returns smoothed states xs (N x 6)."""
    n = len(xf)
    xs = [x.copy() for x in xf]
    Ps = [P.copy() for P in Pf]
    for k in range(n - 2, -1, -1):
        A = AA[k + 1]                       # transition step k -> k+1
        Ppred = Pp[k + 1]                   # P_pred[k+1] = A P_filt[k] A^T + Q
        try:
            Pinv = np.linalg.inv(Ppred)
        except np.linalg.LinAlgError:
            Pinv = np.linalg.pinv(Ppred)
        Ck = Pf[k] @ A.T @ Pinv
        xs[k] = xf[k] + Ck @ (xs[k + 1] - xp[k + 1])
        Ps[k] = Pf[k] + Ck @ (Ps[k + 1] - Pp[k + 1]) @ Ck.T
    return np.array(xs)


GATE_ANISO = np.array([0.4008175, 1.1093333, 0.4015686])   # reference max_distance_mm


def track_kf_rts(P, F, sigma_a=0.03, sigma_z=0.12, gate_chi2=9.0, v0=0.5,
                 max_gap=2, min_len=4, max_step_scale=np.inf):
    """Constant-velocity Kalman FILTER (association identical to track_bubbles.track_kf)
    with per-step internals stored, then RTS-smoothed per track on retire.

    Returns a list of track records, each a dict:
        frames (N,)   frame index of each matched step
        z      (N,3)  raw linked measurement (the un-smoothed detection positions)
        xf     (N,6)  KF forward-filtered state  (pos+vel)
        xs     (N,6)  RTS-smoothed state         (pos+vel)
    Same links across all three, so raw / Savgol(raw) / RTS are apples-to-apples."""
    I3 = np.eye(3)
    H = np.zeros((3, 6)); H[:, :3] = I3
    R = sigma_z ** 2 * I3
    nf = int(F.max()) + 1
    by_frame = [P[F == f] for f in range(nf)]

    X = np.zeros((0, 6)); Pc = np.zeros((0, 6, 6))
    last_f = np.zeros(0, int); missed = np.zeros(0, int)
    # per active-track stored internals (parallel python lists)
    frs, ZZ, XF, PF, XP, PP, AA = [], [], [], [], [], [], []
    done = []

    def retire(mask):
        nonlocal X, Pc, last_f, missed, frs, ZZ, XF, PF, XP, PP, AA
        for i in np.where(mask)[0]:
            if len(XF[i]) >= min_len:
                xs = _rts_smooth(XF[i], PF[i], XP[i], PP[i], AA[i])
                done.append(dict(frames=np.array(frs[i]), z=np.array(ZZ[i]),
                                 xf=np.array(XF[i]), xs=xs))
        keep = ~mask
        X, Pc, last_f, missed = X[keep], Pc[keep], last_f[keep], missed[keep]
        ki = np.where(keep)[0]
        frs = [frs[i] for i in ki]; ZZ = [ZZ[i] for i in ki]
        XF = [XF[i] for i in ki]; PF = [PF[i] for i in ki]
        XP = [XP[i] for i in ki]; PP = [PP[i] for i in ki]; AA = [AA[i] for i in ki]

    for f in range(nf):
        D = by_frame[f]
        T = len(X)
        if T and len(D):
            Xp = np.zeros_like(X); Sp = np.zeros((T, 3, 3)); zpred = np.zeros((T, 3))
            Pp_all = np.zeros_like(Pc)
            for t in range(T):
                A, Q = _A_Q(f - last_f[t], sigma_a)
                xp = A @ X[t]; pp = A @ Pc[t] @ A.T + Q
                Xp[t] = xp; Pp_all[t] = pp
                zpred[t] = H @ xp; Sp[t] = H @ pp @ H.T + R
            C = np.full((T, len(D)), BIG)
            for t in range(T):
                Si = np.linalg.inv(Sp[t])
                y = D - zpred[t]
                C[t] = np.einsum("mi,ij,mj->m", y, Si, y)
            C[C > gate_chi2] = BIG
            # operating-point knob (default off): hard per-frame step (speed) gate on the raw
            # displacement, scaled by the reference's anisotropic max_distance_mm. The
            # Mahalanobis gate alone cannot lower median speed; this is what trades coverage
            # for reference-matched speed. See docs/bubble-tracking-acq0.md.
            if np.isfinite(max_step_scale):
                dtf = np.maximum(1, f - last_f)[:, None]
                sd = np.linalg.norm((D[None, :, :] - X[:, :3][:, None, :]) / GATE_ANISO, axis=2) / dtf
                C[sd > max_step_scale] = BIG
            ri, ci = linear_sum_assignment(C)
            ok = C[ri, ci] < BIG
            ri, ci = ri[ok], ci[ok]
            upd = np.zeros(T, bool)
            for t, m in zip(ri, ci):
                A, Q = _A_Q(f - last_f[t], sigma_a)
                xp = A @ X[t]; pp = A @ Pc[t] @ A.T + Q     # prediction into this step
                S = H @ pp @ H.T + R
                K = pp @ H.T @ np.linalg.inv(S)
                X[t] = xp + K @ (D[m] - H @ xp)             # filtered state
                Pc[t] = (np.eye(6) - K @ H) @ pp
                last_f[t] = f; missed[t] = 0; upd[t] = True
                frs[t].append(f); ZZ[t].append(D[m].copy())
                XF[t].append(X[t].copy()); PF[t].append(Pc[t].copy())
                XP[t].append(xp.copy()); PP[t].append(pp.copy()); AA[t].append(A.copy())
            assigned = np.zeros(len(D), bool); assigned[ci] = True
            missed[~upd] += 1
            retire(missed > max_gap)
        else:
            assigned = np.zeros(len(D), bool)
            if T:
                missed += 1; retire(missed > max_gap)
        # spawn new tracks from unassigned detections
        for m in np.where(~assigned)[0]:
            x0 = np.concatenate([D[m], np.zeros(3)])
            P0 = np.diag([sigma_z ** 2] * 3 + [v0 ** 2] * 3)
            X = np.vstack([X, x0]); Pc = np.concatenate([Pc, P0[None]])
            last_f = np.append(last_f, f); missed = np.append(missed, 0)
            frs.append([f]); ZZ.append([D[m].copy()])
            XF.append([x0.copy()]); PF.append([P0.copy()])
            XP.append([x0.copy()]); PP.append([P0.copy()]); AA.append([np.eye(6)])
    retire(np.ones(len(X), bool))
    return done


# ---------- derive the three trajectory estimates from the records ----------

def tracks_raw(recs):     return [(r["z"], r["frames"]) for r in recs]
def tracks_filt(recs):    return [(r["xf"][:, :3], r["frames"]) for r in recs]
def tracks_rts(recs):     return [(r["xs"][:, :3], r["frames"]) for r in recs]
# Savgol = the BASELINE's shipped post-smoother: track_kf stores the KF-FILTERED position
# in its output, then smooth_tracks() (Savitzky-Golay) is applied on top. Both Savgol and
# RTS therefore post-process the SAME forward filter -> the fairest head-to-head, and this
# reproduces the documented baseline (turning 18.1, speed 40.2, cl 0.55).
def tracks_savgol(recs):  return smooth_tracks(tracks_filt(recs))


def cl_of(tracks):
    """Within-acq orientation coherence cl (mean over cells with >=4 segments). REPORTED,
    not targeted (reference within-acq cl is 0.72)."""
    from flow_diversity import coherence_map, segdirs_from_tracks
    polys = [{"positions": p, "acq_index": 0} for p, _ in tracks if len(p) >= 2]
    m, d, _ = segdirs_from_tracks(polys)
    cl, cn = coherence_map(m, d)
    o = cn >= 4
    return float(cl[o].mean()) if o.any() else float("nan"), int(o.sum())


def summarize(tracks, ref_pos):
    """ref_pos: list of reference position arrays (not (pos,frames) tuples)."""
    sp, ang = phys_plausibility(tracks)
    os1 = onestep_consistency(tracks)
    cl, ncell = cl_of(tracks)
    da, nsh = direction_agreement(tracks, ref_pos)
    return dict(turn=float(np.median(ang)) if len(ang) else float("nan"),
                speed=float(np.median(sp)) if len(sp) else float("nan"),
                onestep=float(np.median(os1)) if len(os1) else float("nan"),
                cl=cl, ncell=ncell, da=da, nsh=nsh)


def main():
    import os as _os
    _os.environ.setdefault("OMP_NUM_THREADS", "1")
    P, F, I, ref = load_acq0()
    ref_polys = [(p, np.arange(len(p))) for p in ref]
    ref_pos = [p for p in ref]

    print(f"acq-0: {len(P)} detections, {int(F.max())+1} frames")
    rsp, rang = phys_plausibility(ref_polys)
    ref_turn, ref_speed = float(np.median(rang)), float(np.median(rsp))
    rcl, rncell = cl_of([(p, np.arange(len(p))) for p in ref])
    print(f"REFERENCE (target): {len(ref)} tracks | turning {ref_turn:.1f} deg | "
          f"speed {ref_speed:.1f} mm/s | cl {rcl:.3f} ({rncell} cells)\n")

    # ---- run our KF (same association as track_kf) at the baseline operating point ----
    sa, g2 = 0.03, 9.0
    recs = track_kf_rts(P, F, sigma_a=sa, gate_chi2=g2)
    st = length_stats(tracks_raw(recs))
    print(f"OUR KF tracker (sigma_a={sa}, gate_chi2={g2}): {st['n']} tracks, {st['pts']} pts, "
          f"mean {st['mean']:.1f}, max {st['mx']}, {st['pts']/len(P)*100:.0f}% linked")
    print(f"   [baseline track_kf: 398 tracks, mean 8.2, max 71, 33% linked]\n")

    raw = tracks_raw(recs); filt = tracks_filt(recs)
    sav = tracks_savgol(recs); rts = tracks_rts(recs)

    S = {n: summarize(t, ref_pos) for n, t in
         [("raw", raw), ("filtered", filt), ("Savgol", sav), ("RTS", rts)]}

    # ---------- MAIN TABLE: trajectory-estimate quality ----------
    print("=" * 74)
    print("TRAJECTORY-ESTIMATE QUALITY  (same track set / same links for all rows)")
    print("=" * 74)
    print(f"  {'estimate':<10} {'turning':>9} {'speed':>8} {'one-step':>9} {'cl':>7} {'dir-agr':>8}")
    print(f"  {'':<10} {'(deg)':>9} {'(mm/s)':>8} {'(mm)':>9} {'':>7} {'(<30d)':>8}")
    print("  " + "-" * 60)
    for n in ("raw", "filtered", "Savgol", "RTS"):
        s = S[n]
        print(f"  {n:<10} {s['turn']:9.1f} {s['speed']:8.1f} {s['onestep']:9.3f} "
              f"{s['cl']:7.3f} {s['da']*100:7.0f}%")
    print("  " + "-" * 60)
    print(f"  {'BASELINE':<10} {18.1:9.1f} {40.2:8.1f} {0.200:9.3f} {0.550:7.3f} {70:7.0f}%   (track_kf Savgol)")
    print(f"  {'REFERENCE':<10} {ref_turn:9.1f} {ref_speed:8.1f} {'--':>9} {rcl:7.3f} {'--':>7}    (target)")
    print("=" * 74)

    # residual closed?
    b_turn, b_speed = 18.1, 40.2
    rts_turn, rts_speed = S["RTS"]["turn"], S["RTS"]["speed"]
    sav_turn, sav_speed = S["Savgol"]["turn"], S["Savgol"]["speed"]
    def pct_closed(base, val, target):
        gap0 = base - target
        return (base - val) / gap0 * 100 if abs(gap0) > 1e-9 else float("nan")
    print("\nTURNING residual toward reference (8.1 deg):")
    print(f"   Savgol {sav_turn:.1f} deg  -> {pct_closed(b_turn, sav_turn, ref_turn):+.0f}% of the 18.1->8.1 gap closed")
    print(f"   RTS    {rts_turn:.1f} deg  -> {pct_closed(b_turn, rts_turn, ref_turn):+.0f}% of the 18.1->8.1 gap closed")
    print("SPEED residual toward reference (26 mm/s):")
    print(f"   Savgol {sav_speed:.1f} mm/s -> {pct_closed(b_speed, sav_speed, ref_speed):+.0f}% of the 40.2->26 gap closed")
    print(f"   RTS    {rts_speed:.1f} mm/s -> {pct_closed(b_speed, rts_speed, ref_speed):+.0f}% of the 40.2->26 gap closed")

    # ---------- HONEST link-quality: what smoothing CANNOT fake ----------
    print("\n" + "=" * 74)
    print("LINK CORRECTNESS  (what smoothing CANNOT fake -- RTS does NOT fix these)")
    print("=" * 74)
    print("  ANCHOR (un-fakeable): one-step PREDICTION error on the FILTERED states = "
          f"{S['filtered']['onestep']:.3f} mm")
    print("     RTS is a BACKWARD pass -- it provably never alters the forward-filtered")
    print("     states, so this number is identical no matter how hard you smooth. It")
    print(f"     stays at the baseline 0.202 mm. Compare the smoothed OUTPUT one-steps:")
    print(f"       raw {S['raw']['onestep']:.3f}  Savgol {S['Savgol']['onestep']:.3f}  "
          f"RTS {S['RTS']['onestep']:.3f}  <- these DROP because smoothing MASKS residual,")
    print("     including any residual from a wrong link. So a low RTS one-step is NOT")
    print("     evidence the links are correct; the filtered 0.202 is the honest figure.")
    print(f"\n  direction agreement with reference:  filtered {S['filtered']['da']*100:.0f}%  "
          f"(raw {S['raw']['da']*100:.0f}% -> Savgol {S['Savgol']['da']*100:.0f}% -> "
          f"RTS {S['RTS']['da']*100:.0f}%)")
    print("     This one DOES rise with smoothing -- cleaner segments give a cleaner per-cell")
    print("     dominant orientation. That is PARTLY genuine estimate improvement and PARTLY")
    print("     masking (a marginal link straightened toward the local trend), so it is a")
    print("     WEAKER honesty guarantee than the filtered one-step. Bottom line: RTS")
    print("     improves the TRAJECTORY ESTIMATE; it does NOT re-associate detections, so it")
    print("     cannot fix a genuinely wrong link -- the filtered anchor is unchanged.")

    # ---------- OPTIONAL: sigma_a sweep -- the process noise matching physiology ----------
    print("\n" + "=" * 74)
    print("SIGMA_A SWEEP (RTS): process-noise that matches physiological smoothness")
    print("=" * 74)
    print("  sigma_a couples BOTH the gate (links) AND the smoothing, so track set shifts.")
    print(f"  {'sig_a':>6} {'ntr':>5} {'link%':>6} | {'RTSturn':>7} {'RTSsp':>6} {'RTScl':>6} "
          f"| {'filt1step':>9} {'filtDA':>6}")
    print("  " + "-" * 66)
    for sga in (0.01, 0.02, 0.03, 0.05, 0.08, 0.15, 0.30):
        rr = track_kf_rts(P, F, sigma_a=sga, gate_chi2=g2)
        sr = length_stats(tracks_raw(rr))
        rt = summarize(tracks_rts(rr), ref_pos)
        ft = summarize(tracks_filt(rr), ref_pos)
        mark = "  <- ~matches ref 8.1 deg" if abs(rt['turn'] - ref_turn) < 1.5 else ""
        print(f"  {sga:6.2f} {sr['n']:5d} {sr['pts']/len(P)*100:6.1f} | "
              f"{rt['turn']:7.1f} {rt['speed']:6.1f} {rt['cl']:6.3f} | "
              f"{ft['onestep']:9.3f} {ft['da']*100:5.0f}%{mark}")
    print("  " + "-" * 66)
    print("  Read: raise sigma_a to let RTS turn MORE (looser model). The sigma_a whose RTS")
    print("  turning ~ 8 deg is the physiologically-matched process noise; note its filtered")
    print("  one-step / dir-agreement (link quality) barely change -- tuning smoothness does")
    print("  NOT change which links are correct.")

    # ---------- render ----------
    try:
        render(filt, rts, ref_pos, sa, g2, S)
        print("\nwrote renders/track_rts_vs_reference.png")
    except Exception as e:
        print(f"\n[render skipped: {e}]")

    return S


def render(filt, rts, ref_pos, sa, g2, S):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    def panel(ax, tracks, title, framed=True):
        segs, cols = [], []
        for item in tracks:
            p, fr = item if framed else (item, np.arange(len(item)))
            if len(p) < 2:
                continue
            xz = p[:, [0, 2]]
            v = np.linalg.norm(np.diff(p, axis=0), axis=1) / (np.diff(fr) * DT + 1e-9)
            for i in range(len(xz) - 1):
                segs.append([xz[i], xz[i + 1]]); cols.append(min(v[i], 40) / 40)
        lc = LineCollection(segs, cmap="jet", linewidths=0.7, alpha=0.85)
        lc.set_array(np.array(cols)); ax.add_collection(lc)
        ax.set_facecolor("black"); ax.autoscale(); ax.set_aspect("equal")
        ax.set_title(title, color="white", fontsize=10.5); ax.tick_params(colors="0.6")

    fig, ax = plt.subplots(1, 3, figsize=(21, 6.6), facecolor="black")
    panel(ax[0], ref_pos, f"Reference acq-0 ({len(ref_pos)})\nturn 8.1 deg  cl 0.72",
          framed=False)
    panel(ax[1], filt, f"Ours: KF FILTERED (forward)\nturn {S['filtered']['turn']:.1f} deg  "
          f"cl {S['filtered']['cl']:.2f}")
    panel(ax[2], rts, f"Ours: RTS SMOOTHED\nturn {S['RTS']['turn']:.1f} deg  "
          f"cl {S['RTS']['cl']:.2f}  speed {S['RTS']['speed']:.0f} mm/s")
    fig.suptitle("RTS smoother vs KF filtered vs reference -- identical acq-0 links, "
                 "speed-colored jet 0-40 mm/s", color="white", fontsize=12.5)
    os.makedirs("renders", exist_ok=True)
    fig.savefig("renders/track_rts_vs_reference.png", dpi=140, facecolor="black",
                bbox_inches="tight")


if __name__ == "__main__":
    main()
