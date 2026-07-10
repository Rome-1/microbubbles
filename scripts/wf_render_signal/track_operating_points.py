"""The two honest operating points of the Kalman bubble tracker, side by side with the
reference (acq-0, identical detections). Per docs/bubble-tracking-acq0.md the only genuine
residual is a gate-vs-coverage SELECTIVITY tradeoff:

  - HIGH-COVERAGE  : loose gate -> track more bubbles, some reach farther (higher median speed).
  - REFERENCE-MATCHED (tight gate): cap the per-frame step so median speed ~= the reference's
    25.8 mm/s -> fewer, slower, cleaner tracks (the reference discards ~69% of detections).

The knob is a hard per-frame step gate (scaled by the reference's anisotropic max_distance_mm),
added on top of the Kalman Mahalanobis gate — the Mahalanobis gate alone does not lower median
speed. Physiology/render use Savgol window 11 (matches the reference's Gaussian sigma=2 smoothing).

CPU numpy/scipy; fleet-safe (nice -15, single-thread, Agg). Output:
renders/track_operating_points.png
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy.optimize import linear_sum_assignment
from track_bubbles import (load_acq0, smooth_tracks, phys_plausibility, direction_agreement,
                           onestep_consistency, length_stats, DT, BIG)

GATE_ANISO = np.array([0.4008175, 1.1093333, 0.4015686])   # reference max_distance_mm


def track_kf_step(P, F, sigma_a=0.05, sigma_z=0.12, gate_chi2=9.0, max_step_scale=2.0,
                  v0=0.5, max_gap=2, min_len=4):
    """track_bubbles.track_kf + a hard per-frame step (speed) gate: reject any link whose
    per-frame displacement (scaled by the reference's anisotropic max_distance) exceeds
    max_step_scale. Small scale -> tight/slow/selective; large -> high coverage."""
    I3 = np.eye(3); H = np.zeros((3, 6)); H[:, :3] = I3; R = sigma_z ** 2 * I3
    nf = int(F.max()) + 1
    by_frame = [P[F == f] for f in range(nf)]

    def A_Q(dt):
        A = np.eye(6); A[:3, 3:] = dt * I3; q = sigma_a ** 2; Q = np.zeros((6, 6))
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
        pts = [pts[i] for i in np.where(keep)[0]]; frs = [frs[i] for i in np.where(keep)[0]]

    for f in range(nf):
        D = by_frame[f]; T = len(X)
        if T and len(D):
            Sp = np.zeros((T, 3, 3)); zpred = np.zeros((T, 3)); Pp = np.zeros_like(Pc); Xp = np.zeros_like(X)
            for t in range(T):
                A, Q = A_Q(f - last_f[t]); xp = A @ X[t]; pp = A @ Pc[t] @ A.T + Q
                Xp[t] = xp; Pp[t] = pp; zpred[t] = H @ xp; Sp[t] = H @ pp @ H.T + R
            C = np.full((T, len(D)), BIG)
            for t in range(T):
                y = D - zpred[t]; C[t] = np.einsum("mi,ij,mj->m", y, np.linalg.inv(Sp[t]), y)
            C[C > gate_chi2] = BIG
            # hard per-frame step (speed) gate on the raw displacement from last position
            dtf = np.maximum(1, f - last_f)[:, None]
            sd = np.linalg.norm((D[None, :, :] - X[:, :3][:, None, :]) / GATE_ANISO, axis=2) / dtf
            C[sd > max_step_scale] = BIG
            ri, ci = linear_sum_assignment(C); ok = C[ri, ci] < BIG; ri, ci = ri[ok], ci[ok]
            upd = np.zeros(T, bool)
            for t, m in zip(ri, ci):
                A, Q = A_Q(f - last_f[t]); xp = A @ X[t]; pp = A @ Pc[t] @ A.T + Q
                S = H @ pp @ H.T + R; K = pp @ H.T @ np.linalg.inv(S)
                X[t] = xp + K @ (D[m] - H @ xp); Pc[t] = (np.eye(6) - K @ H) @ pp
                last_f[t] = f; missed[t] = 0; upd[t] = True
                pts[t].append(X[t][:3].copy()); frs[t].append(f)
            assigned = np.zeros(len(D), bool); assigned[ci] = True
            missed[~upd] += 1; retire(missed > max_gap)
        else:
            assigned = np.zeros(len(D), bool)
            if T:
                missed += 1; retire(missed > max_gap)
        for m in np.where(~assigned)[0]:
            X = np.vstack([X, np.concatenate([D[m], np.zeros(3)])])
            Pc = np.concatenate([Pc, np.diag([sigma_z ** 2] * 3 + [v0 ** 2] * 3)[None]])
            last_f = np.append(last_f, f); missed = np.append(missed, 0)
            pts.append([D[m]]); frs.append([f])
    retire(np.ones(len(X), bool))
    return [(p, fr) for p, fr in done if len(p) >= min_len]


def metrics(tr, ref, P, name):
    st = length_stats(tr); sm = smooth_tracks(tr, window=11)
    sp, ang = phys_plausibility(sm); da, nsh = direction_agreement(sm, ref)
    from flow_diversity import coherence_map, segdirs_from_tracks
    m, d, _ = segdirs_from_tracks([{"positions": p, "acq_index": 0} for p, _ in sm])
    cl, cn = coherence_map(m, d); cl = float(cl[cn >= 4].mean())
    return dict(name=name, n=st["n"], mean=st["mean"], mx=st["mx"], linked=st["pts"] / len(P),
                sp=float(np.median(sp)), sp90=float(np.percentile(sp, 90)),
                turn=float(np.median(ang)), da=da, cl=cl, sm=sm)


def main():
    P, F, I, ref = load_acq0()
    refm = phys_plausibility([(p, np.arange(len(p))) for p in ref])
    ref_sp = float(np.median(refm[0]))
    print(f"acq-0: {len(P)} detections, {len(ref)} reference tracks, ref speed median {ref_sp:.1f} mm/s\n")

    # HIGH-COVERAGE: loose gate, track aggressively
    hi = track_kf_step(P, F, sigma_a=0.10, gate_chi2=16, max_step_scale=2.0)

    # REFERENCE-MATCHED: sweep the step gate down until smoothed median speed ~= reference
    print(f"  tuning tight gate to reference speed ({ref_sp:.1f} mm/s):")
    best = None
    for s in (1.2, 1.0, 0.8, 0.6, 0.45, 0.35):
        tr = track_kf_step(P, F, sigma_a=0.05, gate_chi2=9, max_step_scale=s)
        spmed = float(np.median(phys_plausibility(smooth_tracks(tr, window=11))[0]))
        link = sum(len(p) for p, _ in tr) / len(P)
        print(f"    step_scale={s:4.2f}: speed {spmed:5.1f} mm/s, linked {link*100:4.1f}%")
        if best is None or abs(spmed - ref_sp) < abs(best[1] - ref_sp):
            best = (s, spmed, tr)
    lo = best[2]
    print(f"  -> reference-matched step_scale={best[0]} (speed {best[1]:.1f})\n")

    rows = [metrics(hi, ref, P, "HIGH-COVERAGE (loose)"),
            metrics(lo, ref, P, f"REFERENCE-MATCHED (step<={best[0]})")]
    print(f"  {'operating point':<34} {'trk':>4} {'mean':>5} {'max':>4} {'link%':>6} "
          f"{'spMed':>6} {'sp90':>6} {'turn':>5} {'cl':>5} {'dirAg':>6}")
    print(f"  {'REFERENCE (target)':<34} {len(ref):>4} {10.3:>5.1f} {68:>4} {31.0:>6.1f} "
          f"{ref_sp:>6.1f} {np.percentile(refm[0],90):>6.1f} {np.median(refm[1]):>5.1f} {0.73:>5.2f} {'-':>6}")
    for r in rows:
        print(f"  {r['name']:<34} {r['n']:>4} {r['mean']:>5.1f} {r['mx']:>4} {r['linked']*100:>6.1f} "
              f"{r['sp']:>6.1f} {r['sp90']:>6.1f} {r['turn']:>5.1f} {r['cl']:>5.2f} {r['da']*100:>6.0f}")

    render(ref, rows[0], rows[1])
    print("\n  wrote renders/track_operating_points.png")


def render(ref, hi, lo):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    def panel(ax, tracks, title):
        segs, cols = [], []
        for p, fr in tracks:
            if len(p) < 2:
                continue
            xz = p[:, [0, 2]]; v = np.linalg.norm(np.diff(p, axis=0), axis=1) / (np.diff(fr) * DT + 1e-9)
            for i in range(len(xz) - 1):
                segs.append([xz[i], xz[i + 1]]); cols.append(min(v[i], 40) / 40)
        lc = LineCollection(segs, cmap="jet", linewidths=0.7, alpha=0.85); lc.set_array(np.array(cols))
        ax.add_collection(lc); ax.set_facecolor("black"); ax.autoscale(); ax.set_aspect("equal")
        ax.set_title(title, color="white", fontsize=10.5); ax.tick_params(colors="0.6")

    fig, ax = plt.subplots(1, 3, figsize=(20, 6.2), facecolor="black")
    panel(ax[0], [(p, np.arange(len(p))) for p in ref], f"REFERENCE  ({len(ref)} tracks, speed med 25.8)")
    panel(ax[1], hi["sm"], f"HIGH-COVERAGE  ({hi['n']} tr, {hi['linked']*100:.0f}% linked, "
                           f"speed {hi['sp']:.0f}, cl {hi['cl']:.2f})")
    panel(ax[2], lo["sm"], f"REFERENCE-MATCHED tight gate  ({lo['n']} tr, {lo['linked']*100:.0f}% linked, "
                           f"speed {lo['sp']:.0f}, cl {lo['cl']:.2f})")
    fig.suptitle("Two honest operating points of the Kalman bubble tracker vs reference "
                 "(acq-0, speed-colored jet 0-40 mm/s)", color="white", fontsize=13)
    os.makedirs("renders", exist_ok=True)
    fig.savefig("renders/track_operating_points.png", dpi=140, facecolor="black", bbox_inches="tight")


if __name__ == "__main__":
    main()
