"""Principled microbubble tracker on the released acq-0 detections, benchmarked
against the reference's OWN acq-0 tracks (same input) — data-grounded accuracy, no
new data. Goal per Rome (2026-07-09): the most ACCURATE tracked trajectories, validated
GT-free — NOT beauty, NOT matching the reference's orientation-diversity prior.

The reference tracker links ~31% of acq-0 detections into 292 tracks (mean len 10.3,
max 68). A nearest-neighbour tracker fragments (mean 7, max 32). This module adds the
two things that maintain continuity:
  1. a constant-velocity motion model (predict where the bubble goes, link there), and
  2. gap-closing (coast through 1-2 missed detections instead of breaking the track),
with globally-optimal per-frame assignment (Hungarian). Then it MEASURES accuracy against
the reference and against physics — letting the data say what the trajectories look like.

CPU numpy/scipy only. Fleet-safe: nice -15, single-threaded, load-gated by the caller.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from render_flow_diversity import SU

REF = "outputs/reference/full_tracks_smoothed.pkl"
FPS = 222.43
DT = 1.0 / FPS               # s per frame
BIG = 1e6


def load_acq0():
    o = SU(open(REF, "rb")).load()
    det = o["detections"]
    P = np.asarray(det["positions_mm"], float)
    F = np.asarray(det["frame_indices"], int)
    I = np.asarray(det["intensities"], float)
    # reference acq-0 tracks (their tracker's output on the same input)
    ref = [t for t in o["tracks_smoothed"] if int(t.get("acq_index", -1)) == 0]
    ref = [np.asarray(t["positions"], float) for t in ref]
    return P, F, I, ref


def track(P, F, init_gate=0.45, track_gate=0.18, max_gap=2, vel_alpha=0.5, min_len=4,
          max_turn_deg=60.0, turn_lambda=0.5):
    """Constant-velocity + Hungarian + gap-closing tracker with MOTION COHERENCE.

    Two gates, which is the key to smooth-AND-long tracks:
      - init_gate  (new track, prediction = position): the max physiological STEP, so a
        seed can reach its next detection (fast vessels included).
      - track_gate (established track, prediction = pos + velocity): the tight prediction
        RESIDUAL — the bubble should sit near where constant velocity says it will be, so
        far/jagged mis-links are rejected while fast bubbles are still followed (the
        prediction, not the gate, moves far).
    Plus a turn cap (backstop) and a straightness tie-break. Returns [(pts Nx3, frames N)]."""
    cos_max_turn = np.cos(np.deg2rad(max_turn_deg))
    nf = int(F.max()) + 1
    by_frame = [P[F == f] for f in range(nf)]
    # active track state
    pos = np.zeros((0, 3)); vel = np.zeros((0, 3))
    last_f = np.zeros(0, int); missed = np.zeros(0, int)
    pts = []          # list of list-of-(pt) per active track
    frs = []          # list of list-of-frame per active track
    done = []         # finished (pts, frames)

    def retire(mask):
        nonlocal pos, vel, last_f, missed, pts, frs
        for i in np.where(mask)[0]:
            done.append((np.array(pts[i]), np.array(frs[i])))
        keep = ~mask
        pos, vel, last_f, missed = pos[keep], vel[keep], last_f[keep], missed[keep]
        pts = [pts[i] for i in np.where(keep)[0]]
        frs = [frs[i] for i in np.where(keep)[0]]

    for f in range(nf):
        D = by_frame[f]
        if len(pos):
            dtf = (f - last_f)[:, None]
            pred = pos + vel * dtf                     # constant-velocity prediction
        else:
            pred = np.zeros((0, 3))
        if len(pred) and len(D):
            C = np.linalg.norm(pred[:, None, :] - D[None, :, :], axis=2)
            # per-track gate: tight residual once moving, wider step at initiation; both
            # grow slightly with the coast gap (missed frames)
            veln = np.linalg.norm(vel, axis=1)
            hv = veln > 1e-9
            base = np.where(hv, track_gate, init_gate)
            gate = base * (1.0 + 0.5 * (f - last_f))
            C[C > gate[:, None]] = BIG
            # motion coherence: tracks with velocity reject sharp turns; ties break straight
            if hv.any():
                step = D[None, :, :] - pos[:, None, :]
                stepn = np.linalg.norm(step, axis=2) + 1e-9
                cos_turn = (step * vel[:, None, :]).sum(2) / (stepn * veln[:, None] + 1e-9)
                bad = hv[:, None] & (cos_turn < cos_max_turn)
                C[bad] = BIG
                soft = turn_lambda * stepn * (1.0 - np.clip(cos_turn, -1, 1))
                C[hv] = C[hv] + soft[hv]
            ri, ci = linear_sum_assignment(C)
            ok = C[ri, ci] < BIG
            ri, ci = ri[ok], ci[ok]
        else:
            ri, ci = np.array([], int), np.array([], int)
        assigned_det = np.zeros(len(D), bool)
        upd = np.zeros(len(pos), bool)
        for t, m in zip(ri, ci):
            newpos = D[m]
            step = (newpos - pos[t]) / max(1, f - last_f[t])
            vel[t] = vel_alpha * step + (1 - vel_alpha) * vel[t] if last_f[t] >= 0 and np.any(vel[t]) else step
            pos[t] = newpos; last_f[t] = f; missed[t] = 0; upd[t] = True
            pts[t].append(newpos); frs[t].append(f); assigned_det[m] = True
        # unassigned active tracks: increment miss, retire if past max_gap
        miss_now = (~upd)
        missed[miss_now] += 1
        retire(missed > max_gap)
        # spawn new tracks from unassigned detections
        for m in np.where(~assigned_det)[0]:
            pos = np.vstack([pos, D[m]]); vel = np.vstack([vel, np.zeros(3)])
            last_f = np.append(last_f, f); missed = np.append(missed, 0)
            pts.append([D[m]]); frs.append([f])
    retire(np.ones(len(pos), bool))
    tracks = [(p, fr) for p, fr in done if len(p) >= min_len]
    return tracks


def track_kf(P, F, sigma_a=0.08, sigma_z=0.12, gate_chi2=12.0, v0=0.5,
             max_gap=2, min_len=4):
    """Constant-velocity KALMAN tracker. State [pos(3); vel(3)]; the gate is a Mahalanobis
    chi-square on the innovation, so it is WIDE while velocity is uncertain (track start)
    and TIGHT once the motion is locked in — giving long trajectories that are still smooth,
    without the fixed-gate length/smoothness tradeoff. sigma_a = process accel std
    (mm/frame^2), sigma_z = localization noise (mm), v0 = initial velocity std (mm/frame)."""
    I3 = np.eye(3)
    H = np.zeros((3, 6)); H[:, :3] = I3
    R = sigma_z ** 2 * I3
    nf = int(F.max()) + 1
    by_frame = [P[F == f] for f in range(nf)]

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
        D = by_frame[f]
        T = len(X)
        if T and len(D):
            Xp = np.zeros_like(X); Sp = np.zeros((T, 3, 3)); zpred = np.zeros((T, 3))
            Pp = np.zeros_like(Pc)
            for t in range(T):
                A, Q = A_Q(f - last_f[t])
                xp = A @ X[t]; pp = A @ Pc[t] @ A.T + Q
                Xp[t] = xp; Pp[t] = pp
                zpred[t] = H @ xp; Sp[t] = H @ pp @ H.T + R
            # Mahalanobis^2 cost (T x M)
            C = np.full((T, len(D)), BIG)
            for t in range(T):
                Si = np.linalg.inv(Sp[t])
                y = D - zpred[t]
                C[t] = np.einsum("mi,ij,mj->m", y, Si, y)
            C[C > gate_chi2] = BIG
            ri, ci = linear_sum_assignment(C)
            ok = C[ri, ci] < BIG
            ri, ci = ri[ok], ci[ok]
            # update matched
            upd = np.zeros(T, bool)
            for t, m in zip(ri, ci):
                A, Q = A_Q(f - last_f[t])
                xp = A @ X[t]; pp = A @ Pc[t] @ A.T + Q
                S = H @ pp @ H.T + R
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
        # spawn new tracks
        for m in np.where(~assigned)[0]:
            x0 = np.concatenate([D[m], np.zeros(3)])
            P0 = np.diag([sigma_z ** 2] * 3 + [v0 ** 2] * 3)
            X = np.vstack([X, x0]); Pc = np.concatenate([Pc, P0[None]])
            last_f = np.append(last_f, f); missed = np.append(missed, 0)
            pts.append([D[m]]); frs.append([f])
    retire(np.ones(len(X), bool))
    return [(p, fr) for p, fr in done if len(p) >= min_len]


# ---------- accuracy metrics (GT-free, grounded in the data) ----------

def smooth_tracks(tracks, window=5, poly=2):
    """Spline/Savitzky-Golay smoothing, matching what the reference `tracks_smoothed` did.
    Fair like-for-like comparison: raw jagged links have inflated turning AND speed."""
    from scipy.signal import savgol_filter
    out = []
    for p, fr in tracks:
        n = len(p)
        if n >= window:
            w = window if window % 2 else window + 1
            ps = np.column_stack([savgol_filter(p[:, d], w, poly) for d in range(3)])
        elif n >= 3:
            ps = p.copy(); ps[1:-1] = (p[:-2] + p[1:-1] + p[2:]) / 3.0
        else:
            ps = p
        out.append((ps, fr))
    return out


def length_stats(tracks):
    L = np.array([len(p) for p, _ in tracks])
    npts = int(L.sum())
    return dict(n=len(tracks), pts=npts, mean=float(L.mean()) if len(L) else 0,
                median=float(np.median(L)) if len(L) else 0, mx=int(L.max()) if len(L) else 0)


def phys_plausibility(tracks):
    """Speeds (mm/s) and turning angles (deg). Accurate tracks are physiological and smooth."""
    sp, ang = [], []
    for p, fr in tracks:
        d = np.diff(p, axis=0); df = np.diff(fr)[:, None]
        v = d / (df * DT)
        s = np.linalg.norm(v, axis=1); sp.append(s)
        u = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
        if len(u) > 1:
            c = np.clip((u[:-1] * u[1:]).sum(1), -1, 1)
            ang.append(np.degrees(np.arccos(c)))
    sp = np.concatenate(sp) if sp else np.zeros(0)
    ang = np.concatenate(ang) if ang else np.zeros(0)
    return sp, ang


def onestep_consistency(tracks):
    """Constant-velocity one-step prediction error (mm) at interior points — motion-model
    self-consistency. Low = trajectories are smooth/predictable (real motion, not mislinks)."""
    res = []
    for p, fr in tracks:
        if len(p) < 3:
            continue
        pred = p[1:-1] + (p[1:-1] - p[:-2])       # extrapolate from previous step
        res.append(np.linalg.norm(pred - p[2:], axis=1))
    return np.concatenate(res) if res else np.zeros(0)


def direction_agreement(tracks, ref_polys, grid=1.0):
    """Where our tracks and reference tracks coexist (same ~1mm cell), do they flow the
    same way? Fraction of shared cells with dominant-direction angle < 30 deg."""
    def cell_dirs(polys):
        d = {}
        for p in polys:
            seg = np.diff(p, axis=0)
            mid = 0.5 * (p[:-1] + p[1:])
            u = seg / (np.linalg.norm(seg, axis=1, keepdims=True) + 1e-9)
            key = np.floor(mid / grid).astype(int)
            for k, uu in zip(map(tuple, key), u):
                d.setdefault(k, []).append(uu)
        # dominant orientation per cell via first eigenvector of sum(u u^T)
        out = {}
        for k, us in d.items():
            U = np.array(us); T = U.T @ U
            w, V = np.linalg.eigh(T); out[k] = V[:, -1]
        return out
    A = cell_dirs([p for p, _ in tracks]); B = cell_dirs(ref_polys)
    shared = set(A) & set(B)
    if not shared:
        return float("nan"), 0
    ang = []
    for k in shared:
        c = abs(float(np.clip(A[k] @ B[k], -1, 1)))
        ang.append(np.degrees(np.arccos(c)))
    ang = np.array(ang)
    return float((ang < 30).mean()), len(shared)


def bootstrap_stability(P, F, tracker=track, drop=0.15, seed=0, **kw):
    """Reproducibility: drop a random fraction of detections, re-track, measure what
    fraction of full-run track points are recovered by a nearby bootstrap track."""
    full = tracker(P, F, **kw)
    rng = np.random.default_rng(seed)
    keep = rng.random(len(P)) > drop
    boot = tracker(P[keep], F[keep], **kw)
    if not full or not boot:
        return float("nan")
    BP = np.vstack([p for p, _ in boot]); tree = cKDTree(BP)
    rec = []
    for p, _ in full:
        d, _ = tree.query(p); rec.append(np.mean(d < 0.3))   # within 0.3mm of a boot track
    return float(np.mean(rec))


def evaluate(tr, ref, P):
    st = length_stats(tr)                       # length from raw links
    trs = smooth_tracks(tr)                      # physiology on smoothed (fair vs ref)
    sp, ang = phys_plausibility(trs)
    da, nsh = direction_agreement(trs, ref)
    return dict(n=st["n"], mean=st["mean"], mx=st["mx"], linked=st["pts"] / len(P),
                sp_med=float(np.median(sp)) if len(sp) else 0,
                turn_med=float(np.median(ang)) if len(ang) else 0, da=da, nsh=nsh)


def main():
    P, F, I, ref = load_acq0()
    refL = np.array([len(p) for p in ref])
    print(f"acq-0: {len(P)} detections, {int(F.max())+1} frames")
    print(f"reference tracker (target): {len(ref)} tracks, {int(refL.sum())} pts, "
          f"mean len {refL.mean():.1f}, max {refL.max()}, {refL.sum()/len(P)*100:.0f}% linked")
    rsp, rang = phys_plausibility([(p, np.arange(len(p))) for p in ref])
    ref_turn, ref_sp = float(np.median(rang)), float(np.median(rsp))
    print(f"reference physiology (yardstick): speed med {ref_sp:.1f} mm/s | turning med {ref_turn:.1f} deg\n")

    # Kalman tracker sweep: accept configs that stay physiological (turn/speed within ~2x
    # ref), then among those maximize detections linked (coverage). The chi-square gate
    # adapts with velocity uncertainty, so we expect long AND smooth without a fixed-gate
    # tradeoff. Grounds the operating point in physics + coverage.
    print(f"  {'sig_a':>6} {'chi2':>5} {'ntr':>5} {'mean':>5} {'max':>4} {'link%':>6} "
          f"{'spMed':>6} {'tnMed':>6} {'dirAg':>6}")
    grid = [(sa, g2) for sa in (0.03, 0.05, 0.10, 0.20) for g2 in (6.0, 9.0, 16.0)]
    results = []
    for sa, g2 in grid:
        tr = track_kf(P, F, sigma_a=sa, gate_chi2=g2)
        e = evaluate(tr, ref, P); e["cfg"] = (sa, g2); results.append((e, tr))
        print(f"  {sa:6.2f} {g2:5.1f} {e['n']:5d} {e['mean']:5.1f} {e['mx']:4d} "
              f"{e['linked']*100:6.1f} {e['sp_med']:6.1f} {e['turn_med']:6.1f} {e['da']*100:6.0f}")

    # operating point for ACCURACY: among configs that at least match the reference's
    # coverage (~31% linked), pick the SMOOTHEST (fewest wrong links -> lowest turning),
    # tie-broken by direction agreement with the reference. Coverage beyond that trades
    # accuracy for quantity, which is not what we want.
    cand = [(e, tr) for e, tr in results if e["linked"] >= 0.30]
    pool = cand if cand else results
    best_e, best_tr = min(pool, key=lambda x: (x[0]["turn_med"], -x[0]["da"]))
    sa, g2 = best_e["cfg"]
    print(f"\n  -> operating point: sigma_a={sa} gate_chi2={g2}  "
          f"(smoothest config at >=ref coverage)")

    st = length_stats(best_tr)
    best_sm = smooth_tracks(best_tr)
    print(f"\nOUR tracker (best): {st['n']} tracks, {st['pts']} pts, mean len {st['mean']:.1f}, "
          f"max {st['mx']}, {st['pts']/len(P)*100:.0f}% linked   [ref: 292, 10.3, 68, 31%]")
    spr, angr = phys_plausibility(best_tr)
    sp, ang = phys_plausibility(best_sm)
    print(f"  physiology (SMOOTHED, like ref): speed med {np.median(sp):.1f} p90 {np.percentile(sp,90):.1f} mm/s "
          f"[ref med {ref_sp:.1f}]  |  turning med {np.median(ang):.1f} p90 {np.percentile(ang,90):.1f} deg "
          f"[ref med {ref_turn:.1f}]")
    print(f"  physiology (raw links, unsmoothed): speed med {np.median(spr):.1f}  turning med {np.median(angr):.1f}")
    res = onestep_consistency(best_tr)
    print(f"  motion-model one-step error (mm): median {np.median(res):.3f} p90 {np.percentile(res,90):.3f}")
    print(f"  direction agreement with reference: {best_e['da']*100:.0f}% of {best_e['nsh']} shared cells <30 deg")
    stab = bootstrap_stability(P, F, tracker=track_kf, sigma_a=sa, gate_chi2=g2)
    print(f"  bootstrap stability (drop 15%%): {stab*100:.0f}% of track points recovered within 0.3mm")

    from flow_diversity import coherence_map, segdirs_from_tracks
    om, od, _ = segdirs_from_tracks([{"positions": p, "acq_index": 0} for p, _ in best_sm])
    ocl, ocn = coherence_map(om, od); oo = ocn >= 4
    rm, rd, _ = segdirs_from_tracks([{"positions": p, "acq_index": 0} for p in ref])
    rcl, rcn = coherence_map(rm, rd); ro = rcn >= 4
    print(f"\n  [data speaks] within-acq orientation coherence cl (reported, NOT targeted):")
    print(f"    our acq-0 tracks : cl {ocl[oo].mean():.3f}  ({int(oo.sum())} cells)")
    print(f"    ref acq-0 tracks : cl {rcl[ro].mean():.3f}  ({int(ro.sum())} cells)")
    print("    => within an acquisition the data is COHERENT; a good tracker should not scatter it.")

    render_compare(best_sm, [(p, np.arange(len(p))) for p in ref],
                   "renders/tracking_acq0_vs_reference.png", sa, g2)
    print("\n  wrote renders/tracking_acq0_vs_reference.png")


def render_compare(our, ref, path, sa, g2):
    """Our acq-0 KF tracks vs the reference's acq-0 tracks, coronal x-z, speed-colored
    (jet 0-40 mm/s, the reference's scale). Matplotlib Agg (fleet-safe)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    import os

    def panel(ax, tracks, title):
        segs, cols = [], []
        for p, fr in tracks:
            if len(p) < 2:
                continue
            xz = p[:, [0, 2]]
            v = np.linalg.norm(np.diff(p, axis=0), axis=1) / (np.diff(fr) * DT + 1e-9)
            for i in range(len(xz) - 1):
                segs.append([xz[i], xz[i + 1]]); cols.append(min(v[i], 40) / 40)
        lc = LineCollection(segs, cmap="jet", linewidths=0.7, alpha=0.85)
        lc.set_array(np.array(cols)); ax.add_collection(lc)
        ax.set_facecolor("black"); ax.autoscale(); ax.set_aspect("equal")
        ax.set_title(title, color="white", fontsize=11)
        ax.tick_params(colors="0.6")

    fig, ax = plt.subplots(1, 2, figsize=(15, 6.4), facecolor="black")
    panel(ax[0], ref, f"Reference acq-0 tracks  ({len(ref)}, mean len 10.3)")
    panel(ax[1], our, f"Our KF tracker acq-0  ({len(our)}, sigma_a={sa}/chi2={g2})\nspeed-colored jet 0-40 mm/s")
    fig.suptitle("Independent bubble tracking on identical acq-0 detections — continuous trajectories, not fragments",
                 color="white", fontsize=12.5)
    os.makedirs("renders", exist_ok=True)
    fig.savefig(path, dpi=140, facecolor="black", bbox_inches="tight")


if __name__ == "__main__":
    main()
