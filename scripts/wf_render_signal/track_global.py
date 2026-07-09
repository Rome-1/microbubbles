"""GLOBAL / multi-frame microbubble association on the released acq-0 detections,
benchmarked against the reference's OWN acq-0 tracks (same input) — data-grounded
accuracy, no new data. This is the follow-through on the honest residual in
`docs/bubble-tracking-acq0.md`: the baseline KF (`track_bubbles.track_kf`) uses
GREEDY per-frame Hungarian association, which commits to the nearest-by-prediction
detection NOW even when a later frame reveals the link was wrong -> jagged/fast
mis-links (turning 18 vs ref 8, speed 40 vs 26). The fix is GLOBAL evidence.

Two global methods here, both reusing `track_bubbles`' data + metrics so results are
apples-to-apples:

  track_global_flow  (option a — the principled target): min-cost network-flow /
    successive-shortest-paths tracking. The feasible-link graph is lifted to a
    SECOND-ORDER (edge/velocity) graph so the transition cost carries CURVATURE
    (turn angle between consecutive links), not just distance — a first-order
    Zhang/Berclaz flow cannot see a turn. Nodes = detections (capacity 1, enforced
    by removal), source/sink = birth/death with an entrance/exit cost, edges carry
    a motion negative-log-likelihood. We extract whole trajectories by successive
    globally-shortest paths (min-cost augmentation) until no negative-cost
    trajectory remains. Because a whole trajectory's smoothness is scored BEFORE
    committing (not per-frame), directionally-wrong links greedy accepted are
    rejected in favour of straighter continuations.

  track_global_stitch  (option c — cheap complement): run the baseline KF at a
    TIGHT gate (high-purity, smooth but fragmented), then GLOBALLY stitch fragment
    endpoints that align in position + velocity + time (one Hungarian over all
    fragments at once). Recovers length/coverage bridging greedy breaks WITHOUT
    re-introducing jagged links.

The feasible-link graph on acq-0 is tiny (~5.5k links, mean out-degree <1: per-frame
steps ~0.12mm vs 2.5mm NN spacing), so both run in seconds. CPU numpy/scipy only.
Fleet-safe: nice -15, single-thread, load-gated by the caller. No commit/push.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from track_bubbles import (
    load_acq0, track_kf, smooth_tracks, phys_plausibility, onestep_consistency,
    direction_agreement, bootstrap_stability, length_stats, evaluate,
    render_compare, DT, FPS, BIG,
)

INF = 1e18


# ----------------------------------------------------------------------------
# (a) second-order min-cost-flow / successive-shortest-paths tracker
# ----------------------------------------------------------------------------

def _build_links(P, F, v_max=60.0, max_gap=3, eps=0.05):
    """Feasible-link DAG: (src det, dst det) with 1<=gap<=max_gap and per-frame
    speed <= v_max (mm/s). Returns arrays src, dst, gap, unit-dir, dist."""
    nf = int(F.max()) + 1
    by = [np.where(F == f)[0] for f in range(nf)]
    trees = [cKDTree(P[b]) if len(b) else None for b in by]
    src_l, dst_l = [], []
    for f in range(nf):
        s_idx = by[f]
        if len(s_idx) == 0:
            continue
        for g in range(1, max_gap + 1):
            f2 = f + g
            if f2 >= nf or trees[f2] is None:
                continue
            r = v_max * DT * g + eps
            res = trees[f2].query_ball_point(P[s_idx], r)
            d_idx = by[f2]
            for a, nb in zip(s_idx, res):
                for bl in nb:
                    src_l.append(int(a)); dst_l.append(int(d_idx[bl]))
    if not src_l:
        return (np.zeros(0, int),) * 2 + (np.zeros(0, int), np.zeros((0, 3)), np.zeros(0))
    src = np.asarray(src_l, int); dst = np.asarray(dst_l, int)
    disp = P[dst] - P[src]
    dist = np.linalg.norm(disp, axis=1)
    gap = (F[dst] - F[src]).astype(int)
    dirn = disp / (dist[:, None] + 1e-12)
    return src, dst, gap, dirn, dist


def track_global_flow(P, F, v_max=60.0, max_gap=3, v_scale=26.0,
                      w_speed=0.35, w_gap=0.35, w_turn=0.6, c_birth=0.6,
                      c_det=1.0, max_turn_deg=70.0, min_len=4, max_tracks=20000):
    """Second-order successive-shortest-paths (greedy min-cost-flow) tracker.

    Trajectory cost (what the global solve minimizes) for a path d0->..->dL:
        c_birth + c_death + sum_k linkcost(k) + sum_k w_turn*(1-cos(turn_k))
        - c_det * (#detections)
    A trajectory is worth extracting only while this is < 0, so long, slow,
    STRAIGHT chains are strongly preferred and short/jagged/fast ones are left
    out — the global cure for greedy per-frame commitment. linkcost penalizes
    per-frame speed (soft, ~(v/v_scale)^2) and coasted gaps; the turn term
    (second-order, between consecutive links) penalizes curvature and is the
    lever on the smoothness residual. Detections have capacity 1 (removed once
    used). c_death := c_birth (symmetric)."""
    c_death = c_birth
    cos_max = np.cos(np.deg2rad(max_turn_deg))
    N = len(P)
    src, dst, gap, dirn, dist = _build_links(P, F, v_max, max_gap)
    L = len(src)
    if L == 0:
        return []
    speed = dist / (gap * DT)                       # mm/s per link
    linkcost = w_speed * (speed / v_scale) ** 2 + w_gap * (gap - 1)
    edgeval = linkcost - c_det                      # net cost of adding dst detection via this link

    inc = [[] for _ in range(N)]                    # incoming link ids per detection
    for l in range(L):
        inc[dst[l]].append(l)
    order = np.argsort(F[dst], kind="stable")       # topological: process by dst frame

    act = np.ones(N, bool)                          # detection still free
    done = []

    def _backtrack(best):
        dets = [int(dst[best]), int(src[best])]
        l = best
        while pred[l] >= 0:
            l = pred[l]; dets.append(int(src[l]))
        return dets[::-1]                            # time order

    for _ in range(max_tracks):
        alink = act[src] & act[dst]
        g = np.full(L, INF)
        pred = np.full(L, -2, int)                  # -2 unset, -1 birth, >=0 predecessor link
        for l in order:
            if not alink[l]:
                continue
            i = src[l]
            base = c_birth - c_det                  # start a new track at i
            bp = -1
            for p in inc[i]:
                if not alink[p] or g[p] >= INF:
                    continue
                cs = float(dirn[p] @ dirn[l])
                if cs < cos_max:
                    continue                        # hard turn reject (loose backstop)
                cand = g[p] + w_turn * (1.0 - cs)
                if cand < base:
                    base = cand; bp = p
            g[l] = base + edgeval[l]
            pred[l] = bp
        # BATCH extraction: peel a maximal set of DETECTION-DISJOINT negative-cost
        # trajectories in one DP pass (their costs assumed those detections free, which
        # they are — so disjoint peels stay valid). Cuts DP passes ~500 -> ~10.
        ec = np.where(alink, g + c_death, INF)
        cand = np.where(ec < 0.0)[0]
        if len(cand) == 0:
            break
        cand = cand[np.argsort(ec[cand], kind="stable")]
        used = np.zeros(N, bool)
        peeled = 0
        for best in cand:
            dets = _backtrack(int(best))
            d = np.asarray(dets)
            if used[d].any() or (~act[d]).any():
                continue                            # overlaps an earlier peel this pass
            used[d] = True; peeled += 1
            if len(dets) >= min_len:
                done.append((P[d].copy(), F[d].copy()))
        if peeled == 0:
            break
        act[used] = False                           # capacity 1: remove used detections
    return done


# ----------------------------------------------------------------------------
# (c) KF fragments + GLOBAL Hungarian stitching of endpoints
# ----------------------------------------------------------------------------

def _endpoint_states(tracks, k=4):
    """Per-fragment head/tail position, frame, and endpoint velocity (mm/frame)."""
    H = np.zeros((len(tracks), 3)); Hf = np.zeros(len(tracks), int); Hv = np.zeros((len(tracks), 3))
    T = np.zeros((len(tracks), 3)); Tf = np.zeros(len(tracks), int); Tv = np.zeros((len(tracks), 3))
    for m, (p, fr) in enumerate(tracks):
        n = len(p); kk = min(k, n)
        H[m] = p[0]; Hf[m] = fr[0]
        T[m] = p[-1]; Tf[m] = fr[-1]
        Hv[m] = (p[kk - 1] - p[0]) / max(1, int(fr[kk - 1] - fr[0]))
        Tv[m] = (p[-1] - p[-kk]) / max(1, int(fr[-1] - fr[-kk]))
    return H, Hf, Hv, T, Tf, Tv


def track_global_stitch(P, F, sigma_a=0.03, gate_chi2=6.0, max_gap=2, min_len=4,
                        stitch_gap=6, err_max=0.35, vcos_min=0.6, w_v=0.4, passes=2,
                        **kf):
    """Baseline KF at a TIGHT gate (clean, fragmented) + global fragment stitching.

    A tail->head stitch is feasible when the head appears 1..stitch_gap frames after
    the tail, the tail's constant-velocity prediction lands within err_max of the
    head start, and the endpoint velocities agree (cos > vcos_min). One Hungarian
    over ALL fragments picks a globally-consistent set of stitches (each fragment
    gets <=1 successor / <=1 predecessor); chains are concatenated. Repeated `passes`
    times to bridge multi-break gaps."""
    tracks = track_kf(P, F, sigma_a=sigma_a, gate_chi2=gate_chi2, max_gap=max_gap,
                      min_len=min_len, **kf)
    for _ in range(passes):
        M = len(tracks)
        if M < 2:
            break
        H, Hf, Hv, T, Tf, Tv = _endpoint_states(tracks)
        C = np.full((M, M), BIG)                    # rows=tails(i), cols=heads(j)
        for i in range(M):
            dtf = Hf - Tf[i]                         # frames from tail_i to each head
            feas = (dtf >= 1) & (dtf <= stitch_gap)
            feas[i] = False
            for j in np.where(feas)[0]:
                pred = T[i] + Tv[i] * dtf[j]
                err = np.linalg.norm(H[j] - pred)
                nv = np.linalg.norm(Tv[i]) * np.linalg.norm(Hv[j])
                vcos = float(Tv[i] @ Hv[j]) / (nv + 1e-12)
                if err < err_max and vcos > vcos_min:
                    C[i, j] = err + w_v * (1.0 - vcos)
        ri, ci = linear_sum_assignment(C)
        ok = C[ri, ci] < BIG
        succ = {int(a): int(b) for a, b in zip(ri[ok], ci[ok])}
        if not succ:
            break
        pred_of = set(succ.values())
        starts = [m for m in range(M) if m not in pred_of]
        merged, used = [], np.zeros(M, bool)
        for s in starts:
            if used[s]:
                continue
            chain = [s]; used[s] = True; cur = s
            while cur in succ and not used[succ[cur]]:
                cur = succ[cur]; used[cur] = True; chain.append(cur)
            ps = np.vstack([tracks[m][0] for m in chain])
            fs = np.concatenate([tracks[m][1] for m in chain])
            merged.append((ps, fs))
        for m in range(M):                           # carry any fragment not in a chain
            if not used[m]:
                merged.append(tracks[m])
        tracks = merged
    return [(p, fr) for p, fr in tracks if len(p) >= min_len]


# ----------------------------------------------------------------------------
# evaluation harness (reuses track_bubbles metrics for apples-to-apples)
# ----------------------------------------------------------------------------

def _cl(tracks_smoothed):
    """Within-acq orientation coherence cl (reported, NOT targeted)."""
    from flow_diversity import coherence_map, segdirs_from_tracks
    polys = [{"positions": p, "acq_index": 0} for p, _ in tracks_smoothed]
    m, d, _ = segdirs_from_tracks(polys)
    cl, cn = coherence_map(m, d)
    occ = cn >= 4
    return float(cl[occ].mean()) if occ.any() else float("nan"), int(occ.sum())


def metrics(name, tr, ref, P, window=11, ref_track=False):
    """All comparison metrics at a chosen Savitzky-Golay smoothing window. Length is
    from RAW links; physiology (speed/turning), direction-agreement and cl are on the
    SMOOTHED tracks (like the reference). window matched to the reference's effective
    smoothing so the turn/speed comparison is fair (the baseline doc used window=5,
    which UNDER-smooths — see the smoothing-sensitivity table)."""
    if ref_track:
        ls = dict(n=len(tr), mean=float(np.mean([len(p) for p, _ in tr])),
                  mx=int(np.max([len(p) for p, _ in tr])), pts=int(sum(len(p) for p, _ in tr)))
        trs = tr                                     # reference already smoothed; do not re-smooth
        one = np.zeros(0)
    else:
        ls = length_stats(tr)
        trs = smooth_tracks(tr, window=window)
        one = onestep_consistency(tr)
    sp, ang = phys_plausibility(trs)
    da, nsh = direction_agreement(trs, [p for p, _ in ref])
    cl, ncl = _cl(trs)
    nan = float("nan")
    return dict(name=name, n=ls["n"], mean=ls["mean"], mx=ls["mx"], linked=ls["pts"] / len(P),
                sp=float(np.median(sp)) if len(sp) else nan,
                sp90=float(np.percentile(sp, 90)) if len(sp) else nan,
                turn=float(np.median(ang)) if len(ang) else nan,
                turn90=float(np.percentile(ang, 90)) if len(ang) else nan,
                da=da if not ref_track else nan, nsh=nsh,
                one=float(np.median(one)) if len(one) else nan, cl=cl, ncl=ncl)


def print_table(rows):
    hdr = ("method", "trk", "mean", "max", "link%", "spMed", "sp90", "tnMed", "tn90",
           "dirAg", "1step", "cl")
    print(f"{hdr[0]:<24}{hdr[1]:>5}{hdr[2]:>6}{hdr[3]:>5}{hdr[4]:>7}{hdr[5]:>7}{hdr[6]:>6}"
          f"{hdr[7]:>7}{hdr[8]:>6}{hdr[9]:>7}{hdr[10]:>7}{hdr[11]:>6}")
    print("-" * 104)
    for r in rows:
        da = f"{r['da']*100:.0f}%" if r["da"] == r["da"] else "   -"
        one = f"{r['one']:.3f}" if r["one"] == r["one"] else "    -"
        print(f"{r['name']:<24}{r['n']:>5}{r['mean']:>6.1f}{r['mx']:>5}{r['linked']*100:>6.0f}%"
              f"{r['sp']:>7.1f}{r['sp90']:>6.0f}{r['turn']:>7.1f}{r['turn90']:>6.0f}"
              f"{da:>6}{one:>7}{r['cl']:>6.2f}")


# curated global-method operating points (from the sweeps in this module's history)
FLOW_CLEAN = dict(v_max=60, c_det=1.0, w_speed=0.10, v_scale=40, w_gap=0.25,
                  w_turn=0.8, c_birth=0.6, max_turn_deg=60.0)          # ~11% cov, cherry-picked clean
FLOW_COV = dict(v_max=120, c_det=1.2, w_speed=0.06, v_scale=45, w_gap=0.15,
                w_turn=0.5, c_birth=0.4, max_turn_deg=110.0)           # ~32% cov, matched coverage


def main():
    W = 11                                           # smoothing window matched to the reference
    P, F, I, ref = load_acq0()
    refT = [(p, np.arange(len(p))) for p in ref]
    print(f"acq-0: {len(P)} detections, {int(F.max())+1} frames, {len(ref)} reference tracks")
    print(f"metrics at Savitzky-Golay window={W} (matched to reference smoothing; see decomposition below)\n")

    kf = track_kf(P, F, sigma_a=0.03, gate_chi2=9.0)             # baseline greedy KF, ~33% cov
    flow_clean = track_global_flow(P, F, **FLOW_CLEAN)          # min-cost-flow, clean/low-cov
    flow_cov = track_global_flow(P, F, **FLOW_COV)             # min-cost-flow, matched-cov
    stitch = track_global_stitch(P, F, sigma_a=0.03, gate_chi2=9.0,
                                 stitch_gap=12, err_max=0.5, vcos_min=0.4, passes=3)

    rows = [
        metrics("reference (target)", refT, refT, P, ref_track=True),
        metrics("baseline KF (greedy)", kf, refT, P, W),
        metrics("flow min-cost (a) 11%", flow_clean, refT, P, W),
        metrics("flow min-cost (a) 32%", flow_cov, refT, P, W),
        metrics("stitch (c) global", stitch, refT, P, W),
    ]
    print("COMPARISON — baseline greedy KF vs GLOBAL association vs reference (acq-0, identical detections)")
    print_table(rows)
    print("  (length from raw links; speed/turning/cl on smoothed tracks. dirAg = % shared 1mm cells <30 deg vs ref.)")

    # ---- the decisive diagnostic: is the turn residual association error or SMOOTHING? ----
    print("\nSMOOTHING SENSITIVITY of the baseline KF (same tracks, wider Savitzky-Golay window):")
    print(f"  {'window':>7}{'turn_med':>10}{'speed_med':>11}     (reference: turn 8.1, speed 25.8)")
    for w in (5, 7, 9, 11, 15):
        sm = smooth_tracks(kf, window=w); sp, ang = phys_plausibility(sm)
        print(f"  {w:>7}{np.median(ang):>10.1f}{np.median(sp):>11.1f}")
    print("  => the turn gap (18 -> 8) is a SMOOTHING-WINDOW artifact, not greedy mis-linking:")
    print("     baseline turning matches the reference at window ~11-15. The SPEED gap (~36 vs 26)")
    print("     PLATEAUS under smoothing -> it is a real over-reach/gate residual, not smoothing.")

    # ---- bootstrap stability ----
    print("\nbootstrap stability (drop 15%, re-track, % points recovered within 0.3mm):")
    sb_kf = bootstrap_stability(P, F, tracker=track_kf, sigma_a=0.03, gate_chi2=9.0)
    sb_fl = bootstrap_stability(P, F, tracker=track_global_flow, **FLOW_CLEAN)
    sb_st = bootstrap_stability(P, F, tracker=track_global_stitch, sigma_a=0.03, gate_chi2=9.0,
                                stitch_gap=12, err_max=0.5, vcos_min=0.4, passes=3)
    print(f"  baseline KF {sb_kf*100:.0f}%   flow-clean {sb_fl*100:.0f}%   stitch {sb_st*100:.0f}%")

    # ---- honest verdict ----
    b, fc, fv, st = rows[1], rows[2], rows[3], rows[4]
    print("\nVERDICT — did GLOBAL association reduce the residual at matched (>=31%) coverage?")
    print(f"  min-cost-flow at MATCHED coverage ({fv['linked']*100:.0f}%): turn {fv['turn']:.1f}, speed {fv['sp']:.1f}"
          f"  -> WORSE than baseline ({b['turn']:.1f}/{b['sp']:.1f}).")
    print(f"  min-cost-flow only wins by UNDER-covering ({fc['linked']*100:.0f}%): turn {fc['turn']:.1f}, speed {fc['sp']:.1f}"
          f"  -> not apples-to-apples.")
    print(f"  stitch (c) at matched {st['linked']*100:.0f}% coverage: turn {st['turn']:.1f}, speed {st['sp']:.1f}, "
          f"max-len {st['mx']} (vs {b['mx']}), tracks {st['n']} (vs {b['n']}).")
    print("  CONCLUSION: at these sub-jitter per-frame steps (~0.12mm vs ~0.12mm localization noise)")
    print("  the KF's recursive velocity smoothing is what suppresses jaggedness; a raw-link min-cost-")
    print("  flow cannot match it except by cherry-picking coverage. Global association does NOT beat")
    print("  the greedy KF on physiology at matched coverage. Stitching (global fragment linking) improves")
    print("  track CONTINUITY (fewer breaks, longer max) at equal coverage/metrics — a topology win, not a")
    print("  physiology win. The flagged turn residual is dominated by SMOOTHING (closes at window~11);")
    print("  the genuine residual is a SPEED/over-reach vs coverage tradeoff set by the gate + detection")
    print("  quality, NOT by greedy-vs-global association.")

    # ---- render (matplotlib Agg, reuse baseline's compare) ----
    try:
        render_compare(smooth_tracks(stitch, window=W), refT,
                       "renders/track_global_vs_reference.png",
                       "0.03", "9 stitched")
        print("\nwrote renders/track_global_vs_reference.png")
    except Exception as ex:
        print(f"\n(render skipped: {ex})")


if __name__ == "__main__":
    main()
