"""Multi-vector (crossing-preserving) tractography — the fix for the flow-diversity
gap (docs/flow-diversity-gap.md).

The single-vector field averages each voxel's velocity samples to ONE direction, so
streamlines are locally parallel (orientation coherence cl~0.99) — the "combed" look.
Our RAW samples actually carry the reference's diversity (cl~0.51). This module keeps
it: per voxel, cluster the sample directions into up to K flow MODES (greedy angular
clustering, samples pooled from an in-plane neighborhood to denoise), then integrate
streamlines that, at each step, follow the mode best aligned with the incoming
direction — so two vessels crossing a voxel stay distinct (like multi-fiber dMRI
tractography). Goal: pull streamline cl from 0.99 toward the reference ~0.53 while the
crossings still reproduce split-half (real structure, not noise).

CPU numpy only. Run niced when /proc/loadavg 1-min < 14.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from tractography_field import build_field, GS, SP, ORG

KMAX = 3


def build_multivector(mids, vels, ang_deg=32.0, min_frac=0.18, min_count=5, pool=1):
    """Per-voxel flow modes. Returns modes (GS,K,3) unit dirs, weights (GS,K) sample
    fractions, count (GS). Samples pooled over +/-pool in-plane voxels (denoise)."""
    idx = np.floor((mids - ORG) / SP).astype(int)
    ok = np.all((idx >= 0) & (idx < np.array(GS)), axis=1)
    idx = idx[ok]; v = vels[ok]
    nrm = np.linalg.norm(v, axis=1); good = nrm > 1e-9
    idx = idx[good]; d = v[good] / nrm[good, None]
    # bucket sample rows by voxel (+ in-plane neighbours) via a dict of lists
    from collections import defaultdict
    buck = defaultdict(list)
    lin = (idx[:, 0] * GS[1] + idx[:, 1]) * GS[2] + idx[:, 2]
    order = np.argsort(lin, kind="stable")
    lin_s, d_s, idx_s = lin[order], d[order], idx[order]
    bnd = np.searchsorted(lin_s, np.arange(lin_s[-1] + 2))
    modes = np.zeros(GS + (KMAX, 3)); weights = np.zeros(GS + (KMAX,)); cnt = np.zeros(GS)
    cos_th = np.cos(np.deg2rad(ang_deg))
    # precompute per-voxel sample index ranges
    uniq = np.unique(lin_s)
    starts = np.searchsorted(lin_s, uniq); ends = np.append(starts[1:], len(lin_s))
    span = {int(u): (int(s), int(e)) for u, s, e in zip(uniq, starts, ends)}
    for u, s, e in zip(uniq, starts, ends):
        i, j, k = idx_s[s]
        # pool in-plane neighbours
        rows = [(s, e)]
        if pool:
            for di in range(-pool, pool + 1):
                for dk in range(-pool, pool + 1):
                    if di == 0 and dk == 0:
                        continue
                    ii, kk = i + di, k + dk
                    if 0 <= ii < GS[0] and 0 <= kk < GS[2]:
                        nl = (ii * GS[1] + j) * GS[2] + kk
                        if int(nl) in span:
                            rows.append(span[int(nl)])
        dd = np.concatenate([d_s[a:b] for a, b in rows])
        cnt[i, j, k] = len(dd)
        if len(dd) < min_count:
            continue
        remaining = dd; ntot = len(dd); m = 0
        while len(remaining) >= max(min_count, int(min_frac * ntot)) and m < KMAX:
            mode = remaining.mean(0); mode /= (np.linalg.norm(mode) + 1e-9)
            for _ in range(4):
                sel = remaining @ mode > cos_th
                if sel.sum() == 0:
                    break
                mode = remaining[sel].mean(0); mode /= (np.linalg.norm(mode) + 1e-9)
            sel = remaining @ mode > cos_th
            if sel.sum() < max(min_count, int(min_frac * ntot)):
                break
            modes[i, j, k, m] = mode; weights[i, j, k, m] = sel.sum() / ntot; m += 1
            remaining = remaining[~sel]
    return modes, weights, cnt


def mv_tractogram(modes, weights, cnt, step=0.6, max_steps=600, nseed=4000,
                  seed=1, cthr_q=0.4, align_cos=0.5, min_len_mm=4.0):
    """Integrate crossing-preserving streamlines. At each voxel pick the mode whose
    (signed) direction best matches the incoming heading (>align_cos). Returns list of
    (pts_mm Nx3, speed-proxy N=weight)."""
    nx, ny, nz = GS
    cthr = np.quantile(cnt[cnt > 0], cthr_q)
    seedvox = np.argwhere((cnt >= cthr) & (weights[..., 0] > 0))
    rng = np.random.default_rng(seed)
    seedvox = seedvox[rng.choice(len(seedvox), min(nseed, len(seedvox)), replace=False)]
    SPi = 1.0 / SP

    def pick(p, prev):
        i, j, k = [int(round(x)) for x in p]
        if not (0 <= i < nx and 0 <= j < ny and 0 <= k < nz and cnt[i, j, k] >= cthr):
            return None
        ms = modes[i, j, k]; ws = weights[i, j, k]
        best = None; bestscore = -2
        for m in range(KMAX):
            if ws[m] <= 0:
                continue
            dmm = ms[m]
            if prev is None:
                score = ws[m]                     # seed: dominant mode
                cand = dmm
            else:
                dot = float(dmm @ prev)
                cand = dmm if dot >= 0 else -dmm  # resolve sign toward heading
                score = abs(dot)
                if score < align_cos:
                    continue
            if score > bestscore:
                bestscore = score; best = cand
        return best

    out = []
    for s in seedvox:
        fwd, bwd = [], []
        for sgn, acc in ((1, fwd), (-1, bwd)):
            p = s.astype(float).copy()
            prev = None
            # seed heading from dominant mode
            dom = modes[s[0], s[1], s[2], 0] * sgn
            prev = None
            for step_i in range(max_steps):
                d = pick(p, prev if prev is not None else dom)
                if d is None:
                    break
                acc.append(ORG + p * SP)
                di = d * SPi; di /= (np.linalg.norm(di) + 1e-9)   # index-space step (anisotropy)
                p = p + step * di; prev = d
        pts = [q for q in reversed(bwd)] + [q for q in fwd]
        if len(pts) < 2:
            continue
        pts = np.array(pts)
        if np.linalg.norm(np.diff(pts, axis=0), axis=1).sum() >= min_len_mm:
            out.append((pts, np.ones(len(pts))))
    return out


def mv_streamlets(modes, weights, cnt, half_steps=7, step=0.6, cthr_q=0.4,
                  align_cos=0.4, min_len_mm=1.2):
    """Render primitive: seed ONE short bidirectional streamlet per MODE per occupied
    voxel (so a 2-mode crossing emits two streamlets going different ways). Many short
    diverse primitives -> low cl, like the reference's short tracks. Returns polylines."""
    nx, ny, nz = GS
    cthr = np.quantile(cnt[cnt > 0], cthr_q)
    SPi = 1.0 / SP

    def pick(p, prev):
        i, j, k = [int(round(x)) for x in p]
        if not (0 <= i < nx and 0 <= j < ny and 0 <= k < nz and cnt[i, j, k] >= cthr):
            return None
        ms, ws = modes[i, j, k], weights[i, j, k]
        best, bs = None, align_cos
        for m in range(KMAX):
            if ws[m] <= 0:
                continue
            dot = float(ms[m] @ prev)
            cand = ms[m] if dot >= 0 else -ms[m]
            if abs(dot) > bs:
                bs = abs(dot); best = cand
        return best

    seeds = np.argwhere((cnt >= cthr) & (weights[..., 0] > 0))
    out = []
    for s in seeds:
        for m in range(KMAX):
            if weights[s[0], s[1], s[2], m] <= 0:
                continue
            d0 = modes[s[0], s[1], s[2], m]
            fwd, bwd = [], []
            for sgn, acc in ((1, fwd), (-1, bwd)):
                p = s.astype(float).copy(); prev = d0 * sgn
                for _ in range(half_steps):
                    d = pick(p, prev)
                    if d is None:
                        break
                    acc.append(ORG + p * SP)
                    di = d * SPi; di /= (np.linalg.norm(di) + 1e-9)
                    p = p + step * di; prev = d
            pts = [q for q in reversed(bwd)] + [q for q in fwd]
            if len(pts) >= 2:
                pts = np.array(pts)
                if np.linalg.norm(np.diff(pts, axis=0), axis=1).sum() >= min_len_mm:
                    out.append(pts)
    return out


if __name__ == "__main__":
    from flow_diversity import segdirs_from_streamlines, coherence_map
    FD = "outputs/reference/wf/r2/signal/velocity-field/"
    S = np.load(FD + "samples.npz")
    print("building multi-vector field...")
    modes, weights, cnt = build_multivector(S["mids"], S["vels"])
    nmodes = (weights > 0).sum(-1)
    occ = cnt >= np.quantile(cnt[cnt > 0], 0.4)
    print(f"occupied voxels {int(occ.sum())}; mode histogram (1/2/3): "
          f"{int((nmodes[occ]==1).sum())}/{int((nmodes[occ]==2).sum())}/{int((nmodes[occ]==3).sum())}")
    lines = mv_tractogram(modes, weights, cnt)
    fm, fd = segdirs_from_streamlines(lines)
    cl, c2 = coherence_map(fm, fd)
    o = c2 >= 4
    print(f"multi-vector streamlines: n={len(lines)}  mean_cl={cl[o].mean():.3f} "
          f"crossing_frac={(cl[o]<0.4).mean():.3f}  (target ref~0.53/0.35, was field 0.99/0.00)")
