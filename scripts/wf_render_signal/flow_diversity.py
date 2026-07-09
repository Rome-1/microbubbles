"""Structural flow-DIVERSITY analysis: why the reference render looks like richly
varied vasculature and ours looks like long parallel streamlines.

Hypothesis (Rome's visual observation, 2026-07-09): our build_field AVERAGES
velocity to ONE vector per voxel, and heavy along-vessel smoothing straightens
everything into long parallel streamlines. Real vasculature has crossing/branching
vessels -> many flow directions coexisting locally. Our split-half Dice/cosine
compass REWARDS smooth parallel fields, so it misled us on visual/structural fidelity.

Metric: per-voxel ORIENTATION TENSOR T = mean(d (x) d) over the unit segment
directions passing through the voxel (d(x)d makes it orientation, not signed dir).
Eigenvalues l1>=l2>=l3 (sum 1). Coherence cl = (l1-l2)/(l1+l2+l3):
  cl ~ 1  -> one dominant direction (parallel / coherent)
  cl low  -> dispersed: crossings / branches / diverse flows
Report per source: mean cl, crossing-fraction (cl<0.4), global in-plane orientation
entropy, and split-half reproducibility of the cl map (real crossings reproduce,
noise doesn't). In-plane (x-z) variant sidesteps the coarse elevation axis.

Sources compared, all rasterized to the coarsen-2 grid:
  ref   = reference tracks_smoothed segments (clean linked bubble tracks, all 216 acqs)
  field = OUR graph-regularized field streamlines (what we render)
  raw   = OUR raw per-localization velocity samples (before we averaged them away)

Fleet-safe: numpy/scipy only. Run niced when /proc/loadavg 1-min < 14.
"""
from __future__ import annotations
import sys, os, pickle
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from tractography_field import build_field, GS, SP, ORG
from tractography_pde import regularize, streamlines

FD = "outputs/reference/wf/r2/signal/velocity-field/"
REF = "outputs/reference/full_tracks_smoothed.pkl"
SAFE = {("numpy._core.multiarray", "_reconstruct"), ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "scalar"), ("numpy.core.multiarray", "scalar"),
        ("numpy", "ndarray"), ("numpy", "dtype")}
class SU(pickle.Unpickler):
    def find_class(self, m, n):
        for mm in (m, m.replace("numpy._core", "numpy.core"), m.replace("numpy.core", "numpy._core")):
            if (mm, n) in SAFE:
                return super().find_class(mm, n)
        raise pickle.UnpicklingError(m + "." + n)


def vidx(p):
    return np.floor((p - ORG) / SP).astype(int)


def coherence_map(mids, dirs, inplane=False):
    """Per-voxel orientation coherence cl=(l1-l2)/sum from segment midpoints+unit dirs.
    Returns cl grid (GS), count grid. inplane=True zeros the elevation (y) component."""
    d = dirs.copy().astype(float)
    if inplane:
        d[:, 1] = 0.0
    n = np.linalg.norm(d, axis=1)
    ok = n > 1e-9
    d = d[ok] / n[ok, None]; m = mids[ok]
    idx = vidx(m)
    inb = np.all((idx >= 0) & (idx < np.array(GS)), axis=1)
    idx, d = idx[inb], d[inb]
    lin = (idx[:, 0] * GS[1] + idx[:, 1]) * GS[2] + idx[:, 2]
    nv = GS[0] * GS[1] * GS[2]
    # accumulate outer products d d^T per voxel (6 unique components)
    T = np.zeros((nv, 6)); cnt = np.zeros(nv)
    comp = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    for c, (a, b) in enumerate(comp):
        np.add.at(T[:, c], lin, d[:, a] * d[:, b])
    np.add.at(cnt, lin, 1.0)
    cl = np.zeros(nv)
    good = cnt >= 4
    Tg = T[good] / cnt[good, None]
    mats = np.zeros((Tg.shape[0], 3, 3))
    for c, (a, b) in enumerate(comp):
        mats[:, a, b] = Tg[:, c]; mats[:, b, a] = Tg[:, c]
    w = np.linalg.eigvalsh(mats)                    # ascending l3<=l2<=l1
    l1, l2 = w[:, 2], w[:, 1]
    s = w.sum(1) + 1e-12
    cl[good] = (l1 - l2) / s
    return cl.reshape(GS), cnt.reshape(GS)


def segdirs_from_tracks(tracks):
    """midpoints + unit directions of every track segment; plus acq parity per seg."""
    mids, dirs, par = [], [], []
    for t in tracks:
        p = np.asarray(t["positions"], float)
        if len(p) < 2:
            continue
        dp = np.diff(p, axis=0)
        mids.append(0.5 * (p[:-1] + p[1:])); dirs.append(dp)
        par.append(np.full(len(dp), int(t["acq_index"]) & 1))
    return np.concatenate(mids), np.concatenate(dirs), np.concatenate(par)


def segdirs_from_streamlines(lines):
    mids, dirs = [], []
    for p, _ in lines:
        if len(p) < 2:
            continue
        dp = np.diff(p, axis=0)
        mids.append(0.5 * (p[:-1] + p[1:])); dirs.append(dp)
    return np.concatenate(mids), np.concatenate(dirs)


def orient_entropy(dirs, inplane=True, nbins=36):
    """Global orientation entropy of in-plane angles (0..pi, orientation not dir)."""
    d = dirs.astype(float)
    ang = np.arctan2(d[:, 2], d[:, 0]) % np.pi        # x-z plane angle mod pi
    h, _ = np.histogram(ang, bins=nbins, range=(0, np.pi))
    p = h / (h.sum() + 1e-12); p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(nbins))   # normalized 0..1


def summarize(name, mids, dirs, par=None):
    for tag, ip in (("3D", False), ("xz", True)):
        cl, cnt = coherence_map(mids, dirs, inplane=ip)
        occ = cnt >= 4
        meancl = float(cl[occ].mean())
        crossing = float((cl[occ] < 0.4).mean())
        ent = orient_entropy(dirs, inplane=ip)
        extra = ""
        if par is not None:                            # split-half repro of cl map
            clo, _ = coherence_map(mids[par == 0], dirs[par == 0], inplane=ip)
            cle, _ = coherence_map(mids[par == 1], dirs[par == 1], inplane=ip)
            both = (coherence_map(mids[par == 0], dirs[par == 0], inplane=ip)[1] >= 4) & \
                   (coherence_map(mids[par == 1], dirs[par == 1], inplane=ip)[1] >= 4)
            if both.sum() > 20:
                r = np.corrcoef(clo[both], cle[both])[0, 1]
                extra = f" split-half cl r={r:.3f}"
        print(f"  {name:8s}[{tag}] mean_cl={meancl:.3f} crossing_frac={crossing:.3f} "
              f"orient_entropy={ent:.3f} occ_vox={int(occ.sum())}{extra}")


def main():
    print("loading reference tracks (all 216 acqs)...")
    o = SU(open(REF, "rb")).load()
    rm, rd, rpar = segdirs_from_tracks(o["tracks_smoothed"])
    print("loading our samples + building graph field...")
    S = np.load(FD + "samples.npz")
    V, C = build_field(S["mids"], S["vels"])
    Vr, Cr = regularize(V, C, mode="graph", gamma=1.5, iters=200, dt=0.12, eps_iso=0.2)
    lines = streamlines(Vr, Cr, nseed=4000, min_len_mm=4.0)
    fm, fd = segdirs_from_streamlines(lines)
    print("\nlower mean_cl / higher crossing_frac / higher entropy = MORE diverse flows (more vessel-like):")
    summarize("ref", rm, rd, rpar)
    summarize("field", fm, fd)
    summarize("raw", S["mids"], S["vels"], S["par"])


if __name__ == "__main__":
    main()
