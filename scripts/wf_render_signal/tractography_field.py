"""Regularized-field tractography for 3D ULM — build long, coherent vessels by
integrating streamlines through a confidence-weighted-smoothed velocity field,
WITHOUT detection-tracking (which fragments at our high bubble concentration).

Validated result (acq-0 velocity field, split-half over odd/even acquisitions):
  raw field            -> streamlines max ~17mm (no better than tracking)
  smoothed sigma~2.5   -> vessels up to ~50mm (reference max track ~43mm),
                          split-half occupancy Dice 0.60, direction cosine 0.85
                          => the long vessels REPRODUCE across independent halves.

Key insight: the validated velocity-DIRECTION field (split-half cosine 0.96) is
confident only over short segments; confidence-weighted smoothing fills the gaps
so streamlines integrate across them. Smoothing sigma trades length vs. detail;
choose it by MAXIMIZING split-half reproducibility, not just length.

CPU/numpy+scipy only. Run niced, single process (shared box).
Inputs: outputs/reference/wf/r2/signal/velocity-field/{velocity_field.npy,
confidence.npy, samples.npz(mids,vels,par)}.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter

GS = (138, 13, 77)                                   # coarsen-2 grid (x, elev, z)
SP = np.array([0.4008, 1.1094, 0.4016])              # mm/voxel
ORG = np.array([-27.456, -6.656, 10.0])              # mm origin (x, elev, z)


def build_field(mids: np.ndarray, vels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bin localization samples into the coarse grid: mean velocity + count per voxel."""
    idx = np.floor((mids - ORG) / SP).astype(int)
    ok = np.all((idx >= 0) & (idx < np.array(GS)), axis=1)
    idx, v = idx[ok], vels[ok]
    Vsum = np.zeros(GS + (3,)); Cnt = np.zeros(GS)
    np.add.at(Vsum, (idx[:, 0], idx[:, 1], idx[:, 2]), v)
    np.add.at(Cnt, (idx[:, 0], idx[:, 1], idx[:, 2]), 1)
    with np.errstate(invalid="ignore"):
        V = np.nan_to_num(Vsum / Cnt[..., None])
    return V, Cnt


def smooth_field(V, C, sig=(2.5, 1.4, 2.5)):
    """Confidence-weighted Gaussian smoothing to fill gaps + denoise directions."""
    num = np.stack([gaussian_filter(V[..., c] * C, sig) for c in range(3)], -1)
    den = gaussian_filter(C, sig)[..., None] + 1e-9
    return num / den, gaussian_filter(C, sig)


def tractogram(V, C, cthr_q=0.4, flip=0.3, step=0.6, max_steps=600, nseed=3000, seed=1):
    """RK1 streamline integration through the (anisotropic) unit-direction field.
    Returns (occupancy bool grid, mean-direction grid, per-streamline arclengths mm)."""
    nx, ny, nz, _ = V.shape
    spd = np.linalg.norm(V, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        di = (V / spd[..., None]) / SP                       # index-space dir (anisotropy)
        di /= (np.linalg.norm(di, axis=-1, keepdims=True) + 1e-9)
    cthr = np.quantile(C[C > 0], cthr_q)
    occ = np.zeros((nx, ny, nz), bool)
    DIR = np.zeros((nx, ny, nz, 3)); DN = np.zeros((nx, ny, nz))

    def look(p):
        i, j, k = [int(round(x)) for x in p]
        if 0 <= i < nx and 0 <= j < ny and 0 <= k < nz and C[i, j, k] >= cthr and spd[i, j, k] > 1e-6:
            return di[i, j, k]
        return None

    valid = np.argwhere((C >= cthr) & (spd > 1e-6))
    rng = np.random.default_rng(seed)
    valid = valid[rng.choice(len(valid), min(nseed, len(valid)), replace=False)]
    lens = []
    for s in valid:
        total = 0.0
        for sg in (1, -1):
            p = s.astype(float).copy(); prev = None; steps = 0
            for _ in range(max_steps):
                d = look(p)
                if d is None:
                    break
                d = d * sg
                if prev is not None and np.dot(d, prev) < flip:
                    break
                i, j, k = [int(round(x)) for x in p]
                occ[i, j, k] = True; DIR[i, j, k] += d * sg; DN[i, j, k] += 1
                p = p + step * d; prev = d; steps += 1
            total += np.sqrt(((step * (di[tuple(int(round(x)) for x in np.clip(s, 0, np.array(GS) - 1))]) * SP) ** 2).sum()) * steps if steps else 0
        lens.append(total)
    with np.errstate(invalid="ignore"):
        DIRn = np.nan_to_num(DIR / (np.linalg.norm(DIR, axis=-1, keepdims=True) + 1e-9))
    return occ, DIRn, np.array(lens)


def split_half_validate(mids, vels, par, sig=(2.5, 1.4, 2.5)):
    """Build + integrate on odd vs even acquisitions independently; report agreement."""
    Vo, Co = smooth_field(*build_field(mids[par % 2 == 0], vels[par % 2 == 0]), sig)
    Ve, Ce = smooth_field(*build_field(mids[par % 2 == 1], vels[par % 2 == 1]), sig)
    oo, do, _ = tractogram(Vo, Co); oe, de, _ = tractogram(Ve, Ce)
    dice = 2 * (oo & oe).sum() / (oo.sum() + oe.sum())
    both = oo & oe
    cos = float(np.sum(do[both] * de[both])) / (both.sum() + 1e-9)
    return {"dice_occupancy": float(dice), "direction_cosine": cos}


if __name__ == "__main__":
    FD = "outputs/reference/wf/r2/signal/velocity-field/"
    S = np.load(FD + "samples.npz")
    print("split-half:", split_half_validate(S["mids"], S["vels"], S["par"]))
