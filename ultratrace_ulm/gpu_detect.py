"""GPU (cupy) port of tracking.detect_batch (mb-crr.2/.9 enabler).

The probe measured CPU detection at ~95 s/acq (scipy gaussian_filter +
maximum_filter on a 700x25x225x378 volume), during which the beamforming GPU is
idle. This runs the same z-score peak detection on the GPU (~5 s), so the full
fused per-acq cost drops from ~188 s to ~100 s. Output is identical in structure
to detect_batch: a list of (spatial[int32 N,3], intensities[float32 N],
zscores[float32 N]) per frame.

Numerically mirrors detect_batch, with one dedup: detect_batch smooths the
volume twice (once for z-score, once inside _slice_stats); both use the SAME
(0,0,sigma,sigma) Gaussian, so we smooth once and use it for both -- identical
result, half the filtering work.
"""

from __future__ import annotations

import numpy as np


def detect_batch_gpu(
    magnitude: np.ndarray,         # host (frames, elev, z, x) float32 (already SVD-filtered)
    sigma_threshold: float,
    min_distance: int,
    smoothing_sigma: float,
    frame_chunk: int = 150,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    import cupy as cp
    from cupyx.scipy.ndimage import gaussian_filter, maximum_filter

    if magnitude.ndim != 4:
        raise ValueError(f"Expected volume (frames,elev,z,x), got {magnitude.shape}")
    n_frames, n_elev = int(magnitude.shape[0]), int(magnitude.shape[1])
    empty = (np.empty((0, 3), dtype=np.int32), np.empty(0, dtype=np.float32),
             np.empty(0, dtype=np.float32))
    pool = cp.get_default_memory_pool()
    pool.free_all_blocks()

    # One smoothed volume on the GPU (~6GB). Smooth in place (spatial only).
    sm = cp.asarray(magnitude, dtype=cp.float32)
    if smoothing_sigma > 0:
        tmp = gaussian_filter(sm, sigma=(0, 0, smoothing_sigma, smoothing_sigma))
        del sm
        sm = tmp
        del tmp
        pool.free_all_blocks()

    # Per-elevation-slice mean/std over positive smoothed voxels (mirrors _slice_stats).
    means = cp.zeros(n_elev, dtype=cp.float32)
    stds = cp.ones(n_elev, dtype=cp.float32)
    for e in range(n_elev):
        v = sm[:, e]
        mask = v > 0
        if bool(mask.any()):
            vals = v[mask]
            means[e] = vals.mean()
            s = vals.std()
            stds[e] = s if float(s) > 1e-10 else 1.0

    # z-score in place (no second full volume), then peak-find in frame chunks
    # (frames are independent: the max-filter is size 1 along the frame axis).
    sm -= means[None, :, None, None]
    sm /= (stds[None, :, None, None] + 1e-10)
    fs = 2 * int(min_distance) + 1
    thr = float(sigma_threshold)

    cf, ce, cz, cx, czs = [], [], [], [], []
    for f0 in range(0, n_frames, frame_chunk):
        f1 = min(f0 + frame_chunk, n_frames)
        zc = sm[f0:f1]
        mx = maximum_filter(zc, size=(1, fs, fs, fs))
        pk = (zc == mx) & (zc > thr)
        idx = cp.where(pk)
        if idx[0].size:
            cf.append(cp.asnumpy(idx[0]) + f0)
            ce.append(cp.asnumpy(idx[1])); cz.append(cp.asnumpy(idx[2])); cx.append(cp.asnumpy(idx[3]))
            czs.append(cp.asnumpy(zc[pk]).astype(np.float32, copy=False))
        del zc, mx, pk, idx
        pool.free_all_blocks()
    del sm
    pool.free_all_blocks()

    if not cf:
        return [empty for _ in range(n_frames)]
    frame_ids = np.concatenate(cf)                              # ascending (chunks in order)
    ei, zi, xi = np.concatenate(ce), np.concatenate(cz), np.concatenate(cx)
    zs_pk = np.concatenate(czs)
    spatial = np.stack([ei, zi, xi], axis=1).astype(np.int32, copy=False)
    intensities = magnitude[frame_ids, ei, zi, xi].astype(np.float32, copy=False)
    bounds = np.searchsorted(frame_ids, np.arange(n_frames + 1))
    out = []
    for frame in range(n_frames):
        lo, hi = int(bounds[frame]), int(bounds[frame + 1])
        out.append(empty if lo == hi else (spatial[lo:hi], intensities[lo:hi], zs_pk[lo:hi]))
    return out
