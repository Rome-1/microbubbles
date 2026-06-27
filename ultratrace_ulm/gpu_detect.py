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
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    import cupy as cp
    from cupyx.scipy.ndimage import gaussian_filter, maximum_filter

    if magnitude.ndim != 4:
        raise ValueError(f"Expected volume (frames,elev,z,x), got {magnitude.shape}")
    n_frames, n_elev = int(magnitude.shape[0]), int(magnitude.shape[1])
    empty = (np.empty((0, 3), dtype=np.int32), np.empty(0, dtype=np.float32),
             np.empty(0, dtype=np.float32))

    cp.get_default_memory_pool().free_all_blocks()
    vol = cp.asarray(magnitude, dtype=cp.float32)
    smoothed = gaussian_filter(vol, sigma=(0, 0, smoothing_sigma, smoothing_sigma)) if smoothing_sigma > 0 else vol
    del vol

    # Per-elevation-slice mean/std over positive (smoothed) voxels -- mirrors
    # _slice_stats (which smooths with the same kernel, so we reuse `smoothed`).
    means = cp.zeros(n_elev, dtype=cp.float32)
    stds = cp.ones(n_elev, dtype=cp.float32)
    for e in range(n_elev):
        v = smoothed[:, e]
        mask = v > 0
        if bool(mask.any()):
            vals = v[mask]
            means[e] = vals.mean()
            s = vals.std()
            stds[e] = s if float(s) > 1e-10 else 1.0

    zscore = (smoothed - means[None, :, None, None]) / (stds[None, :, None, None] + 1e-10)
    del smoothed

    fs = 2 * int(min_distance) + 1
    maxf = maximum_filter(zscore, size=(1, fs, fs, fs))
    is_peak = (zscore == maxf) & (zscore > float(sigma_threshold))
    del maxf

    coords = cp.where(is_peak)            # tuple of 4 index arrays, frame-sorted
    zs_pk = cp.asnumpy(zscore[is_peak]).astype(np.float32, copy=False)
    coords = [cp.asnumpy(c) for c in coords]
    del zscore, is_peak
    cp.get_default_memory_pool().free_all_blocks()

    if coords[0].size == 0:
        return [empty for _ in range(n_frames)]

    frame_ids = coords[0]
    spatial = np.stack(coords[1:], axis=1).astype(np.int32, copy=False)  # (N,3)
    # intensities from the ORIGINAL (host) magnitude, matching detect_batch.
    intensities = magnitude[coords[0], coords[1], coords[2], coords[3]].astype(np.float32, copy=False)
    bounds = np.searchsorted(frame_ids, np.arange(n_frames + 1))
    out = []
    for frame in range(n_frames):
        lo, hi = int(bounds[frame]), int(bounds[frame + 1])
        out.append(empty if lo == hi else (spatial[lo:hi], intensities[lo:hi], zs_pk[lo:hi]))
    return out
