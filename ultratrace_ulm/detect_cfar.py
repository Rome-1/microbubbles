"""Local-CFAR detection variants for 3D ULM magnitude volumes.

These functions are opt-in alternatives to ``tracking.detect_batch``. The
drop-in path returns the same per-frame tuples:

    (spatial[int32 N,3], intensities[float32 N], zscores[float32 N])

The confidence path returns one extra array per frame:

    (spatial, intensities, zscores, confidence[uint8 N])

where confidence is 1 for strict/high-threshold detections and 0 for
low-threshold detections. A tracker should allow both confidence levels to
match existing tracks, but should only spawn new tracks from confidence==1.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import (
    gaussian_filter,
    gaussian_filter1d,
    maximum_filter,
    median_filter,
)


DetectionFrame = tuple[np.ndarray, np.ndarray, np.ndarray]
ConfidenceDetectionFrame = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def elevation_bias_profile(
    volume: np.ndarray,
    smooth_sigma: float = 2.0,
    floor_fraction: float = 0.25,
) -> np.ndarray:
    """Return a smooth multiplicative elevation profile normalized to mean 1."""
    data = np.asarray(volume, dtype=np.float32)
    if data.ndim != 4:
        raise ValueError(f"Expected volume (frames,elev,z,x), got {data.shape}")
    profile = np.median(data, axis=(0, 2, 3)).astype(np.float32, copy=False)
    positive = profile[profile > 0]
    if positive.size == 0:
        return np.ones(data.shape[1], dtype=np.float32)
    fill = float(np.median(positive))
    profile = np.where(profile > 0, profile, fill).astype(np.float32, copy=False)
    if smooth_sigma > 0:
        profile = gaussian_filter1d(profile, sigma=float(smooth_sigma), mode="nearest")
    mean = float(np.mean(profile[profile > 0])) if np.any(profile > 0) else 1.0
    profile = profile / max(mean, 1e-12)
    floor = max(float(floor_fraction), 1e-6)
    return np.maximum(profile, floor).astype(np.float32, copy=False)


def debias_elevation(
    volume: np.ndarray,
    smooth_sigma: float = 2.0,
    floor_fraction: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Divide by a smooth elevation-plane brightness profile."""
    profile = elevation_bias_profile(volume, smooth_sigma=smooth_sigma, floor_fraction=floor_fraction)
    corrected = np.asarray(volume, dtype=np.float32) / profile[None, :, None, None]
    return corrected.astype(np.float32, copy=False), profile


def anisotropic_nms_size(
    min_distance: int,
    elev_radius: int | None = None,
    z_radius: int | None = None,
    x_radius: int | None = None,
) -> tuple[int, int, int, int]:
    """Build a ``maximum_filter`` size tuple with independent spatial radii."""
    base = max(0, int(min_distance))
    er = max(0, int(2 * base if elev_radius is None else elev_radius))
    zr = max(0, int(base if z_radius is None else z_radius))
    xr = max(0, int(base if x_radius is None else x_radius))
    return (1, 2 * er + 1, 2 * zr + 1, 2 * xr + 1)


def _global_robust_scale(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float32)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return 1.0
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    scale = 1.4826 * mad
    if scale <= 1e-12:
        scale = float(np.std(vals))
    return scale if scale > 1e-12 else 1.0


def cfar_background_map(
    volume: np.ndarray,
    local_radius: tuple[int, int, int] = (3, 9, 9),
    guard_radius: tuple[int, int, int] = (1, 2, 2),
    global_floor_fraction: float = 0.35,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate local CFAR background and robust scale from time-mean magnitude.

    The reference path uses a scalable median/MAD approximation. A pilot median
    finds compact bright outliers, a guard-band maximum filter expands that
    pilot mask, and guarded voxels are replaced by the pilot background before
    the final median/MAD maps are computed. The global floor prevents empty or
    very quiet tissue from producing near-zero local scales.
    """
    data = np.asarray(volume, dtype=np.float32)
    if data.ndim != 4:
        raise ValueError(f"Expected volume (frames,elev,z,x), got {data.shape}")
    time_mean = data.mean(axis=0, dtype=np.float32)
    lr = tuple(max(0, int(v)) for v in local_radius)
    gr = tuple(max(0, int(v)) for v in guard_radius)
    local_size = tuple(2 * v + 1 for v in lr)
    guard_size = tuple(2 * v + 1 for v in gr)

    pilot_bg = median_filter(time_mean, size=local_size, mode="nearest")
    global_scale = _global_robust_scale(time_mean)
    bright = (time_mean - pilot_bg) > (3.0 * global_scale)
    if any(v > 0 for v in gr):
        bright = maximum_filter(bright.astype(np.uint8), size=guard_size, mode="nearest") > 0
    guarded = np.where(bright, pilot_bg, time_mean).astype(np.float32, copy=False)

    bg = median_filter(guarded, size=local_size, mode="nearest").astype(np.float32, copy=False)
    abs_dev = np.abs(guarded - bg).astype(np.float32, copy=False)
    mad = median_filter(abs_dev, size=local_size, mode="nearest")
    scale = (1.4826 * mad).astype(np.float32, copy=False)
    floor = float(global_floor_fraction) * global_scale
    scale = np.maximum(scale, floor).astype(np.float32, copy=False)
    return bg, scale


def _pack_by_frame(
    volume: np.ndarray,
    score: np.ndarray,
    is_peak: np.ndarray,
    confidence: np.ndarray | None = None,
) -> list[DetectionFrame] | list[ConfidenceDetectionFrame]:
    coords = np.array(np.where(is_peak))
    n_frames = int(volume.shape[0])
    empty3: DetectionFrame = (
        np.empty((0, 3), dtype=np.int32),
        np.empty(0, dtype=np.float32),
        np.empty(0, dtype=np.float32),
    )
    empty4: ConfidenceDetectionFrame = (
        empty3[0],
        empty3[1],
        empty3[2],
        np.empty(0, dtype=np.uint8),
    )
    if coords.size == 0:
        return [empty4 if confidence is not None else empty3 for _ in range(n_frames)]

    frame_ids = coords[0]
    ei, zi, xi = coords[1], coords[2], coords[3]
    spatial = coords[1:].T.astype(np.int32, copy=False)
    intensities = volume[frame_ids, ei, zi, xi].astype(np.float32, copy=False)
    zscores = score[frame_ids, ei, zi, xi].astype(np.float32, copy=False)
    conf = None
    if confidence is not None:
        conf = confidence[frame_ids, ei, zi, xi].astype(np.uint8, copy=False)
    bounds = np.searchsorted(frame_ids, np.arange(n_frames + 1))
    out = []
    for frame in range(n_frames):
        lo, hi = int(bounds[frame]), int(bounds[frame + 1])
        if lo == hi:
            out.append(empty4 if confidence is not None else empty3)
        elif confidence is None:
            out.append((spatial[lo:hi], intensities[lo:hi], zscores[lo:hi]))
        else:
            assert conf is not None
            out.append((spatial[lo:hi], intensities[lo:hi], zscores[lo:hi], conf[lo:hi]))
    return out


def detect_batch_cfar(
    magnitude: np.ndarray,
    sigma_threshold: float,
    min_distance: int,
    smoothing_sigma: float,
    *,
    local_radius: tuple[int, int, int] = (3, 9, 9),
    guard_radius: tuple[int, int, int] = (1, 2, 2),
    global_floor_fraction: float = 0.35,
    nms_radius: tuple[int, int, int] | None = None,
    debias: bool = True,
    debias_smooth_sigma: float = 2.0,
) -> list[DetectionFrame]:
    """Drop-in local CFAR/MAD detector."""
    if magnitude.ndim != 4:
        raise ValueError(f"Expected volume (frames,elev,z,x), got {magnitude.shape}")
    data = np.asarray(magnitude, dtype=np.float32)
    if debias:
        data, _ = debias_elevation(data, smooth_sigma=debias_smooth_sigma)
    bg, scale = cfar_background_map(
        data,
        local_radius=local_radius,
        guard_radius=guard_radius,
        global_floor_fraction=global_floor_fraction,
    )
    smoothed = data
    if smoothing_sigma > 0:
        smoothed = gaussian_filter(smoothed, sigma=(0, 0, smoothing_sigma, smoothing_sigma))
    zscore = (smoothed - bg[None, :, :, :]) / (scale[None, :, :, :] + 1e-10)
    if nms_radius is None:
        size = anisotropic_nms_size(min_distance)
    else:
        size = (1, *(2 * int(v) + 1 for v in nms_radius))
    max_filtered = maximum_filter(zscore, size=size, mode="nearest")
    is_peak = (zscore == max_filtered) & (zscore > float(sigma_threshold))
    return _pack_by_frame(np.asarray(magnitude, dtype=np.float32), zscore, is_peak)  # type: ignore[return-value]


def detect_batch_cfar_confidence(
    magnitude: np.ndarray,
    sigma_threshold: float,
    min_distance: int,
    smoothing_sigma: float,
    *,
    low_sigma_threshold: float | None = None,
    local_radius: tuple[int, int, int] = (3, 9, 9),
    guard_radius: tuple[int, int, int] = (1, 2, 2),
    global_floor_fraction: float = 0.35,
    nms_radius: tuple[int, int, int] | None = None,
    debias: bool = True,
    debias_smooth_sigma: float = 2.0,
) -> list[ConfidenceDetectionFrame]:
    """Local CFAR detector with high/low confidence labels.

    The low threshold is used only to emit continuation candidates. The strict
    threshold remains ``sigma_threshold``.
    """
    low_thr = float(sigma_threshold) - 1.0 if low_sigma_threshold is None else float(low_sigma_threshold)
    if low_thr > float(sigma_threshold):
        raise ValueError("low_sigma_threshold must be <= sigma_threshold")
    data = np.asarray(magnitude, dtype=np.float32)
    if debias:
        data, _ = debias_elevation(data, smooth_sigma=debias_smooth_sigma)
    bg, scale = cfar_background_map(
        data,
        local_radius=local_radius,
        guard_radius=guard_radius,
        global_floor_fraction=global_floor_fraction,
    )
    smoothed = data
    if smoothing_sigma > 0:
        smoothed = gaussian_filter(smoothed, sigma=(0, 0, smoothing_sigma, smoothing_sigma))
    zscore = (smoothed - bg[None, :, :, :]) / (scale[None, :, :, :] + 1e-10)
    if nms_radius is None:
        size = anisotropic_nms_size(min_distance)
    else:
        size = (1, *(2 * int(v) + 1 for v in nms_radius))
    max_filtered = maximum_filter(zscore, size=size, mode="nearest")
    is_peak = (zscore == max_filtered) & (zscore > low_thr)
    confidence = (zscore >= float(sigma_threshold)).astype(np.uint8, copy=False)
    return _pack_by_frame(np.asarray(magnitude, dtype=np.float32), zscore, is_peak, confidence)  # type: ignore[return-value]


def split_confidence_batch(
    batch: list[ConfidenceDetectionFrame],
) -> tuple[list[DetectionFrame], list[DetectionFrame]]:
    """Split confidence output into high-start and low-continuation batches."""
    high: list[DetectionFrame] = []
    low: list[DetectionFrame] = []
    for spatial, intensities, zscores, conf in batch:
        h = conf.astype(bool, copy=False)
        high.append((spatial[h], intensities[h], zscores[h]))
        low.append((spatial[~h], intensities[~h], zscores[~h]))
    return high, low


def detect_batch_cfar_gpu(
    magnitude: np.ndarray,
    sigma_threshold: float,
    min_distance: int,
    smoothing_sigma: float,
    *,
    local_radius: tuple[int, int, int] = (3, 9, 9),
    guard_radius: tuple[int, int, int] = (1, 2, 2),
    global_floor_fraction: float = 0.35,
    nms_radius: tuple[int, int, int] | None = None,
    debias: bool = True,
    debias_smooth_sigma: float = 2.0,
    frame_chunk: int = 150,
) -> list[DetectionFrame]:
    """GPU local-CFAR detector. Imports cupy only inside the function."""
    import cupy as cp
    from cupyx.scipy.ndimage import gaussian_filter as cp_gaussian_filter
    from cupyx.scipy.ndimage import gaussian_filter1d as cp_gaussian_filter1d
    from cupyx.scipy.ndimage import maximum_filter as cp_maximum_filter
    from cupyx.scipy.ndimage import median_filter as cp_median_filter

    if magnitude.ndim != 4:
        raise ValueError(f"Expected volume (frames,elev,z,x), got {magnitude.shape}")
    n_frames = int(magnitude.shape[0])
    empty: DetectionFrame = (
        np.empty((0, 3), dtype=np.int32),
        np.empty(0, dtype=np.float32),
        np.empty(0, dtype=np.float32),
    )
    pool = cp.get_default_memory_pool()
    pool.free_all_blocks()

    data = cp.asarray(magnitude, dtype=cp.float32)
    if debias:
        profile = cp.median(data, axis=(0, 2, 3)).astype(cp.float32)
        fill = cp.median(profile[profile > 0]) if bool(cp.any(profile > 0)) else cp.float32(1.0)
        profile = cp.where(profile > 0, profile, fill)
        if debias_smooth_sigma > 0:
            profile = cp_gaussian_filter1d(profile, sigma=float(debias_smooth_sigma), mode="nearest")
        profile = profile / cp.maximum(profile.mean(), cp.float32(1e-12))
        profile = cp.maximum(profile, cp.float32(0.25))
        data = data / profile[None, :, None, None]

    time_mean = data.mean(axis=0, dtype=cp.float32)
    local_size = tuple(2 * max(0, int(v)) + 1 for v in local_radius)
    guard_size = tuple(2 * max(0, int(v)) + 1 for v in guard_radius)
    pilot_bg = cp_median_filter(time_mean, size=local_size, mode="nearest")
    vals = time_mean[time_mean > 0]
    if int(vals.size) == 0:
        global_scale = cp.float32(1.0)
    else:
        med = cp.median(vals)
        global_scale = cp.float32(1.4826) * cp.median(cp.abs(vals - med))
        global_scale = cp.where(global_scale > 1e-12, global_scale, vals.std())
        global_scale = cp.where(global_scale > 1e-12, global_scale, cp.float32(1.0))
    bright = (time_mean - pilot_bg) > (cp.float32(3.0) * global_scale)
    if any(int(v) > 0 for v in guard_radius):
        bright = cp_maximum_filter(bright.astype(cp.uint8), size=guard_size, mode="nearest") > 0
    guarded = cp.where(bright, pilot_bg, time_mean).astype(cp.float32)
    bg = cp_median_filter(guarded, size=local_size, mode="nearest").astype(cp.float32)
    mad = cp_median_filter(cp.abs(guarded - bg), size=local_size, mode="nearest")
    scale = cp.maximum(cp.float32(1.4826) * mad, cp.float32(global_floor_fraction) * global_scale)
    del time_mean, pilot_bg, vals, bright, guarded, mad
    pool.free_all_blocks()

    sm = data
    if smoothing_sigma > 0:
        tmp = cp_gaussian_filter(sm, sigma=(0, 0, smoothing_sigma, smoothing_sigma))
        del sm
        sm = tmp
    sm = (sm - bg[None, :, :, :]) / (scale[None, :, :, :] + cp.float32(1e-10))
    if nms_radius is None:
        size = anisotropic_nms_size(min_distance)
    else:
        size = (1, *(2 * int(v) + 1 for v in nms_radius))

    cf, ce, cz, cx, czs = [], [], [], [], []
    for f0 in range(0, n_frames, frame_chunk):
        f1 = min(f0 + frame_chunk, n_frames)
        zc = sm[f0:f1]
        mx = cp_maximum_filter(zc, size=size, mode="nearest")
        pk = (zc == mx) & (zc > float(sigma_threshold))
        idx = cp.where(pk)
        if int(idx[0].size):
            cf.append(cp.asnumpy(idx[0]) + f0)
            ce.append(cp.asnumpy(idx[1]))
            cz.append(cp.asnumpy(idx[2]))
            cx.append(cp.asnumpy(idx[3]))
            czs.append(cp.asnumpy(zc[pk]).astype(np.float32, copy=False))
        del zc, mx, pk, idx
        pool.free_all_blocks()
    del data, bg, scale, sm
    pool.free_all_blocks()

    if not cf:
        return [empty for _ in range(n_frames)]
    frame_ids = np.concatenate(cf)
    ei, zi, xi = np.concatenate(ce), np.concatenate(cz), np.concatenate(cx)
    spatial = np.stack([ei, zi, xi], axis=1).astype(np.int32, copy=False)
    intensities = magnitude[frame_ids, ei, zi, xi].astype(np.float32, copy=False)
    zscores = np.concatenate(czs)
    bounds = np.searchsorted(frame_ids, np.arange(n_frames + 1))
    out: list[DetectionFrame] = []
    for frame in range(n_frames):
        lo, hi = int(bounds[frame]), int(bounds[frame + 1])
        out.append(empty if lo == hi else (spatial[lo:hi], intensities[lo:hi], zscores[lo:hi]))
    return out
