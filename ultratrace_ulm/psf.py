"""Empirical PSF mining and matched-filter detection for 3D ULM."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve, gaussian_filter, maximum_filter

from .detect_cfar import DetectionFrame, anisotropic_nms_size, debias_elevation


def _as_radius_tuple(radius: int | tuple[int, int, int]) -> tuple[int, int, int]:
    if isinstance(radius, tuple):
        if len(radius) != 3:
            raise ValueError("radius tuple must be (elev,z,x)")
        return tuple(max(0, int(v)) for v in radius)
    r = max(0, int(radius))
    return (r, r, r)


def _extract_patch(frame: np.ndarray, center: np.ndarray, radius: tuple[int, int, int]) -> np.ndarray | None:
    e, z, x = (int(v) for v in center)
    re, rz, rx = radius
    if e - re < 0 or z - rz < 0 or x - rx < 0:
        return None
    if e + re >= frame.shape[0] or z + rz >= frame.shape[1] or x + rx >= frame.shape[2]:
        return None
    return frame[e - re : e + re + 1, z - rz : z + rz + 1, x - rx : x + rx + 1]


def estimate_empirical_psf(
    magnitude: np.ndarray,
    detections: list[DetectionFrame],
    *,
    patch_radius: int | tuple[int, int, int] = (4, 3, 3),
    max_patches: int = 512,
    min_zscore: float = 5.0,
    isolation_radius: int | tuple[int, int, int] = (5, 4, 4),
) -> np.ndarray:
    """Average bright isolated detection-centered patches into a 3D PSF.

    Detections are assumed to use integer pixel coordinates in ``(elev,z,x)``.
    Each patch is baseline-subtracted by its edge median and normalized by its
    sum before averaging, so very bright bubbles do not dominate the PSF.
    """
    data = np.asarray(magnitude, dtype=np.float32)
    if data.ndim != 4:
        raise ValueError(f"Expected volume (frames,elev,z,x), got {data.shape}")
    radius = _as_radius_tuple(patch_radius)
    iso = _as_radius_tuple(isolation_radius)
    candidates: list[tuple[float, int, np.ndarray]] = []
    for frame_idx, (spatial, _, zscores) in enumerate(detections):
        if len(spatial) == 0:
            continue
        for coord, zscore in zip(spatial, zscores):
            if float(zscore) >= float(min_zscore):
                candidates.append((float(zscore), frame_idx, np.asarray(coord, dtype=np.int32)))
    candidates.sort(key=lambda item: item[0], reverse=True)

    patches = []
    for _, frame_idx, coord in candidates:
        if len(patches) >= int(max_patches):
            break
        frame_dets = detections[frame_idx][0]
        if len(frame_dets) > 1:
            dist = np.abs(frame_dets.astype(np.int32) - coord[None, :])
            near = np.all(dist <= np.asarray(iso, dtype=np.int32)[None, :], axis=1)
            # Allow the candidate itself, reject neighbouring detections inside
            # the PSF support plus guard.
            if int(np.count_nonzero(near)) > 1:
                continue
        patch = _extract_patch(data[frame_idx], coord, radius)
        if patch is None:
            continue
        edge = np.concatenate(
            [
                patch[0].ravel(),
                patch[-1].ravel(),
                patch[:, 0, :].ravel(),
                patch[:, -1, :].ravel(),
                patch[:, :, 0].ravel(),
                patch[:, :, -1].ravel(),
            ]
        )
        p = patch.astype(np.float32, copy=True) - float(np.median(edge))
        p[p < 0] = 0
        total = float(p.sum())
        if total <= 1e-12:
            continue
        patches.append(p / total)

    if not patches:
        raise ValueError("No isolated PSF patches found")
    psf = np.mean(np.stack(patches, axis=0), axis=0)
    psf = np.maximum(psf, 0).astype(np.float32, copy=False)
    psf /= max(float(psf.sum()), 1e-12)
    return psf


def _matched_response(frame: np.ndarray, psf: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    kernel = np.asarray(psf, dtype=np.float32)
    kernel = kernel / max(float(kernel.sum()), 1e-12)
    kernel_zm = kernel - float(kernel.mean())
    kernel_norm = max(float(np.sqrt(np.sum(kernel_zm * kernel_zm))), eps)
    local_mean = convolve(frame, kernel, mode="nearest")
    local_second = convolve(frame * frame, kernel, mode="nearest")
    local_std = np.sqrt(np.maximum(local_second - local_mean * local_mean, 0.0) + eps)
    numerator = convolve(frame, kernel_zm[::-1, ::-1, ::-1], mode="nearest")
    return numerator / (local_std * kernel_norm)


def detect_batch_matched_filter(
    magnitude: np.ndarray,
    sigma_threshold: float,
    min_distance: int,
    smoothing_sigma: float,
    *,
    psf: np.ndarray,
    nms_radius: tuple[int, int, int] | None = None,
    debias: bool = True,
    debias_smooth_sigma: float = 2.0,
) -> list[DetectionFrame]:
    """Drop-in detect_fn that z-scores a normalized matched-filter response."""
    data = np.asarray(magnitude, dtype=np.float32)
    if data.ndim != 4:
        raise ValueError(f"Expected volume (frames,elev,z,x), got {data.shape}")
    work = data
    if debias:
        work, _ = debias_elevation(work, smooth_sigma=debias_smooth_sigma)
    if smoothing_sigma > 0:
        work = gaussian_filter(work, sigma=(0, 0, smoothing_sigma, smoothing_sigma))
    response = np.empty_like(work, dtype=np.float32)
    for frame_idx in range(work.shape[0]):
        response[frame_idx] = _matched_response(work[frame_idx], psf).astype(np.float32, copy=False)
    means = response.mean(axis=(0, 2, 3), keepdims=True)
    stds = response.std(axis=(0, 2, 3), keepdims=True)
    zscore = (response - means) / (stds + 1e-10)
    size = anisotropic_nms_size(min_distance) if nms_radius is None else (1, *(2 * int(v) + 1 for v in nms_radius))
    max_filtered = maximum_filter(zscore, size=size, mode="nearest")
    is_peak = (zscore == max_filtered) & (zscore > float(sigma_threshold))
    coords = np.array(np.where(is_peak))
    empty: DetectionFrame = (
        np.empty((0, 3), dtype=np.int32),
        np.empty(0, dtype=np.float32),
        np.empty(0, dtype=np.float32),
    )
    if coords.size == 0:
        return [empty for _ in range(data.shape[0])]
    frame_ids = coords[0]
    ei, zi, xi = coords[1], coords[2], coords[3]
    spatial = coords[1:].T.astype(np.int32, copy=False)
    intensities = data[frame_ids, ei, zi, xi].astype(np.float32, copy=False)
    zscores = zscore[frame_ids, ei, zi, xi].astype(np.float32, copy=False)
    bounds = np.searchsorted(frame_ids, np.arange(data.shape[0] + 1))
    out: list[DetectionFrame] = []
    for frame in range(data.shape[0]):
        lo, hi = int(bounds[frame]), int(bounds[frame + 1])
        out.append(empty if lo == hi else (spatial[lo:hi], intensities[lo:hi], zscores[lo:hi]))
    return out


def detect_batch_matched_filter_gpu(
    magnitude: np.ndarray,
    sigma_threshold: float,
    min_distance: int,
    smoothing_sigma: float,
    *,
    psf: np.ndarray,
    nms_radius: tuple[int, int, int] | None = None,
    debias: bool = True,
    debias_smooth_sigma: float = 2.0,
    frame_chunk: int = 80,
) -> list[DetectionFrame]:
    """GPU matched-filter detector. Imports cupy only inside the function."""
    import cupy as cp
    from cupyx.scipy.ndimage import convolve as cp_convolve
    from cupyx.scipy.ndimage import gaussian_filter as cp_gaussian_filter
    from cupyx.scipy.ndimage import gaussian_filter1d as cp_gaussian_filter1d
    from cupyx.scipy.ndimage import maximum_filter as cp_maximum_filter

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
        data = data / cp.maximum(profile[None, :, None, None], cp.float32(0.25))
    if smoothing_sigma > 0:
        tmp = cp_gaussian_filter(data, sigma=(0, 0, smoothing_sigma, smoothing_sigma))
        del data
        data = tmp
    kernel = cp.asarray(psf, dtype=cp.float32)
    kernel = kernel / cp.maximum(kernel.sum(), cp.float32(1e-12))
    kernel_zm = kernel - kernel.mean()
    kernel_norm = cp.maximum(cp.sqrt(cp.sum(kernel_zm * kernel_zm)), cp.float32(1e-6))
    kernel_rev = kernel_zm[::-1, ::-1, ::-1]
    size = anisotropic_nms_size(min_distance) if nms_radius is None else (1, *(2 * int(v) + 1 for v in nms_radius))

    cf, ce, cz, cx, czs = [], [], [], [], []
    for f0 in range(0, n_frames, frame_chunk):
        f1 = min(f0 + frame_chunk, n_frames)
        chunk = data[f0:f1]
        resp = cp.empty_like(chunk, dtype=cp.float32)
        for j in range(int(chunk.shape[0])):
            frame = chunk[j]
            local_mean = cp_convolve(frame, kernel, mode="nearest")
            local_second = cp_convolve(frame * frame, kernel, mode="nearest")
            local_std = cp.sqrt(cp.maximum(local_second - local_mean * local_mean, 0.0) + 1e-6)
            resp[j] = cp_convolve(frame, kernel_rev, mode="nearest") / (local_std * kernel_norm)
            del frame, local_mean, local_second, local_std
        means = resp.mean(axis=(0, 2, 3), keepdims=True)
        stds = resp.std(axis=(0, 2, 3), keepdims=True)
        zc = (resp - means) / (stds + cp.float32(1e-10))
        mx = cp_maximum_filter(zc, size=size, mode="nearest")
        pk = (zc == mx) & (zc > float(sigma_threshold))
        idx = cp.where(pk)
        if int(idx[0].size):
            cf.append(cp.asnumpy(idx[0]) + f0)
            ce.append(cp.asnumpy(idx[1]))
            cz.append(cp.asnumpy(idx[2]))
            cx.append(cp.asnumpy(idx[3]))
            czs.append(cp.asnumpy(zc[pk]).astype(np.float32, copy=False))
        del chunk, resp, means, stds, zc, mx, pk, idx
        pool.free_all_blocks()
    del data, kernel, kernel_zm, kernel_norm, kernel_rev
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
