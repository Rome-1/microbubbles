"""Regularized per-block SVD rank policy for region filtering.

This module keeps the existing region tiling/blending contract but replaces the
binary ``global_rank`` vs ``per_block`` choice with a regularized k-map:

``delta=0`` forces every block to the global/median rank.
``delta=np.inf`` allows the smoothed per-block rank map through unchanged.
finite ``delta`` clamps each block to ``global_rank +/- delta``.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter

from .svd import _component_count, filter_svd_3d
from .svd_region import _axis_window, _block_bounds


def rank_from_singular_values(
    singular_values: np.ndarray,
    *,
    method: str = "elbow",
    energy_fraction: float = 0.85,
    min_rank: int = 0,
    max_rank: int | None = None,
) -> int:
    """Choose a low-cutoff rank from a descending singular-value curve."""
    s = np.asarray(singular_values, dtype=np.float64)
    s = s[np.isfinite(s) & (s > 0)]
    if s.size == 0:
        return int(min_rank)
    max_allowed = int(s.size - 1 if max_rank is None else min(max_rank, s.size - 1))
    min_allowed = int(max(0, min(min_rank, max_allowed)))
    if max_allowed <= min_allowed:
        return min_allowed

    if method == "energy":
        frac = float(np.clip(energy_fraction, 0.0, 1.0))
        energy = s**2
        cumulative = np.cumsum(energy) / max(float(energy.sum()), 1e-30)
        k = int(np.searchsorted(cumulative, frac) + 1)
    elif method == "elbow":
        # Knee of log singular-value curve: largest positive curvature away
        # from the endpoints. This is intentionally simple and deterministic.
        y = np.log(s + 1e-30)
        if y.size < 4:
            k = 1
        else:
            d2 = np.gradient(np.gradient(y))
            interior = d2[1:-1]
            k = int(np.argmax(interior) + 1)
    else:
        raise ValueError(f"Unknown rank method: {method!r}")
    return int(np.clip(k, min_allowed, max_allowed))


def _singular_values(matrix: np.ndarray) -> np.ndarray:
    x = np.asarray(matrix, dtype=np.complex64)
    x = x - x.mean(axis=0, keepdims=True)
    cov = x @ x.conj().T
    evals = np.linalg.eigvalsh(cov)
    evals = np.sort(np.maximum(evals.real, 0.0))[::-1]
    return np.sqrt(evals)


def regularize_rank_map(
    raw_k: np.ndarray,
    *,
    delta: float = 0.0,
    global_rank: int | None = None,
    smooth_size: int = 3,
    min_rank: int = 0,
    max_rank: int | None = None,
) -> np.ndarray:
    """Median-smooth and clamp a per-block rank map."""
    raw = np.asarray(raw_k, dtype=np.float64)
    if raw.ndim != 2:
        raise ValueError(f"Expected 2D rank map, got {raw.shape}")
    base = int(round(np.median(raw))) if global_rank is None else int(global_rank)
    work = raw.copy()
    if smooth_size and int(smooth_size) > 1 and min(raw.shape) > 1:
        size = int(smooth_size)
        if size % 2 == 0:
            size += 1
        work = median_filter(work, size=size, mode="nearest")
    if np.isfinite(delta):
        d = max(0.0, float(delta))
        work = np.clip(work, base - d, base + d)
    lo = int(min_rank)
    hi = int(np.max(raw) if max_rank is None else max_rank)
    out = np.rint(np.clip(work, lo, hi)).astype(np.int32, copy=False)
    return out


def block_rank_map(
    data: np.ndarray,
    n_z_blocks: int = 3,
    n_x_blocks: int = 3,
    overlap: float = 0.25,
    *,
    rank_method: str = "elbow",
    energy_fraction: float = 0.85,
    low_cutoff: float | int | None = None,
    n_components: int | None = None,
    delta: float = 0.0,
    global_rank: int | None = None,
    smooth_size: int = 3,
    min_rank: int = 0,
    max_rank: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(regularized_k, raw_k)`` for the z/x region grid."""
    if data.ndim == 3:
        data4 = data[:, None, :, :]
    elif data.ndim == 4:
        data4 = data
    else:
        raise ValueError(f"Expected 3D or 4D compound data, got shape {data.shape}")
    n_frames, _, n_z, n_x = data4.shape
    if n_components is not None or low_cutoff is not None:
        fixed = int(n_components) if n_components is not None else _component_count(low_cutoff, n_frames)
        raw = np.full((n_z_blocks, n_x_blocks), fixed, dtype=np.int32)
        return regularize_rank_map(
            raw,
            delta=delta,
            global_rank=fixed if global_rank is None else global_rank,
            smooth_size=smooth_size,
            min_rank=min_rank,
            max_rank=max_rank,
        ), raw

    z_bounds = _block_bounds(n_z, n_z_blocks, overlap)
    x_bounds = _block_bounds(n_x, n_x_blocks, overlap)
    raw = np.zeros((len(z_bounds), len(x_bounds)), dtype=np.int32)
    max_r = n_frames - 1 if max_rank is None else int(max_rank)
    for iz, (za, zb) in enumerate(z_bounds):
        for ix, (xa, xb) in enumerate(x_bounds):
            matrix = np.asarray(data4[:, :, za:zb, xa:xb], dtype=np.complex64).reshape(n_frames, -1)
            s = _singular_values(matrix)
            raw[iz, ix] = rank_from_singular_values(
                s,
                method=rank_method,
                energy_fraction=energy_fraction,
                min_rank=min_rank,
                max_rank=max_r,
            )
    reg = regularize_rank_map(
        raw,
        delta=delta,
        global_rank=global_rank,
        smooth_size=smooth_size,
        min_rank=min_rank,
        max_rank=max_r,
    )
    return reg, raw


def filter_svd_3d_region_ranked(
    data: np.ndarray,
    n_z_blocks: int = 3,
    n_x_blocks: int = 3,
    overlap: float = 0.25,
    *,
    rank_method: str = "elbow",
    energy_fraction: float = 0.85,
    delta: float = 0.0,
    global_rank: int | None = None,
    smooth_size: int = 3,
    min_rank: int = 0,
    max_rank: int | None = None,
    high_cutoff: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filter with a regularized per-block low-cutoff rank.

    Returns ``(filtered_complex, regularized_k_map, raw_k_map)``.
    """
    if data.ndim == 3:
        data4 = data[:, None, :, :]
        squeeze = True
    elif data.ndim == 4:
        data4 = data
        squeeze = False
    else:
        raise ValueError(f"Expected 3D or 4D compound data, got shape {data.shape}")

    n_frames, _, n_z, n_x = data4.shape
    kmap, raw = block_rank_map(
        data4,
        n_z_blocks=n_z_blocks,
        n_x_blocks=n_x_blocks,
        overlap=overlap,
        rank_method=rank_method,
        energy_fraction=energy_fraction,
        delta=delta,
        global_rank=global_rank,
        smooth_size=smooth_size,
        min_rank=min_rank,
        max_rank=n_frames - 1 if max_rank is None else max_rank,
    )
    z_bounds = _block_bounds(n_z, n_z_blocks, overlap)
    x_bounds = _block_bounds(n_x, n_x_blocks, overlap)
    z_windows = _axis_window(z_bounds, n_z)
    x_windows = _axis_window(x_bounds, n_x)
    acc = np.zeros(data4.shape, dtype=np.complex64)
    wsum = np.zeros((n_z, n_x), dtype=np.float64)

    for iz, (za, zb) in enumerate(z_bounds):
        for ix, (xa, xb) in enumerate(x_bounds):
            block = data4[:, :, za:zb, xa:xb]
            filt = filter_svd_3d(
                block,
                method="fast" if int(kmap[iz, ix]) > 0 else "none",
                n_components=int(kmap[iz, ix]),
                high_cutoff=high_cutoff,
            )
            w2d = np.outer(z_windows[iz][za:zb], x_windows[ix][xa:xb])
            acc[:, :, za:zb, xa:xb] += filt * w2d
            wsum[za:zb, xa:xb] += w2d
    out = (acc / wsum[None, None, :, :]).astype(np.complex64, copy=False)
    if squeeze:
        out = out[:, 0]
    return out, kmap, raw


def filtered_magnitude_region_ranked(
    compound: np.ndarray,
    temporal_sigma: float = 0.0,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Magnitude wrapper for ``filter_svd_3d_region_ranked``."""
    filtered, kmap, raw = filter_svd_3d_region_ranked(compound, **kwargs)
    magnitude = np.abs(filtered).astype(np.float32, copy=False)
    if temporal_sigma > 0:
        magnitude = gaussian_filter1d(magnitude, sigma=temporal_sigma, axis=0)
    return magnitude, kmap, raw


def block_rank_map_gpu(
    data: np.ndarray,
    n_z_blocks: int = 3,
    n_x_blocks: int = 3,
    overlap: float = 0.25,
    *,
    rank_method: str = "elbow",
    energy_fraction: float = 0.85,
    delta: float = 0.0,
    global_rank: int | None = None,
    smooth_size: int = 3,
    min_rank: int = 0,
    max_rank: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """GPU k-map computation. Imports cupy only inside the function."""
    import cupy as cp

    if data.ndim == 3:
        data4 = data[:, None, :, :]
    elif data.ndim == 4:
        data4 = data
    else:
        raise ValueError(f"Expected 3D or 4D compound data, got shape {data.shape}")
    n_frames, _, n_z, n_x = data4.shape
    z_bounds = _block_bounds(n_z, n_z_blocks, overlap)
    x_bounds = _block_bounds(n_x, n_x_blocks, overlap)
    raw = np.zeros((len(z_bounds), len(x_bounds)), dtype=np.int32)
    max_r = n_frames - 1 if max_rank is None else int(max_rank)
    pool = cp.get_default_memory_pool()
    pool.free_all_blocks()
    for iz, (za, zb) in enumerate(z_bounds):
        for ix, (xa, xb) in enumerate(x_bounds):
            block = cp.asarray(np.ascontiguousarray(data4[:, :, za:zb, xa:xb]), dtype=cp.complex64)
            mat = block.reshape(n_frames, -1)
            x = mat - mat.mean(axis=0, keepdims=True)
            cov = x @ x.conj().T
            evals = cp.linalg.eigvalsh(cov)
            s = cp.asnumpy(cp.sqrt(cp.sort(cp.maximum(evals.real, 0.0))[::-1]))
            raw[iz, ix] = rank_from_singular_values(
                s,
                method=rank_method,
                energy_fraction=energy_fraction,
                min_rank=min_rank,
                max_rank=max_r,
            )
            del block, mat, x, cov, evals, s
            pool.free_all_blocks()
    reg = regularize_rank_map(
        raw,
        delta=delta,
        global_rank=global_rank,
        smooth_size=smooth_size,
        min_rank=min_rank,
        max_rank=max_r,
    )
    return reg, raw


def filter_svd_3d_region_ranked_gpu(
    data: np.ndarray,
    n_z_blocks: int = 3,
    n_x_blocks: int = 3,
    overlap: float = 0.25,
    *,
    rank_method: str = "elbow",
    energy_fraction: float = 0.85,
    delta: float = 0.0,
    global_rank: int | None = None,
    smooth_size: int = 3,
    min_rank: int = 0,
    max_rank: int | None = None,
    high_cutoff: float | None = None,
    voxel_chunk: int = 300_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GPU region SVD wrapper using the regularized k-map."""
    import cupy as cp

    from .gpu_svd import filter_svd_3d_gpu

    if data.ndim == 3:
        data4 = data[:, None, :, :]
        squeeze = True
    elif data.ndim == 4:
        data4 = data
        squeeze = False
    else:
        raise ValueError(f"Expected 3D or 4D compound data, got shape {data.shape}")
    _, _, n_z, n_x = data4.shape
    kmap, raw = block_rank_map_gpu(
        data4,
        n_z_blocks=n_z_blocks,
        n_x_blocks=n_x_blocks,
        overlap=overlap,
        rank_method=rank_method,
        energy_fraction=energy_fraction,
        delta=delta,
        global_rank=global_rank,
        smooth_size=smooth_size,
        min_rank=min_rank,
        max_rank=max_rank,
    )
    z_bounds = _block_bounds(n_z, n_z_blocks, overlap)
    x_bounds = _block_bounds(n_x, n_x_blocks, overlap)
    z_windows = _axis_window(z_bounds, n_z)
    x_windows = _axis_window(x_bounds, n_x)
    acc = np.zeros(data4.shape, dtype=np.complex64)
    wsum = np.zeros((n_z, n_x), dtype=np.float64)
    pool = cp.get_default_memory_pool()
    pool.free_all_blocks()
    for iz, (za, zb) in enumerate(z_bounds):
        for ix, (xa, xb) in enumerate(x_bounds):
            block = np.ascontiguousarray(data4[:, :, za:zb, xa:xb])
            k = int(kmap[iz, ix])
            filt = filter_svd_3d_gpu(
                block,
                method="fast" if k > 0 else "none",
                n_components=k,
                high_cutoff=high_cutoff,
                voxel_chunk=voxel_chunk,
            )
            w2d = np.outer(z_windows[iz][za:zb], x_windows[ix][xa:xb])
            acc[:, :, za:zb, xa:xb] += filt * w2d
            wsum[za:zb, xa:xb] += w2d
            del block, filt, w2d
            pool.free_all_blocks()
    out = (acc / wsum[None, None, :, :]).astype(np.complex64, copy=False)
    if squeeze:
        out = out[:, 0]
    return out, kmap, raw
