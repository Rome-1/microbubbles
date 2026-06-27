"""GPU (cupy) port of the region-adaptive SVD clutter filter (mb-crr.12).

Numerically mirrors ``svd_region.filter_svd_3d_region`` /
``svd_region.filtered_magnitude_region``, but filters each spatial block with
the GPU kernel ``gpu_svd.filter_svd_3d_gpu`` (which already handles the
memory-disciplined Gram accumulation + the phase-invariant adaptive cutoff).

Memory discipline (A10G, 24 GB -- these rules cost real failed runs):
  * The blocks are sub-volumes, so each per-block matrix is far smaller than the
    full (700, ~2.13M) complex64 matrix (11.9 GB); only one block is ever on the
    GPU at a time because ``filter_svd_3d_gpu`` returns its result to the host.
  * ``cp.get_default_memory_pool().free_all_blocks()`` at entry AND after every
    block iteration, with the per-block host arrays ``del``'d first so nothing
    keeps a parent array alive.
  * The blend accumulator lives on the HOST (numpy complex64), so the only large
    device allocations are inside ``filter_svd_3d_gpu`` per block.

The CPU path in ``svd_region.py`` is the authoritative reference; this is
validated for equivalence against it on the GPU (Modal).
"""

from __future__ import annotations

import numpy as np


def filter_svd_3d_region_gpu(
    data: np.ndarray,
    n_z_blocks: int = 3,
    n_x_blocks: int = 3,
    overlap: float = 0.25,
    low_cutoff: float = 0.1,
    method: str = "adaptive",
    frame_rate_hz: float | None = None,
    tissue_freq_hz: float = 100.0,
    high_cutoff: float | None = None,
    n_components: int | None = None,
    voxel_chunk: int = 300_000,
) -> np.ndarray:
    """GPU region-adaptive temporal-SVD clutter filter. Returns the blended
    filtered complex volume on the host, same shape as ``data``
    ((F,elev,z,x) or (F,z,x)). Reduces to ``gpu_svd.filter_svd_3d_gpu`` when
    ``n_z_blocks == n_x_blocks == 1``."""
    import cupy as cp

    from .gpu_svd import filter_svd_3d_gpu
    from .svd_region import _axis_window, _block_bounds

    if data.ndim == 3:
        data = data[:, None, :, :]
        squeeze = True
    elif data.ndim == 4:
        squeeze = False
    else:
        raise ValueError(f"Expected 3D or 4D compound data, got shape {data.shape}")

    n_frames, n_elev, n_z, n_x = data.shape
    z_bounds = _block_bounds(n_z, n_z_blocks, overlap)
    x_bounds = _block_bounds(n_x, n_x_blocks, overlap)
    z_windows = _axis_window(z_bounds, n_z)
    x_windows = _axis_window(x_bounds, n_x)

    # Blend accumulator stays on the HOST: each block is filtered on the GPU and
    # returned to host by filter_svd_3d_gpu, so device peak == one block's cost.
    acc = np.zeros(data.shape, dtype=np.complex64)
    wsum = np.zeros((n_z, n_x), dtype=np.float64)

    cp.get_default_memory_pool().free_all_blocks()
    for iz, (za, zb) in enumerate(z_bounds):
        for ix, (xa, xb) in enumerate(x_bounds):
            # Contiguous host copy of just this block (a view of the big host
            # array would needlessly pin it during the GPU transfer).
            block = np.ascontiguousarray(data[:, :, za:zb, xa:xb])
            filt = filter_svd_3d_gpu(
                block,
                low_cutoff=low_cutoff,
                high_cutoff=high_cutoff,
                method=method,
                n_components=n_components,
                frame_rate_hz=frame_rate_hz,
                tissue_freq_hz=tissue_freq_hz,
                voxel_chunk=voxel_chunk,
            )
            w2d = np.outer(z_windows[iz][za:zb], x_windows[ix][xa:xb])
            acc[:, :, za:zb, xa:xb] += filt * w2d  # broadcast over (F, elev)
            wsum[za:zb, xa:xb] += w2d
            del block, filt, w2d
            cp.get_default_memory_pool().free_all_blocks()

    out = (acc / wsum[None, None, :, :]).astype(np.complex64, copy=False)
    if squeeze:
        out = out[:, 0]
    return out


def filtered_magnitude_region_gpu(
    compound: np.ndarray,
    low_cutoff: float = 0.1,
    high_cutoff: float | None = None,
    method: str = "fast",
    temporal_sigma: float = 0.0,
    n_components: int | None = None,
    frame_rate_hz: float | None = None,
    tissue_freq_hz: float = 100.0,
    n_z_blocks: int = 3,
    n_x_blocks: int = 3,
    overlap: float = 0.25,
    voxel_chunk: int = 300_000,
) -> np.ndarray:
    """GPU analogue of ``svd_region.filtered_magnitude_region``: returns float32
    |filtered|, optionally temporally smoothed. Argument order matches the other
    ``filtered_magnitude*`` entry points plus the block params, so the pipeline's
    ``(compound, opts)`` adapter can wire it like ``gpu_svd.filtered_magnitude_gpu``."""
    filtered = filter_svd_3d_region_gpu(
        compound,
        n_z_blocks=n_z_blocks,
        n_x_blocks=n_x_blocks,
        overlap=overlap,
        low_cutoff=low_cutoff,
        method=method,
        frame_rate_hz=frame_rate_hz,
        tissue_freq_hz=tissue_freq_hz,
        high_cutoff=high_cutoff,
        n_components=n_components,
        voxel_chunk=voxel_chunk,
    )
    magnitude = np.abs(filtered).astype(np.float32, copy=False)
    if temporal_sigma > 0:
        from scipy.ndimage import gaussian_filter1d

        magnitude = gaussian_filter1d(magnitude, sigma=temporal_sigma, axis=0)
    return magnitude
