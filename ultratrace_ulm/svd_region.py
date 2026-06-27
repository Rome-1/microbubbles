"""Spatially region-adaptive SVD clutter filter (mb-crr.12), NUMPY reference.

The shipped baseline (``svd.filter_svd_3d``) runs ONE global temporal SVD with
ONE adaptive cutoff over the whole (frames, voxels) matrix. Tissue clutter in 3D
ULM is depth/region-structured (skull shadow, superficial vs deep), so a single
global cutoff leaves region-structured residual clutter -- visible as horizontal
"banding" in sagittal/axial MIPs.

This module partitions the spatial volume into OVERLAPPING blocks along z and x
(elev is kept whole -- only 25 planes), runs the existing ``svd.filter_svd_3d``
on each sub-volume so every block gets its OWN adaptive (spectral-centroid)
cutoff, then blends the overlapping blocks with a smooth windowed
overlap-add. Tissue then gets region-specific suppression.

Design (blend == partition of unity):
  * Each axis is tiled by ``n`` blocks whose stride leaves a fractional
    ``overlap`` between neighbours (see ``_block_bounds``).
  * Each block carries a 1D weight that is 1 in its core and cos^2/sin^2 ramps
    across the overlap with each neighbour (see ``_axis_window``). Matched
    cos^2/sin^2 ramps are complementary, so the per-axis weights sum to exactly
    1 at every index. The 2D block weight is the separable outer product, so the
    2D weights also sum to exactly 1 -- a true partition of unity. We still
    divide by the accumulated weight, so the result is correct (and seam-free)
    even outside the <=2-overlap-per-axis regime.

With ``n_z_blocks == n_x_blocks == 1`` there is a single full-volume block with
unit weight, so this reduces EXACTLY to ``svd.filter_svd_3d`` (the weight is
1.0 and the float32 -> complex128 -> float32 round trip is lossless).

This is the CPU/numpy reference; ``gpu_svd_region`` mirrors it on the GPU.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .svd import _component_count, filter_svd_3d, spectral_centroid_cutoff


def _block_bounds(length: int, n_blocks: int, overlap: float) -> list[tuple[int, int]]:
    """Integer ``(start, end)`` slices tiling ``[0, length)`` with ``n_blocks``
    overlapping blocks. ``overlap`` is the fractional overlap between adjacent
    blocks relative to block size. Guarantees full coverage (no gaps) and that
    block 0 starts at 0 and the last block ends at ``length``."""
    if n_blocks <= 1:
        return [(0, length)]
    overlap = float(np.clip(overlap, 0.0, 0.9))
    # block_size * [(1-overlap)*(n-1) + 1] = length ; stride = block_size*(1-overlap)
    block_size = length / ((1.0 - overlap) * (n_blocks - 1) + 1.0)
    stride = block_size * (1.0 - overlap)
    bounds: list[list[int]] = []
    for i in range(n_blocks):
        start = int(round(i * stride))
        end = int(round(i * stride + block_size))
        start = max(0, min(start, length - 1))
        end = max(start + 1, min(end, length))
        bounds.append([start, end])
    bounds[0][0] = 0
    bounds[-1][1] = length
    return [(a, b) for a, b in bounds]


def _axis_window(bounds: list[tuple[int, int]], length: int) -> list[np.ndarray]:
    """One float64 weight of shape ``(length,)`` per block: 1 in the core,
    cos^2/sin^2 ramps over the overlap with each neighbour. Matched ramps are
    complementary so ``sum(windows)`` is exactly 1 at every index (in the
    <=2-overlap regime); zero outside the block's support."""
    n = len(bounds)
    windows: list[np.ndarray] = []
    for i, (a, b) in enumerate(bounds):
        w = np.zeros(length, dtype=np.float64)
        wi = np.ones(b - a, dtype=np.float64)
        # Left ramp UP across the overlap with the previous block: [a, prev_end).
        left_end = bounds[i - 1][1] if i > 0 else a
        left_end = int(min(max(left_end, a), b))
        if left_end > a:
            t = (np.arange(a, left_end) - a) / float(left_end - a)
            wi[: left_end - a] = np.sin(0.5 * np.pi * t) ** 2
        # Right ramp DOWN across the overlap with the next block: [next_start, b).
        right_start = bounds[i + 1][0] if i < n - 1 else b
        right_start = int(min(max(right_start, a), b))
        if b > right_start:
            t = (np.arange(right_start, b) - right_start) / float(b - right_start)
            wi[right_start - a :] = np.cos(0.5 * np.pi * t) ** 2
        w[a:b] = wi
        windows.append(w)
    return windows


def _resolve_rank(
    matrix: np.ndarray,
    method: str,
    low_cutoff: float,
    n_components: int | None,
    frame_rate_hz: float | None,
    tissue_freq_hz: float,
) -> int:
    """The integer rank k (number of leading temporal components removed) that
    ``svd.filter_svd_3d`` would apply to ``matrix`` ((frames, voxels)) under the
    given ``method`` -- used both for the single global rank and the per-block
    diagnostic so they cannot drift from the filter's own behaviour."""
    n_frames = int(matrix.shape[0])
    if method == "adaptive":
        if frame_rate_hz is None:
            raise ValueError("method='adaptive' requires frame_rate_hz")
        return spectral_centroid_cutoff(matrix, frame_rate_hz, tissue_freq_hz)
    if method == "none":
        return 0
    return int(n_components) if n_components is not None else _component_count(low_cutoff, n_frames)


def region_block_cutoffs(
    data: np.ndarray,
    n_z_blocks: int = 3,
    n_x_blocks: int = 3,
    overlap: float = 0.25,
    cutoff_mode: str = "global_rank",
    low_cutoff: float = 0.1,
    method: str = "adaptive",
    frame_rate_hz: float | None = None,
    tissue_freq_hz: float = 100.0,
    n_components: int | None = None,
) -> np.ndarray:
    """Diagnostic: the integer rank k that ``filter_svd_3d_region`` removes in
    each (z, x) block, as an ``(n_z_blocks, n_x_blocks)`` int array.

    Under ``cutoff_mode="global_rank"`` this is a single global rank broadcast to
    every block (constant). Under ``cutoff_mode="per_block"`` with
    ``method="adaptive"`` each block picks its own spectral-centroid cutoff, so
    the count varies by region (the source of the per-block intensity
    inhomogeneity that ``global_rank`` removes).
    """
    if data.ndim == 3:
        data = data[:, None, :, :]
    elif data.ndim != 4:
        raise ValueError(f"Expected 3D or 4D compound data, got shape {data.shape}")

    n_frames, _, n_z, n_x = data.shape
    z_bounds = _block_bounds(n_z, n_z_blocks, overlap)
    x_bounds = _block_bounds(n_x, n_x_blocks, overlap)
    ks = np.zeros((len(z_bounds), len(x_bounds)), dtype=int)

    if cutoff_mode == "global_rank":
        matrix = np.asarray(data, dtype=np.complex64).reshape(n_frames, -1)
        ks[:] = _resolve_rank(matrix, method, low_cutoff, n_components, frame_rate_hz, tissue_freq_hz)
        return ks
    if cutoff_mode != "per_block":
        raise ValueError(f"Unknown cutoff_mode: {cutoff_mode!r}")

    for iz, (za, zb) in enumerate(z_bounds):
        for ix, (xa, xb) in enumerate(x_bounds):
            bm = np.asarray(data[:, :, za:zb, xa:xb], dtype=np.complex64).reshape(n_frames, -1)
            ks[iz, ix] = _resolve_rank(bm, method, low_cutoff, n_components, frame_rate_hz, tissue_freq_hz)
    return ks


def filter_svd_3d_region(
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
    cutoff_mode: str = "global_rank",
) -> np.ndarray:
    """Region-adaptive temporal-SVD clutter filter.

    Partitions the z and x axes into overlapping blocks (elev kept whole),
    filters each sub-volume with ``svd.filter_svd_3d``, and blends overlaps with
    a smooth partition-of-unity window. Returns the filtered COMPLEX volume,
    same shape as ``data`` ((F,elev,z,x) or (F,z,x)).

    ``cutoff_mode`` controls how the removal rank is chosen per block:

    * ``"global_rank"`` (DEFAULT): compute ONE rank ``k`` on the whole volume
      (the global adaptive spectral-centroid cutoff for ``method="adaptive"``,
      else the fixed ``low_cutoff``/``n_components`` count), then filter every
      block with that same ``k`` via ``n_components=k``. Each block still uses
      its OWN local temporal subspace (from its local covariance), but removes
      the same NUMBER of components everywhere. This keeps the region-adaptivity
      benefit (local tissue subspace per block) while eliminating the per-block
      intensity inhomogeneity that a varying per-block count introduces (bright
      center / dim periphery banding).
    * ``"per_block"``: each block computes its own cutoff (the original
      behaviour). With ``method="adaptive"`` the removal count varies by region.

    Reduces EXACTLY to ``svd.filter_svd_3d`` when ``n_z_blocks == n_x_blocks == 1``
    (both cutoff modes).
    """
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

    # Resolve how each block is filtered. For "global_rank" we pick one rank from
    # the whole volume and apply it as a fixed component COUNT to every block
    # (uniform removal -> no per-block intensity inhomogeneity); each block still
    # gets its own LOCAL subspace because filter_svd_3d works on the local block.
    if cutoff_mode == "global_rank":
        if method == "none":
            block_method, block_ncomp = "none", None
        else:
            matrix = np.asarray(data, dtype=np.complex64).reshape(n_frames, -1)
            global_k = _resolve_rank(
                matrix, method, low_cutoff, n_components, frame_rate_hz, tissue_freq_hz
            )
            del matrix
            block_method, block_ncomp = "fast", global_k
    elif cutoff_mode == "per_block":
        block_method, block_ncomp = method, n_components
    else:
        raise ValueError(f"Unknown cutoff_mode: {cutoff_mode!r}")

    acc = np.zeros(data.shape, dtype=np.complex64)
    wsum = np.zeros((n_z, n_x), dtype=np.float64)

    for iz, (za, zb) in enumerate(z_bounds):
        for ix, (xa, xb) in enumerate(x_bounds):
            block = data[:, :, za:zb, xa:xb]
            filt = filter_svd_3d(
                block,
                low_cutoff=low_cutoff,
                high_cutoff=high_cutoff,
                method=block_method,
                n_components=block_ncomp,
                frame_rate_hz=frame_rate_hz,
                tissue_freq_hz=tissue_freq_hz,
            )
            # Separable 2D weight over (z, x) restricted to this block's support.
            w2d = np.outer(z_windows[iz][za:zb], x_windows[ix][xa:xb])
            acc[:, :, za:zb, xa:xb] += filt * w2d  # broadcast over (F, elev)
            wsum[za:zb, xa:xb] += w2d

    out = (acc / wsum[None, None, :, :]).astype(np.complex64, copy=False)
    if squeeze:
        out = out[:, 0]
    return out


def filtered_magnitude_region(
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
    cutoff_mode: str = "global_rank",
) -> np.ndarray:
    """Region-adaptive analogue of ``svd.filtered_magnitude``: returns float32
    |filtered|, optionally temporally smoothed."""
    filtered = filter_svd_3d_region(
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
        cutoff_mode=cutoff_mode,
    )
    magnitude = np.abs(filtered).astype(np.float32, copy=False)
    if temporal_sigma > 0:
        magnitude = gaussian_filter1d(magnitude, sigma=temporal_sigma, axis=0)
    return magnitude
