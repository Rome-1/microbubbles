"""GPU (cupy) port of the rigid motion estimator/corrector (mb-crr.11).

Numerically mirrors ``motion.py`` -- the low-rank tissue B-mode, the FFT
phase-correlation shift estimate, and the Fourier inverse-warp -- but runs the
per-frame FFTs and the chunked covariance projection on the GPU. The CPU path in
``motion.py`` is the authoritative baseline; this is validated for equivalence
against it on Modal (no local GPU here).

Memory (target A10G 24 GB): the full magnitude volume
``(700, 25, 225, 378)`` float32 is ~6 GB and the complex compound ~12 GB, so we
NEVER hold a second full complex copy. The Gram is accumulated in voxel chunks,
the tissue reconstruction is written into a float32 output in chunks, and the
phase correlation / shift run one frame at a time. ``free_all_blocks`` is called
between chunks/phases and loop views are ``del``'d so the big buffers are
actually reclaimed.

cupy / cupyx imports are kept INSIDE the functions so the module imports fine on
a CPU-only host.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Tissue B-mode (low-rank temporal reconstruction)                            #
# --------------------------------------------------------------------------- #
def tissue_bmode_gpu(
    compound: np.ndarray,
    rank: int = 10,
    voxel_chunk: int = 300_000,
) -> np.ndarray:
    """GPU analogue of :func:`motion.tissue_bmode`.

    Keeps the TOP-``rank`` temporal components (tissue) instead of removing them
    (the clutter filter). Returns the float32 B-mode magnitude on the host.
    """
    import cupy as cp

    if compound.ndim == 3:
        data = compound[:, None, :, :]
        squeeze = True
    elif compound.ndim == 4:
        data = compound
        squeeze = False
    else:
        raise ValueError(f"Expected 3D or 4D compound, got shape {compound.shape}")

    n_frames = int(data.shape[0])
    spatial = data.shape[1:]

    cp.get_default_memory_pool().free_all_blocks()
    mat = cp.asarray(data, dtype=cp.complex64).reshape(n_frames, -1)
    n_vox = mat.shape[1]
    rank = int(max(1, min(rank, n_frames)))

    # Gram accumulated in complex128 (small (F,F)) from complex64 chunk matmuls.
    gram = cp.zeros((n_frames, n_frames), dtype=cp.complex128)
    for s0 in range(0, n_vox, voxel_chunk):
        mc = mat[:, s0:s0 + voxel_chunk]
        gram += (mc @ mc.conj().T).astype(cp.complex128)

    evals, u = cp.linalg.eigh(gram)
    u = u[:, cp.argsort(evals)[::-1]]
    uk = u[:, :rank].astype(cp.complex64)        # (F, rank)
    uk_h = uk.conj().T                           # (rank, F)
    del gram, u, evals

    out = cp.empty((n_frames, n_vox), dtype=cp.float32)
    for s0 in range(0, n_vox, voxel_chunk):
        mc = mat[:, s0:s0 + voxel_chunk]
        recon = uk @ (uk_h @ mc)                 # (F, chunk)
        out[:, s0:s0 + voxel_chunk] = cp.abs(recon).astype(cp.float32)
        del recon

    bmode = cp.asnumpy(out).reshape((n_frames, *spatial))
    del mat, out, uk, uk_h, mc
    cp.get_default_memory_pool().free_all_blocks()
    if squeeze:
        bmode = bmode[:, 0]
    return bmode


# --------------------------------------------------------------------------- #
# Phase-correlation helpers (cupy)                                            #
# --------------------------------------------------------------------------- #
def _subvoxel_peak_gpu(corr):
    """Sub-voxel signed peak of a cupy phase-correlation volume.

    Mirrors ``motion._subvoxel_peak``: integer argmax + per-axis parabolic
    refinement (modular neighbors) + unwrap to signed shift. The tiny per-axis
    scalar work is pulled to the host; only the (small) corr volume is on GPU.
    """
    import cupy as cp

    shape = corr.shape
    flat = int(cp.argmax(corr).item())
    peak = list(np.unravel_index(flat, shape))
    shifts = np.zeros(len(shape), dtype=np.float64)
    y0 = float(corr[tuple(peak)].item())
    for ax, n in enumerate(shape):
        if n >= 3:
            idx_m = list(peak)
            idx_p = list(peak)
            idx_m[ax] = (peak[ax] - 1) % n
            idx_p[ax] = (peak[ax] + 1) % n
            ym = float(corr[tuple(idx_m)].item())
            yp = float(corr[tuple(idx_p)].item())
            denom = ym - 2.0 * y0 + yp
            offset = 0.0 if abs(denom) < 1e-12 else float(
                np.clip(0.5 * (ym - yp) / denom, -0.5, 0.5)
            )
        else:
            offset = 0.0
        coord = peak[ax]
        if coord > n // 2:
            coord -= n
        shifts[ax] = coord + offset
    return shifts


def _phase_correlation_gpu(ref_fft_conj, frame):
    """Signed sub-voxel shift of cupy ``frame`` vs precomputed conj-ref FFT."""
    import cupy as cp

    fr = cp.fft.fftn(frame)
    cross = ref_fft_conj * fr
    mag = cp.abs(cross)
    cross = cp.where(mag > 0, cross / (mag + 1e-12), 0.0 + 0.0j)
    corr = cp.fft.ifftn(cross).real
    shift = _subvoxel_peak_gpu(corr)
    del fr, cross, mag, corr
    return shift


def _resolve_reference_gpu(bmode, reference):
    import cupy as cp

    if isinstance(reference, np.ndarray):
        return cp.asarray(reference, dtype=cp.float64)
    if isinstance(reference, cp.ndarray):
        return reference.astype(cp.float64)
    if reference == "mean":
        return bmode.mean(axis=0).astype(cp.float64)
    if reference == "first":
        return bmode[0].astype(cp.float64)
    if isinstance(reference, (int, np.integer)):
        return bmode[int(reference)].astype(cp.float64)
    raise ValueError(f"Unsupported reference: {reference!r}")


# --------------------------------------------------------------------------- #
# Motion estimation                                                           #
# --------------------------------------------------------------------------- #
def estimate_motion_rigid_gpu(
    bmode: np.ndarray,
    reference="mean",
    upsample: int = 1,
) -> np.ndarray:
    """GPU analogue of :func:`motion.estimate_motion_rigid` (FFT path).

    ``bmode`` may be a host ndarray or a cupy array. Returns host float32
    ``(frames, spatial_ndim)`` shifts in voxels, same convention as the CPU
    version. ``upsample`` is accepted for signature parity; the FFT path refines
    to sub-voxel via parabolic interpolation regardless.
    """
    import cupy as cp

    bmode_cp = cp.asarray(bmode, dtype=cp.float64)
    if bmode_cp.ndim < 2:
        raise ValueError(f"Expected (frames, ...spatial), got shape {bmode_cp.shape}")
    n_frames = int(bmode_cp.shape[0])
    spatial_ndim = bmode_cp.ndim - 1

    ref = _resolve_reference_gpu(bmode_cp, reference)
    ref_fft_conj = cp.conj(cp.fft.fftn(ref))

    shifts = np.zeros((n_frames, spatial_ndim), dtype=np.float64)
    for i in range(n_frames):
        shifts[i] = _phase_correlation_gpu(ref_fft_conj, bmode_cp[i])
        if (i & 63) == 0:
            cp.get_default_memory_pool().free_all_blocks()

    del bmode_cp, ref, ref_fft_conj
    cp.get_default_memory_pool().free_all_blocks()
    return shifts.astype(np.float32)


# --------------------------------------------------------------------------- #
# Motion application                                                          #
# --------------------------------------------------------------------------- #
def apply_motion_gpu(compound: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """GPU analogue of :func:`motion.apply_motion`.

    Inverse-warps each frame by ``-shift`` with ``cupyx.scipy.ndimage.fourier_shift``
    on the complex data, one frame at a time, returning a host complex64 array.
    """
    import cupy as cp
    from cupyx.scipy.ndimage import fourier_shift as cp_fourier_shift

    arr = compound
    if arr.ndim < 2:
        raise ValueError(f"Expected (frames, ...spatial), got shape {arr.shape}")
    n_frames = int(arr.shape[0])
    spatial_ndim = arr.ndim - 1
    shifts = np.asarray(shifts, dtype=np.float64)
    if shifts.shape != (n_frames, spatial_ndim):
        raise ValueError(
            f"shifts shape {shifts.shape} != (frames, spatial_ndim) "
            f"({n_frames}, {spatial_ndim})"
        )

    spatial_axes = tuple(range(spatial_ndim))
    out = np.empty(arr.shape, dtype=np.complex64)
    cp.get_default_memory_pool().free_all_blocks()
    for i in range(n_frames):
        s = shifts[i]
        if not np.any(s):
            out[i] = np.asarray(arr[i], dtype=np.complex64)
            continue
        frame = cp.asarray(arr[i], dtype=cp.complex64)
        frame_fft = cp.fft.fftn(frame, axes=spatial_axes)
        shifted = cp.fft.ifftn(
            cp_fourier_shift(frame_fft, -s), axes=spatial_axes
        ).astype(cp.complex64)
        out[i] = cp.asnumpy(shifted)
        del frame, frame_fft, shifted
        if (i & 63) == 0:
            cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_memory_pool().free_all_blocks()
    return out


# --------------------------------------------------------------------------- #
# Convenience                                                                  #
# --------------------------------------------------------------------------- #
def correct_motion_gpu(
    compound: np.ndarray,
    rank: int = 10,
    max_shift_voxels: float = 8.0,
    reference="mean",
    upsample: int = 1,
    voxel_chunk: int = 300_000,
) -> tuple[np.ndarray, np.ndarray]:
    """GPU analogue of :func:`motion.correct_motion`.

    Returns ``(corrected_compound_host, shifts_host)``.
    """
    import cupy as cp

    bmode = tissue_bmode_gpu(compound, rank=rank, voxel_chunk=voxel_chunk)
    shifts = estimate_motion_rigid_gpu(bmode, reference=reference, upsample=upsample)
    del bmode
    cp.get_default_memory_pool().free_all_blocks()

    shifts = np.asarray(shifts, dtype=np.float32).copy()
    bad = np.any(np.abs(shifts) > float(max_shift_voxels), axis=1)
    shifts[bad] = 0.0
    corrected = apply_motion_gpu(compound, shifts)
    return corrected, shifts
