"""Rigid motion estimation + correction BEFORE clutter filtering (mb-crr.11).

A 4-minute scan carries small probe/head motion that drifts tissue through
voxels. That drift (a) smears vessels across super-resolution pixels and (b)
breaks the temporal-SVD clutter filter, whose whole premise is that tissue is
*stationary* (slow, low-rank) while blood is fast. If the tissue itself moves,
it stops being low-rank and bleeds into the "blood" subspace.

The fix is to estimate the motion from the TISSUE component -- the low-rank
(B-mode) reconstruction, NOT the sparse bubbles -- and inverse-warp each frame
to a common reference before SVD + accumulation.

This module is the numpy/scipy reference implementation. ``gpu_motion`` mirrors
it on cupy. It is import-safe without cupy/h5py/skimage (those are imported lazily
inside the functions that need them).

Conventions
-----------
* ``compound`` is complex ``(frames, elev, z, x)`` (or ``(frames, z, x)``).
* A per-frame shift vector is ordered to match the volume's SPATIAL axes, i.e.
  ``shift[i]`` is the displacement along ``compound`` axis ``i + 1``. For the
  ``(frames, elev, z, x)`` layout that is ``(d_elev, d_z, d_x)`` voxels.
  Keeping the shift in array-axis order means ``apply_motion`` can feed it
  straight to ``fourier_shift`` with no axis bookkeeping.
* ``estimate_motion_rigid`` returns the displacement ``s`` of each frame
  RELATIVE to the reference (frame(x) ~= reference(x - s)); ``apply_motion``
  inverse-warps by ``-s`` to bring every frame onto the reference.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import fourier_shift


# --------------------------------------------------------------------------- #
# Tissue B-mode (low-rank temporal reconstruction)                            #
# --------------------------------------------------------------------------- #
def tissue_bmode(
    compound: np.ndarray,
    rank: int = 10,
    voxel_chunk: int = 200_000,
) -> np.ndarray:
    """Magnitude of the low-rank (top-``rank`` temporal SVD) reconstruction.

    This is the *tissue* / B-mode image: the OPPOSITE of the SVD clutter filter
    in ``svd.py``. The clutter filter keeps components ``[low:]`` (the fast
    residual = blood); here we keep components ``[:rank]`` (the slow, large
    eigenvalues = tissue).

    Reuses the covariance-projection idea from ``svd.filter_svd_3d`` (the "fast"
    path): form the ``(frames, frames)`` Gram of the temporal matrix, take its
    top-``rank`` eigenvectors ``U_k``, and reconstruct ``U_k (U_k^H X)``. The
    Gram and the reconstruction are computed in voxel chunks so the only
    full-size buffers are the input and the float32 output (never a second full
    complex copy).

    Parameters
    ----------
    compound : complex array, ``(frames, elev, z, x)`` or ``(frames, z, x)``.
    rank : number of temporal components to KEEP as tissue.
    voxel_chunk : voxels processed per chunk (memory knob).

    Returns
    -------
    float32 array, same shape as ``compound``: the tissue B-mode magnitude.
    """
    if compound.ndim == 3:
        spatial = compound.shape[1:]
        squeeze = True
        data = compound[:, None, :, :]
        spatial = data.shape[1:]
    elif compound.ndim == 4:
        squeeze = False
        data = compound
        spatial = data.shape[1:]
    else:
        raise ValueError(f"Expected 3D or 4D compound, got shape {compound.shape}")

    n_frames = int(data.shape[0])
    mat = np.asarray(data, dtype=np.complex64).reshape(n_frames, -1)
    n_vox = mat.shape[1]

    rank = int(max(1, min(rank, n_frames)))

    # Gram in complex128 for a stable eigendecomposition (the (F,F) matrix is
    # tiny; the cost is in the chunked matmuls, kept in complex64).
    gram = np.zeros((n_frames, n_frames), dtype=np.complex128)
    for s0 in range(0, n_vox, voxel_chunk):
        mc = mat[:, s0:s0 + voxel_chunk]
        gram += (mc @ mc.conj().T).astype(np.complex128)

    evals, u = np.linalg.eigh(gram)
    u = u[:, np.argsort(evals)[::-1]]
    uk = u[:, :rank].astype(np.complex64)        # (F, rank) top components
    uk_h = uk.conj().T                           # (rank, F)

    out = np.empty((n_frames, n_vox), dtype=np.float32)
    for s0 in range(0, n_vox, voxel_chunk):
        mc = mat[:, s0:s0 + voxel_chunk]
        recon = uk @ (uk_h @ mc)                 # (F, chunk) low-rank tissue
        out[:, s0:s0 + voxel_chunk] = np.abs(recon).astype(np.float32)
        del recon

    bmode = out.reshape((n_frames, *spatial))
    if squeeze:
        bmode = bmode[:, 0]
    return bmode


# --------------------------------------------------------------------------- #
# Phase-correlation helpers                                                    #
# --------------------------------------------------------------------------- #
def _parabolic_offset(y_minus: float, y_center: float, y_plus: float) -> float:
    """Sub-voxel peak offset from three samples (vertex of the fitted parabola)."""
    denom = y_minus - 2.0 * y_center + y_plus
    if abs(denom) < 1e-12:
        return 0.0
    return float(np.clip(0.5 * (y_minus - y_plus) / denom, -0.5, 0.5))


def _subvoxel_peak(corr: np.ndarray) -> np.ndarray:
    """Locate the (sub-voxel, signed) peak of a phase-correlation volume.

    Integer peak via argmax, sub-voxel refinement via per-axis parabolic
    interpolation (neighbors taken modulo the axis length), then each axis is
    unwrapped to a signed shift in ``(-N/2, N/2]``.
    """
    shape = corr.shape
    peak = np.unravel_index(int(np.argmax(corr)), shape)
    shifts = np.zeros(len(shape), dtype=np.float64)
    for ax, p in enumerate(shape):
        if p >= 3:
            idx_m = list(peak)
            idx_p = list(peak)
            idx_m[ax] = (peak[ax] - 1) % p
            idx_p[ax] = (peak[ax] + 1) % p
            offset = _parabolic_offset(
                float(corr[tuple(idx_m)]),
                float(corr[peak]),
                float(corr[tuple(idx_p)]),
            )
        else:
            offset = 0.0
        coord = peak[ax]
        if coord > p // 2:           # unwrap to a signed shift
            coord -= p
        shifts[ax] = coord + offset
    return shifts


def _phase_correlation(ref_fft_conj: np.ndarray, frame: np.ndarray) -> np.ndarray:
    """Signed sub-voxel shift of ``frame`` relative to a reference.

    ``ref_fft_conj = conj(fftn(reference))`` is precomputed once. With
    ``frame(x) = reference(x - s)`` the normalized cross-power spectrum is
    ``exp(-i 2pi k . s / N)``, whose inverse transform peaks at ``x = s``.
    """
    fr = np.fft.fftn(frame)
    cross = ref_fft_conj * fr
    mag = np.abs(cross)
    cross = np.where(mag > 0, cross / (mag + 1e-12), 0.0)
    corr = np.fft.ifftn(cross).real
    return _subvoxel_peak(corr)


def _resolve_reference(bmode: np.ndarray, reference) -> np.ndarray:
    """Pick the reference frame (spatial volume) from the ``reference`` arg."""
    if isinstance(reference, np.ndarray):
        if reference.shape != bmode.shape[1:]:
            raise ValueError(
                f"reference array shape {reference.shape} != frame shape {bmode.shape[1:]}"
            )
        return np.asarray(reference, dtype=np.float64)
    if reference == "mean":
        return bmode.mean(axis=0).astype(np.float64)
    if reference == "first":
        return np.asarray(bmode[0], dtype=np.float64)
    if isinstance(reference, (int, np.integer)):
        return np.asarray(bmode[int(reference)], dtype=np.float64)
    raise ValueError(f"Unsupported reference: {reference!r}")


# --------------------------------------------------------------------------- #
# Motion estimation                                                           #
# --------------------------------------------------------------------------- #
def estimate_motion_rigid(
    bmode: np.ndarray,
    reference="mean",
    upsample: int = 1,
    *,
    method: str = "auto",
) -> np.ndarray:
    """Per-frame rigid 3D translation via FFT phase correlation of the B-mode.

    Each frame's B-mode is correlated against a reference (default the temporal
    mean) to recover its displacement. Shifts are in VOXELS, ordered to match
    the volume's spatial axes -- for ``(frames, elev, z, x)`` that is
    ``(d_elev, d_z, d_x)`` (i.e. (dz, dy/elev, dx) physically, reordered to
    array-axis order so ``apply_motion`` can use it directly).

    Parameters
    ----------
    bmode : real array, ``(frames, elev, z, x)`` or ``(frames, z, x)``
        Tissue B-mode (see :func:`tissue_bmode`).
    reference : ``"mean"`` (default), ``"first"``, an int frame index, or a
        spatial ndarray. ``"mean"`` returns shifts relative to the temporal
        mean (so the per-axis mean shift is ~0).
    upsample : sub-voxel upsampling factor passed to skimage when used. The
        hand-rolled FFT fallback always refines to sub-voxel via parabolic
        interpolation regardless of this value.
    method : ``"auto"`` (skimage if importable, else FFT fallback), ``"fft"``
        (force hand-rolled), or ``"skimage"`` (force skimage; raises if absent).

    Returns
    -------
    float32 array ``(frames, spatial_ndim)`` of per-frame shifts in voxels.
    """
    bmode = np.asarray(bmode, dtype=np.float64)
    if bmode.ndim < 2:
        raise ValueError(f"Expected (frames, ...spatial), got shape {bmode.shape}")
    n_frames = int(bmode.shape[0])
    spatial_ndim = bmode.ndim - 1

    ref = _resolve_reference(bmode, reference)

    use_skimage = False
    pcc = None
    if method in ("auto", "skimage"):
        try:
            from skimage.registration import phase_cross_correlation as pcc  # noqa: N813
            use_skimage = True
        except Exception:
            if method == "skimage":
                raise
            use_skimage = False
    elif method != "fft":
        raise ValueError(f"Unsupported method: {method!r}")

    shifts = np.zeros((n_frames, spatial_ndim), dtype=np.float64)

    if use_skimage:
        up = max(int(round(upsample)), 1)
        for i in range(n_frames):
            shift, _err, _phase = pcc(
                ref, bmode[i], upsample_factor=up, normalization="phase"
            )
            # skimage returns the shift mapping moving->reference; negate to get
            # the frame's displacement relative to the reference (our convention).
            shifts[i] = -np.asarray(shift, dtype=np.float64)
        return shifts.astype(np.float32)

    ref_fft_conj = np.conj(np.fft.fftn(ref))
    for i in range(n_frames):
        shifts[i] = _phase_correlation(ref_fft_conj, bmode[i])
    return shifts.astype(np.float32)


# --------------------------------------------------------------------------- #
# Motion application                                                          #
# --------------------------------------------------------------------------- #
def apply_motion(compound: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """Inverse-warp each frame by the NEGATIVE estimated shift to align them.

    Uses band-limited Fourier shifting (``scipy.ndimage.fourier_shift``) on the
    complex data, frame by frame, preserving complex64 dtype and shape.

    Parameters
    ----------
    compound : complex array, ``(frames, elev, z, x)`` or ``(frames, z, x)``.
    shifts : ``(frames, spatial_ndim)`` per-frame shifts from
        :func:`estimate_motion_rigid` (same axis order as the volume).

    Returns
    -------
    complex64 array, same shape as ``compound``, motion-corrected.
    """
    arr = np.asarray(compound)
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
    for i in range(n_frames):
        s = shifts[i]
        if not np.any(s):
            out[i] = arr[i].astype(np.complex64, copy=False)
            continue
        frame_fft = np.fft.fftn(arr[i], axes=spatial_axes)
        shifted_fft = fourier_shift(frame_fft, -s)   # inverse-warp by -s
        out[i] = np.fft.ifftn(shifted_fft, axes=spatial_axes).astype(np.complex64)
    return out


# --------------------------------------------------------------------------- #
# Convenience: estimate from tissue, clamp outliers, apply                    #
# --------------------------------------------------------------------------- #
def correct_motion(
    compound: np.ndarray,
    rank: int = 10,
    max_shift_voxels: float = 8.0,
    reference="mean",
    upsample: int = 1,
    *,
    method: str = "auto",
    voxel_chunk: int = 200_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate motion from tissue, clamp outliers, inverse-warp the compound.

    1. Build the tissue B-mode (top-``rank`` low-rank reconstruction).
    2. Estimate per-frame rigid shifts from it (phase correlation).
    3. Zero any frame whose shift is implausibly large (any component magnitude
       ``> max_shift_voxels``) -- a robustness guard against correlation peaks
       landing on a wrap-around lobe or a bad frame.
    4. Apply the (clamped) shifts to the COMPLEX compound.

    Returns ``(corrected_compound, shifts)`` where ``shifts`` are the clamped
    per-frame shifts actually applied.
    """
    bmode = tissue_bmode(compound, rank=rank, voxel_chunk=voxel_chunk)
    shifts = estimate_motion_rigid(
        bmode, reference=reference, upsample=upsample, method=method
    )
    shifts = np.asarray(shifts, dtype=np.float32).copy()
    bad = np.any(np.abs(shifts) > float(max_shift_voxels), axis=1)
    shifts[bad] = 0.0
    corrected = apply_motion(compound, shifts)
    return corrected, shifts
