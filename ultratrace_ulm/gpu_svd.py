"""GPU (cupy) port of the SVD clutter filter (mb-crr.2 enabler).

Numerically mirrors ``svd.filter_svd_3d`` / ``svd.filtered_magnitude`` (the "fast"
covariance-projection path and the "adaptive" spectral-centroid cutoff), but runs
on the GPU so the beamforming GPU isn't idle ~80% of each acquisition. The probe
measured the CPU filter at ~292 s/acq; this brings it to ~10-20 s.

Memory: the (frames, voxels) matrix is ~11.9 GB at this dataset's size, so on a
24 GB card we must NOT also materialise a full conjugate copy. The Gram matrix is
accumulated in voxel chunks, and the projection is written back into the matrix
in place per chunk, keeping peak device memory ~ matrix + magnitude (~18 GB).

The CPU path in ``svd.py`` is the authoritative baseline; this is validated for
equivalence against it (see scripts/validate_gpu_svd or the Modal probe).
"""

from __future__ import annotations

import numpy as np

from .svd import _component_count


def _spectral_centroid_cutoff_gpu(gram_centered, n_frames, frame_rate_hz, tissue_freq_hz):
    """Cutoff from the temporal spectral centroid, mirroring
    ``svd.spectral_centroid_cutoff`` but vectorised on the GPU. ``gram_centered``
    is the Gram matrix of the MEAN-SUBTRACTED temporal matrix (x @ x^H)."""
    import cupy as cp

    evals, u = cp.linalg.eigh(gram_centered)
    u = u[:, cp.argsort(evals)[::-1]]  # (F, F), columns = temporal singular vectors
    freqs = cp.fft.rfftfreq(n_frames, d=1.0 / frame_rate_hz)
    spec = cp.abs(cp.fft.rfft(u.real, axis=0)) ** 2  # (Frfft, F)
    spec[0, :] = 0.0  # exclude DC
    total = spec.sum(axis=0)
    centroid = (freqs[:, None] * spec).sum(axis=0) / cp.where(total > 0, total, 1.0)
    above = cp.where(centroid > tissue_freq_hz)[0]
    if above.size:
        return int(above[0].item())
    return max(1, int(round(n_frames * 0.1)))


def filter_svd_3d_gpu(
    data: np.ndarray,
    low_cutoff: float = 0.1,
    high_cutoff: float | None = None,
    method: str = "fast",
    n_components: int | None = None,
    frame_rate_hz: float | None = None,
    tissue_freq_hz: float = 100.0,
    voxel_chunk: int = 300_000,
) -> np.ndarray:
    """GPU fast/adaptive temporal-SVD clutter filter. Returns the filtered
    complex volume on the host, same shape as ``data`` ((F,elev,z,x) or (F,z,x))."""
    import cupy as cp

    if data.ndim == 3:
        data = data[:, None, :, :]
        squeeze = True
    elif data.ndim == 4:
        squeeze = False
    else:
        raise ValueError(f"Expected 3D or 4D compound data, got shape {data.shape}")

    n_frames = int(data.shape[0])
    spatial = data.shape[1:]
    mat = cp.asarray(data, dtype=cp.complex64).reshape(n_frames, -1)
    n_vox = mat.shape[1]

    # Two Gram matrices in one pass: G (raw) drives the projection; Gc
    # (mean-subtracted) drives the adaptive cutoff -- matching the CPU code,
    # where spectral_centroid_cutoff mean-subtracts but the projection does not.
    G = cp.zeros((n_frames, n_frames), dtype=cp.complex64)
    Gc = cp.zeros((n_frames, n_frames), dtype=cp.complex64) if method == "adaptive" else None
    for s0 in range(0, n_vox, voxel_chunk):
        mc = mat[:, s0:s0 + voxel_chunk]
        G += mc @ mc.conj().T
        if Gc is not None:
            xc = mc - mc.mean(axis=0, keepdims=True)
            Gc += xc @ xc.conj().T

    if method == "adaptive":
        if frame_rate_hz is None:
            raise ValueError("method='adaptive' requires frame_rate_hz")
        low = _spectral_centroid_cutoff_gpu(Gc, n_frames, frame_rate_hz, tissue_freq_hz)
        del Gc
    elif method in {"fast", "full", "gpu", "gpu_full", "randomized"}:
        low = int(n_components) if n_components is not None else _component_count(low_cutoff, n_frames)
    elif method == "none":
        return (data[:, 0] if squeeze else data).astype(np.complex64, copy=False)
    else:
        raise ValueError(f"Unsupported SVD method: {method}")

    high = 1.0 if high_cutoff is None else float(high_cutoff)
    high_remove = max(0, min(n_frames, int(round((1.0 - high) * n_frames))))
    if low + high_remove >= n_frames:
        raise ValueError(f"SVD cutoff removes all components: low={low}, high={high_cutoff}")

    evals, u = cp.linalg.eigh(G)
    u = u[:, cp.argsort(evals)[::-1]]
    stop = n_frames - high_remove if high_remove > 0 else n_frames
    uc = u[:, low:stop]            # (F, k)
    uc_h = uc.conj().T             # (k, F)
    # Project in place per chunk: filtered[:,c] = uc @ (uc^H @ mat[:,c]).
    for s0 in range(0, n_vox, voxel_chunk):
        mc = mat[:, s0:s0 + voxel_chunk]
        mat[:, s0:s0 + voxel_chunk] = uc @ (uc_h @ mc)

    out = cp.asnumpy(mat).reshape((n_frames, *spatial))
    del mat, G
    cp.get_default_memory_pool().free_all_blocks()
    if squeeze:
        out = out[:, 0]
    return out.astype(np.complex64, copy=False)


def filtered_magnitude_gpu(
    compound: np.ndarray,
    low_cutoff: float = 0.1,
    high_cutoff: float | None = None,
    method: str = "fast",
    temporal_sigma: float = 0.0,
    n_components: int | None = None,
    frame_rate_hz: float | None = None,
    tissue_freq_hz: float = 100.0,
    voxel_chunk: int = 300_000,
) -> np.ndarray:
    """GPU analogue of ``svd.filtered_magnitude``."""
    filtered = filter_svd_3d_gpu(
        compound, low_cutoff=low_cutoff, high_cutoff=high_cutoff, method=method,
        n_components=n_components, frame_rate_hz=frame_rate_hz,
        tissue_freq_hz=tissue_freq_hz, voxel_chunk=voxel_chunk,
    )
    magnitude = np.abs(filtered).astype(np.float32, copy=False)
    if temporal_sigma > 0:
        from scipy.ndimage import gaussian_filter1d

        magnitude = gaussian_filter1d(magnitude, sigma=temporal_sigma, axis=0)
    return magnitude
