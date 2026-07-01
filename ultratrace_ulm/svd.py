from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d


def _component_count(cutoff: float | int | None, n_frames: int) -> int:
    if cutoff is None:
        return 0
    value = float(cutoff)
    if 0.0 <= value <= 1.0:
        return int(round(value * n_frames))
    return int(round(value))


def _gram_c128(mat: np.ndarray, chunk: int = 300_000) -> np.ndarray:
    """Temporal Gram ``mat @ mat.conj().T`` accumulated in complex128 (chunked).

    Forming ``M Mᴴ`` squares the condition number; with the tissue/blood dynamic
    range (~1e3–1e4) a complex64 Gram loses the RETAINED (blood/bubble) subspace
    and over-suppresses it -> compressed, low-contrast magnitude -> the 2σ
    detector fires on noise and the localizations don't link (mb-a0a: ~16x fewer
    tracks than the GPU path). Accumulating the small (F,F) Gram in complex128
    from complex64 chunks -- exactly as ``gpu_svd.filter_svd_3d_gpu`` does -- keeps
    the cross-chunk sum stable without a full complex128 copy of the (F, n_vox)
    matrix. This makes the CPU path numerically match the (correct) GPU path.
    """
    n = int(mat.shape[0])
    g = np.zeros((n, n), dtype=np.complex128)
    for s0 in range(0, mat.shape[1], chunk):
        mc = mat[:, s0:s0 + chunk]
        g += (mc @ mc.conj().T).astype(np.complex128)
    return g


def spectral_centroid_cutoff(
    matrix: np.ndarray,
    frame_rate_hz: float,
    tissue_freq_hz: float = 100.0,
) -> int:
    """Data-driven tissue/blood SVD boundary via temporal spectral centroid.

    For each temporal singular vector, compute the power-weighted mean
    frequency; the cutoff is the first vector whose centroid exceeds
    ``tissue_freq_hz`` (Ghosh et al., PNAS 2025). Falls back to 10% of frames.
    ``matrix`` is the (frames, voxels) temporal matrix.

    DETERMINISM FIX (mb-crr.2): the singular vectors are complex and a Hermitian
    eigensolver returns each only up to an arbitrary unit-phase. The shipped code
    measured ``|rfft(u.real)|^2``, which is phase-DEPENDENT -- so the cutoff (and
    thus the whole clutter filter) silently changed with the LAPACK/cuSOLVER
    phase convention (CPU and GPU disagreed by ~15x in track count). We instead
    use the phase-INVARIANT full complex spectrum ``|fft(u)|^2`` with the centroid
    taken over ``|freq|``. This is deterministic, GPU/CPU-consistent, and the
    physically correct frequency content of a complex (signed-Doppler) mode.
    """
    n_frames = int(matrix.shape[0])
    x = matrix - matrix.mean(axis=0, keepdims=True)
    cov = _gram_c128(x)  # complex128 Gram (mb-a0a): match the GPU cutoff exactly
    evals, u = np.linalg.eigh(cov)
    u = u[:, np.argsort(evals)[::-1]]
    freqs = np.fft.fftfreq(n_frames, d=1.0 / frame_rate_hz)
    spec = np.abs(np.fft.fft(u, axis=0)) ** 2  # (F, F) phase-invariant
    spec[0, :] = 0.0  # exclude DC
    total = spec.sum(axis=0)
    centroid = (np.abs(freqs)[:, None] * spec).sum(axis=0) / np.where(total > 0, total, 1.0)
    above = np.where(centroid > tissue_freq_hz)[0]
    return int(above[0]) if len(above) else max(1, round(n_frames * 0.1))


def filter_svd_3d(
    data: np.ndarray,
    low_cutoff: float = 0.1,
    high_cutoff: float | None = None,
    method: str = "fast",
    n_components: int | None = None,
    frame_rate_hz: float | None = None,
    tissue_freq_hz: float = 100.0,
    knee_min: int = 1,
    knee_high: bool = False,
) -> np.ndarray:
    """Apply temporal SVD clutter filtering to (frames,elev,z,x) data.

    method="adaptive" picks the low cutoff per-acquisition from the temporal
    spectral centroid (requires frame_rate_hz); method="knee" picks it from the
    data-driven singular-value turning point (``low_cutoff`` becomes the ceiling);
    "fast"/"full" use a fixed low_cutoff (or n_components). "fast" is the
    covariance projection; "full" is the numerically stable SVD. This is the
    authoritative reference for the GPU port in ``gpu_svd`` (mb-3k4).
    """
    if data.ndim == 3:
        data = data[:, None, :, :]
        squeeze = True
    elif data.ndim == 4:
        squeeze = False
    else:
        raise ValueError(f"Expected 3D or 4D compound data, got shape {data.shape}")

    n_frames = int(data.shape[0])
    spatial_shape = data.shape[1:]
    n_vox = int(np.prod(spatial_shape))
    matrix = np.asarray(data, dtype=np.complex64).reshape(n_frames, -1)

    high = 1.0 if high_cutoff is None else float(high_cutoff)
    high_remove = max(0, min(n_frames, int(round((1.0 - high) * n_frames))))

    if method == "adaptive":
        if frame_rate_hz is None:
            raise ValueError("method='adaptive' requires frame_rate_hz")
        low = spectral_centroid_cutoff(matrix, frame_rate_hz, tissue_freq_hz)
        normalized_method = "fast"
    elif method == "knee":
        from .svd_knee import select_svd_cutoffs

        # Raw Gram (not mean-subtracted): matches the basis the projection uses
        # below and the GPU port, so the knee indexes the modes actually removed.
        evals = np.linalg.eigvalsh(matrix @ matrix.conj().T).real
        ceiling = int(n_components) if n_components is not None else _component_count(low_cutoff, n_frames)
        low, knee_high_remove = select_svd_cutoffs(
            evals, n_frames, n_vox, low_min=int(knee_min), low_max=ceiling, high=bool(knee_high),
        )
        high_remove = max(high_remove, knee_high_remove)
        normalized_method = "fast"
    else:
        low = int(n_components) if n_components is not None else _component_count(low_cutoff, n_frames)
        normalized_method = "fast" if method in {"gpu", "gpu_full", "randomized"} else method

    if low + high_remove >= n_frames:
        raise ValueError(
            f"SVD cutoff removes all components: low={low}, high={high_cutoff}"
        )

    if normalized_method == "none":
        filtered = matrix
    elif normalized_method == "fast":
        cov = _gram_c128(matrix)  # complex128 Gram (mb-a0a): preserve the retained blood subspace
        evals, u = np.linalg.eigh(cov)
        u = u[:, np.argsort(evals)[::-1]]
        stop = n_frames - high_remove if high_remove > 0 else n_frames
        uc = u[:, low:stop].astype(np.complex64)  # project in complex64 (memory-safe, matches GPU)
        filtered = uc @ (uc.conj().T @ matrix)
    elif normalized_method == "full":
        u, s, vh = np.linalg.svd(matrix, full_matrices=False)
        s[:low] = 0
        if high_remove > 0:
            s[-high_remove:] = 0
        filtered = (u * s[None, :]) @ vh
    else:
        raise ValueError(f"Unsupported SVD method: {method}")

    out = filtered.reshape((n_frames, *spatial_shape))
    if squeeze:
        out = out[:, 0]
    return out.astype(np.complex64, copy=False)


def filtered_magnitude(
    compound: np.ndarray,
    low_cutoff: float = 0.1,
    high_cutoff: float | None = None,
    method: str = "fast",
    temporal_sigma: float = 0.0,
    n_components: int | None = None,
    frame_rate_hz: float | None = None,
    tissue_freq_hz: float = 100.0,
    knee_min: int = 1,
    knee_high: bool = False,
) -> np.ndarray:
    filtered = filter_svd_3d(
        compound,
        low_cutoff=low_cutoff,
        high_cutoff=high_cutoff,
        method=method,
        n_components=n_components,
        frame_rate_hz=frame_rate_hz,
        tissue_freq_hz=tissue_freq_hz,
        knee_min=knee_min,
        knee_high=knee_high,
    )
    magnitude = np.abs(filtered).astype(np.float32, copy=False)
    if temporal_sigma > 0:
        magnitude = gaussian_filter1d(magnitude, sigma=temporal_sigma, axis=0)
    return magnitude
