"""Determinism properties of the adaptive SVD clutter filter (issue #2)."""

import numpy as np

from ultratrace_ulm.svd import filter_svd_3d, mode_centroids, spectral_centroid_cutoff

FRAME_RATE_HZ = 222.0


def _synthetic(n_frames=128, n_vox=2000, seed=0):
    """Strong low-frequency tissue modes plus weak high-frequency blood modes."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_frames) / FRAME_RATE_HZ
    m = np.zeros((n_frames, n_vox), dtype=np.complex128)
    for freq, amp in ((2.0, 60.0), (5.0, 40.0), (11.0, 25.0)):
        m += amp * np.exp(2j * np.pi * freq * t)[:, None] * rng.standard_normal((1, n_vox))
    for freq, amp in ((105.0, 3.0), (95.0, 2.0)):
        m += amp * np.exp(2j * np.pi * freq * t)[:, None] * rng.standard_normal((1, n_vox))
    m += 0.5 * (rng.standard_normal((n_frames, n_vox))
                + 1j * rng.standard_normal((n_frames, n_vox)))
    return m.astype(np.complex64)


def test_mode_centroids_ignore_eigenvector_phase():
    """A Hermitian eigensolver fixes each eigenvector only up to a unit phase, and
    different LAPACK/cuSOLVER builds pick different ones. The mode score must not
    depend on that choice -- scoring ``|rfft(u.real)|**2`` did."""
    rng = np.random.default_rng(3)
    matrix = _synthetic()
    x = np.asarray(matrix, dtype=np.complex128)
    x = x - x.mean(axis=0, keepdims=True)
    evals, u = np.linalg.eigh(x @ x.conj().T)
    u = u[:, np.argsort(evals)[::-1]]
    rotated = u * np.exp(1j * rng.uniform(0, 2 * np.pi, size=u.shape[1]))[None, :]

    np.testing.assert_allclose(mode_centroids(u, FRAME_RATE_HZ),
                               mode_centroids(rotated, FRAME_RATE_HZ), rtol=1e-6)


def test_cutoff_is_bit_stable_across_repeated_calls():
    matrix = _synthetic()
    cutoffs = {spectral_centroid_cutoff(matrix, FRAME_RATE_HZ) for _ in range(5)}
    assert len(cutoffs) == 1


def test_cutoff_survives_float32_epsilon_perturbation():
    """Cross-run BLAS reduction-order jitter is a float32-epsilon perturbation of the
    input; double precision has to absorb it without moving the cutoff."""
    rng = np.random.default_rng(11)
    matrix = _synthetic()
    base = spectral_centroid_cutoff(matrix, FRAME_RATE_HZ)
    for _ in range(3):
        jitter = (1.0 + 1e-6 * rng.standard_normal(matrix.shape).astype(np.float32))
        assert spectral_centroid_cutoff((matrix * jitter).astype(np.complex64),
                                        FRAME_RATE_HZ) == base


def test_adaptive_filter_output_is_reproducible():
    data = _synthetic(n_frames=64, n_vox=8 * 10 * 12).reshape(64, 8, 10, 12)
    a = filter_svd_3d(data, method="adaptive", frame_rate_hz=FRAME_RATE_HZ)
    b = filter_svd_3d(data, method="adaptive", frame_rate_hz=FRAME_RATE_HZ)
    assert a.dtype == np.complex64
    np.testing.assert_array_equal(a, b)
