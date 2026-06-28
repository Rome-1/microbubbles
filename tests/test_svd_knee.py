"""Tests for the data-driven SVD cutoff selection (mb-3k4).

These are pure-numpy and GPU-free: they validate the knee/MP selection logic and
the CPU ``filter_svd_3d(method='knee')`` path, which is the authoritative
reference the GPU port mirrors.
"""

from __future__ import annotations

import numpy as np

from ultratrace_ulm.svd import filter_svd_3d, spectral_centroid_cutoff
from ultratrace_ulm.svd_knee import (
    combined_low_cutoff,
    mp_noise_cutoff,
    select_svd_cutoffs,
    singular_value_knee,
    spatial_correlation_cutoff,
)


def _l_shaped_spectrum(n_tissue=5, n=200, tail=0.5):
    """Descending singular values: a steep tissue head then a flat noise tail."""
    head = np.geomspace(500.0, 20.0, n_tissue)
    tail_vals = np.full(n - n_tissue, tail) + np.linspace(0.2, 0.0, n - n_tissue)
    return np.concatenate([head, tail_vals])


def test_knee_finds_the_corner_of_an_L():
    s = _l_shaped_spectrum(n_tissue=5, n=200)
    k = singular_value_knee(s, min_rank=1, max_rank=70)
    # The tissue head is ~5 modes; the knee should land in its neighbourhood,
    # nowhere near the old fixed 70-mode floor.
    assert 2 <= k <= 12


def test_knee_respects_min_and_max_guards():
    s = _l_shaped_spectrum(n_tissue=5, n=200)
    assert singular_value_knee(s, min_rank=8, max_rank=70) >= 8
    assert singular_value_knee(s, min_rank=1, max_rank=3) <= 3


def test_knee_is_count_independent():
    """Same physical spectrum at 200 vs 700 frames -> similar knee (the bug the
    fixed-fraction floor had: 10 % scaled the rank with frame count for no
    physical reason)."""
    k_short = singular_value_knee(_l_shaped_spectrum(5, 200), min_rank=1, max_rank=20)
    k_long = singular_value_knee(_l_shaped_spectrum(5, 700), min_rank=1, max_rank=70)
    assert abs(k_short - k_long) <= 3


def test_knee_handles_degenerate_input():
    assert singular_value_knee(np.array([]), min_rank=1, max_rank=10) >= 0
    assert singular_value_knee(np.array([5.0]), min_rank=1, max_rank=10) >= 0
    flat = np.ones(50)
    # A flat curve has no real knee; the guard must keep it sane (<= max_rank).
    assert 0 <= singular_value_knee(flat, min_rank=1, max_rank=10) <= 10


def test_mp_noise_cutoff_separates_signal_from_noise():
    rng = np.random.default_rng(1)
    F, N = 120, 40_000
    # A genuine low-rank-plus-noise matrix so the Marchenko-Pastur law actually
    # governs the noise eigenvalues (N >> F -> a tight noise bulk).
    rank = 6
    U = rng.standard_normal((F, rank)) + 1j * rng.standard_normal((F, rank))
    V = rng.standard_normal((N, rank)) + 1j * rng.standard_normal((N, rank))
    amps = np.array([40.0, 28.0, 18.0, 12.0, 8.0, 6.0])
    X = (U * amps) @ V.conj().T
    X += rng.standard_normal((F, N)) + 1j * rng.standard_normal((F, N))  # sigma=1 noise
    evals = np.linalg.eigvalsh(X @ X.conj().T).real
    high_remove = mp_noise_cutoff(evals, F, N)
    signal_kept = F - high_remove
    assert 4 <= signal_kept <= 9  # ~6 signal modes retained, the noise bulk cut


def test_select_cutoffs_keeps_a_valid_window():
    rng = np.random.default_rng(2)
    F, N = 100, 20_000
    evals = np.sort(rng.random(F) + 0.01)[::-1] ** 2
    low, high_remove = select_svd_cutoffs(evals, F, N, low_min=1, low_max=10, high=True)
    assert 1 <= low <= 10
    assert low + high_remove < F  # at least one mode survives


def _synthetic_volume(n_tissue_modes=4, F=120, seed=0):
    """(F, elev, z, x) volume: strong slow tissue + weak faster blood + noise."""
    rng = np.random.default_rng(seed)
    elev, z, x = 2, 16, 16
    t = np.arange(F)
    vol = 0.05 * (rng.standard_normal((F, elev, z, x)) + 1j * rng.standard_normal((F, elev, z, x)))
    for m, f in enumerate(np.linspace(0.5, 3.0, n_tissue_modes)):  # slow, strong
        sp = rng.standard_normal((elev, z, x)) + 1j * rng.standard_normal((elev, z, x))
        vol += (40.0 / (m + 1)) * np.exp(2j * np.pi * f * t / F)[:, None, None, None] * sp[None]
    for f in (15.0, 22.0):  # faster, weak "blood"
        sp = rng.standard_normal((elev, z, x)) + 1j * rng.standard_normal((elev, z, x))
        vol += 2.0 * np.exp(2j * np.pi * f * t / F)[:, None, None, None] * sp[None]
    return vol.astype(np.complex64)


def test_filter_knee_removes_far_fewer_modes_than_fixed_floor():
    """End-to-end: on a volume with ~4 tissue modes, the knee cut should keep
    much more energy than the fixed 10 %-floor path that removes 12 modes here."""
    vol = _synthetic_volume(n_tissue_modes=4, F=120)
    knee = filter_svd_3d(vol, method="knee", low_cutoff=0.1)          # ceiling 12
    fixed = filter_svd_3d(vol, method="fast", low_cutoff=0.1)          # removes 12
    # Removing fewer (tissue-only) modes leaves strictly more signal energy.
    assert np.linalg.norm(knee) > np.linalg.norm(fixed)
    assert np.all(np.isfinite(np.abs(knee)))
    assert knee.shape == vol.shape


def test_knee_beats_broken_adaptive_floor_on_our_framerate():
    """Regression guard for the actual bug: at 222 Hz the 'adaptive' centroid cut
    falls back to the 10 % floor, while the knee selects a small tissue rank."""
    vol = _synthetic_volume(n_tissue_modes=4, F=200)
    mat = vol.reshape(200, -1)
    adaptive_low = spectral_centroid_cutoff(mat, frame_rate_hz=222.0, tissue_freq_hz=100.0)
    evals = np.linalg.eigvalsh(mat @ mat.conj().T).real
    knee_low, _ = select_svd_cutoffs(evals, 200, mat.shape[1], low_min=1, low_max=20)
    assert adaptive_low == round(200 * 0.1)  # == 20, the silent floor
    assert knee_low < adaptive_low           # knee is data-driven and smaller


# ---------------------------------------------------------------------------
# B18 spatial-singular-vector cutoff (spatial_correlation_cutoff / combined)
# ---------------------------------------------------------------------------


def _smooth_tissue_volume(n_tissue=3, F=100, seed=0):
    """(F, elev, z, x) volume whose TISSUE modes are spatially SMOOTH (low spatial
    frequency, separable cosines, constant across elevation) and strong, with
    weaker spatially-INCOHERENT 'blood' modes and noise on top -- the regime the
    B18 spatial-coherence estimator is built for.
    """
    rng = np.random.default_rng(seed)
    elev, z, x = 2, 24, 24
    t = np.arange(F)
    zc = np.linspace(0.0, np.pi, z)
    xc = np.linspace(0.0, np.pi, x)
    vol = 0.02 * (
        rng.standard_normal((F, elev, z, x)) + 1j * rng.standard_normal((F, elev, z, x))
    )
    for m, f in enumerate(np.linspace(0.5, 2.5, n_tissue)):  # slow, strong, SMOOTH
        smooth = np.cos((m + 1) * zc)[:, None] * np.cos((m + 1) * xc)[None, :]  # (z, x)
        sp = np.broadcast_to(smooth, (elev, z, x)).astype(np.complex64)  # flat in elev
        vol += (50.0 / (m + 1)) * np.exp(2j * np.pi * f * t / F)[:, None, None, None] * sp[None]
    for f in (15.0, 22.0, 30.0):  # faster, weak, spatially INCOHERENT "blood"
        sp = rng.standard_normal((elev, z, x)) + 1j * rng.standard_normal((elev, z, x))
        vol += 2.0 * np.exp(2j * np.pi * f * t / F)[:, None, None, None] * sp[None]
    return vol.astype(np.complex64)


def test_spatial_correlation_cutoff_finds_tissue_count():
    """B18: the spatial-coherence collapse lands at the true smooth-tissue count."""
    vol = _smooth_tissue_volume(n_tissue=3, F=100, seed=0)
    mat = vol.reshape(100, -1)
    cut = spatial_correlation_cutoff(mat, (2, 24, 24), min_rank=1, max_rank=20)
    assert 2 <= cut <= 4  # ~3 smooth tissue modes, blood/noise collapse just after


def test_spatial_correlation_cutoff_accepts_volume_and_precomputed_basis():
    """Two input forms agree: a raw (F,elev,z,x) volume (spatial_shape inferred)
    vs a (frames,voxels) matrix with a precomputed (u, s) temporal basis."""
    vol = _smooth_tissue_volume(n_tissue=3, F=100, seed=2)
    from_volume = spatial_correlation_cutoff(vol, min_rank=1, max_rank=20)

    mat = vol.reshape(100, -1)
    evals, u = np.linalg.eigh(mat @ mat.conj().T)
    order = np.argsort(evals)[::-1]
    u = u[:, order]
    s = np.sqrt(np.maximum(evals[order].real, 0.0))
    from_matrix = spatial_correlation_cutoff(
        mat, (2, 24, 24), u=u, s=s, min_rank=1, max_rank=20
    )
    assert from_volume == from_matrix
    assert 2 <= from_volume <= 4


def test_combined_low_cutoff_is_min_of_knee_and_spatial():
    """L20: combined tissue rank == min(singular-value knee, spatial correlation)."""
    vol = _smooth_tissue_volume(n_tissue=3, F=100, seed=1)
    mat = vol.reshape(100, -1)
    sp_shape = (2, 24, 24)

    evals, u = np.linalg.eigh(mat @ mat.conj().T)
    order = np.argsort(evals)[::-1]
    u = u[:, order]
    s = np.sqrt(np.maximum(evals[order].real, 0.0))

    knee = singular_value_knee(s, min_rank=1, max_rank=20)
    spatial = spatial_correlation_cutoff(mat, sp_shape, u=u, s=s, min_rank=1, max_rank=20)
    combined = combined_low_cutoff(mat, sp_shape, min_rank=1, max_rank=20)

    assert combined == min(knee, spatial)
    assert 1 <= combined <= 20


def test_combined_low_cutoff_respects_guards():
    vol = _smooth_tissue_volume(n_tissue=3, F=100, seed=3)
    mat = vol.reshape(100, -1)
    sp_shape = (2, 24, 24)
    assert combined_low_cutoff(mat, sp_shape, min_rank=6, max_rank=20) >= 6
    assert combined_low_cutoff(mat, sp_shape, min_rank=1, max_rank=2) <= 2


def test_spatial_correlation_cutoff_handles_degenerate_input():
    # Single frame: nothing to remove (max_allowed collapses to 0).
    assert spatial_correlation_cutoff(np.ones((1, 9), dtype=complex), (1, 3, 3),
                                      min_rank=1, max_rank=5) == 0
    # No spatial adjacency on any axis -> coherence undefined -> graceful clamp.
    col = (np.arange(6)[:, None] * np.ones((6, 1))).astype(complex)
    out = spatial_correlation_cutoff(col, (1, 1, 1), min_rank=1, max_rank=4)
    assert 0 <= out <= 4
    # Pure spatially-incoherent noise: no tissue/blood collapse -> stays in-range.
    rng = np.random.default_rng(7)
    noise = (rng.standard_normal((40, 2, 8, 8)) + 1j * rng.standard_normal((40, 2, 8, 8)))
    nm = spatial_correlation_cutoff(noise.reshape(40, -1), (2, 8, 8), min_rank=1, max_rank=10)
    assert 0 <= nm <= 10
    # Mismatched spatial_shape is rejected.
    import pytest

    with pytest.raises(ValueError):
        spatial_correlation_cutoff(np.ones((5, 9), dtype=complex), (2, 2, 2))
