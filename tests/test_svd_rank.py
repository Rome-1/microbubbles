from __future__ import annotations

import numpy as np

from ultratrace_ulm.svd_rank import (
    block_rank_map,
    filter_svd_3d_region_ranked,
    rank_from_singular_values,
    regularize_rank_map,
)


def _synthetic_rank_volume(seed=0):
    rng = np.random.default_rng(seed)
    f, elev, z, x = 24, 2, 18, 18
    t = np.arange(f)
    low = np.exp(2j * np.pi * 2 * t / f)
    mid = np.exp(2j * np.pi * 5 * t / f)
    high = np.exp(2j * np.pi * 9 * t / f)
    vol = 0.05 * (rng.standard_normal((f, elev, z, x)) + 1j * rng.standard_normal((f, elev, z, x)))
    vol[:, :, :9, :9] += 12.0 * low[:, None, None, None]
    vol[:, :, 9:, 9:] += (
        9.0 * low[:, None, None, None]
        + 6.0 * mid[:, None, None, None]
        + 3.0 * high[:, None, None, None]
    )
    return vol.astype(np.complex64)


def test_rank_from_singular_values_elbow_and_energy():
    s = np.array([20.0, 10.0, 2.0, 1.0, 0.5])
    assert rank_from_singular_values(s, method="elbow", min_rank=1, max_rank=4) >= 1
    assert rank_from_singular_values(s, method="energy", energy_fraction=0.9, max_rank=4) == 2


def test_regularize_rank_map_delta_zero_and_infinite():
    raw = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]])
    assert np.all(regularize_rank_map(raw, delta=0, smooth_size=1) == 5)
    assert np.array_equal(regularize_rank_map(raw, delta=np.inf, smooth_size=1), raw)
    clamped = regularize_rank_map(raw, delta=1, smooth_size=1)
    assert clamped.min() >= 4
    assert clamped.max() <= 6


def test_block_rank_map_delta_controls_variation():
    vol = _synthetic_rank_volume()
    k0, raw = block_rank_map(
        vol,
        n_z_blocks=2,
        n_x_blocks=2,
        overlap=0.0,
        rank_method="energy",
        energy_fraction=0.82,
        delta=0,
        smooth_size=1,
    )
    kinf, _ = block_rank_map(
        vol,
        n_z_blocks=2,
        n_x_blocks=2,
        overlap=0.0,
        rank_method="energy",
        energy_fraction=0.82,
        delta=np.inf,
        smooth_size=1,
    )
    assert k0.min() == k0.max()
    assert np.array_equal(kinf, raw)
    assert raw.max() >= raw.min()


def test_filter_svd_3d_region_ranked_shape_and_maps():
    vol = _synthetic_rank_volume()
    out, kmap, raw = filter_svd_3d_region_ranked(
        vol,
        n_z_blocks=2,
        n_x_blocks=2,
        overlap=0.0,
        rank_method="energy",
        energy_fraction=0.82,
        delta=1,
        smooth_size=1,
    )
    assert out.shape == vol.shape
    assert out.dtype == np.complex64
    assert kmap.shape == (2, 2)
    assert raw.shape == (2, 2)
