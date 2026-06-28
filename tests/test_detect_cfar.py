from __future__ import annotations

import numpy as np

from ultratrace_ulm.detect_cfar import (
    anisotropic_nms_size,
    debias_elevation,
    detect_batch_cfar,
    detect_batch_cfar_confidence,
    split_confidence_batch,
)


def test_anisotropic_nms_size_defaults_wide_elevation():
    assert anisotropic_nms_size(2) == (1, 9, 5, 5)
    assert anisotropic_nms_size(2, elev_radius=3, z_radius=1, x_radius=2) == (1, 7, 3, 5)


def test_debias_elevation_reduces_midplane_bias():
    vol = np.ones((4, 5, 8, 8), dtype=np.float32)
    vol[:, 2] *= 5.0
    corrected, profile = debias_elevation(vol, smooth_sigma=0)
    plane_means = corrected.mean(axis=(0, 2, 3))
    assert profile[2] > profile[0]
    assert plane_means.max() / plane_means.min() < 1.01


def test_detect_batch_cfar_detects_injected_peak():
    rng = np.random.default_rng(1)
    vol = rng.normal(1.0, 0.05, size=(3, 5, 20, 20)).astype(np.float32)
    vol[:, 2] *= 2.0
    vol[1, 4, 14, 14] += 4.0
    out = detect_batch_cfar(
        vol,
        sigma_threshold=4.0,
        min_distance=1,
        smoothing_sigma=0.0,
        local_radius=(1, 4, 4),
        guard_radius=(1, 1, 1),
        global_floor_fraction=0.2,
        debias=True,
    )
    assert any(np.any(np.all(frame[0] == np.array([4, 14, 14]), axis=1)) for frame in out)


def test_confidence_batch_splits_low_and_high():
    vol = np.ones((1, 3, 12, 12), dtype=np.float32)
    vol[0, 1, 4, 4] = 3.0
    vol[0, 1, 9, 9] = 5.0
    batch = detect_batch_cfar_confidence(
        vol,
        sigma_threshold=8.0,
        low_sigma_threshold=4.0,
        min_distance=1,
        smoothing_sigma=0.0,
        local_radius=(1, 3, 3),
        guard_radius=(0, 1, 1),
        global_floor_fraction=2.0,
        debias=False,
    )
    high, low = split_confidence_batch(batch)
    assert len(high[0][0]) == 1
    assert len(low[0][0]) == 1
    assert np.array_equal(high[0][0][0], np.array([1, 9, 9], dtype=np.int32))
