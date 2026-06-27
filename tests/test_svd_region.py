"""Local (numpy/scipy, NO GPU) tests for the region-adaptive SVD clutter filter.

Covers: (a) 1x1 blocks == global filter_svd_3d; (b) shape preservation;
(c) seam-free blend (partition-of-unity weights sum to ~1, no NaNs);
(d) per-block tissue suppression with localized low-freq clutter.
"""

from __future__ import annotations

import numpy as np
import pytest

from ultratrace_ulm import svd
from ultratrace_ulm.svd_region import (
    _axis_window,
    _block_bounds,
    filter_svd_3d_region,
    filtered_magnitude_region,
)

FS = 500.0  # frame rate (Hz)
TISSUE_FREQ = 100.0


def _synthetic_volume(F=48, elev=3, z=32, x=32, seed=0):
    """Random complex (F, elev, z, x) volume with mild temporal structure."""
    rng = np.random.default_rng(seed)
    base = (rng.standard_normal((F, elev, z, x)) + 1j * rng.standard_normal((F, elev, z, x)))
    # add a low-freq tissue-like and high-freq blood-like component everywhere
    t = np.arange(F)
    tissue = np.exp(2j * np.pi * 8.0 * t / FS)[:, None, None, None]
    blood = np.exp(2j * np.pi * 180.0 * t / FS)[:, None, None, None]
    vol = base + 10.0 * tissue + 2.0 * blood
    return vol.astype(np.complex64)


# ---------------------------------------------------------------------------
# (a) 1x1 blocks reduce EXACTLY to the global filter
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["adaptive", "fast"])
def test_one_block_equals_global(method):
    data = _synthetic_volume()
    region = filter_svd_3d_region(
        data, n_z_blocks=1, n_x_blocks=1, method=method, frame_rate_hz=FS,
        tissue_freq_hz=TISSUE_FREQ,
    )
    glob = svd.filter_svd_3d(
        data, method=method, frame_rate_hz=FS, tissue_freq_hz=TISSUE_FREQ,
    )
    assert region.shape == glob.shape
    # Single unit-weight block + lossless f32<->f64 round trip => bit-exact.
    assert np.array_equal(region, glob)


# ---------------------------------------------------------------------------
# (b) shape preserved for 4D and 3D input
# ---------------------------------------------------------------------------
def test_shape_preserved_4d():
    data = _synthetic_volume()
    out = filter_svd_3d_region(
        data, n_z_blocks=3, n_x_blocks=3, frame_rate_hz=FS, tissue_freq_hz=TISSUE_FREQ,
    )
    assert out.shape == data.shape
    assert out.dtype == np.complex64


def test_shape_preserved_3d():
    data = _synthetic_volume(elev=1)[:, 0]  # (F, z, x)
    out = filter_svd_3d_region(
        data, n_z_blocks=3, n_x_blocks=2, frame_rate_hz=FS, tissue_freq_hz=TISSUE_FREQ,
    )
    assert out.shape == data.shape


# ---------------------------------------------------------------------------
# (c) seam-free blend: weights are a partition of unity, output is finite
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n,overlap", [(3, 0.25), (2, 0.5), (4, 0.1), (1, 0.25), (5, 0.3)])
def test_axis_window_partition_of_unity(n, overlap):
    length = 32
    bounds = _block_bounds(length, n, overlap)
    # full coverage, no gaps, in-bounds
    assert bounds[0][0] == 0 and bounds[-1][1] == length
    for (a, b), (na, nb) in zip(bounds, bounds[1:]):
        assert na <= b  # overlap or touch -> no gap
        assert a < b  # non-empty
    windows = _axis_window(bounds, length)
    total = np.sum(windows, axis=0)
    assert np.all(total > 0)  # no zero-weight index -> no division by zero
    assert np.allclose(total, 1.0, atol=1e-9)  # partition of unity


def test_blend_no_nan_and_unit_weight_2d():
    data = _synthetic_volume()
    n_z, n_x = data.shape[2], data.shape[3]
    z_bounds = _block_bounds(n_z, 3, 0.25)
    x_bounds = _block_bounds(n_x, 3, 0.25)
    z_windows = _axis_window(z_bounds, n_z)
    x_windows = _axis_window(x_bounds, n_x)
    wsum = np.zeros((n_z, n_x))
    for iz, (za, zb) in enumerate(z_bounds):
        for ix, (xa, xb) in enumerate(x_bounds):
            wsum[za:zb, xa:xb] += np.outer(z_windows[iz][za:zb], x_windows[ix][xa:xb])
    assert np.all(wsum > 0)
    assert np.allclose(wsum, 1.0, atol=1e-9)  # 2D separable partition of unity

    out = filter_svd_3d_region(
        data, n_z_blocks=3, n_x_blocks=3, overlap=0.25, frame_rate_hz=FS,
        tissue_freq_hz=TISSUE_FREQ,
    )
    assert np.all(np.isfinite(out.view(np.float32)))


# ---------------------------------------------------------------------------
# (d) localized low-freq "tissue" is suppressed in its block; "blood" kept
# ---------------------------------------------------------------------------
def test_localized_tissue_suppressed():
    F, elev, z, x = 48, 3, 32, 32
    rng = np.random.default_rng(7)
    t = np.arange(F)
    tissue_t = np.exp(2j * np.pi * 8.0 * t / FS)  # low freq (<100 Hz) -> "tissue"
    blood_t = np.exp(2j * np.pi * 180.0 * t / FS)  # high freq (>100 Hz) -> "blood"

    # tissue confined to a corner block; blood everywhere; small noise.
    spatial_tissue = np.zeros((elev, z, x), dtype=np.complex64)
    spatial_tissue[:, :12, :12] = 1.0
    spatial_blood = np.ones((elev, z, x), dtype=np.complex64)
    noise = 0.1 * (rng.standard_normal((F, elev, z, x)) + 1j * rng.standard_normal((F, elev, z, x)))

    vol = (
        50.0 * tissue_t[:, None, None, None] * spatial_tissue[None]
        + 3.0 * blood_t[:, None, None, None] * spatial_blood[None]
        + noise
    ).astype(np.complex64)

    tissue_mask = np.zeros((z, x), dtype=bool)
    tissue_mask[:12, :12] = True
    blood_mask = np.zeros((z, x), dtype=bool)
    blood_mask[20:, 20:] = True  # region with no tissue

    out = filter_svd_3d_region(
        vol, n_z_blocks=3, n_x_blocks=3, overlap=0.25, method="adaptive",
        frame_rate_hz=FS, tissue_freq_hz=TISSUE_FREQ,
    )

    in_mag = np.abs(vol)
    out_mag = np.abs(out)
    tissue_in = in_mag[:, :, tissue_mask].mean()
    tissue_out = out_mag[:, :, tissue_mask].mean()
    blood_in = in_mag[:, :, blood_mask].mean()
    blood_out = out_mag[:, :, blood_mask].mean()

    # tissue strongly suppressed in its block (>80%)...
    assert tissue_out < 0.2 * tissue_in
    # ...while blood is largely retained where there is no tissue.
    assert blood_out > 0.5 * blood_in


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
