from __future__ import annotations

import numpy as np

from ultratrace_ulm.psf import detect_batch_matched_filter, estimate_empirical_psf


def _gaussian_psf(radius=(2, 2, 2), sigma=(1.2, 0.8, 0.8)):
    grids = np.meshgrid(
        *[np.arange(-r, r + 1, dtype=np.float32) for r in radius],
        indexing="ij",
    )
    psf = np.ones_like(grids[0], dtype=np.float32)
    for grid, sig in zip(grids, sigma):
        psf *= np.exp(-0.5 * (grid / sig) ** 2)
    psf /= psf.sum()
    return psf.astype(np.float32)


def test_estimate_empirical_psf_from_isolated_patches():
    psf = _gaussian_psf()
    vol = np.zeros((2, 7, 16, 16), dtype=np.float32)
    centers = [(0, 3, 7, 7), (1, 3, 10, 10)]
    for f, e, z, x in centers:
        vol[f, e - 2 : e + 3, z - 2 : z + 3, x - 2 : x + 3] += 10.0 * psf
    detections = [
        (np.array([[3, 7, 7]], dtype=np.int32), np.array([10.0], dtype=np.float32), np.array([9.0], dtype=np.float32)),
        (np.array([[3, 10, 10]], dtype=np.int32), np.array([10.0], dtype=np.float32), np.array([8.0], dtype=np.float32)),
    ]
    est = estimate_empirical_psf(vol, detections, patch_radius=(2, 2, 2), min_zscore=5.0)
    assert est.shape == psf.shape
    assert np.isclose(est.sum(), 1.0)
    assert np.unravel_index(np.argmax(est), est.shape) == (2, 2, 2)


def test_matched_filter_detects_inserted_psf_peak():
    psf = _gaussian_psf()
    vol = np.zeros((1, 7, 18, 18), dtype=np.float32)
    vol += 0.05
    vol[0, 3 - 2 : 3 + 3, 9 - 2 : 9 + 3, 9 - 2 : 9 + 3] += 6.0 * psf
    out = detect_batch_matched_filter(
        vol,
        sigma_threshold=3.5,
        min_distance=1,
        smoothing_sigma=0.0,
        psf=psf,
        debias=False,
    )
    assert len(out[0][0]) >= 1
    assert np.any(np.all(out[0][0] == np.array([3, 9, 9]), axis=1))
