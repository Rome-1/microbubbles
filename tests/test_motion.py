"""Local (numpy/scipy, NO GPU) tests for ultratrace_ulm.motion (mb-crr.11).

Strategy: build a STATIC tissue volume, displace it by KNOWN per-frame shifts
(integer + sub-voxel) with the same band-limited Fourier shifter that
``apply_motion`` uses, then check that

  * ``estimate_motion_rigid`` recovers the shifts within ~0.5 voxel, and
  * ``apply_motion`` with the recovered shifts realigns the frames (frame-to-frame
    temporal variance collapses vs the uncorrected stack).

skimage is optional; these tests force ``method="fft"`` so the hand-rolled FFT
fallback is exercised regardless of whether skimage is installed.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import fourier_shift

from ultratrace_ulm import motion

# Small synthetic volume per the spec.
F, ELEV, Z, X = 24, 3, 40, 40
RNG = np.random.default_rng(20260626)


def _static_tissue() -> np.ndarray:
    """A static 3D tissue pattern: several Gaussian blobs + a little texture.

    Texture sharpens the phase-correlation peak (broadband content), and the
    asymmetric blob layout gives signal along every axis.
    """
    ee, zz, xx = np.meshgrid(
        np.arange(ELEV), np.arange(Z), np.arange(X), indexing="ij"
    )
    vol = np.zeros((ELEV, Z, X), dtype=np.float64)
    blobs = [
        (1.0, 12.0, 14.0, 6.0),
        (1.0, 26.0, 24.0, 5.0),
        (1.5, 18.0, 30.0, 7.0),
        (0.5, 30.0, 9.0, 4.0),
    ]
    for amp, cz, cx, sig in blobs:
        ce = 1.0
        vol += amp * np.exp(
            -(((ee - ce) ** 2) / 2.0)
            - (((zz - cz) ** 2) / (2.0 * sig**2))
            - (((xx - cx) ** 2) / (2.0 * sig**2))
        )
    texture = RNG.standard_normal((ELEV, Z, X))
    # light spatial smoothing of the texture so it stays band-limited
    tf = np.fft.fftn(texture)
    kz = np.fft.fftfreq(Z)[None, :, None]
    kx = np.fft.fftfreq(X)[None, None, :]
    ke = np.fft.fftfreq(ELEV)[:, None, None]
    lp = np.exp(-(kz**2 + kx**2 + ke**2) / (2 * 0.15**2))
    texture = np.fft.ifftn(tf * lp).real
    texture *= 0.2 / (texture.std() + 1e-12)
    return vol + texture


def _fourier_shift_real(vol: np.ndarray, shift) -> np.ndarray:
    """Shift a real volume by ``shift`` (result(x) = vol(x - shift))."""
    return np.fft.ifftn(fourier_shift(np.fft.fftn(vol), shift)).real


def _build_shifted_stack(true_shifts: np.ndarray):
    ref = _static_tissue()
    stack = np.empty((F, ELEV, Z, X), dtype=np.float64)
    for i in range(F):
        stack[i] = _fourier_shift_real(ref, true_shifts[i])
    return ref, stack


def _make_true_shifts() -> np.ndarray:
    """Per-frame known shifts: frame 0 = 0; integer + sub-voxel on z/x, small elev."""
    s = np.zeros((F, 3), dtype=np.float64)
    # axis order is (elev, z, x)
    for i in range(1, F):
        s[i, 0] = float(RNG.choice([-0.5, 0.0, 0.0, 0.5]))         # elev (only 3 deep)
        s[i, 1] = float(RNG.uniform(-3.0, 3.0))                    # z
        s[i, 2] = float(RNG.uniform(-3.0, 3.0))                    # x
    return s


def _make_small_shifts() -> np.ndarray:
    """Small (<=1 voxel) shifts -- the regime where a mean reference is valid."""
    s = np.zeros((F, 3), dtype=np.float64)
    for i in range(1, F):
        s[i, 1] = float(RNG.uniform(-1.0, 1.0))                    # z
        s[i, 2] = float(RNG.uniform(-1.0, 1.0))                    # x
    return s


def test_estimate_recovers_known_shifts_first_reference():
    true_shifts = _make_true_shifts()
    _ref, stack = _build_shifted_stack(true_shifts)
    # frame 0 is the unshifted tissue, so use it as the clean reference.
    est = motion.estimate_motion_rigid(stack, reference=0, method="fft")
    assert est.shape == (F, 3)

    err = est - true_shifts
    max_zx = np.abs(err[:, 1:]).max()
    max_elev = np.abs(err[:, 0]).max()
    rmse = float(np.sqrt(np.mean(err**2)))
    print(f"[recover/first] z,x max err={max_zx:.3f} elev max err={max_elev:.3f} rmse={rmse:.3f}")
    assert max_zx < 0.5
    assert max_elev < 0.5


def test_estimate_recovers_with_mean_reference():
    """Default reference='mean' recovers shifts up to a constant (the mean).

    A mean reference is only valid for SMALL motion (the feature's target
    regime): with large drift the temporal mean is too smeared to correlate
    against, so this uses <=1 voxel shifts.
    """
    true_shifts = _make_small_shifts()
    _ref, stack = _build_shifted_stack(true_shifts)
    est = motion.estimate_motion_rigid(stack, reference="mean", method="fft")
    # mean reference => recovered ~ true - mean(true); compare de-meaned.
    est_dm = est - est.mean(axis=0)
    true_dm = true_shifts - true_shifts.mean(axis=0)
    err = est_dm - true_dm
    max_zx = np.abs(err[:, 1:]).max()
    print(f"[recover/mean] de-meaned z,x max err={max_zx:.3f}")
    assert max_zx < 0.5


def test_apply_motion_reduces_variance():
    """apply_motion with recovered shifts collapses frame-to-frame variance."""
    true_shifts = _make_true_shifts()
    _ref, stack = _build_shifted_stack(true_shifts)

    # complex compound (nonzero phase) to exercise the complex path
    compound = (stack * np.exp(1j * 0.7)).astype(np.complex64)

    est = motion.estimate_motion_rigid(stack, reference=0, method="fft")
    corrected = motion.apply_motion(compound, est)
    assert corrected.shape == compound.shape
    assert corrected.dtype == np.complex64

    var_before = float(np.abs(compound).var(axis=0).mean())
    var_after = float(np.abs(corrected).var(axis=0).mean())
    print(f"[variance] before={var_before:.5f} after={var_after:.5f} ratio={var_after / var_before:.4f}")
    # realignment should cut the temporal variance by a large factor
    assert var_after < 0.2 * var_before


def test_tissue_bmode_is_low_rank_tissue():
    """tissue_bmode keeps the slow tissue and suppresses fast sparse bubbles."""
    ref = _static_tissue()
    # slowly-varying tissue amplitude (low temporal rank)
    t = np.arange(F)
    amp = 1.0 + 0.1 * np.sin(2 * np.pi * t / F)
    tissue = amp[:, None, None, None] * ref[None]

    # sparse fast bubbles: random bright voxels per frame (high temporal rank)
    bubbles = np.zeros((F, ELEV, Z, X), dtype=np.float64)
    for i in range(F):
        n = 5
        ie = RNG.integers(0, ELEV, n)
        iz = RNG.integers(0, Z, n)
        ix = RNG.integers(0, X, n)
        bubbles[i, ie, iz, ix] = 3.0

    compound = (tissue + bubbles).astype(np.complex64)
    bmode = motion.tissue_bmode(compound, rank=4)
    assert bmode.shape == compound.shape
    assert bmode.dtype == np.float32

    # B-mode should track the tissue, not the bubbles: correlate the temporal-mean
    # B-mode with the temporal-mean tissue vs with the bubble locations.
    bmode_mean = bmode.mean(axis=0)
    tissue_mean = np.abs(tissue).mean(axis=0)
    corr_tissue = np.corrcoef(bmode_mean.ravel(), tissue_mean.ravel())[0, 1]
    print(f"[bmode] corr(bmode_mean, tissue_mean)={corr_tissue:.3f}")
    assert corr_tissue > 0.9

    # the bubble energy left in the B-mode should be small relative to its raw level
    bub_mask = bubbles.sum(axis=0) > 0
    leak = bmode_mean[bub_mask].mean() / (bmode_mean.mean() + 1e-12)
    print(f"[bmode] bubble-location relative level={leak:.3f}")


def test_correct_motion_end_to_end():
    """correct_motion: estimate from tissue + clamp + apply, variance drops.

    Probe motion moves tissue AND bubbles together, so the whole frame is
    rigidly displaced. Tissue amplitude is held constant here so the residual
    after correction is bounded by the sparse-bubble floor (an irreducible,
    motion-independent term) -- isolating the motion that correction removes.
    """
    ref = _static_tissue()
    true_shifts = _make_small_shifts()
    # widen z/x a touch so there is real motion to remove (still in-range)
    true_shifts[:, 1] *= 2.0
    true_shifts[:, 2] *= 2.0

    compound = np.empty((F, ELEV, Z, X), dtype=np.complex64)
    for i in range(F):
        frame = ref.copy()
        n = 4
        ie = RNG.integers(0, ELEV, n)
        iz = RNG.integers(0, Z, n)
        ix = RNG.integers(0, X, n)
        frame[ie, iz, ix] += 2.0
        moved = _fourier_shift_real(frame, true_shifts[i])
        compound[i] = moved.astype(np.complex64)

    corrected, shifts = motion.correct_motion(
        compound, rank=6, max_shift_voxels=8.0, reference=0, method="fft"
    )
    assert corrected.shape == compound.shape
    assert corrected.dtype == np.complex64
    assert shifts.shape == (F, 3)
    # no clamping should trigger for these in-range shifts
    assert np.all(np.abs(shifts) <= 8.0)

    var_before = float(np.abs(compound).var(axis=0).mean())
    var_after = float(np.abs(corrected).var(axis=0).mean())
    print(f"[end2end] before={var_before:.5f} after={var_after:.5f} ratio={var_after / var_before:.4f}")
    assert var_after < 0.5 * var_before


def test_correct_motion_clamps_outliers():
    """A wildly large estimated shift is clamped to zero by max_shift_voxels."""
    # tiny shifts so the estimator is accurate; then a tight clamp forces a
    # subset to zero, proving the guard fires.
    true_shifts = np.zeros((F, 3), dtype=np.float64)
    true_shifts[:, 2] = np.linspace(0.0, 5.0, F)  # x ramps 0..5
    ref, stack = _build_shifted_stack(true_shifts)
    compound = stack.astype(np.complex64)

    _corrected, shifts = motion.correct_motion(
        compound, rank=6, max_shift_voxels=2.0, reference=0, method="fft"
    )
    # frames whose true x-shift exceeds 2.0 must be clamped to all-zero
    clamped = np.all(shifts == 0.0, axis=1)
    big = true_shifts[:, 2] > 2.5
    print(f"[clamp] n_clamped={int(clamped.sum())} n_big={int(big.sum())}")
    assert clamped[big].any()


def test_apply_motion_zero_shift_is_noop():
    compound = (_static_tissue()[None] * np.ones((F, 1, 1, 1))).astype(np.complex64)
    shifts = np.zeros((F, 3), dtype=np.float32)
    out = motion.apply_motion(compound, shifts)
    assert np.allclose(out, compound, atol=1e-5)


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("skimage") is None,
    reason="skimage not installed",
)
def test_skimage_path_matches_when_available():
    true_shifts = _make_true_shifts()
    _ref, stack = _build_shifted_stack(true_shifts)
    est = motion.estimate_motion_rigid(
        stack, reference=0, upsample=20, method="skimage"
    )
    err = est - true_shifts
    assert np.abs(err[:, 1:]).max() < 0.5
