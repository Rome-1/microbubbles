"""Correctness tests for the J3 injection + null instrument (bead mb-fmj).

The instrument is the measurement device the rest of the program leans on, so
these check the load-bearing physics claims (the PSF is *measured*, the injected
Doppler rate is right, the compounding null lands where the algebra says) rather
than just exercising the code paths.
"""

from __future__ import annotations

import numpy as np
import pytest

from ultratrace_ulm.inject import (
    ComplexPSF,
    Geometry,
    compounding_gain,
    estimate_complex_psf,
    inject_bubbles,
    make_bubble_plan,
    null_block_shuffle,
    null_phase_surrogate,
    null_quiet_crop,
    reference_noise_sigma,
)
from ultratrace_ulm.opsweep import (
    find_peaks,
    match_peaks_to_truth,
    threshold_for_count,
    zscore_field,
)

GEOM = Geometry()


def _make_psf(radius=(2, 5, 5), sigma=(1.0, 1.6, 1.4), k=2.5) -> ComplexPSF:
    """A separable anisotropic Gaussian envelope with a known axial carrier."""
    re, rz, rx = radius
    ge = np.exp(-0.5 * (np.arange(-re, re + 1) / sigma[0]) ** 2)
    gz = np.exp(-0.5 * (np.arange(-rz, rz + 1) / sigma[1]) ** 2)
    gx = np.exp(-0.5 * (np.arange(-rx, rx + 1) / sigma[2]) ** 2)
    env = (ge[:, None, None] * gz[None, :, None] * gx[None, None, :]).astype(np.complex64)
    return ComplexPSF(env=env, k_axial=float(k), radius=(re, rz, rx), n_patches=0)


# --------------------------------------------------------------------------- #
# The PSF is measured, not assumed
# --------------------------------------------------------------------------- #
def test_complex_psf_is_recovered_from_planted_bubbles():
    """estimate_complex_psf recovers a planted anisotropic PSF and its carrier."""
    rng = np.random.default_rng(0)
    truth = _make_psf(k=2.5)
    shape = (6, 11, 60, 60)
    vol = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex64) * 0.01
    dets = []
    kern = truth.sample((0.0, 0.0, 0.0))
    re, rz, rx = truth.radius
    for f in range(shape[0]):
        coords = []
        for i in range(4):
            e = 5
            z = 10 + 12 * i
            x = 10 + 12 * ((i + f) % 4)
            phase = np.exp(1j * rng.uniform(0, 2 * np.pi))  # arbitrary per-bubble phase
            vol[f, e - re:e + re + 1, z - rz:z + rz + 1, x - rx:x + rx + 1] += (kern * phase)
            coords.append((e, z, x))
        arr = np.asarray(coords, dtype=np.int32)
        dets.append((arr, np.ones(len(arr), np.float32), np.full(len(arr), 20.0, np.float32)))

    est = estimate_complex_psf(vol, dets, patch_radius=truth.radius, isolation_radius=(3, 8, 8))
    assert est.n_patches >= 8
    assert est.k_axial == pytest.approx(2.5, abs=0.05)
    a = est.kernel().ravel()
    b = kern.ravel()
    coh = abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))
    assert coh > 0.99, f"recovered PSF coherence {coh:.4f}"


def test_psf_sample_shifts_sub_voxel():
    """A half-voxel offset moves the magnitude centroid by half a voxel."""
    psf = _make_psf()
    rz = psf.radius[1]
    axis = np.arange(-rz, rz + 1, dtype=float)

    def centroid(delta):
        m = np.abs(psf.sample(delta))
        prof = m.sum(axis=(0, 2))
        return float((prof * axis).sum() / prof.sum())

    assert centroid((0.0, 0.0, 0.0)) == pytest.approx(0.0, abs=1e-6)
    assert centroid((0.0, 0.5, 0.0)) == pytest.approx(0.5, abs=0.05)
    assert centroid((0.0, -0.25, 0.0)) == pytest.approx(-0.25, abs=0.05)


# --------------------------------------------------------------------------- #
# The physics the SVD actually sees
# --------------------------------------------------------------------------- #
def test_injected_axial_motion_produces_the_right_doppler():
    """At a fixed voxel the injected phase rotates at k_axial * v_z, i.e. Doppler.

    This is the property that makes injection meaningful: a slow bubble lands
    near DC and is correctly eaten by the clutter filter, a fast one does not.
    If this were wrong the whole instrument would be measuring the wrong thing.
    """
    psf = _make_psf(k=2.5)
    geom = Geometry()
    v_z = 5.0  # mm/s axial: slow enough that the bubble stays inside the PSF support
    shape = (32, 11, 40, 40)
    vol = np.zeros(shape, dtype=np.complex64)
    sigma = np.ones((shape[1], 1), dtype=np.float32)
    edges = np.array([0, shape[2]])
    plan = [
        type(make_bubble_plan(shape, geom, n_bubbles=1, seed=0)[0])(
            bubble_id=0, start_evzx=(5.0, 20.0, 20.0), velocity_mms=(0.0, v_z, 0.0),
            f0=0, lifetime=20, snr_db=40.0, phase=0.3, speed_mms=v_z, axial_frac=1.0, z_band=0,
        )
    ]
    out, truth = inject_bubbles(vol, plan, psf, geom, sigma, edges, compounding_gain_on=False)
    series = out[:20, 5, 20, 20]
    d = np.angle(series[1:] * np.conj(series[:-1]))
    measured = float(np.median(d))
    vox_per_frame = geom.mms_to_voxels_per_frame((0.0, v_z, 0.0))[1]
    expected = -psf.k_axial * vox_per_frame  # phase = k*(z_fixed - z0(t)), z0 grows
    assert measured == pytest.approx(expected, abs=0.02), (measured, expected)


def test_compounding_null_sits_at_the_algebraic_speed():
    geom = Geometry()
    v_null = geom.wavelength_mm * geom.frame_rate_hz / 2.0
    assert v_null == pytest.approx(88.97, abs=0.02)
    assert float(compounding_gain(0.0, geom)) == pytest.approx(1.0, abs=1e-9)
    assert float(compounding_gain(v_null, geom)) < 1e-6
    # The measured bulk loses between ~1.7 and ~7.5 dB; check the shape is a
    # monotone roll-off up to the null, not something else.
    speeds = np.linspace(1.0, v_null - 1.0, 40)
    g = compounding_gain(speeds, geom)
    assert np.all(np.diff(g) < 0)
    assert float(compounding_gain(30.0, geom)) == pytest.approx(0.836, abs=0.01)


def test_injection_hits_the_requested_snr():
    """Peak voxel magnitude / reference sigma equals the requested SNR."""
    psf = _make_psf()
    geom = Geometry()
    shape = (8, 11, 40, 40)
    vol = np.zeros(shape, dtype=np.complex64)
    sigma = np.full((shape[1], 1), 0.25, dtype=np.float32)
    edges = np.array([0, shape[2]])
    B = type(make_bubble_plan(shape, geom, n_bubbles=1, seed=0)[0])
    for snr in (0.0, 6.0, 12.0):
        plan = [B(bubble_id=0, start_evzx=(5.0, 20.0, 20.0), velocity_mms=(0.0, 0.0, 0.0),
                  f0=0, lifetime=1, snr_db=snr, phase=0.0, speed_mms=0.0, axial_frac=0.0, z_band=0)]
        out, _ = inject_bubbles(vol, plan, psf, geom, sigma, edges, compounding_gain_on=False)
        peak = float(np.abs(out[0]).max())
        assert peak / 0.25 == pytest.approx(10 ** (snr / 20.0), rel=1e-3)


def test_injection_leaves_the_volume_otherwise_untouched():
    psf = _make_psf()
    geom = Geometry()
    rng = np.random.default_rng(1)
    shape = (10, 11, 50, 50)
    vol = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex64)
    sigma, edges = reference_noise_sigma(vol, ref_rank=2, n_z_bands=2)
    plan = make_bubble_plan(shape, geom, n_bubbles=6, lifetime=4, seed=3)
    out, truth = inject_bubbles(vol, plan, psf, geom, sigma, edges)
    assert truth["frame"].size > 0
    changed = np.flatnonzero(np.any(np.abs(out - vol) > 0, axis=(1, 2, 3)))
    assert set(changed.tolist()) <= set(np.unique(truth["frame"]).tolist())
    # The truth table carries the strata the sweep stratifies on.
    for key in ("speed_mms", "snr_db", "z_band", "axial_frac", "gain_db", "z_mm"):
        assert truth[key].size == truth["frame"].size


def test_plan_spans_the_required_speed_range_including_the_null():
    geom = Geometry()
    plan = make_bubble_plan((240, 25, 154, 275), geom, n_bubbles=300, seed=0)
    speeds = np.unique([b.speed_mms for b in plan])
    assert speeds.min() <= 5.0 and speeds.max() >= 130.0
    assert np.any(np.isclose(speeds, 88.97, atol=0.01)), "the compounding null must be sampled"
    # Both axial and in-plane directions must be present: the null is axial-only.
    fr = np.array([b.axial_frac for b in plan])
    assert (fr > 0.99).sum() > 20 and (fr < 0.01).sum() > 20
    # Depth is stratified.
    assert len(set(b.z_band for b in plan)) >= 4


def test_planned_tracks_stay_inside_the_volume():
    geom = Geometry()
    shape = (240, 25, 154, 275)
    for b in make_bubble_plan(shape, geom, n_bubbles=200, seed=7):
        per = geom.mms_to_voxels_per_frame(b.velocity_mms)
        end = np.asarray(b.start_evzx) + per * (b.lifetime - 1)
        for p in (np.asarray(b.start_evzx), end):
            assert np.all(p >= -1e-6) and np.all(p <= np.array(shape[1:]) - 1 + 1e-6), (b, p)


# --------------------------------------------------------------------------- #
# Nulls: what each one preserves and what it breaks
# --------------------------------------------------------------------------- #
def test_block_shuffle_preserves_every_frame_exactly():
    rng = np.random.default_rng(0)
    vol = (rng.normal(size=(40, 3, 8, 8)) + 1j * rng.normal(size=(40, 3, 8, 8))).astype(np.complex64)
    out = null_block_shuffle(vol, block=8, seed=1)
    assert out.shape == vol.shape
    key = lambda v: (float(v[0, 0, 0].real), float(v[0, 0, 0].imag))
    assert sorted(map(key, vol)) == sorted(map(key, out))  # a permutation of WHOLE frames,
    assert np.allclose(np.sort(np.abs(out).ravel()), np.sort(np.abs(vol).ravel()))  # so every
    assert not np.array_equal(out, vol)                     # spatial statistic is untouched


def test_block_shuffle_block_size_controls_how_much_temporal_structure_survives():
    """block=1 destroys temporal coherence; large blocks largely preserve it.

    This is the dial the sweep must report against, because a null that destroys
    the clutter structure the SVD relies on leaves tissue residue and OVERSTATES
    false alarms -- making every arm look worse, and the harshest arm look best.
    """
    n = 64
    t = np.arange(n)
    slow = np.exp(2j * np.pi * 2 * t / n)  # a slow "tissue" mode
    vol = (slow[:, None, None, None] * np.ones((1, 2, 4, 4))).astype(np.complex64)

    def lag1(v):
        a = v.reshape(v.shape[0], -1)
        num = abs(np.vdot(a[:-1].ravel(), a[1:].ravel()))
        return num / (np.linalg.norm(a[:-1]) * np.linalg.norm(a[1:]))

    assert lag1(vol) > 0.98
    assert lag1(null_block_shuffle(vol, block=16, seed=0)) > 0.8
    assert lag1(null_block_shuffle(vol, block=1, seed=0)) < 0.5


def test_phase_surrogate_preserves_the_per_voxel_power_spectrum():
    rng = np.random.default_rng(2)
    vol = (rng.normal(size=(32, 2, 5, 5)) + 1j * rng.normal(size=(32, 2, 5, 5))).astype(np.complex64)
    out = null_phase_surrogate(vol, seed=3)
    p_in = np.abs(np.fft.fft(vol, axis=0)) ** 2
    p_out = np.abs(np.fft.fft(out, axis=0)) ** 2
    assert np.allclose(p_in, p_out, rtol=1e-3, atol=1e-3)
    # ...while raising the tissue's RANK from the number of spatial modes to the
    # number of frequency bins it occupies, so a fixed-rank cut removes far less
    # of it and the null is HARDER than real data. The mechanism is exact: after
    # per-voxel phase randomization the array is still supported on the same
    # frequency bins, so rank <= bandwidth-in-bins. A monochromatic clutter mode
    # would therefore survive with rank 1 intact; a realistically broadband one
    # (here 30 bins against a rank-8 cut) does not.
    rng2 = np.random.default_rng(11)
    n_t, n_v = 64, 400
    spec = np.zeros((n_t, 3), dtype=complex)
    spec[:30] = rng2.normal(size=(30, 3)) + 1j * rng2.normal(size=(30, 3))
    temporal = np.fft.ifft(spec, axis=0)                      # 3 band-limited tissue profiles
    spatial = rng2.normal(size=(3, n_v)) + 1j * rng2.normal(size=(3, n_v))
    x = (temporal @ spatial) * 50.0   # tissue clutter sits far above blood/noise
    x = x + rng2.normal(size=(n_t, n_v)) + 1j * rng2.normal(size=(n_t, n_v))
    tissue = x.reshape(n_t, 1, 20, 20).astype(np.complex64)

    def resid_after_rank(v, k=8):
        a = v.reshape(v.shape[0], -1)
        s = np.linalg.svd(a, compute_uv=False)
        return float((s[k:] ** 2).sum() / (s**2).sum())

    before = resid_after_rank(tissue)
    after = resid_after_rank(null_phase_surrogate(tissue, seed=1))
    assert after > 3 * before, (before, after)


def test_quiet_crop_finds_the_quiet_region():
    mag = np.ones((4, 2, 40, 40), dtype=np.float32)
    mag[:, :, :20, :20] = 10.0  # loud quadrant
    vol = mag.astype(np.complex64)
    crop, (z0, x0) = null_quiet_crop(vol, mag, crop=(16, 16))
    assert crop.shape == (4, 2, 16, 16)
    assert z0 >= 20 or x0 >= 20


# --------------------------------------------------------------------------- #
# Matched-rate scoring
# --------------------------------------------------------------------------- #
def test_threshold_for_count_matches_density_exactly():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(20, 6, 30, 30)).astype(np.float32)
    peaks = find_peaks(z, min_distance=2, floor=-10.0)
    for target in (50, 200, 500):
        if target > peaks.zscore.size:
            continue
        t = threshold_for_count(peaks, target)
        assert int(np.count_nonzero(peaks.zscore >= t)) == target
    assert threshold_for_count(peaks, peaks.zscore.size + 1) == float("inf")


def test_matched_density_is_immune_to_a_rescaled_noise_floor():
    """The J2 failure mode: two arms with different noise floors.

    At a fixed threshold the arm with the smaller denominator wins trivially. At
    matched density the two are identical, which is the whole point.
    """
    rng = np.random.default_rng(4)
    mag = np.abs(rng.normal(size=(12, 4, 30, 30))).astype(np.float32)
    a = find_peaks(zscore_field(mag, mode="per_elev", smoothing_sigma=0.0), floor=-10.0)
    b = find_peaks(zscore_field(3.0 * mag, mode="per_elev", smoothing_sigma=0.0), floor=-10.0)
    at_fixed = (a.density_per_frame(1.5), b.density_per_frame(1.5))
    ta, tb = threshold_for_count(a, 100), threshold_for_count(b, 100)
    assert a.density_per_frame(ta) == b.density_per_frame(tb)
    assert at_fixed[0] == pytest.approx(at_fixed[1], rel=0.2)  # z-score already de-scales


def test_per_elev_zband_flattens_a_depth_varying_noise_floor():
    """The specific prediction of the pooled-_slice_stats concern.

    With a noise floor that falls with depth (attenuation), pooling one std per
    elevation plane over all depths over-thresholds the shallow half and
    under-thresholds the deep half, so detections pile up at one end. The
    depth-banded denominator should even them out.
    """
    rng = np.random.default_rng(5)
    n_z = 120
    gain = np.linspace(3.0, 0.5, n_z).astype(np.float32)  # loud shallow, quiet deep
    mag = np.abs(rng.normal(size=(24, 4, n_z, 60)).astype(np.float32)) * gain[None, None, :, None]

    def split(mode):
        z = zscore_field(mag, mode=mode, smoothing_sigma=0.0, n_z_bands=8)
        p = find_peaks(z, min_distance=2, floor=-10.0)
        t = threshold_for_count(p, 400)
        sel = p.at(t)
        return float(np.mean(sel.z < n_z / 2))

    frac_shallow_pooled = split("per_elev")
    frac_shallow_banded = split("per_elev_zband")
    assert frac_shallow_pooled > 0.95, frac_shallow_pooled
    assert abs(frac_shallow_banded - 0.5) < 0.15, frac_shallow_banded


def test_spatial_tgc_also_flattens_it():
    rng = np.random.default_rng(6)
    n_z = 120
    gain = np.linspace(3.0, 0.5, n_z).astype(np.float32)
    mag = np.abs(rng.normal(size=(24, 4, n_z, 60)).astype(np.float32)) * gain[None, None, :, None]
    z = zscore_field(mag, mode="spatial_tgc", smoothing_sigma=0.0,
                     voxel_mm=(0.5547, 0.2, 0.2), tgc_sigma_lambda=4.0)
    p = find_peaks(z, min_distance=2, floor=-10.0).at(threshold_for_count(
        find_peaks(z, min_distance=2, floor=-10.0), 400))
    # Looser than the banded arm: the smoothed gain map is edge-biased by the
    # Gaussian's boundary handling, which the banded estimator does not suffer.
    assert abs(float(np.mean(p.z < n_z / 2)) - 0.5) < 0.27


def test_match_peaks_to_truth_recovers_a_loud_injection():
    """End-to-end: inject loud bubbles into noise, they come back at high recovery."""
    rng = np.random.default_rng(9)
    geom = Geometry()
    shape = (24, 11, 80, 80)
    vol = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex64)
    sigma, edges = reference_noise_sigma(vol, ref_rank=1, n_z_bands=2)
    psf = _make_psf()
    plan = make_bubble_plan(shape, geom, n_bubbles=12, lifetime=6,
                            snr_db=(24.0,), speeds_mms=(20.0,), n_z_bands=2, seed=2)
    inj, truth = inject_bubbles(vol, plan, psf, geom, sigma, edges, compounding_gain_on=False)
    z = zscore_field(np.abs(inj), mode="per_elev", smoothing_sigma=0.0)
    peaks = find_peaks(z, min_distance=2, floor=3.0)
    hit = match_peaks_to_truth(truth, peaks)
    assert hit.mean() > 0.9, hit.mean()
    # The comparison that matters is against the un-injected volume AT THE SAME
    # DETECTION DENSITY -- the chance rate of a peak landing inside the match
    # tolerance. A raw "does it fire" check would only measure how permissive
    # the floor is.
    z0 = zscore_field(np.abs(vol), mode="per_elev", smoothing_sigma=0.0)
    p0 = find_peaks(z0, min_distance=2, floor=-10.0)
    chance = match_peaks_to_truth(truth, p0.at(threshold_for_count(p0, peaks.zscore.size)))
    assert chance.mean() < 0.4 and hit.mean() > 2.5 * chance.mean(), (hit.mean(), chance.mean())


# --------------------------------------------------------------------------- #
# The clutter-filter arm
# --------------------------------------------------------------------------- #
def test_filter_bank_shares_one_basis_and_orders_the_cutoffs():
    """The three low-cutoff arms are nested: knee_spatial <= knee <= rank24."""
    from ultratrace_ulm.opsweep import svd_filter_bank

    rng = np.random.default_rng(0)
    n_f = 60
    # 3 strong slow "tissue" modes + weak fast "blood" + noise.
    t = np.arange(n_f)
    tissue = sum(
        np.exp(2j * np.pi * f * t / n_f)[:, None] * (rng.normal(size=(1, 900)) * 60.0)
        for f in (1, 2, 3)
    )
    blood = np.exp(2j * np.pi * 20 * t / n_f)[:, None] * (rng.normal(size=(1, 900)) * 2.0)
    noise = rng.normal(size=(n_f, 900)) + 1j * rng.normal(size=(n_f, 900))
    vol = (tissue + blood + noise).reshape(n_f, 1, 30, 30).astype(np.complex64)

    specs = [(f"{m}{'_mp' if h else ''}", m, h)
             for m in ("rank24", "knee", "knee_spatial") for h in (False, True)]
    got = {name: (c, mag) for name, c, mag in svd_filter_bank(vol, specs, ceiling_frac=0.1)}
    assert set(got) == {s[0] for s in specs}
    c = {k: v[0] for k, v in got.items()}
    assert c["rank24"]["low"] == round(0.1 * n_f)          # the unfired fallback, reproduced
    assert c["knee"]["low"] <= c["rank24"]["low"]           # 24 is a CEILING for the knee
    assert c["knee_spatial"]["low"] <= c["knee"]["low"]     # L20 min() rule
    assert c["knee"]["low"] >= 1                            # the DC mode is always removed
    # The MP switch only ever removes trailing modes, never leading ones.
    for m in ("rank24", "knee", "knee_spatial"):
        assert c[f"{m}_mp"]["low"] == c[m]["low"]
        assert c[f"{m}_mp"]["high_remove"] >= c[m]["high_remove"] == 0
    for _, mag in got.values():
        assert mag.shape == vol.shape and mag.dtype == np.float32


def test_filter_bank_keeping_more_modes_retains_more_energy():
    from ultratrace_ulm.opsweep import svd_filter_bank

    rng = np.random.default_rng(1)
    n_f = 40
    t = np.arange(n_f)
    vol = (np.exp(2j * np.pi * t / n_f)[:, None] * (rng.normal(size=(1, 400)) * 40.0)
           + rng.normal(size=(n_f, 400)) + 1j * rng.normal(size=(n_f, 400))
           ).reshape(n_f, 1, 20, 20).astype(np.complex64)
    out = {n: (c, m) for n, c, m in
           svd_filter_bank(vol, [("a", "rank24", False), ("b", "knee", False)])}
    assert out["b"][0]["low"] < out["a"][0]["low"]
    assert float((out["b"][1] ** 2).sum()) > float((out["a"][1] ** 2).sum())
