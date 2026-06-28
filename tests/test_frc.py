"""Tests for the split-half FRC/FSC reproducibility arbiter (ultratrace_ulm.frc).

Pure-numpy, seeded, small grids. The design mirrors the real use case: a
REPRODUCIBLE method recovers the SAME structure in both independent halves (FRC
stays high to high frequency -> fine resolution, high repro_score); a SPURIOUS
method puts DIFFERENT structure in each half (FRC collapses -> coarse/no
resolution, low repro_score). Ordering + noise-coarsening + degenerate-input
behaviour are all asserted.
"""

import numpy as np
import pytest

from ultratrace_ulm.frc import (
    frc_curve,
    reconstruct_density,
    split_half_frc,
    split_tracks,
    _half_bit_threshold,
)

FPA = 700  # frames per acquisition (matches the real geometry)

# Common reconstruction grid for all split-half tests (small -> fast FFTs).
BMIN = np.array([-0.5, -0.5, -0.5])
BMAX = np.array([6.5, 4.5, 6.5])
VOXEL = 0.1
_KW = dict(bounds_min=BMIN, bounds_max=BMAX, voxel_mm=VOXEL, split="acq_parity")

# A few thin, well-separated 3D "vessels" (line segments) spanning all 3 axes so
# every projection plane (xz, yz, xy) sees real structure.
_VESSELS = [
    ((0.5, 0.5, 0.5), (5.5, 3.5, 5.5)),
    ((5.5, 0.5, 0.5), (0.5, 3.5, 5.5)),
    ((0.5, 2.0, 5.0), (5.5, 2.0, 1.0)),
    ((3.0, 0.5, 0.5), (3.0, 3.5, 5.5)),
]

_EVEN_ACQS = np.arange(0, 40, 2)
_ODD_ACQS = np.arange(1, 40, 2)


def _points_on_lines(lines, n, rng, jitter):
    """Sample ``n`` jittered points along each line segment."""
    out = []
    for p0, p1 in lines:
        p0 = np.asarray(p0, float)
        p1 = np.asarray(p1, float)
        t = rng.random(n)
        seg = p0[None, :] + t[:, None] * (p1 - p0)[None, :]
        seg = seg + rng.normal(0.0, jitter, seg.shape)
        out.append(seg)
    return np.concatenate(out)


def _pack_tracks(points, acqs, rng, track_len=10):
    """Chop a point cloud into short tracks, each assigned to an acq in ``acqs``
    (so acq-parity splitting recovers the intended half)."""
    points = points[rng.permutation(len(points))]
    tracks = []
    for i in range(0, len(points), track_len):
        chunk = points[i:i + track_len]
        if len(chunk) == 0:
            continue
        acq = int(acqs[rng.integers(len(acqs))])
        f0 = acq * FPA + int(rng.integers(0, max(1, FPA - len(chunk))))
        frames = np.arange(f0, f0 + len(chunk))
        tracks.append({
            "positions": chunk.astype(np.float32),
            "frames": frames.astype(np.int64),
            "length": int(len(chunk)),
            "intensities": np.full(len(chunk), 5.0, np.float32),
        })
    return tracks


def _build_reproducible(jitter=0.15, seed=0, n=1200):
    """Even and odd halves sample the SAME vessels with INDEPENDENT jitter."""
    pa = _points_on_lines(_VESSELS, n, np.random.default_rng(seed + 1), jitter)
    pb = _points_on_lines(_VESSELS, n, np.random.default_rng(seed + 2), jitter)
    ta = _pack_tracks(pa, _EVEN_ACQS, np.random.default_rng(seed + 3))
    tb = _pack_tracks(pb, _ODD_ACQS, np.random.default_rng(seed + 4))
    return ta + tb


def _build_spurious(seed=0, n=1200):
    """Even and odd halves sample DIFFERENT random vessel sets -> no shared
    structure beyond DC."""
    ra = np.random.default_rng(seed + 10)
    rb = np.random.default_rng(seed + 20)
    lines_a = [((ra.uniform(0, 6), ra.uniform(0, 4), ra.uniform(0, 6)),
                (ra.uniform(0, 6), ra.uniform(0, 4), ra.uniform(0, 6))) for _ in range(4)]
    lines_b = [((rb.uniform(0, 6), rb.uniform(0, 4), rb.uniform(0, 6)),
                (rb.uniform(0, 6), rb.uniform(0, 4), rb.uniform(0, 6))) for _ in range(4)]
    pa = _points_on_lines(lines_a, n, np.random.default_rng(seed + 11), 0.12)
    pb = _points_on_lines(lines_b, n, np.random.default_rng(seed + 21), 0.12)
    ta = _pack_tracks(pa, _EVEN_ACQS, np.random.default_rng(seed + 13))
    tb = _pack_tracks(pb, _ODD_ACQS, np.random.default_rng(seed + 23))
    return ta + tb


# ---------------------------------------------------------------------------
# reconstruct_density
# ---------------------------------------------------------------------------


def test_reconstruct_density_counts_and_shape():
    bmin = np.array([0.0, 0.0, 0.0])
    bmax = np.array([1.0, 1.0, 1.0])
    # 3 points in one voxel, 1 in another (voxel 0.25 -> 4x4x4 grid).
    pts = np.array([
        [0.1, 0.1, 0.1],
        [0.1, 0.1, 0.1],
        [0.1, 0.1, 0.1],
        [0.6, 0.6, 0.6],
    ])
    dens = reconstruct_density(pts, bmin, bmax, voxel_mm=0.25)
    assert dens.shape == (4, 4, 4)
    assert dens.sum() == 4.0
    assert dens[0, 0, 0] == 3.0  # weighted by COUNT
    assert dens[2, 2, 2] == 1.0


def test_reconstruct_density_clamps_grid_size():
    # Tiny voxel over a large extent would explode; grid must clamp to <= 512.
    bmin = np.array([0.0, 0.0, 0.0])
    bmax = np.array([1000.0, 1000.0, 1000.0])
    dens = reconstruct_density(np.array([[1.0, 1.0, 1.0]]), bmin, bmax, voxel_mm=0.001)
    assert max(dens.shape) <= 512


def test_reconstruct_density_empty():
    dens = reconstruct_density(np.empty((0, 3)), np.zeros(3), np.ones(3), voxel_mm=0.1)
    assert dens.sum() == 0.0
    assert dens.ndim == 3


# ---------------------------------------------------------------------------
# frc_curve core
# ---------------------------------------------------------------------------


def test_half_bit_threshold_formula_and_asymptote():
    # Spot value: n_q = 100 -> sqrt = 10.
    val = float(_half_bit_threshold(np.array([100.0]))[0])
    expected = (0.2071 + 1.9102 / 10.0) / (1.2071 + 0.9102 / 10.0)
    assert val == pytest.approx(expected)
    # Large n_q asymptote ~ 0.2071/1.2071.
    big = float(_half_bit_threshold(np.array([1e9]))[0])
    assert big == pytest.approx(0.2071 / 1.2071, abs=1e-4)


def test_frc_identical_maps_is_one():
    rng = np.random.default_rng(0)
    m = rng.random((48, 48))
    curve = frc_curve(m, m, VOXEL)
    frc = curve["frc"]
    nq = curve["n_q"]
    populated = nq > 0
    # FRC of a map with itself is identically 1 wherever the ring has content.
    assert np.allclose(frc[populated], 1.0, atol=1e-6)
    # Never crosses below threshold -> Nyquist-limited (finite) resolution.
    assert np.isfinite(curve["resolution_mm_half_bit"])


def test_frc_empty_map_is_null():
    curve = frc_curve(np.zeros((16, 16)), np.zeros((16, 16)), VOXEL)
    assert not np.isfinite(curve["resolution_mm_half_bit"])
    assert curve["frc"].size == 0


# ---------------------------------------------------------------------------
# split_half_frc: reproducible vs spurious
# ---------------------------------------------------------------------------


def test_reproducible_high_frc_fine_resolution():
    res = split_half_frc(_build_reproducible(jitter=0.15), FPA, **_KW)
    assert res["repro_score"] > 0.5
    # Every plane + 3D resolves to fine (sub-mm) detail.
    for key in ("xz", "yz", "xy", "fsc3d"):
        r = res[key]["resolution_mm_half_bit"]
        assert np.isfinite(r)
        assert r < 1.0
    # Both halves sampled.
    assert res["meta"]["n_points_a"] > 0
    assert res["meta"]["n_points_b"] > 0


def test_spurious_collapses():
    res = split_half_frc(_build_spurious(), FPA, **_KW)
    # Different structure in each half -> correlation collapses near 0.
    assert res["repro_score"] < 0.25
    # No reproducible structure -> 3D resolution is unresolved (inf).
    assert not np.isfinite(res["fsc3d"]["resolution_mm_half_bit"])


def test_ordering_reproducible_beats_spurious():
    rep = split_half_frc(_build_reproducible(jitter=0.15), FPA, **_KW)
    spur = split_half_frc(_build_spurious(), FPA, **_KW)
    # The whole point: repro_score RANKS real structure above artifacts.
    assert rep["repro_score"] > spur["repro_score"] + 0.3
    # And reproducible resolution is finer (smaller) than spurious (inf).
    assert (rep["fsc3d"]["resolution_mm_half_bit"]
            < spur["fsc3d"]["resolution_mm_half_bit"])


def test_more_noise_coarsens_resolution():
    low = split_half_frc(_build_reproducible(jitter=0.15), FPA, **_KW)
    high = split_half_frc(_build_reproducible(jitter=0.45), FPA, **_KW)
    # More independent localization noise -> the fine detail stops reproducing
    # -> coarser (larger) resolution_mm and a lower repro_score.
    assert (high["fsc3d"]["resolution_mm_half_bit"]
            > low["fsc3d"]["resolution_mm_half_bit"])
    assert high["repro_score"] < low["repro_score"]


def test_random_split_also_reproduces():
    # The same shared structure must score high under the random-split fallback.
    tracks = _build_reproducible(jitter=0.15)
    res = split_half_frc(tracks, FPA, bounds_min=BMIN, bounds_max=BMAX,
                         voxel_mm=VOXEL, split="random", seed=7)
    assert res["repro_score"] > 0.4


def test_split_tracks_acq_parity():
    a = {"frames": np.array([0, 1, 2])}        # acq 0 -> A
    b = {"frames": np.array([FPA, FPA + 1])}   # acq 1 -> B
    c = {"frames": np.array([2 * FPA])}        # acq 2 -> A
    ta, tb = split_tracks([a, b, c], FPA, "acq_parity")
    assert len(ta) == 2 and len(tb) == 1


# ---------------------------------------------------------------------------
# degenerate inputs
# ---------------------------------------------------------------------------


def test_empty_tracks():
    res = split_half_frc([], FPA, **_KW)
    assert res["repro_score"] == 0.0
    assert not np.isfinite(res["fsc3d"]["resolution_mm_half_bit"])
    assert res["meta"]["n_points_a"] == 0


def test_single_track():
    # One track lands in one half -> the other half is empty -> graceful null.
    one = _build_reproducible(jitter=0.15)[:1]
    res = split_half_frc(one, FPA, **_KW)
    assert res["repro_score"] == 0.0
    for key in ("xz", "yz", "xy", "fsc3d"):
        assert not np.isfinite(res[key]["resolution_mm_half_bit"])


def test_split_half_frc_default_bounds_no_crash():
    # No bounds given -> union bounds derived internally; must run end to end.
    res = split_half_frc(_build_reproducible(jitter=0.15), FPA,
                         voxel_mm=VOXEL, split="acq_parity")
    assert res["repro_score"] > 0.4
    assert "grid_shape" in res["meta"]
