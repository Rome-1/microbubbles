"""Local pytest for ultratrace_ulm.bench (pure numpy, no GPU/h5py/cupy).

Builds a TINY synthetic .bin with the exact header+index+points byte layout
(see export_tracks_bin in ultratrace_ulm/tracking.py) with known track lengths
and straight-line positions, then asserts the metrics come out as expected.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultratrace_ulm.bench import (  # noqa: E402
    compare,
    density_metrics,
    format_table,
    load_tracks_bin,
    saturation_metrics,
    track_metrics,
)

_MAGIC = 0x554C4D54
_HEADER_FMT = "<IIIIf3f3fIIf"
_INDEX_FMT = "<IIHH"

# Known design: 4 straight tracks along +x (y=z=0), 1 mm steps, frame step 1.
TRACK_LENGTHS = [3, 12, 25, 60]      # mean = 25.0
N_ACQS = 2                           # tracks_per_acq = 4/2 = 2.0
FRAME_RATE = 10.0
FRAMES_PER_ACQ = 100


def _write_synthetic_bin(path: Path) -> None:
    tracks = []
    for L in TRACK_LENGTHS:
        x = np.arange(L, dtype=np.float32)           # 0,1,...,L-1 mm
        y = np.zeros(L, dtype=np.float32)
        z = (x * 0.01).astype(np.float32)            # tiny z spread for histogram
        frames = np.arange(L, dtype=np.float32)
        speed = np.zeros(L, dtype=np.float32)        # stored, but recomputed in bench
        tracks.append(np.column_stack([x, y, z, frames, speed]).astype(np.float32))

    all_pts = np.concatenate(tracks, axis=0)
    total_points = all_pts.shape[0]
    bmin = all_pts[:, 0:3].min(axis=0)
    bmax = all_pts[:, 0:3].max(axis=0)

    header = struct.pack(
        _HEADER_FMT,
        _MAGIC, 4, len(tracks), total_points, 0.0,
        float(bmin[0]), float(bmin[1]), float(bmin[2]),
        float(bmax[0]), float(bmax[1]), float(bmax[2]),
        FRAMES_PER_ACQ, N_ACQS, FRAME_RATE,
    )
    with open(path, "wb") as fp:
        fp.write(header + b"\x00" * (64 - len(header)))
        offset = 0
        for i, t in enumerate(tracks):
            fp.write(struct.pack(_INDEX_FMT, offset, t.shape[0], i % N_ACQS, 0))
            offset += t.shape[0]
        fp.write(all_pts.tobytes())


def test_load_tracks_bin(tmp_path):
    path = tmp_path / "synth.bin"
    _write_synthetic_bin(path)
    data = load_tracks_bin(path)

    assert data["n_tracks"] == 4
    assert data["total_points"] == sum(TRACK_LENGTHS)
    assert data["n_acqs"] == N_ACQS
    assert data["frame_rate"] == FRAME_RATE
    assert list(data["track_lengths"]) == TRACK_LENGTHS
    assert data["positions"].shape == (sum(TRACK_LENGTHS), 3)
    # offsets are cumulative
    assert list(data["track_offsets"]) == [0, 3, 15, 40]


def test_track_metrics(tmp_path):
    path = tmp_path / "synth.bin"
    _write_synthetic_bin(path)
    m = track_metrics(load_tracks_bin(path))

    assert m["n_tracks"] == 4
    assert m["total_points"] == 100
    assert m["tracks_per_acq"] == 2.0
    assert m["track_len_mean"] == 25.0
    assert m["mean_points_per_track"] == 25.0

    # Length-gate fractions: lengths [3,12,25,60]
    assert m["frac_len_ge_10"] == 0.75
    assert m["frac_len_ge_20"] == 0.5
    assert m["frac_len_ge_35"] == 0.25
    assert m["frac_len_ge_50"] == 0.25

    # Fragmentation: 1 short (<10), 2 long (>=20) -> 0.5
    assert m["n_short_lt10"] == 1
    assert m["n_long_ge20"] == 2
    assert m["frag_short_over_long"] == 0.5

    # Straight lines -> straightness == 1.0; curv length = (L-1) mm in x, plus a
    # tiny z spread (0.01/mm), so allow a small tolerance.
    assert m["straightness"] == pytest.approx(1.0, abs=1e-3)
    # mean curvilinear length ~ mean(L-1) = mean([2,11,24,59]) = 24.0
    assert m["mean_curvilinear_len_mm"] == pytest.approx(24.0, abs=0.05)
    # segment speed = |dp| * frame_rate / dframe ~= 1mm * 10 / 1 = 10 mm/s
    assert m["mean_speed_mm_s"] == pytest.approx(10.0, abs=0.05)
    assert np.isfinite(m["median_speed_mm_s"])


def test_density_metrics(tmp_path):
    path = tmp_path / "synth.bin"
    _write_synthetic_bin(path)
    d = density_metrics(load_tracks_bin(path), bins=32)

    assert d["n_bins"] == 32
    assert 0.0 < d["occupied_fraction"] <= 1.0
    assert d["occupied_bins"] > 0
    for k in ("contrast_cnr_proxy", "entropy_bits", "max_entropy_bits",
              "entropy_ratio", "peak_density"):
        assert np.isfinite(d[k]), f"{k} not finite: {d[k]}"


def test_compare_and_table(tmp_path):
    p1 = tmp_path / "a.bin"
    p2 = tmp_path / "b.bin"
    _write_synthetic_bin(p1)
    _write_synthetic_bin(p2)
    comp = compare([str(p1), str(p2)], ["a", "b"])
    assert comp["labels"] == ["a", "b"]
    assert set(comp["runs"]) == {"a", "b"}
    # saturation group is part of every run.
    assert "saturation" in comp["runs"]["a"]

    table = format_table(comp)
    assert "metric" in table
    assert "n_tracks" in table
    assert "delta" in table
    assert "sat_frac_of_ceiling" in table  # saturation rows present
    # identical inputs -> zero delta column for n_tracks
    assert "| n_tracks | 4 | 4 | 0 |" in table


# ---------------------------------------------------------------------------
# Saturation metric (Hingot 2019 coverage saturation, no ground truth)
# ---------------------------------------------------------------------------

_SAT_BINS = 16          # 16 x 16 = 256 coronal bins
_SAT_FPA = 10           # frames per acquisition (acq = frame // fpa)


def _write_event_bin(path: Path, events) -> None:
    """Write a .bin whose coronal occupancy follows ``events``.

    ``events`` is a list of (ix, iz, acq): a localization placed at the CENTRE of
    coronal bin (ix, iz) in acquisition ``acq``. Bounds are fixed to
    (0,0,0)-(bins,1,bins) so a point at (ix+0.5, *, iz+0.5) lands in bin (ix,iz).
    One track per acquisition (track geometry is irrelevant to saturation, which
    keys off per-point frame // fpa).
    """
    by_acq: dict[int, list] = {}
    for ix, iz, acq in events:
        by_acq.setdefault(int(acq), []).append((int(ix), int(iz)))
    acqs = sorted(by_acq)

    tracks = []
    for acq in acqs:
        pts = by_acq[acq]
        arr = np.zeros((len(pts), 5), dtype=np.float32)
        for j, (ix, iz) in enumerate(pts):
            arr[j] = (ix + 0.5, 0.5, iz + 0.5, acq * _SAT_FPA, 0.0)
        tracks.append(arr)

    all_pts = np.concatenate(tracks, axis=0)
    total_points = all_pts.shape[0]
    header = struct.pack(
        _HEADER_FMT,
        _MAGIC, 4, len(tracks), total_points, 0.0,
        0.0, 0.0, 0.0,
        float(_SAT_BINS), 1.0, float(_SAT_BINS),
        _SAT_FPA, len(acqs), 10.0,
    )
    with open(path, "wb") as fp:
        fp.write(header + b"\x00" * (64 - len(header)))
        offset = 0
        for acq, t in zip(acqs, tracks):
            fp.write(struct.pack(_INDEX_FMT, offset, t.shape[0], acq, 0))
            offset += t.shape[0]
        fp.write(all_pts.tobytes())


def _distinct_bins(n: int):
    """First ``n`` distinct coronal (ix, iz) bins, row-major over the grid."""
    return [(k // _SAT_BINS, k % _SAT_BINS) for k in range(n)]


def test_saturation_plateau(tmp_path):
    """Coverage that fills the bed early then plateaus = real, efficient."""
    # New bins per acq decay to zero, then 10 tail acqs add NO new coverage
    # (points fall in already-occupied bin (0,0)) -> rise-then-flat curve.
    new_per_acq = [20, 16, 13, 10, 8, 6, 5, 4, 3, 2]
    bins = _distinct_bins(sum(new_per_acq))
    events = []
    cur = 0
    for acq, k in enumerate(new_per_acq):
        for ix, iz in bins[cur:cur + k]:
            events.append((ix, iz, acq))
        cur += k
    for acq in range(10, 20):          # tail: reuse bin (0,0)
        events.append((0, 0, acq))

    path = tmp_path / "sat.bin"
    _write_event_bin(path, events)
    m = saturation_metrics(load_tracks_bin(path), bins=_SAT_BINS)

    assert m["sat_n_acqs"] == 20
    # Flat tail -> late slope is zero -> ratio ~ 0 (plateaued).
    assert m["sat_late_early_slope_ratio"] == pytest.approx(0.0, abs=1e-9)
    # Final render is near its asymptote, and saturates well before the end.
    assert m["sat_fraction_of_ceiling"] > 0.85
    assert np.isfinite(m["sat_acqs_to_90pct"]) and 0 < m["sat_acqs_to_90pct"] < 20
    # Exponential explains a plateau better than a straight line does.
    assert m["sat_r2"] > 0.9
    assert m["sat_r2"] > m["sat_linear_r2"]
    assert 0.0 < m["sat_ceiling_fraction"] <= 1.0
    assert len(m["curve_occupied_fraction"]) == 20


def test_saturation_linear_clutter(tmp_path):
    """Coverage that climbs linearly and never plateaus = clutter signature."""
    per_acq = 5                         # same # of FRESH bins every acquisition
    bins = _distinct_bins(per_acq * 20)
    events = []
    cur = 0
    for acq in range(20):
        for ix, iz in bins[cur:cur + per_acq]:
            events.append((ix, iz, acq))
        cur += per_acq

    path = tmp_path / "clutter.bin"
    _write_event_bin(path, events)
    m = saturation_metrics(load_tracks_bin(path), bins=_SAT_BINS)

    assert m["sat_n_acqs"] == 20
    # Constant growth -> late slope == early slope -> ratio ~ 1 (never plateaus).
    assert m["sat_late_early_slope_ratio"] == pytest.approx(1.0, abs=1e-9)
    # A straight line fits at least as well as the saturating model.
    assert m["sat_linear_r2"] > 0.99
    assert m["sat_linear_r2"] >= m["sat_r2"] - 0.05
    # Still climbing at the end -> far from the (extrapolated) ceiling.
    assert m["sat_fraction_of_ceiling"] < 0.7


def test_saturation_too_few_acqs(tmp_path):
    """< 3 acquisitions: report the raw curve but no fit (graceful)."""
    events = [(0, 0, 0), (1, 1, 0), (2, 2, 1)]   # only 2 distinct acqs
    path = tmp_path / "few.bin"
    _write_event_bin(path, events)
    m = saturation_metrics(load_tracks_bin(path), bins=_SAT_BINS)

    assert m["sat_n_acqs"] == 2
    assert np.isnan(m["sat_rate_kappa"])
    assert np.isnan(m["sat_acqs_to_90pct"])
    assert m["sat_final_occupied_fraction"] > 0.0
    assert len(m["curve_acq"]) == 2
