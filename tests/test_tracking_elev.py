"""Tests for the elevation-anisotropic Kalman option in kalman_tracking_3d.

Our 25 elevation planes are synthesized from a single physical receive row, so
the y (elevation) axis is far less reliable than x/z. The tracker can down-weight
elevation by inflating the measurement noise R[y,y] (elev_meas_factor) and by
widening the hard box gate along y (elev_gate_factor). Defaults (axis 1, factors
1.0) must reproduce the isotropic tracker exactly.
"""

import numpy as np
import pytest

from ultratrace_ulm.tracking import kalman_tracking_3d


def _make_detections(seed=0, n=40, jump_frames=(10, 20, 30), jump=0.9, jitter=0.05):
    """One bubble per frame: smooth in x/z, with large elevation (y) jumps.

    x and z advance by 0.1 mm/frame (well inside any reasonable gate). The y
    coordinate is flat except for sustained jumps at ``jump_frames`` that exceed
    the isotropic box gate. With small per-axis gates these jumps fragment the
    track; widening / down-weighting elevation keeps it whole.
    """
    rng = np.random.default_rng(seed)
    dets = []
    y = 0.0
    for f in range(n):
        x = 0.1 * f
        z = 8.0 + 0.1 * f
        if f in jump_frames:
            y += jump  # elevation jump larger than the isotropic gate
        y_obs = y + rng.normal(0.0, jitter)
        dets.append(np.array([[x, y_obs, z]], dtype=np.float64))
    return dets


# Small per-axis gate so the 0.9 mm elevation jumps exceed it (0.5 mm).
GATE = (0.5, 0.5, 0.5)
COMMON = dict(max_distance_mm=GATE, max_gap=3, min_track_length=3, max_cost=1e5)


def _summary(tracks):
    lengths = sorted((int(t["length"]) for t in tracks), reverse=True)
    return len(tracks), (lengths[0] if lengths else 0)


def test_backward_compat_defaults_identical():
    """Default factors (axis 1, 1.0/1.0) must reproduce the isotropic tracker
    exactly: same track count and byte-identical positions/frames."""
    dets = _make_detections()
    base = kalman_tracking_3d(dets, **COMMON)
    withargs = kalman_tracking_3d(
        dets, elev_axis=1, elev_meas_factor=1.0, elev_gate_factor=1.0, **COMMON
    )
    assert len(base) == len(withargs)
    for a, b in zip(base, withargs):
        assert a["length"] == b["length"]
        assert np.array_equal(a["positions"], b["positions"])
        assert np.array_equal(a["frames"], b["frames"])


def test_backward_compat_with_intensities_identical():
    """Same backward-compat guarantee on the intensities-carrying path."""
    dets = _make_detections()
    ints = [np.array([1.0], dtype=np.float64) for _ in dets]
    base = kalman_tracking_3d(dets, intensities=ints, **COMMON)
    withargs = kalman_tracking_3d(
        dets, intensities=ints, elev_axis=1, elev_meas_factor=1.0,
        elev_gate_factor=1.0, **COMMON,
    )
    assert len(base) == len(withargs)
    for a, b in zip(base, withargs):
        assert np.array_equal(a["positions"], b["positions"])
        assert np.array_equal(a["intensities"], b["intensities"])


def test_elevation_downweight_reduces_fragmentation():
    """Isotropic tracking fragments the bubble at each elevation jump; the
    down-weighted run (meas_factor=10, gate_factor=3) keeps it as one long
    track."""
    dets = _make_detections()
    iso = kalman_tracking_3d(dets, **COMMON)
    aniso = kalman_tracking_3d(
        dets, elev_meas_factor=10.0, elev_gate_factor=3.0, **COMMON
    )

    n_iso, maxlen_iso = _summary(iso)
    n_aniso, maxlen_aniso = _summary(aniso)

    # Isotropic must actually fragment (sanity: the scenario bites).
    assert n_iso > 1
    # Down-weighting elevation yields fewer, longer tracks.
    assert n_aniso < n_iso
    assert maxlen_aniso > maxlen_iso
    # Concretely: the whole 40-frame bubble survives as a single track.
    assert n_aniso == 1
    assert maxlen_aniso == 40


def test_gate_vs_R_attribution():
    """Attribution of which knob does the work in this scenario.

    The hard box gate fires BEFORE the Mahalanobis stage, so when the binding
    constraint is the gate, widening it (elev_gate_factor) recovers the track,
    while R inflation alone (elev_meas_factor) changes nothing -- R only enters
    the Kalman update/Mahalanobis cost downstream of the gate. We assert this
    explicitly so the caveat is captured in code.
    """
    dets = _make_detections()
    iso = kalman_tracking_3d(dets, **COMMON)
    gate_only = kalman_tracking_3d(
        dets, elev_meas_factor=1.0, elev_gate_factor=3.0, **COMMON
    )
    r_only = kalman_tracking_3d(
        dets, elev_meas_factor=10.0, elev_gate_factor=1.0, **COMMON
    )

    n_iso, maxlen_iso = _summary(iso)
    n_gate, maxlen_gate = _summary(gate_only)
    n_r, maxlen_r = _summary(r_only)

    # Gate widening alone recovers the full track.
    assert maxlen_gate > maxlen_iso
    assert n_gate < n_iso
    # R inflation alone is a no-op here (gate is the binding constraint): the
    # output is identical to the isotropic run.
    assert n_r == n_iso
    assert maxlen_r == maxlen_iso
    assert len(r_only) == len(iso)
    for a, b in zip(r_only, iso):
        assert np.array_equal(a["positions"], b["positions"])


def test_elev_axis_out_of_range_raises():
    """elev_axis must index a coordinate (0=x, 1=y, 2=z)."""
    dets = _make_detections(n=5, jump_frames=())
    with pytest.raises(ValueError):
        kalman_tracking_3d(dets, elev_axis=3, **COMMON)
    with pytest.raises(ValueError):
        kalman_tracking_3d(dets, elev_axis=-1, **COMMON)


def test_factors_below_one_allowed():
    """Factors < 1 are permitted (tightening elevation): the call must succeed
    and return a valid track list."""
    dets = _make_detections(jump_frames=())  # smooth, no jumps
    out = kalman_tracking_3d(
        dets, elev_meas_factor=0.5, elev_gate_factor=0.5, **COMMON
    )
    assert isinstance(out, list)
    # Smooth single bubble with a tighter elevation gate still tracks as one.
    assert len(out) == 1
    assert out[0]["length"] == 40


def test_elev_axis_zero_targets_x():
    """elev_axis is configurable: pointing it at x (axis 0) should widen the
    x gate instead of y. With jumps placed on x, axis-0 widening recovers the
    track while the default axis-1 (y) widening does not."""
    rng = np.random.default_rng(1)
    dets = []
    x = 0.0
    for f in range(40):
        if f in (10, 20, 30):
            x += 0.9  # jumps on the X axis now
        dets.append(
            np.array([[x + rng.normal(0, 0.05), 0.0, 8.0 + 0.1 * f]], dtype=np.float64)
        )
    # Widening y (default axis) does nothing for an x-axis jump -> fragments.
    y_widen = kalman_tracking_3d(
        dets, elev_axis=1, elev_meas_factor=10.0, elev_gate_factor=3.0, **COMMON
    )
    # Widening x (axis 0) recovers the track.
    x_widen = kalman_tracking_3d(
        dets, elev_axis=0, elev_meas_factor=10.0, elev_gate_factor=3.0, **COMMON
    )
    assert len(x_widen) < len(y_widen)
    assert max(t["length"] for t in x_widen) > max(t["length"] for t in y_widen)
