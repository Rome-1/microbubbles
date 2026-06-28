"""Tests for the intensity-aware association cost in kalman_tracking_3d.

A real microbubble's echo intensity is roughly continuous frame-to-frame, so
brightness is a free association cue. When ``intensity_cost_weight > 0`` (and
per-detection intensities are supplied), the Hungarian cost gains a term
``weight * |log(cand+eps) - log(track_ref+eps)|`` where ``track_ref`` is a
running EMA of the track's matched intensities. This keeps a track matched to
detections of similar brightness at geometric ambiguities (e.g. crossings),
reducing identity swaps. ``intensity_cost_weight = 0.0`` (default) must
reproduce the geometry-only tracker byte-for-byte.
"""

import numpy as np

from ultratrace_ulm.tracking import kalman_tracking_3d

GATE = (1.0, 1.0, 1.0)
COMMON = dict(max_distance_mm=GATE, max_gap=3, min_track_length=3, max_cost=1e5)

BRIGHT = 10.0
DIM = 1.0


def _make_crossing(n=24, Y=0.15, vx=0.25, z0=8.0, noise=0.05, seed=3):
    """Two bubbles crossing in elevation (y) while advancing in x.

    Bubble P (bright, intensity 10) sweeps y from +Y down to -Y; bubble Q (dim,
    intensity 1) sweeps y from -Y up to +Y. Both advance in x at ``vx`` per
    frame, so the dominant motion is along x and the (small, noisy) y-velocity
    is hard to extrapolate through the crossing. The detection order is held
    fixed as [P, Q] every frame, so any identity swap comes from the assignment
    logic, not detection ordering. With position ``noise`` the geometry-only
    cost becomes ambiguous near the crossing and the tracker swaps the two
    identities back and forth; the brightness gap (10 vs 1) is a clean cue to
    keep them apart.
    """
    rng = np.random.default_rng(seed)
    dets, ints = [], []
    for f in range(n):
        t = f / (n - 1)
        yP = Y - 2 * Y * t
        yQ = -Y + 2 * Y * t
        x = vx * f
        nP = rng.normal(0.0, noise, 3) if noise else np.zeros(3)
        nQ = rng.normal(0.0, noise, 3) if noise else np.zeros(3)
        P = [x + nP[0], yP + nP[1], z0 + nP[2]]
        Q = [x + nQ[0], yQ + nQ[1], z0 + nQ[2]]
        dets.append(np.array([P, Q], dtype=np.float64))
        ints.append(np.array([BRIGHT, DIM], dtype=np.float64))
    return dets, ints


def _intensity_flips(tracks):
    """Total number of brightness midpoint-crossings within tracks.

    Each crossing of the bright/dim midpoint inside a single output track is an
    identity swap: the track jumped from following one bubble to the other.
    """
    mid = 0.5 * (BRIGHT + DIM)
    total = 0
    for t in tracks:
        it = t.get("intensities")
        if it is None or len(it) < 2:
            continue
        side = np.asarray(it) >= mid
        total += int(np.sum(side[1:] != side[:-1]))
    return total


# ---------------------------------------------------------------------------
# Backward compatibility: weight 0.0 must reproduce the geometry-only tracker.
# ---------------------------------------------------------------------------

def test_backward_compat_weight_zero_identical_no_intensities():
    """weight=0.0 with no intensities == default call, byte-for-byte."""
    dets, _ = _make_crossing()
    base = kalman_tracking_3d(dets, **COMMON)
    same = kalman_tracking_3d(dets, intensity_cost_weight=0.0, **COMMON)
    assert len(base) == len(same)
    for a, b in zip(base, same):
        assert a["length"] == b["length"]
        assert np.array_equal(a["positions"], b["positions"])
        assert np.array_equal(a["frames"], b["frames"])


def test_backward_compat_weight_zero_identical_with_intensities():
    """weight=0.0 with intensities supplied == default call, byte-for-byte.

    The intensity-cost path and the EMA reference are gated entirely behind
    ``intensity_cost_weight > 0``, so providing intensities at weight 0 cannot
    change the geometry-only result (positions, frames, or carried intensities).
    """
    dets, ints = _make_crossing()
    base = kalman_tracking_3d(dets, intensities=ints, **COMMON)
    same = kalman_tracking_3d(dets, intensities=ints, intensity_cost_weight=0.0, **COMMON)
    assert len(base) == len(same)
    for a, b in zip(base, same):
        assert np.array_equal(a["positions"], b["positions"])
        assert np.array_equal(a["frames"], b["frames"])
        assert np.array_equal(a["intensities"], b["intensities"])


# ---------------------------------------------------------------------------
# Effect: a positive weight reduces identity swaps at the crossing.
# ---------------------------------------------------------------------------

def test_intensity_cost_reduces_identity_swaps():
    """Geometry alone swaps the two crossing bubbles; the intensity term keeps
    each track locked onto its own brightness."""
    dets, ints = _make_crossing()
    geo = kalman_tracking_3d(dets, intensities=ints, intensity_cost_weight=0.0, **COMMON)
    itn = kalman_tracking_3d(dets, intensities=ints, intensity_cost_weight=5.0, **COMMON)

    geo_flips = _intensity_flips(geo)
    itn_flips = _intensity_flips(itn)

    # Sanity: the scenario actually bites -- geometry-only swaps identities.
    assert geo_flips > 0
    # The intensity cue removes the swaps entirely.
    assert itn_flips == 0
    assert itn_flips < geo_flips


def test_intensity_cost_keeps_bright_track_bright():
    """With the intensity cost on, each surviving track stays on one brightness
    for its whole length (no mid-track brightness jumps)."""
    dets, ints = _make_crossing()
    itn = kalman_tracking_3d(dets, intensities=ints, intensity_cost_weight=5.0, **COMMON)

    # Exactly the two bubbles, each a clean single-brightness track.
    assert len(itn) == 2
    mid = 0.5 * (BRIGHT + DIM)
    sides = sorted(bool(np.mean(t["intensities"]) >= mid) for t in itn)
    # One bright track, one dim track.
    assert sides == [False, True]
    for t in itn:
        it = np.asarray(t["intensities"])
        # No track mixes bright and dim detections.
        assert np.all(it >= mid) or np.all(it < mid)


# ---------------------------------------------------------------------------
# Degenerate: no intensities -> weight is silently ignored, no crash.
# ---------------------------------------------------------------------------

def test_weight_ignored_when_no_intensities():
    """A positive weight with intensities=None is a no-op (cannot compute an
    intensity term) and must match the geometry-only run exactly."""
    dets, _ = _make_crossing()
    geo = kalman_tracking_3d(dets, intensity_cost_weight=0.0, **COMMON)
    weighted = kalman_tracking_3d(dets, intensity_cost_weight=7.0, **COMMON)
    assert len(geo) == len(weighted)
    for a, b in zip(geo, weighted):
        assert np.array_equal(a["positions"], b["positions"])
        assert np.array_equal(a["frames"], b["frames"])
