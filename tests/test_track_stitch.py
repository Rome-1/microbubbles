"""Local test for post-hoc track stitching (agreed idea #1)."""
import numpy as np

from ultratrace_ulm.track_stitch import stitch_tracks


def _track(p0, vel, f0, n, inten=5.0):
    fr = np.arange(f0, f0 + n, dtype=np.float32)
    pos = (np.asarray(p0, float)[None, :] + np.outer(np.arange(n), vel)).astype(np.float32)
    return {"positions": pos, "frames": fr, "intensities": np.full(n, inten, np.float32),
            "length": n}


def test_rejoins_fragments():
    # One straight bubble path split into 3 fragments with 2-3 frame gaps.
    vel = np.array([0.05, 0.0, 0.05])  # moves in x,z; flat elevation
    a = _track([0, 0, 8], vel, 0, 8)
    b = _track([0 + vel[0] * 11, 0, 8 + vel[2] * 11], vel, 11, 8)   # gap 3
    c = _track([0 + vel[0] * 21, 0, 8 + vel[2] * 21], vel, 21, 8)   # gap 2
    out = stitch_tracks([a, b, c], max_gap=12)
    assert len(out) == 1, f"expected 1 stitched track, got {len(out)}"
    assert out[0]["length"] == 24
    assert out[0].get("stitched_from") == 3
    # frames strictly increasing after merge
    fr = out[0]["frames"]
    assert np.all(np.diff(fr) > 0)


def test_does_not_merge_distinct_bubbles():
    # Two bubbles far apart, opposite directions — must NOT stitch.
    a = _track([0, 0, 8], [0.05, 0, 0.05], 0, 8)
    b = _track([20, 0, 30], [-0.05, 0, -0.05], 11, 8)
    out = stitch_tracks([a, b], max_gap=12)
    assert len(out) == 2


def test_respects_gap_cap():
    vel = np.array([0.05, 0.0, 0.05])
    a = _track([0, 0, 8], vel, 0, 8)
    far = _track([0 + vel[0] * 40, 0, 8 + vel[2] * 40], vel, 40, 8)  # gap 32 > cap
    out = stitch_tracks([a, far], max_gap=12)
    assert len(out) == 2


def test_velocity_disagreement_blocks_stitch():
    # Aligned in space/time but moving in opposite directions at the seam.
    a = _track([0, 0, 8], [0.1, 0, 0], 0, 8)
    b = _track([0.1 * 9, 0, 8], [-0.1, 0, 0], 10, 8)
    out = stitch_tracks([a, b], max_gap=12)
    assert len(out) == 2
