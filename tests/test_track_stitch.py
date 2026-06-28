import numpy as np
import pytest

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


def test_parallel_nearby_distinct_bubbles_do_not_false_merge():
    # Distinct bubbles can be nearly indistinguishable in x/z over a dropout gap.
    # This is the specific false-merge risk: the loose elevation tolerance accepts
    # a separate but parallel bubble and inflates track length.
    vel = np.array([0.05, 0.0, 0.05])
    a = _track([0, 0.0, 8], vel, 0, 8)
    predicted_after_gap = np.asarray([0, 0.0, 8]) + vel * 11
    b = _track([predicted_after_gap[0], 1.5, predicted_after_gap[2]], vel, 11, 8)
    out = stitch_tracks([a, b], max_gap=12)
    assert len(out) == 2


def test_crossing_fragments_keep_velocity_consistent_links():
    # Two bubbles cross near the seam. Position-only stitching would allow an ID
    # swap, but the velocity-agreement gate should reject the swapped links.
    vel_a = np.array([0.1, 0.0, 0.0])
    vel_b = np.array([-0.1, 0.0, 0.0])
    a0 = _track([0.0, 0.0, 8.0], vel_a, 0, 8)
    b0 = _track([1.4, 0.0, 8.0], vel_b, 0, 8)
    a1 = _track([1.0, 0.0, 8.0], vel_a, 10, 8)
    b1 = _track([0.4, 0.0, 8.0], vel_b, 10, 8)

    out = stitch_tracks([a0, b0, a1, b1], max_gap=12)

    assert len(out) == 2
    assert sorted(t.get("stitched_from", 1) for t in out) == [2, 2]
    velocities = []
    for track in out:
        pos = np.asarray(track["positions"])
        frames = np.asarray(track["frames"])
        velocities.append((pos[-1, 0] - pos[0, 0]) / (frames[-1] - frames[0]))
    assert sorted(np.sign(v) for v in velocities) == [-1.0, 1.0]
