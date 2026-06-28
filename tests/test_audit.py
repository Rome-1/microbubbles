from __future__ import annotations

import numpy as np

from ultratrace_ulm.audit import coverage_audit, track_positions_to_indices


def test_track_positions_to_indices_with_grid():
    elev = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    z = np.array([10.0, 11.0, 12.0, 13.0], dtype=np.float32)
    x = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float32)
    yy, zz, xx = np.meshgrid(elev, z, x, indexing="ij")
    pos = np.array([[1.01, 0.1, 11.9]], dtype=np.float32)  # tracking order (x,y,z)
    idx = track_positions_to_indices(pos, (3, 4, 5), grid_x=xx, grid_y=yy, grid_z=zz)
    assert np.array_equal(idx[0], np.array([1, 2, 2]))


def test_coverage_audit_counts_accepted_and_rejected():
    beamformed = np.ones((3, 2, 4, 5), dtype=np.complex64)
    filtered = np.ones((3, 2, 4, 5), dtype=np.float32) * 0.5
    detections = [
        (np.array([[0, 1, 1], [1, 2, 2]], dtype=np.int32), np.ones(2, dtype=np.float32), np.ones(2, dtype=np.float32)),
        (np.array([[1, 2, 3]], dtype=np.int32), np.ones(1, dtype=np.float32), np.ones(1, dtype=np.float32)),
        (np.empty((0, 3), dtype=np.int32), np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)),
    ]
    tracks = [
        {
            "positions": np.array([[0, 1, 1], [1, 2, 3]], dtype=np.float32),
            "frames": np.array([0, 1], dtype=np.int32),
        }
    ]
    audit = coverage_audit(beamformed, filtered, detections, tracks, n_z_blocks=2, n_x_blocks=2, overlap=0.0)
    summary = audit["summary"]
    assert summary["total_detections"] == 3
    assert summary["accepted_detections"] == 2
    assert summary["rejected_detections"] == 1
    assert np.isclose(summary["post_svd_energy_fraction"], 0.25)
    assert audit["maps"]["detection_density"].shape == (2, 4, 5)
    assert audit["blocks"]["detection_density"].shape == (2, 2)
