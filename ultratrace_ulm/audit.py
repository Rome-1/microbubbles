"""Per-stage coverage audit utilities for the 3D ULM pipeline."""

from __future__ import annotations

import numpy as np

from .svd_region import _block_bounds


def _energy_map(volume: np.ndarray) -> np.ndarray:
    data = np.asarray(volume)
    if data.ndim == 4:
        return np.mean(np.abs(data) ** 2, axis=0).astype(np.float32, copy=False)
    if data.ndim == 3:
        return (np.abs(data) ** 2).astype(np.float32, copy=False)
    raise ValueError(f"Expected 3D or 4D volume, got {data.shape}")


def _density_from_detections(
    detections: list,
    shape: tuple[int, int, int],
) -> tuple[np.ndarray, int, list[np.ndarray]]:
    density = np.zeros(shape, dtype=np.int32)
    coords_by_frame: list[np.ndarray] = []
    total = 0
    for item in detections:
        spatial = item[0] if isinstance(item, tuple) else item
        coords = np.asarray(spatial, dtype=np.int32)
        if coords.size == 0:
            coords = np.empty((0, 3), dtype=np.int32)
        coords_by_frame.append(coords)
        if len(coords) == 0:
            continue
        valid = np.all((coords >= 0) & (coords < np.asarray(shape)[None, :]), axis=1)
        c = coords[valid]
        np.add.at(density, (c[:, 0], c[:, 1], c[:, 2]), 1)
        total += int(len(c))
    return density, total, coords_by_frame


def _axis_indices_from_grid(values: np.ndarray, axis_values: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis_values, dtype=np.float64)
    vals = np.asarray(values, dtype=np.float64)
    if len(axis) < 2:
        return np.zeros(vals.shape, dtype=np.int32)
    ascending = axis[-1] >= axis[0]
    work_axis = axis if ascending else axis[::-1]
    idx = np.searchsorted(work_axis, vals)
    idx = np.clip(idx, 1, len(work_axis) - 1)
    left = work_axis[idx - 1]
    right = work_axis[idx]
    choose_right = np.abs(vals - right) < np.abs(vals - left)
    out = idx - 1 + choose_right.astype(np.int32)
    if not ascending:
        out = len(axis) - 1 - out
    return out.astype(np.int32, copy=False)


def track_positions_to_indices(
    positions: np.ndarray,
    shape: tuple[int, int, int],
    *,
    grid_x: np.ndarray | None = None,
    grid_y: np.ndarray | None = None,
    grid_z: np.ndarray | None = None,
) -> np.ndarray:
    """Convert track positions to ``(elev,z,x)`` indices.

    Without grids, positions are assumed to already be index-like. With grids,
    positions are assumed to be tracking coordinates ``(x,y,z)`` in millimetres.
    """
    pos = np.asarray(positions, dtype=np.float32)
    if pos.size == 0:
        return np.empty((0, 3), dtype=np.int32)
    if grid_x is None or grid_y is None or grid_z is None:
        return np.rint(pos).astype(np.int32, copy=False)
    x_axis = np.asarray(grid_x[0, 0, :], dtype=np.float64)
    y_axis = np.asarray(grid_y[:, 0, 0], dtype=np.float64)
    z_axis = np.asarray(grid_z[0, :, 0], dtype=np.float64)
    xi = _axis_indices_from_grid(pos[:, 0], x_axis)
    ei = _axis_indices_from_grid(pos[:, 1], y_axis)
    zi = _axis_indices_from_grid(pos[:, 2], z_axis)
    out = np.stack([ei, zi, xi], axis=1)
    return np.clip(out, 0, np.asarray(shape, dtype=np.int32) - 1).astype(np.int32, copy=False)


def _match_tracks_to_detections(
    coords_by_frame: list[np.ndarray],
    tracks: list[dict],
    shape: tuple[int, int, int],
    *,
    grid_x: np.ndarray | None = None,
    grid_y: np.ndarray | None = None,
    grid_z: np.ndarray | None = None,
    tolerance_vox: float = 1.5,
) -> tuple[np.ndarray, int]:
    accepted = np.zeros(shape, dtype=np.int32)
    used = [np.zeros(len(c), dtype=bool) for c in coords_by_frame]
    matched = 0
    tol2 = float(tolerance_vox) ** 2
    for track in tracks:
        positions = np.asarray(track.get("positions", []), dtype=np.float32)
        frames = np.asarray(track.get("frames", []), dtype=np.int64)
        if len(positions) == 0 or len(frames) == 0:
            continue
        idx_pos = track_positions_to_indices(
            positions,
            shape,
            grid_x=grid_x,
            grid_y=grid_y,
            grid_z=grid_z,
        )
        for pos_idx, frame_idx in zip(idx_pos, frames):
            if frame_idx < 0 or frame_idx >= len(coords_by_frame):
                continue
            dets = coords_by_frame[int(frame_idx)]
            if len(dets) == 0:
                continue
            free = ~used[int(frame_idx)]
            if not np.any(free):
                continue
            free_idx = np.where(free)[0]
            diff = dets[free_idx].astype(np.float32) - pos_idx[None, :].astype(np.float32)
            dist2 = np.sum(diff * diff, axis=1)
            best = int(np.argmin(dist2))
            if float(dist2[best]) <= tol2:
                det_i = int(free_idx[best])
                used[int(frame_idx)][det_i] = True
                coord = dets[det_i]
                accepted[coord[0], coord[1], coord[2]] += 1
                matched += 1
    return accepted, matched


def _block_reduce_sum(map3: np.ndarray, n_z_blocks: int, n_x_blocks: int, overlap: float) -> np.ndarray:
    if map3.ndim == 3:
        spatial = map3.sum(axis=0)
    elif map3.ndim == 2:
        spatial = map3
    else:
        raise ValueError(f"Expected 2D/3D map, got {map3.shape}")
    z_bounds = _block_bounds(spatial.shape[0], n_z_blocks, overlap)
    x_bounds = _block_bounds(spatial.shape[1], n_x_blocks, overlap)
    out = np.zeros((len(z_bounds), len(x_bounds)), dtype=np.float64)
    for iz, (za, zb) in enumerate(z_bounds):
        for ix, (xa, xb) in enumerate(x_bounds):
            out[iz, ix] = float(spatial[za:zb, xa:xb].sum())
    return out


def coverage_audit(
    beamformed: np.ndarray,
    filtered_magnitude: np.ndarray,
    detections: list,
    tracks: list[dict],
    *,
    grid_x: np.ndarray | None = None,
    grid_y: np.ndarray | None = None,
    grid_z: np.ndarray | None = None,
    n_z_blocks: int = 3,
    n_x_blocks: int = 3,
    overlap: float = 0.25,
    accepted_tolerance_vox: float = 1.5,
) -> dict:
    """Report where coverage is lost across beamform/SVD/detect/track stages."""
    beam_energy = _energy_map(beamformed)
    svd_energy = _energy_map(filtered_magnitude)
    if beam_energy.shape != svd_energy.shape:
        raise ValueError(f"Energy map shape mismatch: {beam_energy.shape} vs {svd_energy.shape}")
    density, total_detections, coords_by_frame = _density_from_detections(detections, beam_energy.shape)
    accepted_density, accepted = _match_tracks_to_detections(
        coords_by_frame,
        tracks,
        beam_energy.shape,
        grid_x=grid_x,
        grid_y=grid_y,
        grid_z=grid_z,
        tolerance_vox=accepted_tolerance_vox,
    )
    rejected = max(0, int(total_detections) - int(accepted))
    rejected_density = density - np.minimum(density, accepted_density)
    beam_total = float(beam_energy.sum())
    svd_total = float(svd_energy.sum())
    occupied = int(np.count_nonzero(density))
    spatial_voxels = int(np.prod(density.shape))

    return {
        "summary": {
            "beamformed_energy": beam_total,
            "post_svd_energy": svd_total,
            "post_svd_energy_fraction": svd_total / beam_total if beam_total > 0 else float("nan"),
            "total_detections": int(total_detections),
            "accepted_detections": int(accepted),
            "rejected_detections": int(rejected),
            "acceptance_fraction": accepted / total_detections if total_detections else float("nan"),
            "occupied_voxels": occupied,
            "occupied_fraction": occupied / spatial_voxels if spatial_voxels else float("nan"),
            "n_tracks": int(len(tracks)),
            "track_points": int(sum(len(t.get("frames", [])) for t in tracks)),
        },
        "maps": {
            "beamformed_energy": beam_energy,
            "post_svd_energy": svd_energy,
            "detection_density": density,
            "accepted_density": accepted_density,
            "rejected_density": rejected_density,
        },
        "blocks": {
            "beamformed_energy": _block_reduce_sum(beam_energy, n_z_blocks, n_x_blocks, overlap),
            "post_svd_energy": _block_reduce_sum(svd_energy, n_z_blocks, n_x_blocks, overlap),
            "detection_density": _block_reduce_sum(density, n_z_blocks, n_x_blocks, overlap),
            "accepted_density": _block_reduce_sum(accepted_density, n_z_blocks, n_x_blocks, overlap),
            "rejected_density": _block_reduce_sum(rejected_density, n_z_blocks, n_x_blocks, overlap),
        },
    }
