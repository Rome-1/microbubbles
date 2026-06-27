"""Objective benchmark metrics for a 3D ULM track pipeline (no ground truth).

This module computes *self-contained* track-quality and render-quality measures
from the compact ``.bin`` track exports, so an "insight" run can be compared
quantitatively against a "baseline" run. We have NO ground-truth vasculature,
so every number here is a PROXY, not a measure of correctness.

JUDGEMENT NOTES (read before trusting any single number):
  * These are PROXIES, not truth. Nothing here proves a run is "correct".
  * More tracks is NOT necessarily better. A large jump in track count can mean
    real recovered microvessels, but it can equally mean clutter/false tracks.
    A ~15x track explosion is almost always clutter, not signal -- flag it.
  * For ULM, what is generally *good*: longer tracks, straighter per-track paths
    (each bubble pass through a vessel segment is short and roughly linear),
    and a higher-contrast / more structured density render (vessels stand out
    from a diffuse background).
  * The comparison table is DECISION SUPPORT for a human, not a verdict. Use it
    to quantify and localize differences and to flag suspicious changes.

Pure numpy. No cupy / h5py / GPU. Safe to run locally.
"""

from __future__ import annotations

import struct
from typing import Sequence

import numpy as np

# Byte layout (must match ultratrace_ulm.tracking.export_tracks_bin, version 4):
#   header  "<IIIIf3f3fIIf"  ->  magic, version, n_tracks, total_points,
#           max_speed, bounds_min(3), bounds_max(3), frames_per_acq, n_acqs,
#           frame_rate_hz   (56 bytes, zero-padded to 64)
#   index   n_tracks * "<IIHH"  ->  (point_offset, length, acq_index, pad)  12B
#   points  total_points * 5 float32  ->  [x, y, z, frame, speed]  (x/y/z in mm)
_HEADER_FMT = "<IIIIf3f3fIIf"
_HEADER_PAD = 64
_INDEX_FMT = "<IIHH"
_INDEX_SIZE = 12
_MAGIC = 0x554C4D54


def load_tracks_bin(path) -> dict:
    """Parse a ``.bin`` track export into a flat numpy dict.

    Returns a dict with:
      positions      (N, 3) float32, mm
      frames         (N,)   float32, frame index of each point
      speeds         (N,)   float32, per-point speed (mm/s) as stored in the file
      track_lengths  (n_tracks,) int, points per track
      track_offsets  (n_tracks,) int, start index of each track into positions
      track_acqs     (n_tracks,) int, acquisition index of each track
      n_tracks, total_points, n_acqs, frames_per_acq, frame_rate
      bounds_min, bounds_max  (3,) float32, mm
    """
    with open(path, "rb") as fp:
        buf = fp.read()

    (magic, version, n_tracks, total_points, max_speed,
     bmin_x, bmin_y, bmin_z, bmax_x, bmax_y, bmax_z,
     frames_per_acq, n_acqs, frame_rate) = struct.unpack_from(_HEADER_FMT, buf, 0)
    if magic != _MAGIC:
        raise ValueError(f"{path}: bad magic 0x{magic:08x} (expected 0x{_MAGIC:08x})")

    index_off = _HEADER_PAD
    points_off = _HEADER_PAD + n_tracks * _INDEX_SIZE

    # Per-track index: (offset, length, acq, pad). Read as a structured view.
    idx = np.frombuffer(
        buf, dtype=np.dtype([("offset", "<u4"), ("length", "<u4"),
                             ("acq", "<u2"), ("pad", "<u2")]),
        count=n_tracks, offset=index_off,
    )
    track_offsets = idx["offset"].astype(np.int64)
    track_lengths = idx["length"].astype(np.int64)
    track_acqs = idx["acq"].astype(np.int64)

    pts = np.frombuffer(buf, dtype="<f4", count=total_points * 5,
                        offset=points_off).reshape(total_points, 5)

    return {
        "positions": np.ascontiguousarray(pts[:, 0:3]),
        "frames": np.ascontiguousarray(pts[:, 3]),
        "speeds": np.ascontiguousarray(pts[:, 4]),
        "track_lengths": track_lengths,
        "track_offsets": track_offsets,
        "track_acqs": track_acqs,
        "n_tracks": int(n_tracks),
        "total_points": int(total_points),
        "n_acqs": int(n_acqs),
        "frames_per_acq": int(frames_per_acq),
        "frame_rate": float(frame_rate),
        "max_speed": float(max_speed),
        "bounds_min": np.array([bmin_x, bmin_y, bmin_z], dtype=np.float32),
        "bounds_max": np.array([bmax_x, bmax_y, bmax_z], dtype=np.float32),
    }


def _per_track_geometry(data: dict):
    """Compute per-track curvilinear length, end-to-end distance, and pooled
    per-segment speeds (mm/s). Returns (curv_len, e2e_dist, seg_speeds).

    curv_len / e2e_dist are (n_tracks,) arrays in mm. seg_speeds is a 1D pooled
    array over all interior segments (mm/s), derived from positions + frames +
    frame_rate: speed = |dp| * frame_rate / max(dframe, 1).
    """
    positions = data["positions"]
    frames = data["frames"]
    offsets = data["track_offsets"]
    lengths = data["track_lengths"]
    frame_rate = data["frame_rate"] or 1.0

    n = len(offsets)
    curv_len = np.zeros(n, dtype=np.float64)
    e2e_dist = np.zeros(n, dtype=np.float64)
    seg_speed_chunks = []
    for i in range(n):
        o = int(offsets[i])
        L = int(lengths[i])
        if L < 2:
            continue
        p = positions[o:o + L]
        f = frames[o:o + L]
        d = np.diff(p, axis=0)
        seg = np.linalg.norm(d, axis=1)
        curv_len[i] = seg.sum()
        e2e_dist[i] = float(np.linalg.norm(p[-1] - p[0]))
        df = np.diff(f).astype(np.float64)
        df[df == 0] = 1.0
        seg_speed_chunks.append(seg * frame_rate / df)

    seg_speeds = (np.concatenate(seg_speed_chunks)
                  if seg_speed_chunks else np.zeros(0, dtype=np.float64))
    return curv_len, e2e_dist, seg_speeds


def track_metrics(data: dict) -> dict:
    """Self-contained (no ground truth) track-quality proxies.

    Longer + straighter tracks are generally better for ULM, but raw track
    COUNT is ambiguous (more can be signal or clutter) -- interpret deltas with
    the judgement notes in this module's docstring.
    """
    lengths = data["track_lengths"].astype(np.float64)
    n_tracks = int(data["n_tracks"])
    total_points = int(data["total_points"])
    n_acqs = max(int(data["n_acqs"]), 1)

    curv_len, e2e_dist, seg_speeds = _per_track_geometry(data)

    # Length-distribution stats. Fraction of tracks at/above each length gate is
    # the cleanest "are tracks long?" proxy (longer = better persistence).
    if n_tracks:
        len_mean = float(lengths.mean())
        len_median = float(np.median(lengths))
        len_p90 = float(np.percentile(lengths, 90))
        frac_ge = {f"frac_len_ge_{g}": float((lengths >= g).mean())
                   for g in (10, 20, 35, 50)}
    else:
        len_mean = len_median = len_p90 = 0.0
        frac_ge = {f"frac_len_ge_{g}": 0.0 for g in (10, 20, 35, 50)}

    # Fragmentation proxy: many short tracks relative to long ones suggests the
    # tracker is breaking real paths into fragments (or producing junk stubs).
    n_short = int((lengths < 10).sum())
    n_long = int((lengths >= 20).sum())
    frag_ratio = float(n_short / n_long) if n_long else float("inf") if n_short else 0.0

    # Straightness proxy: end-to-end / curvilinear, averaged over tracks with
    # nonzero path length. ~1.0 = straight; lower = more wandering/curling. A
    # short bubble pass through a vessel segment should be fairly straight, so
    # higher is generally cleaner (very low can indicate noisy/looping tracks).
    valid = curv_len > 0
    straightness = float((e2e_dist[valid] / curv_len[valid]).mean()) if valid.any() else 0.0

    return {
        "n_tracks": n_tracks,
        "total_points": total_points,
        "tracks_per_acq": float(n_tracks / n_acqs),
        "mean_points_per_track": float(total_points / n_tracks) if n_tracks else 0.0,
        "track_len_mean": len_mean,
        "track_len_median": len_median,
        "track_len_p90": len_p90,
        **frac_ge,
        "mean_curvilinear_len_mm": float(curv_len.mean()) if n_tracks else 0.0,
        "median_curvilinear_len_mm": float(np.median(curv_len)) if n_tracks else 0.0,
        "mean_speed_mm_s": float(seg_speeds.mean()) if seg_speeds.size else 0.0,
        "median_speed_mm_s": float(np.median(seg_speeds)) if seg_speeds.size else 0.0,
        "frag_short_over_long": frag_ratio,
        "n_short_lt10": n_short,
        "n_long_ge20": n_long,
        "straightness": straightness,
    }


def density_metrics(data: dict, bins: int = 256) -> dict:
    """Render-quality proxies from the coronal (x vs z) track-density histogram.

    The coronal plane (x horizontal, z depth) is the canonical ULM render view.
    We bin all track points into a 2D count histogram over the data bounds, then
    derive proxies for how "structured" / vessel-like the render is:

      occupied_fraction : nonzero bins / total bins. Lower can mean tighter
          concentration on vessels; very low can mean sparse/undersampled.
      contrast_cnr_proxy: 99th-percentile bin count / median NONZERO bin count.
          A contrast/CNR stand-in -- higher means bright structures sit well
          above the typical occupied background (vessels pop out).
      entropy_bits      : Shannon entropy (base 2) of the normalized density
          over occupied bins. Lower-ish entropy = density concentrated on a few
          structures; higher = diffuse/smeared. Reported with max_entropy_bits
          (= log2(#occupied bins)) for scale.

    All proxies; none proves the render is correct.
    """
    positions = data["positions"]
    x = positions[:, 0]
    z = positions[:, 2]
    bmin = data["bounds_min"]
    bmax = data["bounds_max"]
    xr = [float(bmin[0]), float(bmax[0])]
    zr = [float(bmin[2]), float(bmax[2])]
    # Guard against degenerate (zero-width) ranges.
    if xr[1] <= xr[0]:
        xr[1] = xr[0] + 1.0
    if zr[1] <= zr[0]:
        zr[1] = zr[0] + 1.0

    h, _, _ = np.histogram2d(x, z, bins=bins, range=[xr, zr])
    total_bins = int(h.size)
    nonzero = h[h > 0]
    n_occ = int(nonzero.size)
    occupied_fraction = float(n_occ / total_bins) if total_bins else 0.0

    if n_occ:
        p99 = float(np.percentile(h, 99))
        med_nz = float(np.median(nonzero))
        contrast = float(p99 / med_nz) if med_nz > 0 else float("inf")
        p = nonzero / nonzero.sum()
        entropy = float(-(p * np.log2(p)).sum())
        max_entropy = float(np.log2(n_occ)) if n_occ > 1 else 0.0
    else:
        contrast = 0.0
        entropy = 0.0
        max_entropy = 0.0

    return {
        "n_bins": int(bins),
        "occupied_bins": n_occ,
        "occupied_fraction": occupied_fraction,
        "peak_density": float(h.max()) if h.size else 0.0,
        "contrast_cnr_proxy": contrast,
        "entropy_bits": entropy,
        "max_entropy_bits": max_entropy,
        "entropy_ratio": float(entropy / max_entropy) if max_entropy > 0 else 0.0,
    }


def compare(paths: Sequence[str], labels: Sequence[str], bins: int = 256) -> dict:
    """Run track_metrics + density_metrics on each path; return a structured
    comparison: {"labels": [...], "paths": [...], "runs": {label: {...}}}.
    """
    if len(paths) != len(labels):
        raise ValueError(f"got {len(paths)} paths but {len(labels)} labels")
    runs = {}
    for path, label in zip(paths, labels):
        data = load_tracks_bin(path)
        runs[label] = {
            "path": str(path),
            "track": track_metrics(data),
            "density": density_metrics(data, bins=bins),
        }
    return {"labels": list(labels), "paths": [str(p) for p in paths], "runs": runs}


# Ordered (metric_key, human_label, group) for the table. Keeping it explicit
# controls row order and lets us pick the metrics worth eyeballing.
_TABLE_ROWS = [
    ("n_tracks", "n_tracks", "track"),
    ("total_points", "total_points", "track"),
    ("tracks_per_acq", "tracks_per_acq", "track"),
    ("mean_points_per_track", "mean_points_per_track", "track"),
    ("track_len_mean", "track_len_mean", "track"),
    ("track_len_median", "track_len_median", "track"),
    ("track_len_p90", "track_len_p90", "track"),
    ("frac_len_ge_10", "frac_len>=10", "track"),
    ("frac_len_ge_20", "frac_len>=20", "track"),
    ("frac_len_ge_35", "frac_len>=35", "track"),
    ("frac_len_ge_50", "frac_len>=50", "track"),
    ("mean_curvilinear_len_mm", "mean_curv_len_mm", "track"),
    ("median_curvilinear_len_mm", "median_curv_len_mm", "track"),
    ("mean_speed_mm_s", "mean_speed_mm_s", "track"),
    ("median_speed_mm_s", "median_speed_mm_s", "track"),
    ("frag_short_over_long", "frag_short/long", "track"),
    ("straightness", "straightness", "track"),
    ("occupied_fraction", "occupied_fraction", "density"),
    ("peak_density", "peak_density", "density"),
    ("contrast_cnr_proxy", "contrast_cnr_proxy", "density"),
    ("entropy_bits", "entropy_bits", "density"),
    ("entropy_ratio", "entropy_ratio", "density"),
]


def _fmt(v) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if v == float("inf"):
        return "inf"
    if v == float("-inf"):
        return "-inf"
    av = abs(v)
    if av != 0 and (av >= 1e5 or av < 1e-3):
        return f"{v:.3e}"
    if av >= 100:
        return f"{v:.1f}"
    return f"{v:.4g}"


def format_table(comparison: dict) -> str:
    """Render a comparison (from `compare`) as a markdown table.

    Rows = metrics, columns = labels, final column = delta of the LAST label vs
    the FIRST (last - first). Delta is decision support: the sign/size flags
    where the runs differ -- it does not say which run is "right".
    """
    labels = comparison["labels"]
    runs = comparison["runs"]
    first, last = labels[0], labels[-1]

    header = ["metric"] + list(labels)
    if len(labels) >= 2:
        header.append(f"delta ({last}-{first})")
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]

    for key, name, group in _TABLE_ROWS:
        cells = [name]
        vals = []
        for lab in labels:
            v = runs[lab][group].get(key, float("nan"))
            vals.append(v)
            cells.append(_fmt(v))
        if len(labels) >= 2:
            try:
                delta = vals[-1] - vals[0]
                cells.append(_fmt(delta))
            except (TypeError, ValueError):
                cells.append("")
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)
