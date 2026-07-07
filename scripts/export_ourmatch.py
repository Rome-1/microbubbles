"""Re-filter OUR composite v3 tracks.bin with the reference's recipe, then
bundle it with the blog-style jet/0-38mm/s viewer, for an apples-to-apples
comparison against outputs/reference/viewer_blogmatch/.

We don't have a local pickle for the composite pipeline run (it only ran on
Modal); but the v3 export already stores raw per-point positions + frames
(unsmoothed), so we can recompute the reference's velocity gate directly from
the shipped tracks.bin without needing the source pickle. Recipe (per the
grounding doc, docs/ideation/20-reference-artifacts-received.md): min_length
~=25 + a velocity cap of ~0.40 mm/frame to drop teleport links (our tracker's
gate is looser than the reference's `max_distance_mm` cap, so a fraction of
"long" tracks are actually short real tracks bridged by a a non-physiological
jump -- this is the single largest known contributor to our render's haze,
per docs/ideation/13-track-length-gap.md and 14-overlinking-speed-gap.md).

Usage:
    python3 scripts/export_ourmatch.py \
        renders/track_viewer_composite/data/tracks.bin \
        outputs/reference/viewer_ourmatch \
        [--min-length 25] [--speed-cap 0.40]
"""
from __future__ import annotations

import argparse
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
VIEWER_HTML = REPO_ROOT / "ultratrace_ulm" / "web" / "track_viewer" / "index_blogstyle.html"

MAGIC = 0x554C4D54
VERSION = 3


def read_tracks_bin_v3(path: Path):
    """Parse a v3 tracks.bin into a list of per-track point arrays (unsmoothed
    positions/frames as originally exported, plus the display-speed & intensity
    columns already baked in by the original exporter)."""
    raw = path.read_bytes()
    magic, version, n_tracks, total_points, _max_speed = struct.unpack_from("<IIIIf", raw, 0)
    if magic != MAGIC:
        raise SystemExit(f"bad magic 0x{magic:x} in {path}")
    if version != VERSION:
        raise SystemExit(f"expected v3, got v{version} in {path}")

    table_off = 64
    table = np.frombuffer(raw, dtype="<u4", count=n_tracks * 2, offset=table_off).reshape(n_tracks, 2)
    point_off = table_off + n_tracks * 8
    pts = np.frombuffer(raw, dtype="<f4", count=total_points * 6, offset=point_off).reshape(total_points, 6)

    tracks = []
    for offset, length in table:
        block = pts[offset:offset + length]
        tracks.append({
            "positions": block[:, 0:3].copy(),
            "frames": block[:, 3].copy(),
            "speed": block[:, 4].copy(),        # already-smoothed display speed, mm/frame
            "intensity": block[:, 5].copy(),
            "length": int(length),
        })
    print(f"read {n_tracks} tracks ({total_points} points) from {path}")
    return tracks


def raw_max_speed_mm_frame(t: dict) -> float:
    pos, frames = t["positions"], t["frames"]
    if len(pos) < 2:
        return 0.0
    dp = np.diff(pos, axis=0)
    df = np.diff(frames.astype(np.float64))
    df[df == 0] = 1.0
    return float(np.linalg.norm(dp / df[:, None], axis=1).max())


def write_filtered_v3(tracks: list, output_path: Path):
    """Pack pre-computed (position, frame, speed, intensity) columns straight
    through -- no re-smoothing -- so this is a pure re-filter of the original
    composite export, not a re-derivation."""
    n_tracks = len(tracks)
    total_points = sum(t["length"] for t in tracks)
    all_mins = np.min([t["positions"].min(axis=0) for t in tracks], axis=0).astype(np.float32)
    all_maxs = np.max([t["positions"].max(axis=0) for t in tracks], axis=0).astype(np.float32)
    max_speed = max(float(t["speed"].max()) if len(t["speed"]) else 0.0 for t in tracks)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as fp:
        header = struct.pack("<IIIIf3f3f", MAGIC, VERSION, n_tracks, total_points,
                              max_speed, *all_mins.tolist(), *all_maxs.tolist())
        fp.write(header + b"\x00" * (64 - len(header)))
        offset = 0
        for t in tracks:
            fp.write(struct.pack("<II", offset, t["length"]))
            offset += t["length"]
        for t in tracks:
            point_data = np.column_stack([t["positions"], t["frames"][:, None],
                                           t["speed"][:, None], t["intensity"][:, None]])
            fp.write(point_data.astype(np.float32).tobytes())
    print(f"wrote {n_tracks} tracks ({total_points} points) -> {output_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source_bin", type=Path)
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--min-length", type=int, default=25)
    ap.add_argument("--speed-cap", type=float, default=0.40)
    args = ap.parse_args()

    tracks = read_tracks_bin_v3(args.source_bin)
    kept = [t for t in tracks if t["length"] >= args.min_length]
    kept = [t for t in kept if raw_max_speed_mm_frame(t) <= args.speed_cap]
    print(f"filter min_length>={args.min_length} & max_speed<={args.speed_cap} mm/frame "
          f"-> {len(kept)}/{len(tracks)} tracks kept")

    out_dir = args.output_dir
    write_filtered_v3(kept, out_dir / "data" / "tracks.bin")
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(VIEWER_HTML, out_dir / "index.html")
    print(f"wrote our-match viewer bundle -> {out_dir}")


if __name__ == "__main__":
    main()
