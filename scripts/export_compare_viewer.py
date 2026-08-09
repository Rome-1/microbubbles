"""Export the Aleph reference tracks and our full 216-acquisition recreation
into one A/B 3D viewer bundle.

Writes three v3 ``tracks.bin`` payloads (same packer the single-dataset viewer
uses, ``ultratrace_ulm.track_viewer_export.write_tracks_bin_v3``) plus the
comparison page:

    data/reference.bin        reference tracks, length >= --min-length (15)
    data/recreation.bin       our tracks,       length >= --min-length (15)
    data/reference_min5.bin   reference at its released length >= 5 floor

The default view is >=15-vs->=15: the reference release filtered at
``min_track_length: 5`` while our run used 15, so comparing the shipped
reference against our output would credit the reference with thousands of
5-point stubs we never emitted. ``reference_min5.bin`` is kept as a third,
clearly-labelled dataset so the released number is still inspectable.

Frame bookkeeping: reference tracks carry globally-numbered frames
(acq 215 starts at 51600 = 215 x 240). Our per-batch pickles restart at 0 in
every file, so each batch's frames are shifted by ``start_acq * frames_per_acq``
before packing -- otherwise the viewer's chronological reveal sweep would
replay all 18 batches on top of each other.

Only the un-smoothed ``tracks`` key is read from every pickle. The CLI also
wrote ``tracks_XXXX_smoothed.pkl`` next to each batch; those are the same
trajectories and loading both double-counts, so this script globs the exact
``tracks_[0-9]*.pkl`` shape and refuses anything unexpected.

Usage:
    python3 scripts/export_compare_viewer.py \
        --reference outputs/reference/full_tracks_smoothed.pkl \
        --recreation-dir outputs/corrected_full \
        --output-dir renders/compare_viewer
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ultratrace_ulm.runtime import load_pickle  # noqa: E402
from ultratrace_ulm.track_viewer_export import write_tracks_bin_v3  # noqa: E402

VIEWER_HTML = REPO_ROOT / "ultratrace_ulm" / "web" / "track_viewer" / "compare.html"
BATCH_RE = re.compile(r"^tracks_(\d{4})\.pkl$")


def track_length(t: dict) -> int:
    return int(t.get("length", len(t["positions"])))


def summarize(name: str, tracks: list) -> dict:
    lengths = np.array([track_length(t) for t in tracks])
    return {
        "dataset": name,
        "n_tracks": int(lengths.size),
        "n_tracks_ge35": int((lengths >= 35).sum()),
        "mean_length": round(float(lengths.mean()), 2),
        "total_points": int(lengths.sum()),
    }


def load_reference(path: Path, min_length: int) -> list:
    data = load_pickle(path)
    tracks = [t for t in data["tracks"] if track_length(t) >= min_length]
    print(f"reference: {len(tracks)} tracks with length >= {min_length} ({path.name})")
    return tracks


def load_recreation(directory: Path, min_length: int) -> list:
    """Concatenate the per-batch pickles, shifting each batch's frames global.

    Batch file ``tracks_BBBB.pkl`` holds acquisitions BBBB..BBBB+11 with frames
    restarting at 0, so the shift is ``BBBB * frames_per_acq``.
    """
    batches = sorted(p for p in directory.iterdir() if BATCH_RE.match(p.name))
    if not batches:
        raise SystemExit(f"no tracks_XXXX.pkl batches in {directory}")

    tracks: list = []
    for path in batches:
        start_acq = int(BATCH_RE.match(path.name).group(1))
        data = load_pickle(path)
        frames_per_acq = int(data.get("frames_per_acq") or 240)
        offset = start_acq * frames_per_acq
        kept = 0
        for t in data["tracks"]:
            if track_length(t) < min_length:
                continue
            shifted = dict(t)
            shifted["frames"] = np.asarray(t["frames"], dtype=np.float64) + offset
            tracks.append(shifted)
            kept += 1
        print(f"  {path.name}: acqs {start_acq}-{start_acq + 11}, +{offset} frames, {kept} tracks")
    print(f"recreation: {len(tracks)} tracks with length >= {min_length} "
          f"from {len(batches)} batches")
    return tracks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", type=Path,
                    default=REPO_ROOT / "outputs" / "reference" / "full_tracks_smoothed.pkl")
    ap.add_argument("--recreation-dir", type=Path,
                    default=REPO_ROOT / "outputs" / "corrected_full")
    ap.add_argument("--output-dir", type=Path,
                    default=REPO_ROOT / "renders" / "compare_viewer")
    ap.add_argument("--min-length", type=int, default=15,
                    help="fair-comparison length floor applied to BOTH datasets")
    ap.add_argument("--reference-floor", type=int, default=5,
                    help="the reference release's own min_track_length, exported separately")
    args = ap.parse_args()

    out_dir = args.output_dir
    data_dir = out_dir / "data"

    ref_all = load_reference(args.reference, args.reference_floor)
    ref_fair = [t for t in ref_all if track_length(t) >= args.min_length]
    rec = load_recreation(args.recreation_dir, args.min_length)

    stats = [
        summarize(f"reference (>={args.min_length})", ref_fair),
        summarize(f"recreation (>={args.min_length})", rec),
        summarize(f"reference (>={args.reference_floor}, as released)", ref_all),
    ]

    write_tracks_bin_v3(ref_fair, data_dir / "reference.bin")
    write_tracks_bin_v3(rec, data_dir / "recreation.bin")
    write_tracks_bin_v3(ref_all, data_dir / "reference_min5.bin")

    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(VIEWER_HTML, out_dir / "index.html")
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")

    print()
    for s in stats:
        print(f"{s['dataset']:<42} tracks={s['n_tracks']:>7,}  >=35={s['n_tracks_ge35']:>6,}  "
              f"mean_len={s['mean_length']:>6.2f}  points={s['total_points']:>9,}")
    print(f"\nwrote comparison viewer bundle -> {out_dir}")


if __name__ == "__main__":
    main()
