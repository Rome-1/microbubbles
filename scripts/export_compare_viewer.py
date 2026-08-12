"""Export the Aleph reference, our 216-acquisition recreation, and our improved
operating point into one three-way 3D comparison viewer bundle.

Writes three v3 ``tracks.bin`` payloads (same packer the single-dataset viewer
uses, ``ultratrace_ulm.track_viewer_export.write_tracks_bin_v3``) plus a
``meta.json`` and the comparison page:

    data/reference.bin    Aleph reference,      native floor len >= 5
    data/recreation.bin   our 216-acq redo,     native floor len >= 15
    data/improved.bin     our operating point,  native floor len >= 8

Each dataset is packed once, at its OWN native floor. The viewer applies any
higher floor in the shader from the per-track length it already reads out of the
track table, so the matched-floor view, the native-floor view and the >=35 view
are the same geometry seen through three cutoffs -- not three different exports
that could drift apart.

Why the floors matter: the three runs published at different ``min_track_length``
settings (5, 15, 8). Drawn at their native floors the improved run looks denser
than the recreation partly because it simply emits shorter tracks, which is a
publishing choice and not a signal difference. So the viewer DEFAULTS to a
matched floor of >=15 across all three; the native-floor view is available but
labelled as not like-for-like; and the >=35 subset -- immune to every floor
here -- is the headline number.

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
        --improved-dir outputs/operating_point/tracks \
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

MATCHED_FLOOR = 15
HEADLINE_FLOOR = 35


def track_length(t: dict) -> int:
    return int(t.get("length", len(t["positions"])))


def summarize(tracks: list, floor: int) -> dict:
    """Counts for the subset of ``tracks`` at or above ``floor``."""
    lengths = np.array([track_length(t) for t in tracks], dtype=np.int64)
    lengths = lengths[lengths >= floor]
    if lengths.size == 0:
        return {"floor": floor, "n_tracks": 0, "n_tracks_ge35": 0,
                "mean_length": 0.0, "total_points": 0}
    return {
        "floor": floor,
        "n_tracks": int(lengths.size),
        "n_tracks_ge35": int((lengths >= HEADLINE_FLOOR).sum()),
        "mean_length": round(float(lengths.mean()), 2),
        "total_points": int(lengths.sum()),
    }


def load_reference(path: Path, min_length: int) -> list:
    data = load_pickle(path)
    tracks = [t for t in data["tracks"] if track_length(t) >= min_length]
    print(f"reference: {len(tracks)} tracks with length >= {min_length} ({path.name})")
    return tracks


def load_batched(name: str, directory: Path, min_length: int) -> list:
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
    print(f"{name}: {len(tracks)} tracks with length >= {min_length} "
          f"from {len(batches)} batches")
    return tracks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", type=Path,
                    default=REPO_ROOT / "outputs" / "reference" / "full_tracks_smoothed.pkl")
    ap.add_argument("--recreation-dir", type=Path,
                    default=REPO_ROOT / "outputs" / "corrected_full")
    ap.add_argument("--improved-dir", type=Path,
                    default=REPO_ROOT / "outputs" / "operating_point" / "tracks")
    ap.add_argument("--output-dir", type=Path,
                    default=REPO_ROOT / "renders" / "compare_viewer")
    ap.add_argument("--reference-floor", type=int, default=5,
                    help="the Aleph release's own min_track_length")
    ap.add_argument("--recreation-floor", type=int, default=15,
                    help="the recreation run's own min_track_length")
    ap.add_argument("--improved-floor", type=int, default=8,
                    help="the improved operating point's own min_track_length")
    ap.add_argument("--matched-floor", type=int, default=MATCHED_FLOOR,
                    help="like-for-like floor applied to ALL THREE in the default view")
    args = ap.parse_args()

    out_dir = args.output_dir
    data_dir = out_dir / "data"

    loaders = [
        ("reference", "REFERENCE (Aleph)", args.reference_floor,
         lambda: load_reference(args.reference, args.reference_floor),
         None),
        ("recreation", "RECREATION (ours, 216 acq)", args.recreation_floor,
         lambda: load_batched("recreation", args.recreation_dir, args.recreation_floor),
         99.5),
        ("improved", "IMPROVED (physical gate)", args.improved_floor,
         lambda: load_batched("improved", args.improved_dir, args.improved_floor),
         95.2),
    ]

    meta = {
        "matched_floor": args.matched_floor,
        "headline_floor": HEADLINE_FLOOR,
        "datasets": [],
    }

    for key, label, native_floor, load, purity in loaders:
        tracks = load()
        write_tracks_bin_v3(tracks, data_dir / f"{key}.bin")
        entry = {
            "key": key,
            "label": label,
            "file": f"data/{key}.bin",
            "native_floor": native_floor,
            # Purity: fraction of tracks surviving a frame-permutation null test.
            # Not measured for the reference release -- we only have its output.
            "purity_pct": purity,
            "purity_note": (
                None if purity is None else
                "vs. frame-permutation null; 0 null chains reached 35 frames"
            ),
            "stats": {
                "native": summarize(tracks, native_floor),
                "matched": summarize(tracks, args.matched_floor),
                "headline": summarize(tracks, HEADLINE_FLOOR),
            },
        }
        meta["datasets"].append(entry)
        print()

    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(VIEWER_HTML, out_dir / "index.html")
    (data_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    # stats.json kept as the flat, human-readable roll-up.
    (out_dir / "stats.json").write_text(json.dumps(meta, indent=2) + "\n")

    hdr = f"{'dataset':<28}{'floor':>7}{'tracks':>10}{'>=35':>8}{'mean':>8}{'points':>11}"
    for view in ("matched", "native", "headline"):
        print(f"\n--- {view} floor ---")
        print(hdr)
        for d in meta["datasets"]:
            s = d["stats"][view]
            print(f"{d['label']:<28}{s['floor']:>7}{s['n_tracks']:>10,}"
                  f"{s['n_tracks_ge35']:>8,}{s['mean_length']:>8.2f}{s['total_points']:>11,}")
    print(f"\nwrote three-way comparison viewer bundle -> {out_dir}")


if __name__ == "__main__":
    main()
