"""Re-export the reference pickle to match the Aleph blog's s1 track render.

The blog's shipped v6 binary (`outputs/reference/tracks_v6.bin`) renders 2,294
tracks for the "s1" tab. We don't have their render-time filter/resample code
(the front end is unreleased), but we DO have their full tracked population
(`outputs/reference/full_tracks_smoothed.pkl`, 50,456 tracks). This script
empirically reproduces the s1 population size via a (min_length, velocity-cap)
sweep against the pickle, then writes a v3 viewer bundle with the blog's exact
color recipe: jet colormap, fixed 0-38 mm/s "flow speed" scale (38 mm/s @
222.43 Hz frame rate = 0.1709 mm/frame), small points, pure-black background.

Sweep result (see docs/ideation for the write-up): no single min_length cutoff
on tracks_smoothed reproduces 2,294 exactly (closest: length>=28 -> 2,324,
off by 30). Adding the reference's OWN per-frame velocity gate as a second
filter (max raw finite-difference speed <= 0.48 mm/frame, close to the
tracking pipeline's max_distance_mm x/z component of 0.4008) tightens this to
length>=28 & speed<=0.48 -> 2,295 tracks -- one track off the shipped count.
That's the default recipe below.

Usage:
    python3 scripts/export_blogmatch.py \
        outputs/reference/full_tracks_smoothed.pkl \
        outputs/reference/viewer_blogmatch \
        [--min-length 28] [--speed-cap 0.48]
"""
from __future__ import annotations

import argparse
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
VIEWER_HTML = REPO_ROOT / "ultratrace_ulm" / "web" / "track_viewer" / "index_blogstyle.html"

# --- Safe unpickler: numpy-globals-only whitelist (per grounding doc; the
# pickle was opcode-scanned and only uses numpy reconstruction globals). ---
_SAFE = {
    ("numpy._core.multiarray", "_reconstruct"), ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "scalar"), ("numpy.core.multiarray", "scalar"),
    ("numpy", "ndarray"), ("numpy", "dtype"),
}


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        for m in (module, module.replace("numpy._core", "numpy.core"), module.replace("numpy.core", "numpy._core")):
            if (m, name) in _SAFE:
                return super().find_class(m, name)
        raise pickle.UnpicklingError(f"BLOCKED unpickling global: {module}.{name}")


def safe_load_pickle(path: Path) -> dict:
    with path.open("rb") as fp:
        return _SafeUnpickler(fp).load()


def track_max_speed_mm_frame(t: dict) -> float:
    """Raw (unsmoothed) finite-difference speed, mm/frame -- used only for the
    velocity-gate filter, independent of the viewer's display smoothing."""
    pos = np.asarray(t["positions"])
    if len(pos) < 2:
        return 0.0
    frames = np.asarray(t["frames"], dtype=np.float64)
    dp = np.diff(pos, axis=0)
    df = np.diff(frames)
    df[df == 0] = 1.0
    vel = np.linalg.norm(dp / df[:, None], axis=1)
    return float(vel.max())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pickle_path", type=Path)
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--min-length", type=int, default=28)
    ap.add_argument("--speed-cap", type=float, default=0.48, help="mm/frame; None-like via a huge number to disable")
    ap.add_argument("--sigma", type=float, default=10.0, help="display-speed smoothing sigma (matches viewer_ref)")
    args = ap.parse_args()

    data = safe_load_pickle(args.pickle_path)
    all_tracks = data.get("tracks_smoothed") or data["tracks"]
    print(f"loaded {len(all_tracks)} tracks from {args.pickle_path}")

    def _len(t):
        return int(t.get("length", len(t["positions"])))

    kept = [t for t in all_tracks if _len(t) >= args.min_length]
    if args.speed_cap:
        kept = [t for t in kept if track_max_speed_mm_frame(t) <= args.speed_cap]
    print(f"filter min_length>={args.min_length} & max_speed<={args.speed_cap} mm/frame "
          f"-> {len(kept)} tracks (blog s1 shipped: 2,294)")

    from ultratrace_ulm.track_viewer_export import write_tracks_bin_v3

    out_dir = args.output_dir
    write_tracks_bin_v3(kept, out_dir / "data" / "tracks.bin", sigma=args.sigma)

    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(VIEWER_HTML, out_dir / "index.html")
    print(f"wrote blog-match viewer bundle -> {out_dir}")


if __name__ == "__main__":
    main()
