"""Thin CLI for the ULM benchmark-metrics module (ultratrace_ulm.bench).

Compares two-or-more track .bin exports with self-contained, no-ground-truth
track-quality + render-quality proxies. Prints a markdown table and writes the
full comparison as JSON next to the first input file.

Usage:
  python scripts/bench.py outputs/viz/baseline6_min5.bin \
      outputs/viz/b6gd_kpred_min5.bin --labels baseline,kalman_pred

These metrics are PROXIES for human decision support, not proof of correctness.
More tracks is not automatically better; longer + straighter + higher-contrast
density is generally better for ULM. Treat large deltas as flags, not verdicts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as `python scripts/bench.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultratrace_ulm.bench import compare, format_table  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bins", nargs="+", help="track .bin export files to compare")
    p.add_argument("--labels", default=None,
                   help="comma-separated labels, one per .bin (default: filenames)")
    p.add_argument("--bins-2d", type=int, default=256,
                   help="histogram bins per axis for density metrics (default 256)")
    p.add_argument("--out", default=None,
                   help="JSON output path (default: <first_bin>.bench.json)")
    args = p.parse_args(argv)

    paths = args.bins
    if args.labels:
        labels = [s.strip() for s in args.labels.split(",")]
        if len(labels) != len(paths):
            p.error(f"got {len(paths)} bins but {len(labels)} labels")
    else:
        labels = [Path(b).stem for b in paths]

    comparison = compare(paths, labels, bins=args.bins_2d)

    table = format_table(comparison)
    print(table)

    out_path = Path(args.out) if args.out else Path(paths[0]).with_suffix(".bench.json")
    out_path.write_text(json.dumps(comparison, indent=2, default=float) + "\n")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
