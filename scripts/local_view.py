"""Local renderer for the 3D SVD power volumes produced by the Modal `volume3d`
function (mb-crr.5 / DIFFVIZ). Runs LOCALLY so Rome can see results.

The 4D beamformed shards (~11GB) stay on Modal; `volume3d` reduces each to a
small (elev, z, x) float32 power volume (~8.5MB) that we pull local. This script
renders orthogonal max-intensity projections (MIPs), and signed baseline-vs-
insight diffs, as PNGs.

Usage:
  # single volume -> 3 ortho MIPs
  python scripts/local_view.py volume viz/baseline/acq_0000_power.npy \
      --axes viz/baseline/acq_0000_axes.npz --out outputs/baseline_acq0.png

  # baseline vs insight -> baseline | insight | signed diff
  python scripts/local_view.py diff viz/baseline/acq_0000_power.npy \
      viz/psf/acq_0000_power.npy --axes viz/baseline/acq_0000_axes.npz \
      --out outputs/diff_psf_acq0.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs, no display needed
import matplotlib.pyplot as plt
import numpy as np


def _load_axes(axes_path: Path | None):
    if axes_path and axes_path.exists():
        d = np.load(axes_path)
        return d["z"], d["elev"], d["x"]
    return None, None, None


def _mips(vol: np.ndarray):
    """Max-intensity projections of an (elev, z, x) volume along each axis."""
    return {
        "coronal (max elev) [z x]": vol.max(axis=0),   # (z, x)
        "sagittal (max x) [elev z]": vol.max(axis=2),   # (elev, z)
        "axial (max z) [elev x]": vol.max(axis=1),      # (elev, x)
    }


def _logn(img: np.ndarray, floor_pct: float = 50.0) -> np.ndarray:
    """Log-compress and normalise to [0,1] for display (vasculature spans orders
    of magnitude). Floor at a low percentile to suppress background."""
    img = np.asarray(img, dtype=np.float64)
    floor = np.percentile(img, floor_pct)
    img = np.clip(img - floor, 0, None)
    img = np.log1p(img)
    m = img.max()
    return img / m if m > 0 else img


def render_volume(power_path: Path, axes_path: Path | None, out_path: Path):
    vol = np.load(power_path)
    mips = _mips(vol)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (title, img) in zip(axes, mips.items()):
        ax.imshow(_logn(img).T, origin="lower", cmap="inferno", aspect="auto")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"SVD power MIPs — {power_path.name}", fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"wrote {out_path}  (volume {vol.shape}, max {vol.max():.3g})")


def render_diff(base_path: Path, insight_path: Path, out_path: Path):
    base = np.load(base_path)
    ins = np.load(insight_path)
    if base.shape != ins.shape:
        raise SystemExit(f"shape mismatch: baseline {base.shape} vs insight {ins.shape}")
    # 3 ortho planes x {baseline, insight, signed diff}. Sagittal/axial carry the
    # depth-structured clutter banding, so show all three (not just coronal).
    planes = {
        "coronal [z x]": (base.max(0).T, ins.max(0).T),
        "sagittal [z elev]": (base.max(2), ins.max(2)),   # (elev,z)->row elev
        "axial [x elev]": (base.max(1), ins.max(1)),
    }
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    for r, (title, (b, i)) in enumerate(planes.items()):
        bn, inn = _logn(b), _logn(i)
        diff = inn - bn
        lim = float(np.abs(diff).max()) or 1.0
        axes[r, 0].imshow(bn, origin="lower", cmap="inferno", aspect="auto")
        axes[r, 0].set_ylabel(title, fontsize=10)
        axes[r, 1].imshow(inn, origin="lower", cmap="inferno", aspect="auto")
        im = axes[r, 2].imshow(diff, origin="lower", cmap="seismic", vmin=-lim, vmax=lim, aspect="auto")
        fig.colorbar(im, ax=axes[r, 2], fraction=0.046)
        if r == 0:
            axes[r, 0].set_title("baseline"); axes[r, 1].set_title("insight")
            axes[r, 2].set_title("insight − baseline (red=more, blue=less)")
        for c in range(3):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    cor = _logn(planes["coronal [z x]"][1]) - _logn(planes["coronal [z x]"][0])
    frac = float((np.abs(cor) > 0.05 * (np.abs(cor).max() or 1)).mean())
    fig.suptitle(f"DIFF (3 planes) — {insight_path.parent.name} vs baseline", fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    print(f"wrote {out_path}  (coronal changed area {frac:.1%})")


def load_bin(path: Path):
    """Parse a tracks_min*.bin (the compact ULM track export)."""
    import struct

    raw = path.read_bytes()
    magic, ver, n_tracks, total_points, max_speed = struct.unpack_from("<IIIIf", raw, 0)
    bmin = struct.unpack_from("<3f", raw, 20)
    bmax = struct.unpack_from("<3f", raw, 32)
    off = 64 + n_tracks * 12  # 64B header + per-track index (offset,length,acq,pad)
    pts = np.frombuffer(raw, dtype="<f4", count=total_points * 5, offset=off).reshape(total_points, 5)
    return pts, np.array(bmin), np.array(bmax), int(n_tracks), float(max_speed)


def _hist(a, b, abins, bbins, bins=300):
    h, _, _ = np.histogram2d(a, b, bins=bins, range=[abins, bbins])
    return h


def render_tracks(bin_path: Path, out_path: Path):
    """ULM track-density MIPs: histogram all track points in 3 ortho planes.
    This is the actual vascular render (vs the SVD power volume)."""
    pts, bmin, bmax, n_tracks, max_speed = load_bin(bin_path)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    xr, yr, zr = [float(bmin[0]), float(bmax[0])], [float(bmin[1]), float(bmax[1])], [float(bmin[2]), float(bmax[2])]
    panels = {
        "coronal [x z]": (_hist(x, z, xr, zr), "x (mm)", "z (mm)"),
        "axial [x y]": (_hist(x, y, xr, yr), "x (mm)", "y (mm)"),
        "sagittal [z y]": (_hist(z, y, zr, yr), "z (mm)", "y (mm)"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (title, (h, xl, yl)) in zip(axes, panels.items()):
        ax.imshow(np.log1p(h).T, origin="lower", cmap="magma", aspect="auto")
        ax.set_title(title, fontsize=10); ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"ULM track density — {bin_path.name} "
                 f"({n_tracks} tracks, {len(pts)} pts)", fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"wrote {out_path}  ({n_tracks} tracks, {len(pts)} points)")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("volume")
    v.add_argument("power"); v.add_argument("--axes", default=None); v.add_argument("--out", required=True)
    d = sub.add_parser("diff")
    d.add_argument("baseline"); d.add_argument("insight")
    d.add_argument("--axes", default=None); d.add_argument("--out", required=True)
    t = sub.add_parser("tracks")
    t.add_argument("bin"); t.add_argument("--out", required=True)
    a = p.parse_args()
    if a.cmd == "volume":
        render_volume(Path(a.power), Path(a.axes) if a.axes else None, Path(a.out))
    elif a.cmd == "tracks":
        render_tracks(Path(a.bin), Path(a.out))
    else:
        render_diff(Path(a.baseline), Path(a.insight), Path(a.out))


if __name__ == "__main__":
    main()
