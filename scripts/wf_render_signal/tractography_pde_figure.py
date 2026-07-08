"""Diagnostic figure for mb-ki9: Gaussian-baseline vs graph-along-vessel
streamlines, side by side (coronal x-z projection, jet-colored by flow speed).
Visual companion to the split-half numbers (0.61/0.84 -> 0.71/0.95).

Fleet-safe: matplotlib Agg only (NO headless chrome). Run niced, single process.
Output: renders/pde_vs_gaussian_streamlines.png
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from tractography_field import build_field, ORG, SP, GS
from tractography_pde import regularize, streamlines

FD = "outputs/reference/wf/r2/signal/velocity-field/"


def panel(ax, lines, title, vmax):
    ax.set_facecolor("black")
    segs, cols = [], []
    for pts, sps in lines:
        xz = pts[:, [0, 2]]                              # coronal: x (horiz) vs z (depth)
        for a in range(len(xz) - 1):
            segs.append([xz[a], xz[a + 1]]); cols.append(0.5 * (sps[a] + sps[a + 1]))
    lc = LineCollection(segs, cmap="jet", array=np.array(cols), linewidths=0.6, alpha=0.85)
    lc.set_clim(0, vmax)
    ax.add_collection(lc)
    ax.set_xlim(ORG[0], ORG[0] + GS[0] * SP[0]); ax.set_ylim(ORG[2], ORG[2] + GS[2] * SP[2])
    ax.set_aspect("equal"); ax.set_title(title, color="white", fontsize=11)
    ax.tick_params(colors="0.6", labelsize=7)
    for sp in ax.spines.values():
        sp.set_color("0.3")
    return lc


def main():
    S = np.load(FD + "samples.npz")
    V, C = build_field(S["mids"], S["vels"])            # full field (both halves) for display
    Vg, Cg = regularize(V, C, mode="gauss")
    Vr, Cr = regularize(V, C, mode="graph", gamma=1.5, iters=200, dt=0.12, eps_iso=0.2)
    lg = streamlines(Vg, Cg, nseed=4000)
    lr = streamlines(Vr, Cr, nseed=4000)
    len_g = [np.linalg.norm(np.diff(p, axis=0), axis=1).sum() for p, _ in lg]
    len_r = [np.linalg.norm(np.diff(p, axis=0), axis=1).sum() for p, _ in lr]
    print(f"gauss: {len(lg)} streamlines, median {np.median(len_g):.1f}mm, max {np.max(len_g):.1f}mm")
    print(f"graph: {len(lr)} streamlines, median {np.median(len_r):.1f}mm, max {np.max(len_r):.1f}mm")
    vmax = float(np.percentile([s for _, sps in lr for s in sps], 98))

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), facecolor="black")
    panel(axes[0], lg, f"Isotropic Gaussian  (split-half Dice 0.61 / cos 0.84)\n"
                       f"{len(lg)} lines · median {np.median(len_g):.0f} · max {np.max(len_g):.0f} mm", vmax)
    lc = panel(axes[1], lr, f"Graph along-vessel  (Dice 0.71 / cos 0.95)\n"
                            f"{len(lr)} lines · median {np.median(len_r):.0f} · max {np.max(len_r):.0f} mm", vmax)
    cb = fig.colorbar(lc, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("flow speed (field units)", color="white"); cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cb.ax, "yticklabels"), color="white")
    fig.suptitle("mb-ki9  ·  along-vessel field regularization vs isotropic Gaussian  ·  acq-0 velocity field",
                 color="white", fontsize=12)
    os.makedirs("renders", exist_ok=True)
    fig.savefig("renders/pde_vs_gaussian_streamlines.png", dpi=140, facecolor="black", bbox_inches="tight")
    print("wrote renders/pde_vs_gaussian_streamlines.png")


if __name__ == "__main__":
    main()
