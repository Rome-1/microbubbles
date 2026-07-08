"""mb-4k2 composite render: ONE figure, four channels the reference blog's
speed-only point cloud cannot show, all on the graph-regularized (mb-ki9) field:

  A  STRUCTURE + SPEED   streamlines jet-colored by flow speed (blog-equivalent)
  B  FLOW DIRECTION      streamlines colored by 3D orientation (DTI-style RGB:
                         red=lateral x, green=elevation, blue=depth z)
  C  ARTERY / VEIN       validated antiparallel-adjacency A/V voxels (red/blue)
                         over faint vasculature (split-half-reproducible pairs)
  D  PERFUSION / FLUX    speed x count projected along elevation (throughput map)

Fleet-safe: matplotlib Agg only (NO headless chrome). Run niced, single process.
Output: renders/composite_science.png
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
AV = "outputs/reference/wf/r2/signal/artery-vein/av_labels.npy"
XLIM = (ORG[0], ORG[0] + GS[0] * SP[0])
ZLIM = (ORG[2], ORG[2] + GS[2] * SP[2])


def _frame(ax, title):
    ax.set_facecolor("black"); ax.set_xlim(*XLIM); ax.set_ylim(*ZLIM)
    ax.set_aspect("equal"); ax.set_title(title, color="white", fontsize=10.5, pad=6)
    ax.tick_params(colors="0.55", labelsize=6.5)
    for s in ax.spines.values():
        s.set_color("0.3")


def _segments(lines):
    """Flatten polylines to (segments, midpoint-speed, unit-tangent) in x-z."""
    segs, spd, tan = [], [], []
    for pts, sps in lines:
        xz = pts[:, [0, 2]]
        t = np.diff(pts, axis=0); tn = t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-9)
        for a in range(len(xz) - 1):
            segs.append([xz[a], xz[a + 1]]); spd.append(0.5 * (sps[a] + sps[a + 1])); tan.append(tn[a])
    return segs, np.array(spd), np.array(tan)


def panel_speed(ax, segs, spd, vmax):
    _frame(ax, "A · structure + flow speed  (blog-equivalent readout)")
    lc = LineCollection(segs, cmap="jet", array=spd, linewidths=0.55, alpha=0.85); lc.set_clim(0, vmax)
    ax.add_collection(lc); return lc


def panel_direction(ax, segs, tan):
    _frame(ax, "B · flow DIRECTION  (R=lateral · G=elevation · B=depth)")
    rgb = np.clip(np.abs(tan[:, [0, 1, 2]] if tan.shape[1] == 3 else tan), 0, 1)
    lc = LineCollection(segs, colors=rgb, linewidths=0.6, alpha=0.9); ax.add_collection(lc)


def panel_av(ax, segs, lbl):
    _frame(ax, "C · artery / vein  (validated antiparallel pairs · red=artery blue=vein)")
    ax.add_collection(LineCollection(segs, colors=(0.35, 0.35, 0.35), linewidths=0.3, alpha=0.5))
    xs = ORG[0] + (np.arange(GS[0]) + 0.5) * SP[0]; zs = ORG[2] + (np.arange(GS[2]) + 0.5) * SP[2]
    for val, col in ((1, "#ff3838"), (2, "#3aa0ff")):
        ii, _, kk = np.where(lbl == val)
        ax.scatter(xs[ii], zs[kk], s=9, c=col, alpha=0.9, edgecolors="none",
                   label="artery" if val == 1 else "vein")
    lg = ax.legend(loc="upper right", fontsize=7, framealpha=0.3, labelcolor="white")
    lg.get_frame().set_facecolor("black")


def panel_flux(ax, V, C):
    _frame(ax, "D · perfusion / flux  (speed x count, projected through elevation)")
    flux = (np.linalg.norm(V, axis=-1) * C).sum(axis=1).T          # (z, x) after sum over elev
    im = ax.imshow(flux, origin="lower", extent=[*XLIM, *ZLIM], aspect="equal",
                   cmap="inferno", vmax=np.percentile(flux[flux > 0], 99))
    return im


def main():
    S = np.load(FD + "samples.npz")
    V, C = build_field(S["mids"], S["vels"])
    Vr, Cr = regularize(V, C, mode="graph", gamma=1.5, iters=200, dt=0.12, eps_iso=0.2)
    lines = streamlines(Vr, Cr, nseed=4500)
    segs, spd, tan = _segments(lines)
    lbl = np.load(AV)
    vmax = float(np.percentile(spd, 98))
    print(f"{len(lines)} streamlines, {len(segs)} segments; A/V voxels {(lbl>0).sum()}")

    fig, ax = plt.subplots(2, 2, figsize=(15, 11), facecolor="black")
    lc = panel_speed(ax[0, 0], segs, spd, vmax)
    panel_direction(ax[0, 1], segs, tan)
    panel_av(ax[1, 0], segs, lbl)
    im = panel_flux(ax[1, 1], Vr, Cr)
    fig.colorbar(lc, ax=ax[0, 0], fraction=0.046, pad=0.02).set_label("speed", color="0.8")
    fig.colorbar(im, ax=ax[1, 1], fraction=0.046, pad=0.02).set_label("flux (a.u.)", color="0.8")
    for cb in fig.axes[-2:]:
        cb.yaxis.set_tick_params(color="0.7"); plt.setp(cb.get_yticklabels(), color="0.8")
    fig.suptitle("3D-ULM composite · graph-regularized velocity field (mb-ki9) · acq-0 · "
                 "structure + 3 readouts the reference lacks", color="white", fontsize=12.5, y=0.98)
    os.makedirs("renders", exist_ok=True)
    fig.savefig("renders/composite_science.png", dpi=140, facecolor="black", bbox_inches="tight")
    print("wrote renders/composite_science.png")


if __name__ == "__main__":
    main()
