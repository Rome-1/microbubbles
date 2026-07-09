"""Side-by-side render exposing the flow-DIVERSITY gap (Rome's observation):
reference tracks (richly varied flows) vs our graph-field streamlines (long parallel).

Colors each segment by its IN-PLANE direction (x-z angle -> hue). Diverse/crossing
vasculature shows many hues intermixed; a combed parallel field shows locally-uniform
hue. Coronal x-z projection, matched view. Fleet-safe: matplotlib Agg, no chrome.
Output: renders/flow_diversity_ref_vs_ours.png
"""
from __future__ import annotations
import sys, os, pickle
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import hsv_to_rgb
from tractography_field import build_field, ORG, SP, GS
from tractography_pde import regularize, streamlines

FD = "outputs/reference/wf/r2/signal/velocity-field/"
REF = "outputs/reference/full_tracks_smoothed.pkl"
XLIM = (ORG[0], ORG[0] + GS[0] * SP[0]); ZLIM = (ORG[2], ORG[2] + GS[2] * SP[2])
SAFE = {("numpy._core.multiarray", "_reconstruct"), ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "scalar"), ("numpy.core.multiarray", "scalar"),
        ("numpy", "ndarray"), ("numpy", "dtype")}
class SU(pickle.Unpickler):
    def find_class(self, m, n):
        for mm in (m, m.replace("numpy._core", "numpy.core"), m.replace("numpy.core", "numpy._core")):
            if (mm, n) in SAFE:
                return super().find_class(mm, n)
        raise pickle.UnpicklingError(m + "." + n)


def polylines_to_segs(polys):
    segs, cols = [], []
    for p in polys:
        xz = p[:, [0, 2]]
        d = np.diff(p, axis=0)
        ang = (np.arctan2(d[:, 2], d[:, 0]) % np.pi) / np.pi           # 0..1 hue from x-z orientation
        for a in range(len(xz) - 1):
            segs.append([xz[a], xz[a + 1]])
            cols.append(hsv_to_rgb([ang[a], 0.85, 1.0]))
    return segs, cols


def panel(ax, polys, title):
    ax.set_facecolor("black")
    segs, cols = polylines_to_segs(polys)
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=0.5, alpha=0.8))
    ax.set_xlim(*XLIM); ax.set_ylim(*ZLIM); ax.set_aspect("equal")
    ax.set_title(title, color="white", fontsize=11)
    ax.tick_params(colors="0.55", labelsize=7)
    for s in ax.spines.values():
        s.set_color("0.3")


def main():
    o = SU(open(REF, "rb")).load()
    tr = o["tracks_smoothed"]
    # match the blog's length-filtered subset: longer tracks only
    refpolys = [np.asarray(t["positions"], float) for t in tr if int(t.get("length", len(t["positions"]))) >= 12]
    # cap for render weight
    if len(refpolys) > 6000:
        rng = np.random.default_rng(0); refpolys = [refpolys[i] for i in rng.choice(len(refpolys), 6000, replace=False)]
    S = np.load(FD + "samples.npz")
    V, C = build_field(S["mids"], S["vels"])
    Vr, Cr = regularize(V, C, mode="graph", gamma=1.5, iters=200, dt=0.12, eps_iso=0.2)
    ourpolys = [p for p, _ in streamlines(Vr, Cr, nseed=4000, min_len_mm=4.0)]
    print(f"reference (len>=12): {len(refpolys)} tracks;  ours: {len(ourpolys)} streamlines")

    fig, ax = plt.subplots(1, 2, figsize=(15, 6.2), facecolor="black")
    panel(ax[0], refpolys, f"Aleph reference tracks  ({len(refpolys)}, len≥12 frames)\ncolored by in-plane flow direction")
    panel(ax[1], ourpolys, f"Our graph-field streamlines  ({len(ourpolys)})\ncolored by in-plane flow direction")
    fig.suptitle("Flow diversity: reference (varied, crossing flows) vs ours (long parallel) · hue = x-z orientation",
                 color="white", fontsize=12.5)
    # direction-hue key
    cax = fig.add_axes([0.46, 0.02, 0.08, 0.03]); grad = hsv_to_rgb(np.stack([np.linspace(0,1,180), np.ones(180)*0.85, np.ones(180)], 1))[None]
    cax.imshow(grad, aspect="auto"); cax.set_xticks([0,179]); cax.set_xticklabels(["0°","180°"], color="0.7", fontsize=7); cax.set_yticks([])
    os.makedirs("renders", exist_ok=True)
    fig.savefig("renders/flow_diversity_ref_vs_ours.png", dpi=140, facecolor="black", bbox_inches="tight")
    print("wrote renders/flow_diversity_ref_vs_ours.png")


if __name__ == "__main__":
    main()
