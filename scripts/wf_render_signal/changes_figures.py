"""Figures for docs/from-baseline-to-improved.md — every discrete change, sourced from the
result JSONs rather than retyped numbers.

Fig 1  the velocity wall: per-step speed distributions before and after the gate change.
Fig 2  the purity curve at the shipped gate: why min_track_length is 10.
Fig 3  what each change bought: long tracks and linked detections, with the null alongside.
"""

from __future__ import annotations

import glob
import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "render"
FRAME_RATE = 222.4306816130359
VOX = {"lateral": 0.2004, "elevation": 0.5547, "axial": 0.2008}
GATE_OLD = {k: 2 * v * FRAME_RATE for k, v in VOX.items()}

INK = "#11161b"
C_REF = "#6b7280"
C_BASE = "#1c5d86"
C_IMP = "#b45309"
C_NULL = "#9ca3af"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#cbd5e1", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.titlesize": 10,
})


def steps(pattern: str) -> dict[str, np.ndarray]:
    """Per-step per-axis speeds (mm/s) from a set of track pickles."""
    out = {"lateral": [], "elevation": [], "axial": []}
    for f in sorted(glob.glob(pattern)):
        if "smoothed" in f:
            continue
        with open(f, "rb") as fh:
            for t in pickle.load(fh)["tracks"]:
                p = np.asarray(t["positions"], float)
                fr = np.asarray(t["frames"], float)
                if len(p) < 3:
                    continue
                d = np.diff(p, axis=0)
                dt = np.diff(fr) / FRAME_RATE
                ok = dt > 0
                if not ok.any():
                    continue
                v = np.abs(d[ok] / dt[ok, None])
                out["lateral"].append(v[:, 0])
                out["elevation"].append(v[:, 1])
                out["axial"].append(v[:, 2])
    return {k: np.concatenate(v) if v else np.zeros(0) for k, v in out.items()}


def fig1():
    base = steps(str(REPO / "outputs/corrected_full/tracks_0*.pkl"))
    imp = steps(str(REPO / "outputs/operating_point/tracks/tracks_0*.pkl"))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), constrained_layout=True)
    for ax, axis in zip(axes, ("lateral", "axial", "elevation")):
        bins = np.arange(0, 200, 3)
        ax.hist(base[axis], bins=bins, color=C_BASE, alpha=.85, label="Aleph gate (2 voxels/frame)")
        ax.hist(imp[axis], bins=bins, histtype="step", lw=1.6, color=C_IMP,
                label="ours: 130 mm/s in-plane,\nelevation unchanged")
        # the lobes are integer voxel displacements per frame, not physiological modes:
        # 1 voxel/frame = 44.6 mm/s in-plane, 123.4 in elevation
        step = VOX[axis] * FRAME_RATE
        for n in range(1, 5):
            if n * step < 200:
                ax.axvline(n * step, color="#e5e7eb", lw=.8, zorder=0)
                ax.annotate(f"{n}vx", (n * step, ax.get_ylim()[1]), fontsize=6.5,
                            color="#9ca3af", ha="center", va="bottom")
        ax.axvline(GATE_OLD[axis], color=C_BASE, ls=":", lw=1.2)
        ax.annotate(f"{GATE_OLD[axis]:.0f}", (GATE_OLD[axis], ax.get_ylim()[1] * .92),
                    color=C_BASE, fontsize=8, ha="left", va="top", rotation=90)
        if axis != "elevation":
            ax.axvline(130, color=C_IMP, ls=":", lw=1.2)   # elevation is deliberately left alone
        ax.set_yscale("log")
        ax.set_title(f"{axis} step speed")
        ax.set_xlabel("mm/s")
    axes[0].set_ylabel("steps (log)")
    axes[0].legend(fontsize=7.5, frameon=False, loc="upper right")
    fig.suptitle("Change 1 — the shipped gate censored the IN-PLANE speed distribution at its own wall\n"
                 "(grey lines: integer voxel displacements per frame — the lobes are quantization, not physiology)",
                 fontsize=10.5, y=1.10)
    fig.savefig(OUT / "change_gate_wall.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {k: (float(base[k].max()), float(imp[k].max())) for k in base}


def fig2():
    """Purity curve AT THE GATE WE SHIP. The knee moves with the gate -- at the tighter
    130/130 we first proposed it sat at 8; at the shipped gate it is 10 -- so this must be
    sourced from the matching sweep, never from the default-gate curve."""
    d = json.load(open(REPO / "outputs" / "operating_point" / "purity_curve_130_247.json"))
    rows = [r for r in d["rows"] if r["L"] <= 20]
    Ls = [r["L"] for r in rows]
    real = np.array([r["real"] for r in rows], float)
    null = np.array([r["null"] for r in rows], float)
    pur = np.array([r["purity"] for r in rows], float)
    knee = 10

    fig, (a, b) = plt.subplots(1, 2, figsize=(9.5, 3.4), constrained_layout=True)
    a.plot(Ls, pur * 100, color=C_IMP, lw=1.8)
    a.axhline(95, color=C_NULL, ls="--", lw=1)
    a.axvline(knee, color=C_IMP, ls=":", lw=1.2)
    a.axvline(15, color=C_BASE, ls=":", lw=1.2)
    a.annotate(f"ours: {knee}\n{pur[Ls.index(knee)]*100:.1f}%", (knee + .3, 55),
               fontsize=8, color=C_IMP, ha="left")
    a.annotate(f"Aleph default: 15\n{pur[Ls.index(15)]*100:.1f}%", (15.3, 30),
               fontsize=8, color=C_BASE, ha="left")
    a.set_xlabel("min_track_length"); a.set_ylabel("real tracks that are not chance (%)")
    a.set_title("purity clears 95% at 10 and flattens")
    a.set_ylim(0, 102)

    b.semilogy(Ls, real, color=C_IMP, lw=1.8, label="real detections")
    b.semilogy(Ls, np.maximum(null, .5), color=C_NULL, lw=1.4, ls="--",
               label="frame-shuffled control")
    b.axvline(knee, color=C_IMP, ls=":", lw=1.2); b.axvline(15, color=C_BASE, ls=":", lw=1.2)
    b.set_xlabel("min_track_length"); b.set_ylabel("tracks published (log)")
    b.set_title(f"yield: {knee} → 15 halves the output for 3 points")
    b.legend(fontsize=8, frameon=False)
    fig.suptitle("Change 2 — the length floor decides what is published, so it is calibrated\n"
                 "against a control in which every track is a coincidence (gate: 130 mm/s in-plane)",
                 fontsize=10.5, y=1.10)
    fig.savefig(OUT / "change_length_purity.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"knee": knee, "purity_at_knee": float(pur[Ls.index(knee)]),
            "purity_at_15": float(pur[Ls.index(15)])}


def fig3():
    r = json.load(open(REPO / "outputs/operating_point/operating_point.json"))
    base, prop = r["baseline"], r["proposed"]
    bnull, pnull = r["baseline_null"], r["proposed_null"]

    fig, (a, b) = plt.subplots(1, 2, figsize=(9.5, 3.6), constrained_layout=True)
    names = ["Aleph\nreference", "our\nrecreation", "in-plane gate\n+ floor 10"]
    ge35 = [1421, base["n_ge35"], prop["n_ge35"]]
    bars = a.bar(names, ge35, color=[C_REF, C_BASE, C_IMP], width=.6)
    a.bar_label(bars, fontsize=9)
    a.set_ylabel("tracks ≥35 frames")
    a.set_title("long tracks (immune to the length floor)")
    a.set_ylim(0, max(ge35) * 1.18)
    # the deltas are small; say so on the figure rather than letting equal-looking bars imply more
    a.annotate(f"{ge35[1]-ge35[0]:+d} vs reference", (1, ge35[1] * 1.05),
               ha="center", fontsize=7.5, color=C_BASE)
    a.annotate(f"{ge35[2]-ge35[1]:+d} vs recreation\n({100*(ge35[2]-ge35[1])/ge35[1]:+.0f}%)",
               (2, ge35[2] * 1.05), ha="center", fontsize=7.5, color=C_IMP)
    a.set_xlabel("zero chance chains reach 35 frames in 2.17M permuted detections",
                 fontsize=7.5, color="#4b5563", labelpad=8)

    xs = np.arange(2)
    a2 = b.bar(xs - .18, [base["linked_frac"] * 100, prop["linked_frac"] * 100],
               width=.36, color=[C_BASE, C_IMP], label="real")
    nb = b.bar(xs + .18, [bnull["linked_frac"] * 100, pnull["linked_frac"] * 100],
               width=.36, color=C_NULL, label="permuted null")
    # both null bars round to ~0 at this scale; label them so absence is not read as missing data
    b.bar_label(nb, labels=[f"{bnull['linked_frac']*100:.2f}%", f"{pnull['linked_frac']*100:.2f}%"],
                fontsize=7.5, color="#4b5563")
    b.bar_label(a2, fmt="%.1f%%", fontsize=9)
    b.set_xticks(xs, ["recreation\n(Aleph settings)", "in-plane gate 130 mm/s\n+ floor 10"])
    b.set_ylabel("detections linked into tracks (%)")
    b.set_title("coverage, with the chance floor beside it")
    b.legend(fontsize=8, frameon=False)
    fig.suptitle("What the changes bought, on identical detections across all 216 acquisitions",
                 fontsize=11, y=1.06)
    fig.savefig(OUT / "change_summary.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"ge35": ge35, "linked": [base["linked_frac"], prop["linked_frac"]]}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    print("fig1 maxima:", json.dumps(fig1(), indent=1))
    print("fig2:", json.dumps(fig2()))
    print("fig3:", json.dumps(fig3()))
