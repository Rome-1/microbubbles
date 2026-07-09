"""Does the multi-vector fix restore reference-like flow diversity? Render the
multi-vector short-streamlet reconstruction next to the reference tracks (both
direction-hue) and report orientation coherence cl. Fleet-safe matplotlib Agg.
Output: renders/multivector_vs_reference.png
"""
from __future__ import annotations
import sys, os, pickle
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tractography_field import GS, SP, ORG
from tractography_multivector import build_multivector, mv_streamlets
from flow_diversity import segdirs_from_streamlines, coherence_map
from render_flow_diversity import panel, SU

FD = "outputs/reference/wf/r2/signal/velocity-field/"
REF = "outputs/reference/full_tracks_smoothed.pkl"


def main():
    S = np.load(FD + "samples.npz")
    modes, weights, cnt = build_multivector(S["mids"], S["vels"], ang_deg=30, min_frac=0.15, min_count=5)
    sl = mv_streamlets(modes, weights, cnt, half_steps=7)
    fm, fd = segdirs_from_streamlines([(p, None) for p in sl])
    cl, c2 = coherence_map(fm, fd); o = c2 >= 4
    mv_cl = float(cl[o].mean()); mv_cx = float((cl[o] < 0.4).mean())
    print(f"multi-vector streamlets: n={len(sl)}  mean_cl={mv_cl:.3f} crossing_frac={mv_cx:.3f} "
          f"(field was 0.99/0.00; reference 0.53/0.35)")

    o2 = SU(open(REF, "rb")).load()
    tr = o2["tracks_smoothed"]
    refpolys = [np.asarray(t["positions"], float) for t in tr if int(t.get("length", len(t["positions"]))) >= 12]
    rng = np.random.default_rng(0)
    if len(refpolys) > 6000:
        refpolys = [refpolys[i] for i in rng.choice(len(refpolys), 6000, replace=False)]

    fig, ax = plt.subplots(1, 2, figsize=(15, 6.2), facecolor="black")
    panel(ax[0], refpolys, f"Aleph reference tracks  (cl≈0.53, crossings 35%)\ncolored by in-plane direction")
    panel(ax[1], sl, f"Our MULTI-VECTOR streamlets  (cl={mv_cl:.2f}, crossings {mv_cx*100:.0f}%)\n"
                     f"{len(sl)} short per-mode primitives · colored by direction")
    fig.suptitle("Multi-vector fix vs reference · recovering crossing/diverse flows (was cl 0.99, combed)",
                 color="white", fontsize=12.5)
    os.makedirs("renders", exist_ok=True)
    fig.savefig("renders/multivector_vs_reference.png", dpi=140, facecolor="black", bbox_inches="tight")
    print("wrote renders/multivector_vs_reference.png")


if __name__ == "__main__":
    main()
