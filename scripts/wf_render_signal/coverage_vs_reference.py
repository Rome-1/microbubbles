"""mb-ska: quantitative coverage of our field-tractography vs the reference
tracks, on a shared grid, against a spatial null.

Turns "our vessels match the reference" into a hard, GT-free number:
  * Rasterize all 50,456 reference smoothed tracks onto our coarsen-2 grid ->
    reference occupancy R.
  * Rasterize our streamline occupancy O (Gaussian baseline vs mb-ki9 graph field).
  * Report Dice(O,R), recall = |O&R|/|R|, precision = |O&R|/|O|.
  * NULL: 200 random 3-D circular shifts of O (preserve O's own spatial
    structure/volume, destroy alignment with R) -> null Dice distribution.
    Report observed Dice percentile + z vs null. A circular-shift null is
    stricter than a uniform-random null because it keeps O's clustering.

Fleet-safe: numpy only, no chrome. Run niced, single process.
Output: outputs/reference/wf/r2/signal/coverage_vs_reference.json
"""
from __future__ import annotations
import sys, os, json, pickle
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy.ndimage import binary_dilation
from tractography_field import build_field, GS, SP, ORG, smooth_field, tractogram
from tractography_pde import regularize

FD = "outputs/reference/wf/r2/signal/velocity-field/"
REF = "outputs/reference/full_tracks_smoothed.pkl"
SAFE = {("numpy._core.multiarray", "_reconstruct"), ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "scalar"), ("numpy.core.multiarray", "scalar"),
        ("numpy", "ndarray"), ("numpy", "dtype")}


class SU(pickle.Unpickler):
    def find_class(self, m, n):
        for mm in (m, m.replace("numpy._core", "numpy.core"), m.replace("numpy.core", "numpy._core")):
            if (mm, n) in SAFE:
                return super().find_class(mm, n)
        raise pickle.UnpicklingError(m + "." + n)


def rasterize(pts):
    """mm points -> bool occupancy on our coarsen-2 grid."""
    idx = np.floor((pts - ORG) / SP).astype(int)
    ok = np.all((idx >= 0) & (idx < np.array(GS)), axis=1)
    idx = idx[ok]
    occ = np.zeros(GS, bool)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return occ


def ref_occupancy():
    o = SU(open(REF, "rb")).load()
    pts = np.concatenate([np.asarray(t["positions"], float) for t in o["tracks_smoothed"]])
    return rasterize(pts), len(o["tracks_smoothed"]), len(pts)


def metrics(O, R):
    inter = int((O & R).sum())
    dice = 2 * inter / (O.sum() + R.sum() + 1e-9)
    return {"dice": float(dice), "recall": float(inter / (R.sum() + 1e-9)),
            "precision": float(inter / (O.sum() + 1e-9)), "occ_ours": int(O.sum())}


def null_dice(O, R, n=200, seed=0):
    rng = np.random.default_rng(seed)
    ds = []
    for _ in range(n):
        sh = [int(rng.integers(1, GS[a] - 1)) for a in range(3)]      # nonzero shift each axis
        Os = np.roll(O, sh, axis=(0, 1, 2))
        ds.append(2 * (Os & R).sum() / (Os.sum() + R.sum() + 1e-9))
    return np.array(ds)


def main():
    R, ntr, npts = ref_occupancy()
    print(f"reference: {ntr} tracks, {npts} pts -> {int(R.sum())} occupied voxels on our grid")
    S = np.load(FD + "samples.npz")
    V, C = build_field(S["mids"], S["vels"])
    out = {"reference_tracks": ntr, "reference_occ_vox": int(R.sum())}
    for name, (Vr, Cr) in {
        "gauss_baseline": smooth_field(V, C, (2.5, 1.4, 2.5)),
        "graph_mb_ki9": regularize(V, C, mode="graph", gamma=1.5, iters=200, dt=0.12, eps_iso=0.2),
    }.items():
        occ, _, _ = tractogram(Vr, Cr)
        m = metrics(occ, R)
        # 1-voxel-tolerance (fair to coarsen-2 rasterization + localization slop):
        # recall = ref voxels within 1 vox of ours; precision = our voxels within 1 vox of ref
        Rdil = binary_dilation(R, iterations=1)
        Odil = binary_dilation(occ, iterations=1)
        m["recall_tol1"] = float((Odil & R).sum() / (R.sum() + 1e-9))
        m["precision_tol1"] = float((occ & Rdil).sum() / (occ.sum() + 1e-9))
        nd = null_dice(occ, R)
        m["null_dice_mean"] = float(nd.mean()); m["null_dice_sd"] = float(nd.std())
        m["dice_over_null"] = float(m["dice"] / (nd.mean() + 1e-9))
        m["z_vs_null"] = float((m["dice"] - nd.mean()) / (nd.std() + 1e-9))
        m["pctile_vs_null"] = float((m["dice"] > nd).mean())
        out[name] = m
        print(f"{name:16s} dice={m['dice']:.3f} recall={m['recall']:.3f}({m['recall_tol1']:.3f}) "
              f"prec={m['precision']:.3f}({m['precision_tol1']:.3f}) "
              f"| null {m['null_dice_mean']:.3f}±{m['null_dice_sd']:.3f} -> {m['dice_over_null']:.1f}x z={m['z_vs_null']:.1f}")
    json.dump(out, open(FD + "coverage_vs_reference.json", "w"), indent=2)
    print("wrote", FD + "coverage_vs_reference.json")


if __name__ == "__main__":
    main()
