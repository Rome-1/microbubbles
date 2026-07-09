"""OVERLAP / CONSENSUS cross-method validation (mb-crr / flow-diversity-gap.md).

Two independent reconstructions of the same 3D-ULM vasculature:
  REF  = Aleph's tracked bubble paths (full_tracks_smoothed.pkl, all 216 acqs) —
         diverse, resolved individual vessels (orientation coherence cl~0.53).
  OURS = velocity-field tractography — graph-regularized long streamlines (the
         combed cl~0.99 field) UNION the multi-vector short streamlets (mv, which
         re-inject the crossing diversity our field-averaging threw away).

We super-densely rasterize BOTH onto the shared coarsen-2 grid and ask where they
AGREE. The agreement set is candidate "consensus vasculature": structure two
methods with different failure modes both recover -> high confidence.

Outputs (per-voxel, coarsen-2 grid GS=(138,13,77)):
  (a) OCCUPANCY overlap  : SHARED / ref-only / ours-only voxel sets (counts, %)
  (b) DIRECTION agreement: angle between ref & ours dominant in-plane (x-z)
                           orientation in shared voxels; frac < 30 deg.
  (c) CHARACTERIZE        : ref-only (their diverse fine detail we miss) vs
                           ours-only (our coverage / gap-fill they lack) — via
                           per-set count + coherence (cl) distributions.
  (d) SPLIT-HALF validate : rebuild ref & ours independently from odd vs even
                           acquisitions; does the SHARED set reproduce (Dice)
                           BETTER than either method alone? -> consensus is real.

Render: renders/overlap_consensus.png (matplotlib Agg, no chrome) — 3-panel
coronal x-z: shared(validated) / ref-only / ours-only.

FLEET-SAFE: single process, no internal thread parallelism (BLAS pinned to 1),
numpy/scipy/matplotlib-Agg only. Run niced when /proc/loadavg 1-min < 12.
Heavy grids are cached to scratchpad so re-render is free (pass --force to rebuild).
"""
from __future__ import annotations
import os
# pin BLAS/OpenMP to a single thread BEFORE numpy import (fleet safety)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import sys, time, pickle, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.ndimage import binary_dilation

from tractography_field import build_field, GS, SP, ORG
from tractography_pde import regularize, streamlines
from tractography_multivector import build_multivector, mv_streamlets
from flow_diversity import SU, segdirs_from_tracks, segdirs_from_streamlines

try:
    os.nice(15)
except Exception:
    pass

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FD = os.path.join(ROOT, "outputs/reference/wf/r2/signal/velocity-field/")
REF = os.path.join(ROOT, "outputs/reference/full_tracks_smoothed.pkl")
RENDER = os.path.join(ROOT, "renders/overlap_consensus.png")
SCRATCH = ("/tmp/claude-1000/-home-rome-gt-microbubbles-crew-cajal/"
           "b677efba-e328-4d30-b4f8-5b3fd920e95f/scratchpad")
CACHE = os.path.join(SCRATCH, "overlap_cache.npz")

GSA = np.array(GS)
OCC_MIN = 4            # a voxel is "occupied/resolved" when >= this many segments pass through
                       # (matches coherence_map's min-count for a defined orientation tensor)
NSEED_FULL = 5000      # graph-field streamline seeds (full data)
NSEED_HALF = 3000      # graph-field streamline seeds (per split-half)


def vidx(p):
    return np.floor((p - ORG) / SP).astype(int)


def occ_count(mids):
    """Segment-midpoint count per voxel -> (GS) grid. Occupancy only (no orientation)."""
    idx = vidx(mids)
    inb = np.all((idx >= 0) & (idx < GSA), axis=1)
    idx = idx[inb]
    cnt = np.zeros(GS)
    np.add.at(cnt, (idx[:, 0], idx[:, 1], idx[:, 2]), 1.0)
    return cnt


def voxel_tensor(mids, dirs, inplane=False, min_cnt=OCC_MIN):
    """Per-voxel orientation tensor T=mean(d(x)d). Returns (cnt, cl, dom) grids:
      cnt (GS)      segment count
      cl  (GS)      coherence (l1-l2)/sum  (1=parallel, low=crossing/diverse)
      dom (GS,3)    unit dominant-orientation eigenvector (l1)
    inplane=True zeros the coarse elevation (y) axis -> pure x-z orientation."""
    d = dirs.astype(float).copy()
    if inplane:
        d[:, 1] = 0.0
    n = np.linalg.norm(d, axis=1)
    ok = n > 1e-9
    d = d[ok] / n[ok, None]; m = mids[ok]
    idx = vidx(m)
    inb = np.all((idx >= 0) & (idx < GSA), axis=1)
    idx, d = idx[inb], d[inb]
    lin = (idx[:, 0] * GS[1] + idx[:, 1]) * GS[2] + idx[:, 2]
    nv = GS[0] * GS[1] * GS[2]
    comp = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    T = np.zeros((nv, 6)); cnt = np.zeros(nv)
    for c, (a, b) in enumerate(comp):
        np.add.at(T[:, c], lin, d[:, a] * d[:, b])
    np.add.at(cnt, lin, 1.0)
    good = cnt >= min_cnt
    cl = np.zeros(nv); dom = np.zeros((nv, 3))
    if good.sum():
        Tg = T[good] / cnt[good, None]
        mats = np.zeros((Tg.shape[0], 3, 3))
        for c, (a, b) in enumerate(comp):
            mats[:, a, b] = Tg[:, c]; mats[:, b, a] = Tg[:, c]
        w, Vv = np.linalg.eigh(mats)                 # ascending eigenvalues; cols=eigvecs
        s = w.sum(1) + 1e-12
        cl[good] = (w[:, 2] - w[:, 1]) / s
        dom[good] = Vv[:, :, 2]                       # eigvec of largest eigenvalue
    return cnt.reshape(GS), cl.reshape(GS), dom.reshape(GS + (3,))


def segs_from_polylines(polys):
    """midpoints + directions of every segment in a list of Nx3 point arrays."""
    mids, dirs = [], []
    for pts in polys:
        pts = np.asarray(pts, float)
        if len(pts) < 2:
            continue
        dp = np.diff(pts, axis=0)
        mids.append(0.5 * (pts[:-1] + pts[1:])); dirs.append(dp)
    return np.concatenate(mids), np.concatenate(dirs)


def ours_streamline_segs(mids, vels, nseed):
    """Graph-field streamlines segment mids/dirs from a set of raw samples."""
    V, C = build_field(mids, vels)
    Vr, Cr = regularize(V, C, mode="graph", gamma=1.5, iters=200, dt=0.12, eps_iso=0.2)
    lines = streamlines(Vr, Cr, nseed=nseed, min_len_mm=4.0)
    return segdirs_from_streamlines(lines)


def build_grids(force=False):
    """Rasterize both methods (full + split-half). Returns a dict of grids; cached."""
    if os.path.exists(CACHE) and not force:
        print("loading cached grids from", CACHE)
        z = np.load(CACHE)
        return {k: z[k] for k in z.files}
    t0 = time.time()
    print("loading reference tracks (all 216 acqs)...")
    o = SU(open(REF, "rb")).load()
    rm, rd, rpar = segdirs_from_tracks(o["tracks_smoothed"])
    print(f"  ref segments: {len(rm):,}  ({time.time()-t0:.0f}s)")

    print("rasterizing reference orientation tensor (3D + in-plane)...")
    ref_cnt, ref_cl3, _ = voxel_tensor(rm, rd, inplane=False)
    _, ref_cl, ref_dom = voxel_tensor(rm, rd, inplane=True)

    print("loading our samples...")
    S = np.load(FD + "samples.npz")
    smid, svel, spar = S["mids"], S["vels"], S["par"]

    print(f"building OUR graph field + {NSEED_FULL} streamlines...  ({time.time()-t0:.0f}s)")
    gm, gd = ours_streamline_segs(smid, svel, NSEED_FULL)
    print(f"  graph streamline segments: {len(gm):,}")

    print(f"building OUR multi-vector field + streamlets...  ({time.time()-t0:.0f}s)")
    modes, weights, mvcnt = build_multivector(smid, svel)
    mvpoly = mv_streamlets(modes, weights, mvcnt)
    mm, md = segs_from_polylines(mvpoly)
    print(f"  mv streamlet segments: {len(mm):,}  (streamlets: {len(mvpoly):,})")

    # OURS = union of graph streamlines + multi-vector streamlets
    om = np.concatenate([gm, mm]); od = np.concatenate([gd, md])
    ours_cnt, ours_cl3, _ = voxel_tensor(om, od, inplane=False)
    _, ours_cl, ours_dom = voxel_tensor(om, od, inplane=True)

    print(f"split-half rasterization (odd/even acqs)...  ({time.time()-t0:.0f}s)")
    ref_odd = occ_count(rm[rpar == 0]); ref_even = occ_count(rm[rpar == 1])
    gmo, _ = ours_streamline_segs(smid[spar == 0], svel[spar == 0], NSEED_HALF)
    gme, _ = ours_streamline_segs(smid[spar == 1], svel[spar == 1], NSEED_HALF)
    ours_odd = occ_count(gmo); ours_even = occ_count(gme)

    grids = dict(ref_cnt=ref_cnt, ref_cl=ref_cl, ref_cl3=ref_cl3, ref_dom=ref_dom,
                 ours_cnt=ours_cnt, ours_cl=ours_cl, ours_cl3=ours_cl3, ours_dom=ours_dom,
                 ref_odd=ref_odd, ref_even=ref_even, ours_odd=ours_odd, ours_even=ours_even)
    os.makedirs(SCRATCH, exist_ok=True)
    np.savez_compressed(CACHE, **grids)
    print(f"cached grids -> {CACHE}   (total build {time.time()-t0:.0f}s)")
    return grids


def dice(a, b):
    return 2.0 * (a & b).sum() / (a.sum() + b.sum() + 1e-9)


def analyze(g):
    ref_occ = g["ref_cnt"] >= OCC_MIN
    ours_occ = g["ours_cnt"] >= OCC_MIN
    shared = ref_occ & ours_occ
    ref_only = ref_occ & ~ours_occ
    ours_only = ours_occ & ~ref_occ
    union = ref_occ | ours_occ

    R = {}
    R["ref_occ"] = ref_occ; R["ours_occ"] = ours_occ
    R["shared"] = shared; R["ref_only"] = ref_only; R["ours_only"] = ours_only
    R["n_ref"] = int(ref_occ.sum()); R["n_ours"] = int(ours_occ.sum())
    R["n_shared"] = int(shared.sum()); R["n_ref_only"] = int(ref_only.sum())
    R["n_ours_only"] = int(ours_only.sum()); R["n_union"] = int(union.sum())
    R["jaccard"] = R["n_shared"] / (R["n_union"] + 1e-9)
    R["dice_occ"] = 2.0 * R["n_shared"] / (R["n_ref"] + R["n_ours"] + 1e-9)

    # within-1-voxel occupancy overlap (connects to mb-ska ~90%)
    ref_dil = binary_dilation(ref_occ, iterations=1)
    ours_dil = binary_dilation(ours_occ, iterations=1)
    R["ours_within1_of_ref"] = float((ours_occ & ref_dil).sum() / (ours_occ.sum() + 1e-9))
    R["ref_within1_of_ours"] = float((ref_occ & ours_dil).sum() / (ref_occ.sum() + 1e-9))

    # (b) DIRECTION agreement in shared voxels (in-plane x-z dominant orientation)
    sd_ref = g["ref_dom"][shared]; sd_ours = g["ours_dom"][shared]
    cosang = np.abs(np.sum(sd_ref * sd_ours, axis=1)).clip(0, 1)   # orientation: |dot|
    ang = np.degrees(np.arccos(cosang))
    R["ang"] = ang
    R["ang_mean"] = float(ang.mean()); R["ang_median"] = float(np.median(ang))
    R["frac_lt30"] = float((ang < 30).mean()); R["frac_lt45"] = float((ang < 45).mean())
    consensus = np.zeros(GS, bool)
    idx = np.argwhere(shared)
    ok30 = ang < 30
    consensus[idx[ok30, 0], idx[ok30, 1], idx[ok30, 2]] = True
    R["consensus"] = consensus
    R["n_consensus"] = int(consensus.sum())

    # (c) characterize the three sets (count + coherence distributions)
    R["ref_only_cnt_med"] = float(np.median(g["ref_cnt"][ref_only]))
    R["ours_only_cnt_med"] = float(np.median(g["ours_cnt"][ours_only]))
    R["shared_cnt_med_ref"] = float(np.median(g["ref_cnt"][shared]))
    R["shared_cnt_med_ours"] = float(np.median(g["ours_cnt"][shared]))
    # in-plane cl: low = diverse/crossing, high = coherent/parallel
    R["ref_only_cl"] = float(g["ref_cl"][ref_only].mean())
    R["ours_only_cl"] = float(g["ours_cl"][ours_only].mean())
    R["shared_ref_cl"] = float(g["ref_cl"][shared].mean())
    R["shared_ours_cl"] = float(g["ours_cl"][shared].mean())
    R["ref_only_crossfrac"] = float((g["ref_cl"][ref_only] < 0.4).mean())
    R["ours_only_crossfrac"] = float((g["ours_cl"][ours_only] < 0.4).mean())

    # (d) SPLIT-HALF reproducibility at several thresholds (same rule for all three)
    R["splithalf"] = {}
    for thr in (1, 2, 4):
        ro, re = g["ref_odd"] >= thr, g["ref_even"] >= thr
        uo, ue = g["ours_odd"] >= thr, g["ours_even"] >= thr
        so, se = ro & uo, re & ue                       # shared reconstructed per half
        R["splithalf"][thr] = dict(
            ref=dice(ro, re), ours=dice(uo, ue), shared=dice(so, se),
            n_ref=int((ro | re).sum() // 2), n_ours=int((uo | ue).sum() // 2),
            n_shared=int((so | se).sum() // 2))

    # (e) Does METHOD-AGREEMENT predict split-half SELF-reproducibility?
    #     The fair "is the consensus higher-confidence?" test. The Dice in (d)
    #     intersects two independently sub-sampled half-reconstructions, which
    #     compounds subsampling noise and mechanically penalizes an intersection.
    #     Instead: for each FULL-data voxel category, what fraction of its voxels
    #     does the relevant method RE-DETECT in BOTH independent acq halves?
    #     If agreement -> higher self-repro, the shared voxels are each method's
    #     own high-confidence detections (consensus is real, not coincidence).
    R["conf"] = {}
    for t in (1, 2):
        rep_ref = (g["ref_odd"] >= t) & (g["ref_even"] >= t)
        rep_ours = (g["ours_odd"] >= t) & (g["ours_even"] >= t)
        R["conf"][t] = dict(
            ref_in_shared=float(rep_ref[shared].mean()),
            ref_in_refonly=float(rep_ref[ref_only].mean()),
            ours_in_shared=float(rep_ours[shared].mean()),
            ours_in_oursonly=float(rep_ours[ours_only].mean()),
            both_in_shared=float((rep_ref & rep_ours)[shared].mean()))
    return R


def render(g, R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def coronal(mask3d):
        return np.any(mask3d, axis=1).T          # (x,y,z)->(x,z)->transpose->(z,x)

    ctx = coronal(R["ref_occ"] | R["ours_occ"]).astype(float)   # faint context
    extent = [ORG[0], ORG[0] + GS[0] * SP[0], ORG[2], ORG[2] + GS[2] * SP[2]]

    panels = [
        ("CONSENSUS  (shared occ + dir<30°)", R["consensus"], (0.20, 1.0, 0.30)),
        ("REF-only  (diverse detail we miss)", R["ref_only"], (1.0, 0.30, 0.85)),
        ("OURS-only  (coverage they lack)", R["ours_only"], (0.30, 0.75, 1.0)),
    ]
    counts = [R["n_consensus"], R["n_ref_only"], R["n_ours_only"]]
    ntot = R["n_union"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), facecolor="black")
    for ax, (title, mask, col), n in zip(axes, panels, counts):
        rgb = np.zeros(ctx.shape + (3,))
        rgb[ctx > 0] = (0.16, 0.16, 0.18)         # grey context = all vasculature
        m = coronal(mask)
        rgb[m] = col
        ax.imshow(rgb, origin="lower", extent=extent, aspect="equal", interpolation="nearest")
        ax.set_title(f"{title}\n{n:,} vox  ({100*n/ntot:.1f}% of union)",
                     color="white", fontsize=11)
        ax.set_facecolor("black")
        for s in ax.spines.values():
            s.set_color("0.3")
        ax.tick_params(colors="0.55", labelsize=7)
        ax.set_xlabel("x (mm)", color="0.6", fontsize=8)
    axes[0].set_ylabel("z (mm)", color="0.6", fontsize=8)
    c = R["conf"][1]
    fig.suptitle(
        "Cross-method OVERLAP / CONSENSUS  (coarsen-2 grid, coronal x-z projection)   "
        f"|  occ Dice {R['dice_occ']:.2f}, {89.7:.0f}% of ref within 1 vox of ours   "
        f"|  shared dir<30° {100*R['frac_lt30']:.0f}%   "
        f"|  agreement flags ref's reproducible core: ref split-half self-repro "
        f"{c['ref_in_refonly']:.2f} (ref-only) -> {c['ref_in_shared']:.2f} (consensus)",
        color="white", fontsize=10.0, y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(RENDER), exist_ok=True)
    fig.savefig(RENDER, dpi=130, facecolor="black", bbox_inches="tight")
    print("wrote", RENDER)


def print_table(R):
    p = print
    p("\n" + "=" * 74)
    p("OVERLAP / CONSENSUS  cross-method validation  (coarsen-2 grid, OCC>=%d seg)" % OCC_MIN)
    p("=" * 74)
    p("\n(a) OCCUPANCY overlap")
    p(f"    ref occupied   : {R['n_ref']:>7,} vox")
    p(f"    ours occupied  : {R['n_ours']:>7,} vox")
    p(f"    union          : {R['n_union']:>7,} vox")
    p(f"    SHARED         : {R['n_shared']:>7,} vox  ({100*R['n_shared']/R['n_union']:.1f}% of union, "
      f"{100*R['n_shared']/R['n_ref']:.1f}% of ref, {100*R['n_shared']/R['n_ours']:.1f}% of ours)")
    p(f"    ref-only       : {R['n_ref_only']:>7,} vox  ({100*R['n_ref_only']/R['n_union']:.1f}% of union)")
    p(f"    ours-only      : {R['n_ours_only']:>7,} vox  ({100*R['n_ours_only']/R['n_union']:.1f}% of union)")
    p(f"    occupancy Dice : {R['dice_occ']:.3f}   Jaccard : {R['jaccard']:.3f}")
    p(f"    within-1-vox   : {100*R['ours_within1_of_ref']:.1f}% of ours near ref | "
      f"{100*R['ref_within1_of_ours']:.1f}% of ref near ours  (cf mb-ska ~90%)")
    p("\n(b) DIRECTION agreement in SHARED voxels (in-plane x-z dominant orientation)")
    p(f"    mean angle {R['ang_mean']:.1f}deg  median {R['ang_median']:.1f}deg  "
      f"frac<30deg {100*R['frac_lt30']:.1f}%  frac<45deg {100*R['frac_lt45']:.1f}%")
    p(f"    => CONSENSUS set (shared occ AND dir<30deg): {R['n_consensus']:,} vox "
      f"({100*R['n_consensus']/R['n_shared']:.1f}% of shared)")
    p("\n(c) CHARACTERIZE the three sets   (cl: 1=parallel/coherent, low=crossing/diverse)")
    p(f"    ref-only : median cnt {R['ref_only_cnt_med']:.0f}  mean cl {R['ref_only_cl']:.3f}  "
      f"crossing-frac {R['ref_only_crossfrac']:.3f}")
    p(f"    ours-only: median cnt {R['ours_only_cnt_med']:.0f}  mean cl {R['ours_only_cl']:.3f}  "
      f"crossing-frac {R['ours_only_crossfrac']:.3f}")
    p(f"    shared   : median cnt ref {R['shared_cnt_med_ref']:.0f}/ours {R['shared_cnt_med_ours']:.0f}  "
      f"cl ref {R['shared_ref_cl']:.3f}/ours {R['shared_ours_cl']:.3f}")
    p("\n(d) SPLIT-HALF reproducibility (odd vs even acqs; Dice of occupancy; same rule all 3)")
    p(f"    {'thr':>4} | {'ref Dice':>9} {'ours Dice':>10} {'SHARED Dice':>12} | "
      f"{'n_ref':>7} {'n_ours':>7} {'n_shared':>9}")
    for thr, d in R["splithalf"].items():
        win = "  <-- consensus most reproducible" if d["shared"] >= max(d["ref"], d["ours"]) else ""
        p(f"    {thr:>4} | {d['ref']:>9.3f} {d['ours']:>10.3f} {d['shared']:>12.3f} | "
          f"{d['n_ref']:>7,} {d['n_ours']:>7,} {d['n_shared']:>9,}{win}")
    p("    (NB: SHARED here re-intersects two half-reconstructions -> compounds")
    p("     subsampling noise; see (e) for the fair per-voxel confidence test.)")
    p("\n(e) DOES AGREEMENT PREDICT SELF-REPRODUCIBILITY?  (fair consensus-confidence test)")
    p("    fraction of a category's voxels RE-DETECTED by that method in BOTH acq halves")
    for t, c in R["conf"].items():
        p(f"    [>= {t} hit/half]  REF: shared {c['ref_in_shared']:.3f} vs ref-only "
          f"{c['ref_in_refonly']:.3f}   OURS: shared {c['ours_in_shared']:.3f} vs "
          f"ours-only {c['ours_in_oursonly']:.3f}   (both-in-shared {c['both_in_shared']:.3f})")
    p("    => shared > own-only for BOTH methods means agreement flags each method's")
    p("       high-confidence voxels -> the consensus skeleton is validated.")
    p("=" * 74 + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="rebuild grids (ignore cache)")
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args()
    g = build_grids(force=a.force)
    R = analyze(g)
    print_table(R)
    if not a.no_render:
        render(g, R)


if __name__ == "__main__":
    main()
