"""ODF-style flow-diversity fix (option 1, spatial/data-only) — pushing our render's
orientation coherence cl from the multi-vector fix-v1's 0.83 down toward the reference's
0.53, WITHOUT new data and WITHOUT temporal tracking.

WHY v1 stalled (docs/flow-diversity-gap.md §6): v1 clusters each voxel's velocity
samples into up to 3 modes and integrates ONE streamlet per mode along the mode MEAN.
Two problems: (i) greedy dominant-first clustering + in-plane pooling DENOISES away
minority near-crossings, so ~74% of voxels resolve to a single mode; (ii) following the
mode MEAN collapses the intrinsic within-vessel angular spread — a single-mode voxel then
renders as perfectly parallel (cl~1). Averaged, that lands at 0.83. But our RAW samples
already sit at cl~0.51 (= the reference): the diversity is IN THE DATA, thrown away by
denoising+mean-following.

THIS module represents the measured local direction distribution faithfully instead of
collapsing it:
  (1) ODF / peak estimation per voxel from the raw sample directions.
  (2) SPLIT-HALF-REPRODUCIBILITY GATE on every peak: a peak is drawn only if BOTH the
      odd- and even-acquisition subsets of that voxel contribute members to it (real
      crossing, not a noise spike). This is what keeps the recovered diversity REAL.
  (3) SAMPLE-PROPORTIONAL seeding: streaklets per peak ∝ peak weight, so the rendered
      per-voxel direction distribution matches the data's.
  (4) PRESERVED within-peak angular spread via a knob alpha∈[0,1]: each streaklet step's
      direction is drawn from the peak's own member sample directions and blended toward
      the peak mean by (1-alpha). alpha=0 => mode-mean (combed, v1-like); alpha=1 => full
      measured raw dispersion. We SWEEP alpha and pick the operating point whose mean cl
      matches the reference, then VALIDATE that the recovered cl pattern reproduces
      split-half (r comparable to the reference's ~0.32 — signal, not noise).

Objective: minimize cl-distance to the reference (mean cl -> 0.53, KS(cl-dist) -> 0),
raise crossing-fraction toward 0.35, keep split-half cl r > 0.

Output: renders/odf_vs_reference.png + a printed metrics table.
CPU numpy/scipy only, single process. Run niced when /proc/loadavg 1-min < 12.
"""
from __future__ import annotations
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from collections import defaultdict
from tractography_field import GS, SP, ORG, build_field
from flow_diversity import coherence_map, segdirs_from_tracks
from render_flow_diversity import SU

FD = "outputs/reference/wf/r2/signal/velocity-field/"
REF = "outputs/reference/full_tracks_smoothed.pkl"


# ------------------------------------------------------------------ ODF build
def _voxel_spans(mids):
    """Sort in-bounds samples by linear voxel id; return idx(N,3), lin(N), order-applied
    dirs handled by caller. Returns (idx, d_unit, lin, uniq, starts, ends, keep_mask)."""
    idx = np.floor((mids - ORG) / SP).astype(int)
    ok = np.all((idx >= 0) & (idx < np.array(GS)), axis=1)
    return idx, ok


def build_odf(mids, vels, par, pool=0, ang_deg=38.0, min_frac=0.10, min_count=5,
              kmax=4, repro_min=2, gate=True, members_mode="peak"):
    """Per-voxel gated ODF peaks from raw sample directions.

    members_mode: "peak" = each peak's member dirs are its own angular-cone samples
      (denoised); "full" = every kept voxel's peaks carry the voxel's FULL measured
      direction distribution (faithful — preserves the intrinsic within-vessel spread
      the reference has). The split-half GATE still decides which voxels/peaks are drawn.

    Returns:
      peaks : list of dicts {ijk (3,), dir (3,) canonical-sign unit, w float (normalized
              over kept peaks), members (M,3) sign-aligned unit dirs}
      cnt   : (GS) own-voxel sample count (confidence / seeding density)
    """
    idx, ok = _voxel_spans(mids)
    idx = idx[ok]; v = vels[ok]; p = par[ok]
    nrm = np.linalg.norm(v, axis=1); good = nrm > 1e-9
    idx = idx[good]; d = (v[good] / nrm[good, None]).astype(np.float64); p = p[good]

    lin = (idx[:, 0] * GS[1] + idx[:, 1]) * GS[2] + idx[:, 2]
    order = np.argsort(lin, kind="stable")
    lin_s, d_s, p_s, idx_s = lin[order], d[order], p[order], idx[order]
    uniq = np.unique(lin_s)
    starts = np.searchsorted(lin_s, uniq); ends = np.append(starts[1:], len(lin_s))
    span = {int(u): (int(s), int(e)) for u, s, e in zip(uniq, starts, ends)}

    cnt = np.zeros(GS)
    for u, s, e in zip(uniq, starts, ends):
        i, j, k = idx_s[s]; cnt[i, j, k] = e - s

    cos_th = np.cos(np.deg2rad(ang_deg))
    peaks = []
    for u, s, e in zip(uniq, starts, ends):
        i, j, k = idx_s[s]
        rows = [(s, e)]
        if pool:                                        # optional in-plane pooling
            for di in range(-pool, pool + 1):
                for dk in range(-pool, pool + 1):
                    if di == 0 and dk == 0:
                        continue
                    ii, kk = i + di, k + dk
                    if 0 <= ii < GS[0] and 0 <= kk < GS[2]:
                        nl = int((ii * GS[1] + j) * GS[2] + kk)
                        if nl in span:
                            rows.append(span[nl])
        D = np.concatenate([d_s[a:b] for a, b in rows])
        P = np.concatenate([p_s[a:b] for a, b in rows])
        ntot = len(D)
        if ntot < min_count:
            continue
        remain = np.ones(ntot, bool)
        gidx = np.arange(ntot)
        vox_peaks = []
        thr = max(min_count, int(min_frac * ntot))
        while remain.sum() >= thr and len(vox_peaks) < kmax:
            R = D[remain]
            mode = R.mean(0); mode /= (np.linalg.norm(mode) + 1e-9)
            for _ in range(5):                          # orientation mean-shift
                al = R * np.sign(R @ mode)[:, None]
                sel = np.abs(R @ mode) > cos_th
                if sel.sum() == 0:
                    break
                mode = al[sel].mean(0); mode /= (np.linalg.norm(mode) + 1e-9)
            sel = np.abs(R @ mode) > cos_th
            if sel.sum() < thr:
                break
            gi = gidx[remain][sel]                       # global member indices
            ph = P[gi]
            n0 = int((ph == 0).sum()); n1 = int((ph == 1).sum())
            keep = (not gate) or (n0 >= repro_min and n1 >= repro_min)
            if keep:
                mem = D[gi] * np.sign(D[gi] @ mode)[:, None]   # sign-align to peak
                vox_peaks.append({"ijk": np.array([i, j, k]), "dir": mode.copy(),
                                  "w": float(sel.sum()) / ntot, "members": mem})
            remain[gi] = False                            # consume assigned samples
        if vox_peaks:                                     # renormalize kept weights
            wsum = sum(pk["w"] for pk in vox_peaks) + 1e-9
            for pk in vox_peaks:
                pk["w"] /= wsum
                if members_mode == "full":                # faithful: full voxel spread
                    pk["members"] = D.copy()
            peaks.extend(vox_peaks)
    return peaks, cnt


# --------------------------------------------------------- streaklet generation
def odf_streaklets(peaks, cnt, alpha=1.0, base_n=8, nseg=3, step_mm=0.35,
                   cthr_q=0.4, seed=1):
    """Sample-proportional, spread-preserving streaklets. Each streaklet picks ONE
    heading (so it is a CLEAN short arc, like a real track that follows one bubble);
    the voxel's flow DIVERSITY comes from many streaklets choosing DIFFERENT headings,
    exactly as the reference gets low cl from many distinct-but-smooth tracks crossing a
    voxel. alpha in [0,1] sets how far each heading may stray from the peak mean toward a
    drawn measured sample direction: 0 -> all headings = peak mean (combed, v1-like);
    1 -> headings = the measured direction distribution (matches the reference's spread).
    Returns pts (Ntot, nseg+1, 3) polyline array and per-streaklet confidence."""
    cthr = np.quantile(cnt[cnt > 0], cthr_q)
    rng = np.random.default_rng(seed)
    blocks = []; confs = []
    for pk in peaks:
        i, j, k = pk["ijk"]
        c = cnt[i, j, k]
        if c < cthr:
            continue
        n_p = max(1, int(round(base_n * pk["w"])))
        base = pk["dir"]; mem = pk["members"]; M = len(mem)
        m = mem[rng.integers(0, M, n_p)]                      # one sample dir per streaklet
        m = m * np.sign(m @ base)[:, None]                    # align to peak sign
        h = base[None, :] + alpha * (m - base[None, :])       # per-streaklet heading
        h /= (np.linalg.norm(h, axis=1, keepdims=True) + 1e-9)
        sign = rng.choice(np.array([-1.0, 1.0]), n_p)[:, None]
        h = sign * h
        center = ORG + (pk["ijk"][None, :] + rng.uniform(-0.35, 0.35, (n_p, 3))) * SP
        s0 = center - 0.5 * nseg * step_mm * h                # center the arc on the voxel
        steps = np.arange(nseg + 1)[None, :, None]            # (1, nseg+1, 1)
        pts = s0[:, None, :] + steps * step_mm * h[:, None, :]  # (n_p, nseg+1, 3)
        blocks.append(pts); confs.append(np.full(n_p, c))
    if not blocks:
        return np.zeros((0, nseg + 1, 3)), np.zeros(0)
    return np.concatenate(blocks, 0), np.concatenate(confs)


def cl_of(pts):
    """mean cl, crossing-frac, per-voxel cl values (occupied), from a polyline array."""
    if len(pts) == 0:
        return np.nan, np.nan, np.array([])
    dp = np.diff(pts, axis=1).reshape(-1, 3)
    mid = (0.5 * (pts[:, :-1] + pts[:, 1:])).reshape(-1, 3)
    cl, cnt = coherence_map(mid, dp)
    occ = cnt >= 4
    vals = cl[occ]
    return float(vals.mean()), float((vals < 0.4).mean()), vals


# ------------------------------------------------------------------- driver
def main():
    t0 = time.time()
    from scipy.stats import ks_2samp

    print("loading reference tracks (all 216 acqs) for the target cl distribution...")
    o = SU(open(REF, "rb")).load()
    rm, rd, _ = segdirs_from_tracks(o["tracks_smoothed"])
    rcl, rcnt = coherence_map(rm, rd)
    ref_vals = rcl[rcnt >= 4]
    ref_mean = float(ref_vals.mean()); ref_cx = float((ref_vals < 0.4).mean())
    print(f"  reference: mean_cl={ref_mean:.3f} crossing_frac={ref_cx:.3f} "
          f"({len(ref_vals)} occupied voxels)")

    print("loading our samples; raw-sample baseline...")
    S = np.load(FD + "samples.npz")
    mids, vels, par = S["mids"].astype(float), S["vels"].astype(float), S["par"]
    raw_cl, raw_cnt = coherence_map(mids, vels)
    raw_vals = raw_cl[raw_cnt >= 4]
    print(f"  raw samples: mean_cl={raw_vals.mean():.3f} crossing_frac={(raw_vals<0.4).mean():.3f}")

    # live fix-v1 baseline (multi-vector short streamlets)
    try:
        from tractography_multivector import build_multivector, mv_streamlets
        from flow_diversity import segdirs_from_streamlines
        mo, mw, mc = build_multivector(mids, vels, ang_deg=30, min_frac=0.15, min_count=5)
        v1 = mv_streamlets(mo, mw, mc, half_steps=7)
        v1m, v1d = segdirs_from_streamlines([(pp, None) for pp in v1])
        v1cl, v1cnt = coherence_map(v1m, v1d); v1o = v1cnt >= 4
        v1_mean = float(v1cl[v1o].mean()); v1_cx = float((v1cl[v1o] < 0.4).mean())
        v1_ks = float(ks_2samp(v1cl[v1o], ref_vals).statistic)
        print(f"  fix-v1 (multi-vector): mean_cl={v1_mean:.3f} crossing_frac={v1_cx:.3f} "
              f"KS={v1_ks:.3f}  n={len(v1)}")
    except Exception as ex:
        v1_mean = 0.83; v1_cx = 0.05; v1_ks = float("nan")
        print(f"  fix-v1 baseline (from doc): 0.83/0.05  [live recompute skipped: {ex}]")

    print(f"\nbuilding ODF (gated peaks) ... [{time.time()-t0:.0f}s]")
    peaks = {"peak": build_odf(mids, vels, par, pool=0, gate=True, members_mode="peak")[0],
             "full": build_odf(mids, vels, par, pool=0, gate=True, members_mode="full")[0]}
    _, cnt = build_field(mids, vels)
    pk = peaks["peak"]
    nvox = len(set(tuple(p["ijk"]) for p in pk)); n2 = len(pk) - nvox
    print(f"  {nvox} occupied voxels carry {len(pk)} split-half-REPRODUCED peaks "
          f"({n2} extra crossing/branch peaks beyond the dominant one)")

    # ---- sweep spread knob alpha for both member modes -> find cl==reference point.
    #   members=peak : each peak's own denoised angular cone (structured, higher cl)
    #   members=full : the voxel's FULL measured distribution (faithful, reaches raw cl)
    print(f"\nsweeping spread knob alpha (0=combed peak-mean, 1=measured dispersion): [{time.time()-t0:.0f}s]")
    print(f"  {'mode':>5} {'alpha':>5} {'mean_cl':>8} {'cross_frac':>10} {'KS_to_ref':>9} {'n_streak':>9}")
    rows = []
    for mode in ("peak", "full"):
        for a in [0.0, 0.5, 0.85, 1.0]:
            pts, conf = odf_streaklets(peaks[mode], cnt, alpha=a, base_n=8)
            mcl, mcx, vals = cl_of(pts)
            ks = float(ks_2samp(vals, ref_vals).statistic)
            rows.append((mode, a, mcl, mcx, ks, len(pts), pts, conf))
            print(f"  {mode:>5} {a:5.2f} {mcl:8.3f} {mcx:10.3f} {ks:9.3f} {len(pts):9d}")

    # best = (mode, alpha) whose mean_cl is closest to the reference mean
    best = min(rows, key=lambda r: abs(r[2] - ref_mean))
    mode_b, a_b, mcl_b, mcx_b, ks_b, n_b, pts_b, conf_b = best
    print(f"\n  -> best: members={mode_b} alpha={a_b:.2f}: mean_cl={mcl_b:.3f} (ref {ref_mean:.3f}), "
          f"crossing_frac={mcx_b:.3f} (ref {ref_cx:.3f}), KS={ks_b:.3f}")

    # ---- split-half reproducibility of the recovered cl pattern (real, not noise)
    print(f"\nsplit-half validation at members={mode_b} alpha={a_b:.2f} (recon from odd vs even acqs): [{time.time()-t0:.0f}s]")
    po, _ = build_odf(mids[par == 0], vels[par == 0], par[par == 0], pool=0, gate=False, members_mode=mode_b)
    pe, _ = build_odf(mids[par == 1], vels[par == 1], par[par == 1], pool=0, gate=False, members_mode=mode_b)
    pts_o, _ = odf_streaklets(po, cnt, alpha=a_b, base_n=8, seed=11)
    pts_e, _ = odf_streaklets(pe, cnt, alpha=a_b, base_n=8, seed=22)

    def clgrid(pts):
        dp = np.diff(pts, axis=1).reshape(-1, 3)
        mid = (0.5 * (pts[:, :-1] + pts[:, 1:])).reshape(-1, 3)
        return coherence_map(mid, dp)
    clo, cno = clgrid(pts_o); cle, cne = clgrid(pts_e)
    both = (cno >= 4) & (cne >= 4)
    sh_r = float(np.corrcoef(clo[both], cle[both])[0, 1]) if both.sum() > 20 else float("nan")
    print(f"  split-half cl r={sh_r:.3f} over {int(both.sum())} shared voxels "
          f"(reference ~0.32, raw ~0.36 => recovered diversity is REAL if comparable)")

    # ---- gate 1': occupancy split-half Dice of the ODF object (scorecard §4).
    # Same odd/even reconstructions; does the OBJECT'S occupancy reproduce like the field's 0.71?
    def occ_grid(pts):
        P = np.asarray(pts).reshape(-1, 3)
        idx = np.floor((P - np.array(ORG)) / np.array(SP)).astype(int)
        ok = np.all((idx >= 0) & (idx < np.array(GS)), axis=1)
        g = np.zeros(GS, bool); i = idx[ok]
        g[i[:, 0], i[:, 1], i[:, 2]] = True
        return g
    go, ge = occ_grid(pts_o), occ_grid(pts_e)
    inter = int((go & ge).sum()); tot = int(go.sum()) + int(ge.sum())
    occ_dice = (2.0 * inter / tot) if tot else float("nan")
    print(f"  [gate 1'] ODF occupancy split-half Dice={occ_dice:.3f} "
          f"(field baseline 0.71; expect ~equal — same occupied voxels, multi-directional)")

    # ---- render: our best next to the reference, direction-hue, sparse+thin
    print(f"\nrendering renders/odf_vs_reference.png [{time.time()-t0:.0f}s]")
    render(pts_b, conf_b, o["tracks_smoothed"], f"{mode_b} α={a_b:.2f}", mcl_b, mcx_b, ks_b, sh_r,
           ref_mean, ref_cx)

    # ---- final scorecard
    def gap(x):
        return f"{(v1_mean - x) / (v1_mean - ref_mean) * 100:.0f}%" if v1_mean != ref_mean else "-"
    print("\n================= SCORECARD (target = reference) =================")
    print(f"  {'reconstruction':32s} {'mean_cl':>8} {'cross':>6} {'KS_ref':>7} {'gap_closed':>10}")
    print(f"  {'single-vector field (combed)':32s} {0.99:8.2f} {0.00:6.2f} {'—':>7} {'0%':>10}")
    print(f"  {'fix-v1 multi-vector':32s} {v1_mean:8.2f} {v1_cx:6.2f} {v1_ks:7.3f} {'0%':>10}")
    print(f"  {'THIS ODF fix (best)':32s} {mcl_b:8.2f} {mcx_b:6.2f} {ks_b:7.3f} {gap(mcl_b):>10}")
    print(f"  {'reference (target)':32s} {ref_mean:8.2f} {ref_cx:6.2f} {0.0:7.3f} {'100%':>10}")
    print(f"  split-half cl r (ours) = {sh_r:.3f}   [ref ~0.32]")
    print(f"  done in {time.time()-t0:.0f}s")


# ------------------------------------------------------------------- render
def render(pts, conf, tracks, label, mcl, mcx, ks, sh_r, ref_mean, ref_cx):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import hsv_to_rgb
    XLIM = (ORG[0], ORG[0] + GS[0] * SP[0]); ZLIM = (ORG[2], ORG[2] + GS[2] * SP[2])

    def polys_to_segs(polys):
        segs, cols = [], []
        for p in polys:
            xz = p[:, [0, 2]]; dv = np.diff(p, axis=0)
            ang = (np.arctan2(dv[:, 2], dv[:, 0]) % np.pi) / np.pi
            for a in range(len(xz) - 1):
                segs.append([xz[a], xz[a + 1]]); cols.append(hsv_to_rgb([ang[a], 0.85, 1.0]))
        return segs, cols

    def panel(ax, polys, title, lw, al):
        ax.set_facecolor("black")
        segs, cols = polys_to_segs(polys)
        ax.add_collection(LineCollection(segs, colors=cols, linewidths=lw, alpha=al))
        ax.set_xlim(*XLIM); ax.set_ylim(*ZLIM); ax.set_aspect("equal")
        ax.set_title(title, color="white", fontsize=11)
        ax.tick_params(colors="0.55", labelsize=7)
        for s in ax.spines.values():
            s.set_color("0.3")

    rng = np.random.default_rng(0)
    # reference: length-filtered subset (their render is a length-filtered subset)
    refp = [np.asarray(t["positions"], float) for t in tracks
            if int(t.get("length", len(t["positions"]))) >= 12]
    if len(refp) > 6000:
        refp = [refp[i] for i in rng.choice(len(refp), 6000, replace=False)]

    # ours: SPARSE + high-confidence — keep top-confidence streaklets, subsample, thin
    order = np.argsort(-conf)                       # highest confidence first
    keep = order[: min(len(order), 12000)]
    if len(keep) > 7000:
        keep = keep[rng.choice(len(keep), 7000, replace=False)]
    ourp = [pts[i] for i in keep]

    fig, ax = plt.subplots(1, 2, figsize=(15, 6.2), facecolor="black")
    panel(ax[0], refp, f"Aleph reference tracks  (cl≈{ref_mean:.2f}, crossings {ref_cx*100:.0f}%)\n"
                       f"{len(refp)} tracks · hue = in-plane direction", lw=0.5, al=0.75)
    panel(ax[1], ourp, f"Our ODF fix ({label})  cl={mcl:.2f}, crossings {mcx*100:.0f}%, KS={ks:.2f}\n"
                       f"{len(ourp)} sparse high-conf streaklets · split-half r={sh_r:.2f}", lw=0.35, al=0.55)
    fig.suptitle("ODF fix vs reference — proportional, split-half-gated, spread-preserving streaklets "
                 "(was combed cl 0.99 / fix-v1 0.83)", color="white", fontsize=12)
    cax = fig.add_axes([0.46, 0.02, 0.08, 0.03])
    grad = hsv_to_rgb(np.stack([np.linspace(0, 1, 180), np.ones(180) * 0.85, np.ones(180)], 1))[None]
    cax.imshow(grad, aspect="auto"); cax.set_xticks([0, 179])
    cax.set_xticklabels(["0°", "180°"], color="0.7", fontsize=7); cax.set_yticks([])
    os.makedirs("renders", exist_ok=True)
    fig.savefig("renders/odf_vs_reference.png", dpi=140, facecolor="black", bbox_inches="tight")
    print("  wrote renders/odf_vs_reference.png")


if __name__ == "__main__":
    main()
