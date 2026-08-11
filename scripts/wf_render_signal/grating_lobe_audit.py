"""J5 -- elevation grating-lobe audit (mb-ne3).

The receive aperture is 8 physical rows at ``element_pitch_y_m = 1.664 mm``.  With
c = 1600 m/s and tx_freq = 2.0 MHz the wavelength is 0.800 mm, so the elevation
row pitch is 2.08 lambda -- four times the lambda/2 that would keep the visible
region grating-lobe free.  A coherent delay-and-sum over those rows is therefore
ambiguous: a scatterer at direction-sine s = y / r is indistinguishable, to the
elevation aperture alone, from one at

    s_ghost = s_true +/- lambda / pitch = s_true +/- 0.4808

at the *same* round-trip range r.  The replica is not at the same z: it sits on
the constant-r arc, so z_ghost = r * sqrt(1 - s_ghost^2).  For an on-axis
scatterer that is 28.7 degrees off-axis and 12% shallower.

This script asks whether those replicas are actually in the detection cloud.
There are no free parameters -- the offset is fixed by lambda / pitch -- so every
test is confirm-or-refute against a matched control offset that no lobe predicts.

Three measurements:

1. GEOMETRY.  Where can a replica even land?  Elevation is reconstructed over 25
   planes spanning +/-6.66 mm, so the replica of a scatterer at (y, z) is only
   inside the volume when |y +/- 0.4808 * r| <= Y_max.  Since |y| <= Y_max, that
   needs 0.4808 * r <= 2 * Y_max, i.e. r <= 27.7 mm.  Beyond that range *no*
   scatterer at *any* elevation has an in-volume replica and the data cannot
   speak to the question at all.  Panel A quantifies the fraction of the
   elevation range that is testable at each depth.

2. PAIR AUTOCORRELATION.  Within a frame, histogram ds = |s_i - s_j| over pairs
   that share x and r (the two coordinates a lobe preserves), stratified by r
   band.  A lobe shows up as a bump at ds = 0.4808.  Null: the identical
   statistic with the two detections drawn from frames 101 apart in the same
   acquisition -- same vessel geometry, same density structure, but no pair can
   be a replica of the other.

3. CONDITIONAL REPLICA TEST.  For every detection, compute the two predicted
   replica positions and count same-frame detections within a tolerance sphere.
   Compare against control offsets at ds = 0.36 and 0.60 (equidistant from
   0.4808, predicted by nothing) evaluated on the same arc with the same
   in-volume requirement, and against the cross-frame null.  This is the
   sharpest test: the predicted location is exact, so a lobe with even a few
   percent detection efficiency produces a large excess over the ~0.002/mm^3
   ambient detection density.

Local, CPU only.  Writes outputs/render/grating_lobe_audit.png.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

# ---------------------------------------------------------------- geometry ----

SPEED_OF_SOUND_M_S = 1600.0
TX_FREQ_HZ = 2.0e6
ELEMENT_PITCH_Y_M = 0.001664

LAMBDA_MM = SPEED_OF_SOUND_M_S / TX_FREQ_HZ * 1e3          # 0.800 mm
PITCH_MM = ELEMENT_PITCH_Y_M * 1e3                          # 1.664 mm
DS_LOBE = LAMBDA_MM / PITCH_MM                              # 0.4808 sine offset
THETA_LOBE_DEG = float(np.degrees(np.arcsin(DS_LOBE)))      # 28.74 deg

# Reconstructed elevation support: 25 planes, 0.5547 mm pitch.
ELEV_PITCH_MM = 0.5547
N_ELEV = 25
Y_MAX_MM = 0.5 * (N_ELEV - 1) * ELEV_PITCH_MM               # 6.656 mm
# Detections are subpixel-localized and never reach the outermost plane centres;
# require replicas to land where a detection could actually have been reported.
Y_DET_MM = 6.40
Z_MIN_MM, Z_MAX_MM = 10.2, 40.5

# Control offsets, equidistant from the predicted one.  Deliberately NOT 0.96,
# which is the second grating order and therefore physics-predicted too.
CONTROL_OFFSETS = (0.36, 0.60)
SECONDARY_CONTROLS = (0.30, 0.66)

# Narrow range bands.  Width matters: the largest reachable offset is
# 2 * Y / r, which sweeps from 0.99 to 0.36 across the depth range, and the
# same-frame/null ratio ramps up steeply as probes approach it (both members of
# an edge-to-edge pair sit in the sparse outer elevation planes, where the
# cross-frame null underestimates same-frame co-occurrence).  Wide bands blend
# ramps at different edges and can fake an interior peak.  With 2 mm bands the
# edge is sharp and moves band to band, so it separates cleanly from a grating
# lobe, which must sit at lambda/d in every band regardless of where the edge is.
R_BANDS = ((12.0, 14.0), (14.0, 16.0), (16.0, 18.0), (18.0, 20.0), (20.0, 22.0),
           (22.0, 24.0), (24.0, 26.0), (26.0, 28.0), (28.0, 32.0), (32.0, 41.0))


def band_edge(lo: float, hi: float) -> float:
    """Largest elevation-sine offset with any in-volume probe in this band."""
    return 2.0 * Y_DET_MM / (0.5 * (lo + hi))

FRAMES_PER_ACQ = 240
NULL_FRAME_LAG = 101       # coprime-ish with 240, well beyond any bubble transit
PAIR_TOL_X_MM = 1.5
PAIR_TOL_R_MM = 1.5
MATCH_TOL_MM = 0.8         # replica search radius, ~4x the lateral voxel pitch

BATCHES = ("0000", "0012", "0024", "0036", "0048")


def sine_of(y: np.ndarray, r: np.ndarray) -> np.ndarray:
    return y / r


def replica_position(y: np.ndarray, z: np.ndarray, ds: float):
    """Replica of (y, z) displaced by ``ds`` in direction sine on the same arc."""
    r = np.hypot(y, z)
    s_g = y / r + ds
    ok = np.abs(s_g) < 1.0
    y_g = np.where(ok, r * s_g, np.nan)
    z_g = np.where(ok, r * np.sqrt(np.clip(1.0 - s_g**2, 0.0, None)), np.nan)
    return y_g, z_g


def in_volume(y_g: np.ndarray, z_g: np.ndarray) -> np.ndarray:
    return (np.abs(y_g) <= Y_DET_MM) & (z_g >= Z_MIN_MM) & (z_g <= Z_MAX_MM)


def testable_fraction(r: float, ds: float = DS_LOBE) -> float:
    """Fraction of elevation positions at range r whose replica is in-volume.

    Parameterize by the true elevation y in [-Y, Y] (constrained also by
    |y| <= r).  Either sign of the offset counts.
    """
    y_lim = min(Y_DET_MM, r)
    ys = np.linspace(-y_lim, y_lim, 2001)
    zs = np.sqrt(np.clip(r**2 - ys**2, 0.0, None))
    ok = np.zeros_like(ys, dtype=bool)
    for sgn in (+1.0, -1.0):
        y_g, z_g = replica_position(ys, zs, sgn * ds)
        ok |= in_volume(y_g, z_g) & (zs >= Z_MIN_MM) & (zs <= Z_MAX_MM)
    return float(ok.mean())


# -------------------------------------------------------------------- data ----


@dataclass
class Batch:
    name: str
    pos: np.ndarray          # (N, 3) x, y, z in mm
    intensity: np.ndarray
    frame: np.ndarray
    acq: np.ndarray

    @property
    def r(self) -> np.ndarray:
        return np.hypot(self.pos[:, 1], self.pos[:, 2])


def load_batch(path: Path) -> Batch:
    d = np.load(path)
    return Batch(
        name=path.stem.split("_")[-1],
        pos=d["positions_mm"].astype(np.float64),
        intensity=d["intensities"].astype(np.float64),
        frame=d["frame_indices"].astype(np.int64),
        acq=d["acq_indices"].astype(np.int64),
    )


def frame_groups(b: Batch) -> dict[int, np.ndarray]:
    order = np.argsort(b.frame, kind="stable")
    f_sorted = b.frame[order]
    bounds = np.flatnonzero(np.diff(f_sorted)) + 1
    return {
        int(f_sorted[sl[0]]): sl
        for sl in np.split(order, bounds)
        if sl.size
    }


def null_frame(f: int) -> int:
    """Frame in the same acquisition, NULL_FRAME_LAG away (wrapped)."""
    acq, local = divmod(f, FRAMES_PER_ACQ)
    return acq * FRAMES_PER_ACQ + (local + NULL_FRAME_LAG) % FRAMES_PER_ACQ


# ------------------------------------------------- test 2: pair statistics ----

DS_BINS = np.linspace(0.0, 1.0, 101)
DS_CENTRES = 0.5 * (DS_BINS[:-1] + DS_BINS[1:])


def pair_histograms(b: Batch):
    """(n_bands, n_bins) ds histograms for real same-frame and cross-frame null."""
    groups = frame_groups(b)
    r_all = b.r
    s_all = sine_of(b.pos[:, 1], r_all)
    x_all = b.pos[:, 0]

    real = np.zeros((len(R_BANDS), len(DS_CENTRES)))
    null = np.zeros_like(real)

    def accumulate(idx_a: np.ndarray, idx_b: np.ndarray, out, same: bool):
        if idx_a.size == 0 or idx_b.size == 0:
            return
        dx = np.abs(x_all[idx_a][:, None] - x_all[idx_b][None, :])
        dr = np.abs(r_all[idx_a][:, None] - r_all[idx_b][None, :])
        keep = (dx <= PAIR_TOL_X_MM) & (dr <= PAIR_TOL_R_MM)
        if same:
            keep &= np.triu(np.ones_like(keep, dtype=bool), k=1)
        ii, jj = np.nonzero(keep)
        if ii.size == 0:
            return
        ia, jb = idx_a[ii], idx_b[jj]
        ds = np.abs(s_all[ia] - s_all[jb])
        rm = 0.5 * (r_all[ia] + r_all[jb])
        for bi, (lo, hi) in enumerate(R_BANDS):
            m = (rm >= lo) & (rm < hi)
            if m.any():
                out[bi] += np.histogram(ds[m], bins=DS_BINS)[0]

    for f, idx in groups.items():
        accumulate(idx, idx, real, same=True)
        idx_n = groups.get(null_frame(f))
        if idx_n is not None:
            accumulate(idx, idx_n, null, same=False)
    return real, null


# --------------------------------------- test 3: conditional replica search ----


# The decisive statistic.  A fixed pair of control offsets is NOT a matched
# control: detection density falls toward the elevation edges, and a larger |ds|
# pushes the probe point closer to the edge, so the probe rate has a strong
# smooth trend in ds that has nothing to do with grating lobes.  Sweeping ds
# continuously separates the two: the density trend is smooth, a grating lobe is
# a delta-like spike pinned at lambda/d with no width beyond localization jitter.
DS_SWEEP = np.round(np.arange(0.10, 0.9401, 0.01), 4)


def conditional_sweep(b: Batch, bright_only: bool = False):
    """Same-frame / cross-frame probe-hit counts vs elevation-sine offset.

    Returns dict of (n_ds, n_bands) arrays: seeds, real, null.  The cross-frame
    null is corrected for the target frame's occupancy -- expected hits scale
    linearly with how many detections the probed frame holds, and per-frame
    counts are overdispersed (Var/mean^2 ~ 1.5), which would otherwise bias the
    null low by a constant factor.
    """
    groups = frame_groups(b)
    trees = {f: cKDTree(b.pos[idx]) for f, idx in groups.items()}
    counts = {f: idx.size for f, idx in groups.items()}

    thr = np.quantile(b.intensity, 0.95) if bright_only else -np.inf
    nds, nb = len(DS_SWEEP), len(R_BANDS)
    seeds = np.zeros((nds, nb))
    real = np.zeros((nds, nb))
    null = np.zeros((nds, nb))

    for f, idx in groups.items():
        sel = idx[b.intensity[idx] >= thr] if bright_only else idx
        if sel.size == 0:
            continue
        x, y, z = b.pos[sel, 0], b.pos[sel, 1], b.pos[sel, 2]
        r = np.hypot(y, z)
        band = np.full(sel.size, -1, dtype=int)
        for bi, (lo, hi) in enumerate(R_BANDS):
            band[(r >= lo) & (r < hi)] = bi
        valid_seed = band >= 0
        if not valid_seed.any():
            continue

        tree_real = trees[f]
        f_null = null_frame(f)
        tree_null = trees.get(f_null)
        occ = counts[f_null] / counts[f] if tree_null is not None else np.nan

        # Build every probe point for this frame in one shot: (ds, sign, seed).
        s_seed = y / r
        qx, qy, qz, qi, qd = [], [], [], [], []
        for di, ds in enumerate(DS_SWEEP):
            for sgn in (+1.0, -1.0):
                s_g = s_seed + sgn * ds
                good = np.abs(s_g) < 1.0
                rg = np.where(good, r, np.nan)
                y_g = rg * s_g
                z_g = rg * np.sqrt(np.clip(1.0 - s_g**2, 0.0, None))
                ok = good & in_volume(y_g, z_g) & valid_seed
                if not ok.any():
                    continue
                qx.append(x[ok]); qy.append(y_g[ok]); qz.append(z_g[ok])
                qi.append(band[ok]); qd.append(np.full(int(ok.sum()), di))
        if not qx:
            continue
        q = np.stack([np.concatenate(qx), np.concatenate(qy), np.concatenate(qz)], axis=1)
        qi = np.concatenate(qi); qd = np.concatenate(qd)
        flat = qd * nb + qi
        np.add.at(seeds.ravel(), flat, 1.0)
        hr = tree_real.query_ball_point(q, MATCH_TOL_MM, return_length=True, workers=1)
        np.add.at(real.ravel(), flat, hr.astype(float))
        if tree_null is not None:
            hn = tree_null.query_ball_point(q, MATCH_TOL_MM, return_length=True, workers=1)
            np.add.at(null.ravel(), flat, hn.astype(float) / occ)
    return dict(seeds=seeds, real=real, null=null)


# A replica population does not produce a delta function in ds.  Two effects set
# its width: the elevation mainlobe (aperture extent 7 * 1.664 = 11.65 mm, so a
# sine FWHM of ~0.886 * lambda / L = 0.061, broadened by the Tukey taper and
# narrowed by the two-way response) and the 0.8 mm match tolerance, which at
# r ~ 20 mm is a +/-0.04 boxcar in ds.  Convolved, expect FWHM ~0.11-0.14, so the
# signal window must be ~+/-0.05 and -- critically -- the baseline shoulders must
# sit OUTSIDE that, or the bump fits away its own baseline.
SIG_HALF = 0.05
SHOULDER_IN, SHOULDER_OUT = 0.13, 0.28


def local_peak_test(rate: np.ndarray, centre: float = DS_LOBE,
                    half: float = SIG_HALF):
    """Excess of ``rate`` in a window at ``centre`` over a flanking baseline.

    Returns (observed, baseline, ratio, signal_mask).
    """
    d = DS_SWEEP
    sig = np.abs(d - centre) <= half
    off = np.abs(d - centre)
    flank = (off >= SHOULDER_IN) & (off <= SHOULDER_OUT)
    ok = np.isfinite(rate)
    if (sig & ok).sum() < 5 or (flank & ok).sum() < 10:
        return np.nan, np.nan, np.nan, sig
    coef = np.polyfit(d[flank & ok], rate[flank & ok], 1)
    base = np.polyval(coef, d[sig & ok]).mean()
    obs = rate[sig & ok].mean()
    return obs, base, (obs / base if base > 0 else np.nan), sig


def bump_apex(rate: np.ndarray, lo: float = 0.30, hi: float = 0.70):
    """Location of the maximum of ``rate`` within [lo, hi], parabola-refined.

    The discriminator that matters: a grating lobe pins the apex at lambda/d in
    every range band, while any elevation-edge or in-volume-selection artifact
    would put it near the largest reachable offset, 2 * Y_max / r, which moves
    with the band.
    """
    m = (DS_SWEEP >= lo) & (DS_SWEEP <= hi) & np.isfinite(rate)
    if m.sum() < 5:
        return np.nan, np.nan
    idx = np.flatnonzero(m)
    k = idx[np.argmax(rate[idx])]
    if k <= 0 or k >= len(DS_SWEEP) - 1 or not np.isfinite(rate[k - 1:k + 2]).all():
        return float(DS_SWEEP[k]), float(rate[k])
    y0, y1, y2 = rate[k - 1], rate[k], rate[k + 1]
    denom = y0 - 2 * y1 + y2
    shift = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
    step = DS_SWEEP[1] - DS_SWEEP[0]
    return float(DS_SWEEP[k] + shift * step), float(y1)


# ------------------------------------------------------------------ stats -----


def poisson_ratio_ci(k1: float, k2: float, conf: float = 0.95):
    """CI on k1/k2 for two independent Poisson counts (Gaussian on log)."""
    if k1 <= 0 or k2 <= 0:
        return (np.nan, np.nan)
    lo = np.exp(np.log(k1 / k2) - 1.96 * np.sqrt(1.0 / k1 + 1.0 / k2))
    hi = np.exp(np.log(k1 / k2) + 1.96 * np.sqrt(1.0 / k1 + 1.0 / k2))
    return (lo, hi)


# ------------------------------------------------------------------- main -----


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--det-dir", default="outputs/corrected_full")
    ap.add_argument("--out", default="outputs/render/grating_lobe_audit.png")
    ap.add_argument("--json-out", default="outputs/render/grating_lobe_audit.json")
    ap.add_argument("--cache", default="outputs/render/grating_lobe_audit_cache.npz",
                    help="reuse the sweep/pair counts if present; delete to recompute")
    args = ap.parse_args()

    det_dir = Path(args.det_dir)
    batches = [load_batch(det_dir / f"detections_{n}.npz") for n in BATCHES]
    print(f"loaded {len(batches)} batches, {sum(b.pos.shape[0] for b in batches):,} detections")

    cache = Path(args.cache)
    keys = ("seeds", "real", "null")
    if cache.exists():
        z = np.load(cache)
        real_hist, null_hist = z["real_hist"], z["null_hist"]
        cond_all = [{k: z[f"all_{k}_{i}"] for k in keys} for i in range(len(BATCHES))]
        cond_bright = [{k: z[f"bright_{k}_{i}"] for k in keys} for i in range(len(BATCHES))]
        print(f"reused counts from {cache}")
    else:
        real_hist = np.zeros((len(R_BANDS), len(DS_CENTRES)))
        null_hist = np.zeros_like(real_hist)
        cond_all, cond_bright = [], []
        for b in batches:
            rh, nh = pair_histograms(b)
            real_hist += rh
            null_hist += nh
            cond_all.append(conditional_sweep(b))
            cond_bright.append(conditional_sweep(b, bright_only=True))
            print(f"  {b.name}: pairs real={rh.sum():.0f} null={nh.sum():.0f} "
                  f"probes={cond_all[-1]['seeds'].sum():.0f}")
        blob = dict(real_hist=real_hist, null_hist=null_hist)
        for i in range(len(BATCHES)):
            for k in keys:
                blob[f"all_{k}_{i}"] = cond_all[i][k]
                blob[f"bright_{k}_{i}"] = cond_bright[i][k]
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **blob)
        print(f"cached counts to {cache}")

    def pool(cond_list):
        return {k: np.sum([c[k] for c in cond_list], axis=0)
                for k in ("seeds", "real", "null")}

    cond_pooled = pool(cond_all)
    cond_pooled_bright = pool(cond_bright)

    def rate_of(pooled, band_sel):
        s = pooled["seeds"][:, band_sel].sum(axis=1)
        rr = pooled["real"][:, band_sel].sum(axis=1)
        nn = pooled["null"][:, band_sel].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (np.where(s > 200, rr / s, np.nan),
                    np.where(s > 200, nn / s, np.nan), s)

    # -------------------------------------------------------------- figure ----
    fig = plt.figure(figsize=(16.5, 9.5))
    gs = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.27,
                          left=0.055, right=0.985, top=0.90, bottom=0.075)

    # A: geometry -- where can a replica land at all?
    ax = fig.add_subplot(gs[0, 0])
    rr = np.linspace(10.2, 41.0, 400)
    frac = np.array([testable_fraction(r) for r in rr])
    ax.plot(rr, 100 * frac, color="#c1121f", lw=2.2, label="replica in-volume")
    ax.fill_between(rr, 0, 100 * frac, color="#c1121f", alpha=0.13)
    r_cut = 2 * Y_DET_MM / DS_LOBE
    ax.axvline(r_cut, color="k", ls="--", lw=1.2)
    ax.annotate(f"r = {r_cut:.1f} mm\nno replica beyond", xy=(r_cut, 60),
                xytext=(r_cut + 1.0, 62), fontsize=8.5, va="center")
    ax.set_xlabel("round-trip range r = hypot(y, z)  [mm]")
    ax.set_ylabel("% of elevation range with in-volume replica")
    ax.set_title("A. Where the test can speak\n"
                 fr"$\Delta s = \lambda/d = {DS_LOBE:.4f}$  ({THETA_LOBE_DEG:.1f}$^\circ$ on axis)",
                 fontsize=10.5)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)

    # B: pooled ds histogram, real vs null, for the lobe-supporting bands.
    ax = fig.add_subplot(gs[0, 1])
    support = [i for i, (lo, hi) in enumerate(R_BANDS) if lo < r_cut]
    r_sup = real_hist[support].sum(axis=0)
    n_sup = null_hist[support].sum(axis=0)
    ax.step(DS_CENTRES, r_sup / max(r_sup.sum(), 1), where="mid",
            color="#023e8a", lw=1.8, label=f"same frame (n={r_sup.sum():.0f})")
    ax.step(DS_CENTRES, n_sup / max(n_sup.sum(), 1), where="mid",
            color="#8d99ae", lw=1.6, ls="--", label=f"cross-frame null (n={n_sup.sum():.0f})")
    ax.axvspan(DS_LOBE - 0.03, DS_LOBE + 0.03, color="#c1121f", alpha=0.18, zorder=0)
    for v in CONTROL_OFFSETS:
        ax.axvspan(v - 0.03, v + 0.03, color="#606c38", alpha=0.12, zorder=0)
    ax.set_xlabel(r"$|\Delta s|$ between same-$x$, same-$r$ detections")
    ax.set_ylabel("normalized density")
    ax.set_title(f"B. Elevation-sine pair autocorrelation\nbands with r < {r_cut:.1f} mm; "
                 "red = predicted, olive = control", fontsize=10.5)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25)

    # C: the discriminator.  Per-band same-frame/null ratio curves.  A grating
    # lobe pins the bump at lambda/d in every band; an elevation-edge or
    # in-volume-selection artifact would ride the largest reachable offset
    # 2 * Y / r, which moves left as the band deepens.
    ax = fig.add_subplot(gs[0, 2])
    cmap = plt.get_cmap("viridis")
    apex_rows = []
    for bi, (lo, hi) in enumerate(R_BANDS):
        s = cond_pooled["seeds"][:, bi]
        if s.sum() < 5000:
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            cur = np.where((s > 100) & (cond_pooled["null"][:, bi] > 5),
                           cond_pooled["real"][:, bi]
                           / np.maximum(cond_pooled["null"][:, bi], 1e-9), np.nan)
        col = cmap(bi / max(len(R_BANDS) - 1, 1))
        ax.plot(DS_SWEEP, cur, color=col, lw=1.6, label=f"r {lo:.0f}-{hi:.0f}")
        apex, val = bump_apex(cur)
        r_mid = 0.5 * (lo + hi)
        edge = 2 * Y_DET_MM / r_mid
        if np.isfinite(apex):
            ax.plot([apex], [val], marker="v", ms=7, color=col, mec="k", mew=0.6)
        ax.plot([min(edge, 0.94)], [1.05], marker="|", ms=9, color=col)
        apex_rows.append(dict(band=[lo, hi], apex=apex, value=val,
                              edge_offset=float(edge)))
    ax.axvline(DS_LOBE, color="#c1121f", lw=1.6, ls="--")
    ax.axhline(1.0, color="k", lw=0.9)
    ax.set_xlabel(r"probe offset $\Delta s$")
    ax.set_ylabel("same-frame / cross-frame-null hit ratio")
    ax.set_title(r"C. Bump apex vs band. $\blacktriangledown$ = apex, "
                 r"$|$ = edge offset $2Y/r$" "\n"
                 "lobe pins the apex; an edge artifact would track the ticks",
                 fontsize=10.5)
    ax.legend(fontsize=7.5, ncol=2)
    ax.grid(alpha=0.25)

    # D: the decisive sweep -- probe-hit rate as a continuous function of ds.
    # Both curves carry the same geometry (which seeds have an in-volume probe,
    # where that probe sits in the depth/elevation density field); only the
    # same-frame curve can carry replicas.  Their ratio is the test.
    ax = fig.add_subplot(gs[1, 0])
    band_sel = [i for i, (lo, hi) in enumerate(R_BANDS) if lo < r_cut]
    rr, nn, ss = rate_of(cond_pooled, band_sel)
    rr_b, nn_b, ss_b = rate_of(cond_pooled_bright, band_sel)
    ax.plot(DS_SWEEP, 1e3 * rr, color="#c1121f", lw=2.0, label="same frame")
    ax.plot(DS_SWEEP, 1e3 * nn, color="#8d99ae", lw=1.8, ls="--",
            label="cross-frame null (occupancy-corrected)")
    ax.axvline(DS_LOBE, color="k", lw=1.4, ls=":")
    ax.set_xlabel(r"probe offset $\Delta s$ in elevation direction-sine")
    ax.set_ylabel("probe hits per 1000 probes")
    ax.set_title(r"D. Conditional replica sweep, $r<%.1f$ mm" % r_cut
                 + "\nboth curves share the geometry; only red can hold replicas",
                 fontsize=10.5)
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(alpha=0.25)
    ax2 = ax.twinx()
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio_curve = rr / nn
    ax2.plot(DS_SWEEP, ratio_curve, color="#023e8a", lw=1.5, alpha=0.85)
    ax2.set_ylabel("same-frame / null", color="#023e8a", fontsize=9)
    ax2.tick_params(axis="y", labelcolor="#023e8a", labelsize=8)
    ax2.axvspan(DS_LOBE - 0.025, DS_LOBE + 0.025, color="#c1121f", alpha=0.15, zorder=0)

    # E: local-baseline peak ratio per batch x band x seed-set, computed on the
    # same-frame/null ratio curve so the smooth geometry+density trend is
    # divided out before the peak test rather than fitted away.
    ax = fig.add_subplot(gs[1, 1])
    excess_rows = []
    for cl, tag in ((cond_all, "all"), (cond_bright, "bright")):
        for bidx, c in enumerate(cl):
            for bi, (lo, hi) in enumerate(R_BANDS):
                if lo >= r_cut:
                    continue
                s = c["seeds"][:, bi]
                with np.errstate(invalid="ignore", divide="ignore"):
                    rate = np.where((s > 100) & (c["null"][:, bi] > 5),
                                    c["real"][:, bi] / np.maximum(c["null"][:, bi], 1e-9),
                                    np.nan)
                obs, base, ratio, sig = local_peak_test(rate)
                if not np.isfinite(ratio):
                    continue
                k_sig = c["real"][sig, bi].sum()
                k_eff = base * c["seeds"][sig, bi].sum()
                lo_ci, hi_ci = poisson_ratio_ci(k_sig, max(k_eff, 1e-9))
                excess_rows.append(dict(set=tag, batch=BATCHES[bidx], band=[lo, hi],
                                        obs=obs, base=base, ratio=ratio,
                                        k_sig=k_sig, k_expected=k_eff,
                                        ci=[lo_ci, hi_ci]))
    if excess_rows:
        yv = np.arange(len(excess_rows))
        rat = np.array([e["ratio"] for e in excess_rows])
        cis = np.clip(np.array([e["ci"] for e in excess_rows]), 1e-3, 1e3)
        col = ["#c1121f" if e["set"] == "all" else "#023e8a" for e in excess_rows]
        ax.errorbar(rat, yv, xerr=np.abs(cis.T - rat), fmt="none",
                    ecolor="#adb5bd", elinewidth=1.1, zorder=3)
        ax.scatter(rat, yv, c=col, s=20, zorder=4)
        ax.axvline(1.0, color="k", lw=1.2)
        ax.set_yticks(yv)
        ax.set_yticklabels([f"{e['set'][:5]} {e['batch']} r{e['band'][0]:.0f}-{e['band'][1]:.0f}"
                            for e in excess_rows], fontsize=6.0)
        ax.set_xscale("log")
        ax.set_xlim(0.3, 4.0)
        ax.set_xlabel(r"peak ratio at $\lambda/d$ vs local baseline")
    ax.set_title(f"E. Peak test on same-frame/null ratio ({len(excess_rows)} tests)\n"
                 "1.0 = no lobe; red all dets, blue bright-only", fontsize=10.5)
    ax.grid(alpha=0.25, axis="x")

    # F: detection density in (y, z) -- a strong lobe would mirror bright
    # structure toward the elevation edges at shallow depth.
    ax = fig.add_subplot(gs[1, 2])
    ypool = np.concatenate([b.pos[:, 1] for b in batches])
    zpool = np.concatenate([b.pos[:, 2] for b in batches])
    h, ye, ze = np.histogram2d(ypool, zpool, bins=[49, 80],
                               range=[[-Y_MAX_MM, Y_MAX_MM], [10.0, 40.8]])
    hn = h / np.maximum(h.sum(axis=0, keepdims=True), 1)
    im = ax.pcolormesh(ze, ye, hn, cmap="magma", shading="auto")
    zz = np.linspace(10.2, 27.5, 200)
    ax.plot(zz, np.minimum(zz * np.tan(np.radians(THETA_LOBE_DEG)), Y_MAX_MM),
            color="#4cc9f0", lw=1.6, ls="--", label="replica of on-axis scatterer")
    ax.plot(zz, -np.minimum(zz * np.tan(np.radians(THETA_LOBE_DEG)), Y_MAX_MM),
            color="#4cc9f0", lw=1.6, ls="--")
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("elevation y [mm]")
    ax.set_title("F. Detection density, depth-normalized", fontsize=10.5)
    ax.legend(fontsize=7.5, loc="upper right")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    fig.suptitle(
        f"J5 elevation grating-lobe audit  |  8 rows @ {PITCH_MM:.3f} mm = "
        fr"{PITCH_MM/LAMBDA_MM:.2f}$\lambda$  ($\lambda$ = {LAMBDA_MM:.3f} mm, c = 1600 m/s, "
        f"f = 2.0 MHz)  |  {sum(b.pos.shape[0] for b in batches):,} detections, "
        f"{len(BATCHES)} batches / 60 acquisitions",
        fontsize=12.5, y=0.965,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")

    # --------------------------------------------------------- text report ----
    def ratio_curve_of(pooled, bi):
        s = pooled["seeds"][:, bi]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where((s > 100) & (pooled["null"][:, bi] > 5),
                            pooled["real"][:, bi] / np.maximum(pooled["null"][:, bi], 1e-9),
                            np.nan)

    def band_report(pooled, tag):
        print(f"\n== replica sweep, local-peak test on same-frame/null at "
              f"ds={DS_LOBE:.4f} [{tag}] ==")
        print(f"{'band':>10} {'probes':>10} {'obs':>8} {'baseline':>9} {'peak':>7} "
              f"{'95% CI':>17} {'excess hits':>12}")
        rows = []
        for bi, (lo, hi) in enumerate(R_BANDS):
            s = pooled["seeds"][:, bi]
            if s.sum() < 1:
                print(f"{lo:4.0f}-{hi:<4.0f} {0:>10.0f}   (no in-volume replica possible)")
                rows.append(dict(band=[lo, hi], probes=0.0, testable=False))
                continue
            obs, base, ratio, sig = local_peak_test(ratio_curve_of(pooled, bi))
            n_sig_probes = pooled["seeds"][sig, bi].sum()
            k_sig = pooled["real"][sig, bi].sum()
            # Expected same-frame hits if the ratio held its local baseline.
            k_exp = base * pooled["null"][sig, bi].sum() if np.isfinite(base) else np.nan
            lo_ci, hi_ci = (poisson_ratio_ci(k_sig, max(k_exp, 1e-9))
                            if np.isfinite(k_exp) else (np.nan, np.nan))
            print(f"{lo:4.0f}-{hi:<4.0f} {n_sig_probes:>10.0f} {obs:>8.3f} "
                  f"{base:>9.3f} {ratio:>7.3f} [{lo_ci:>6.3f},{hi_ci:>6.3f}] "
                  f"{k_sig - k_exp:>12.1f}")
            rows.append(dict(band=[lo, hi], probes=float(n_sig_probes), testable=True,
                             obs=float(obs), baseline=float(base), ratio=float(ratio),
                             k_sig=float(k_sig), k_expected=float(k_exp),
                             ci=[float(lo_ci), float(hi_ci)]))
        return rows

    rows_all = band_report(cond_pooled, "all detections")
    rows_bright = band_report(cond_pooled_bright, "bright top-5%")

    # False-centre calibration.  The peak statistic is only meaningful if it does
    # NOT return ~the same value when pointed at offsets physics does not predict.
    # False centres must sit far enough away that their signal window does not
    # overlap the bump, otherwise the calibration measures the bump twice.
    print("\n== false-centre calibration of the peak statistic (pooled, r < r_cut) ==")
    centres = np.round(np.arange(0.16, 0.781, 0.01), 4)
    centres = centres[np.abs(centres - DS_LOBE) >= 2 * SIG_HALF + 0.04]
    fc = []
    for cvals, tag in ((ratio_curve, "same-frame/null ratio"),
                       (rr, "raw same-frame rate")):
        vals = np.array([local_peak_test(cvals, centre=c)[2] for c in centres])
        ok = np.isfinite(vals)
        at_lobe = local_peak_test(cvals, centre=DS_LOBE)[2]
        rank = int((vals[ok] >= at_lobe).sum())
        print(f"  {tag:24s} at lambda/d = {at_lobe:6.3f} | {ok.sum()} false centres: "
              f"median {np.nanmedian(vals):.3f}, 95th pct {np.nanpercentile(vals[ok], 95):.3f}, "
              f"max {np.nanmax(vals):.3f} | exceeded by {rank}")
        fc.append(dict(curve=tag, at_lobe=float(at_lobe),
                       median=float(np.nanmedian(vals)),
                       p95=float(np.nanpercentile(vals[ok], 95)),
                       maximum=float(np.nanmax(vals)),
                       n_exceeding=rank, n=int(ok.sum())))

    print("\n== bump apex per range band (the lobe-vs-edge-artifact discriminator) ==")
    print(f"{'band':>10} {'probes':>10} {'apex ds':>8} {'value':>7} {'2Y/r':>7} "
          f"{'apex - lambda/d':>16}")
    for row in apex_rows:
        lo, hi = row["band"]
        pr = cond_pooled["seeds"][:, [i for i, b in enumerate(R_BANDS)
                                      if list(b) == [lo, hi]][0]].sum()
        print(f"{lo:4.0f}-{hi:<4.0f} {pr:>10.0f} {row['apex']:>8.3f} "
              f"{row['value']:>7.3f} {row['edge_offset']:>7.3f} "
              f"{row['apex'] - DS_LOBE:>+16.3f}")

    # Sensitivity floor on the ratio curve: how big a replica population would
    # have shown up?  A replica present in a fraction eta of parent detections
    # adds eta * probes hits on top of the null-predicted background.
    obs, base, ratio, sig = local_peak_test(ratio_curve)
    n_probe = cond_pooled["seeds"][sig][:, band_sel].sum()
    k_bg = base * cond_pooled["null"][sig][:, band_sel].sum()
    k_obs = cond_pooled["real"][sig][:, band_sel].sum()
    eta_3sigma = 3.0 * np.sqrt(max(k_bg, 1.0)) / n_probe
    print(f"\npooled testable bands: probes={n_probe:.0f} background={k_bg:.1f} "
          f"observed={k_obs:.1f} peak ratio={ratio:.4f}")
    print(f"3-sigma sensitivity floor: a replica present in "
          f"{100*eta_3sigma:.3f}% of parent detections would have been seen")

    # Pair-histogram bump test at the predicted offset vs controls.
    print("\n== pair autocorrelation, local-baseline peak at the predicted offset ==")
    print(f"{'band':>10} {'pairs':>8} {'obs':>9} {'baseline':>9} {'ratio':>7} "
          f"{'95% CI':>17}")
    pair_rows = []

    def hist_peak(hist):
        """Local-peak ratio on a ds histogram, same shoulders as the sweep."""
        d = DS_CENTRES
        sig = np.abs(d - DS_LOBE) <= SIG_HALF
        off = np.abs(d - DS_LOBE)
        flank = (off >= SHOULDER_IN) & (off <= SHOULDER_OUT)
        if hist[flank].sum() < 30 or sig.sum() < 2:
            return np.nan, np.nan, np.nan
        coef = np.polyfit(d[flank], hist[flank], 1)
        base = np.polyval(coef, d[sig]).sum()
        return float(hist[sig].sum()), float(base), float(hist[sig].sum() / base)

    for bi, (lo, hi) in enumerate(R_BANDS):
        rr_b = real_hist[bi]
        if rr_b.sum() < 500:
            print(f"{lo:4.0f}-{hi:<4.0f} {rr_b.sum():>8.0f}   (too few pairs)")
            continue
        o, bse, rt = hist_peak(rr_b)
        if not np.isfinite(rt):
            print(f"{lo:4.0f}-{hi:<4.0f} {rr_b.sum():>8.0f}   "
                  f"(offset unreachable at this range)")
            pair_rows.append(dict(band=[lo, hi], pairs=float(rr_b.sum()), testable=False))
            continue
        lc, hc = poisson_ratio_ci(o, max(bse, 1e-9))
        print(f"{lo:4.0f}-{hi:<4.0f} {rr_b.sum():>8.0f} {o:>9.0f} {bse:>9.1f} "
              f"{rt:>7.3f} [{lc:>6.3f},{hc:>6.3f}]")
        pair_rows.append(dict(band=[lo, hi], pairs=float(rr_b.sum()), testable=True,
                              obs=o, baseline=bse, ratio=rt, ci=[float(lc), float(hc)]))

    print(f"\ngeometry: replica leaves the reconstructed volume for r > {r_cut:.2f} mm "
          f"(all elevations); on-axis scatterers lose their replica for "
          f"z > {Y_DET_MM/np.tan(np.radians(THETA_LOBE_DEG)):.2f} mm")
    for r in (11, 14, 18, 22, 26, 28, 32):
        print(f"   r = {r:>2} mm : {100*testable_fraction(float(r)):5.1f}% of elevation testable")

    report = dict(
        lambda_mm=LAMBDA_MM, pitch_mm=PITCH_MM, ds_lobe=DS_LOBE,
        theta_lobe_deg=THETA_LOBE_DEG, r_cut_mm=float(r_cut),
        onaxis_z_cut_mm=float(Y_DET_MM / np.tan(np.radians(THETA_LOBE_DEG))),
        match_tol_mm=MATCH_TOL_MM, control_offsets=list(CONTROL_OFFSETS),
        conditional_all=rows_all, conditional_bright=rows_bright,
        pair_autocorr=pair_rows, per_test_excess=excess_rows,
        n_tests=len(excess_rows),
        pooled_peak_ratio=float(ratio), pooled_probes=float(n_probe),
        eta_3sigma=float(eta_3sigma),
        false_centre_calibration=fc, bump_apex_by_band=apex_rows,
        ds_sweep=DS_SWEEP.tolist(),
        sweep_rate_all=rr.tolist(),
        sweep_rate_null=nn.tolist(),
        sweep_ratio=ratio_curve.tolist(),
    )
    jp = Path(args.json_out)
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(report, indent=2, default=float))
    print(f"wrote {jp}")


if __name__ == "__main__":
    main()
