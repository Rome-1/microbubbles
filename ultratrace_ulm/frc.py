"""Split-half Fourier Ring/Shell Correlation (FRC/FSC) for 3D ULM tracks.

This is the *no-ground-truth arbiter* for our 3D ULM reconstructions. We never
observe the true vasculature, so we cannot directly measure whether a tracking /
detection setting that produces MORE or LONGER tracks recovered real structure
or merely manufactured artifacts. Track count and length cannot tell those
apart -- both go up for real microvessels AND for clutter.

FRC (Fourier Ring Correlation; Hingot 2021, Heiles 2022 for ULM; van Heel 2005
for the half-bit threshold) sidesteps ground truth. Split the localizations into
two INDEPENDENT halves that sample the SAME vasculature, reconstruct a density
map from each, and measure their correlation as a function of spatial frequency:

  * A method that recovers REAL structure -> the two half-maps agree out to high
    spatial frequency -> FRC stays near 1 to high q -> fine resolution_mm.
  * A method that adds SPURIOUS tracks -> the spurious parts differ between the
    two independent halves (noise does not reproduce) -> FRC falls to ~0 sooner
    -> coarser resolution_mm and a lower repro_score.

So FRC simultaneously gives a *resolution* (how fine is the reproducible detail)
and a *reproducibility score* (how much of the signal reproduces at all). Both
are usable to RANK competing methods on the same data with no ground truth.

------------------------------------------------------------------------------
CAVEAT for our anisotropic-elevation geometry (READ THIS)
------------------------------------------------------------------------------
Our y axis is SYNTHESIZED elevation: 25 elevation planes reconstructed from one
physical receive row, sampled ~3.8x coarser than x/z (dy ~ 0.555 mm vs
dx ~ dz ~ 0.146 mm). y is the LEAST reliable axis. Therefore:

  * We report FRC per projection plane (xz, yz, xy) AND a 3D FSC, NOT just one
    number. The xz plane (lateral x vs depth z) is the trustworthy reference.
    The yz and xy planes -- the ones that INVOLVE elevation -- will show coarser
    resolution_mm and lower correlation, and that is EXPECTED, not a bug.
  * The point of scrutinising the y-involving planes is exactly to see whether a
    method (e.g. elevation-anisotropic Kalman, an SVD cutoff change) IMPROVES or
    DEGRADES elevation reproducibility. If a change lifts yz/xy FRC without
    hurting xz, it genuinely helped the weak axis. If it only inflates track
    count while yz/xy FRC stays flat or drops, the extra tracks are not real
    elevation structure.

This module is pure numpy/scipy, CPU-only. Every number is a reproducibility
PROXY; none proves correctness. Use it as a ranking arbiter, not an oracle.
"""

from __future__ import annotations

import numpy as np

from .runtime import load_pickle

# ----------------------------------------------------------------------------
# Grid / density reconstruction
# ----------------------------------------------------------------------------

_MAX_N = 512  # hard clamp on voxels-per-axis (keeps FFTs sane)


def _grid_params(
    bounds_min,
    bounds_max,
    voxel_mm: float,
    max_n: int = _MAX_N,
) -> tuple[float, np.ndarray]:
    """Effective ISOTROPIC voxel size (mm) and per-axis voxel count.

    A single voxel size is used for all three axes so that any plane projection
    is isotropic in the reconstructed grid -- that lets FRC bin by physical
    frequency cleanly. If the requested ``voxel_mm`` would push any axis past
    ``max_n`` voxels, the voxel is COARSENED (uniformly) so the largest axis
    lands at ~``max_n``; this keeps the grid isotropic while clamping its size.
    """
    bmin = np.asarray(bounds_min, dtype=np.float64).reshape(-1)
    bmax = np.asarray(bounds_max, dtype=np.float64).reshape(-1)
    extent = np.maximum(bmax - bmin, 0.0)
    max_extent = float(np.max(extent)) if extent.size else 0.0
    v = float(voxel_mm)
    if not np.isfinite(v) or v <= 0:
        v = 0.07
    if max_extent > 0:
        v = max(v, max_extent / float(max_n))
    n = np.where(extent > 0, np.ceil(extent / v), 1).astype(np.int64)
    n = np.clip(n, 1, max_n)
    return v, n


def reconstruct_density(
    points_xyz: np.ndarray,
    bounds_min,
    bounds_max,
    voxel_mm: float = 0.07,
) -> np.ndarray:
    """Bin localization points into a 3D super-resolution density grid.

    Args:
        points_xyz: (N, 3) float, localization positions in mm as (x, y, z).
        bounds_min / bounds_max: length-3 mm lower/upper grid corners.
        voxel_mm: target super-res voxel size (mm). Default 0.07 mm ~ dx/2, i.e.
            a few times finer than the tracking grid (dx ~ dz ~ 0.146 mm). If the
            requested voxel would exceed ``_MAX_N`` (512) voxels on any axis, it
            is uniformly coarsened (see ``_grid_params``).

    Returns:
        density (nx, ny, nz) float32, voxel-occupancy COUNTS (weighted by the
        number of localizations falling in each voxel). Axis 0 = x, 1 = y, 2 = z,
        so the plane projections used by FRC are: xy = sum over z (axis 2),
        xz = sum over y (axis 1), yz = sum over x (axis 0).

    Counts (not max) are used so the map is a true localization-density image,
    matching how ULM super-resolution renders accumulate microbubble events.
    """
    v, n = _grid_params(bounds_min, bounds_max, voxel_mm)
    shape = (int(n[0]), int(n[1]), int(n[2]))
    density = np.zeros(shape, dtype=np.float32)
    pts = np.asarray(points_xyz, dtype=np.float64)
    if pts.size == 0:
        return density
    pts = pts.reshape(-1, 3)
    bmin = np.asarray(bounds_min, dtype=np.float64).reshape(-1)
    idx = np.floor((pts - bmin[None, :]) / v).astype(np.int64)
    idx = np.clip(idx, 0, np.asarray(shape, dtype=np.int64)[None, :] - 1)
    np.add.at(density, (idx[:, 0], idx[:, 1], idx[:, 2]), 1.0)
    return density


# ----------------------------------------------------------------------------
# FRC / FSC core (ring + shell correlation, half-bit threshold)
# ----------------------------------------------------------------------------


def _apodize(shape: tuple[int, ...]) -> np.ndarray:
    """Separable Hann window over an ND map.

    FRC needs apodization: a hard map edge injects a cross-shaped spectral
    leakage that artificially correlates the two halves at high frequency and
    inflates resolution. A Hann taper kills that edge.
    """
    w = np.ones(shape, dtype=np.float64)
    for ax, nn in enumerate(shape):
        win = np.hanning(nn) if nn >= 2 else np.ones(nn)
        shp = [1] * len(shape)
        shp[ax] = nn
        w = w * win.reshape(shp)
    return w


def _freq_magnitude(shape: tuple[int, ...], voxel: float) -> np.ndarray:
    """Radial spatial-frequency magnitude (cycles/mm) for every FFT voxel."""
    q2 = np.zeros(shape, dtype=np.float64)
    for ax, nn in enumerate(shape):
        f = np.fft.fftfreq(nn, d=voxel)
        shp = [1] * len(shape)
        shp[ax] = nn
        q2 = q2 + f.reshape(shp) ** 2
    return np.sqrt(q2)


def _half_bit_threshold(n_q: np.ndarray) -> np.ndarray:
    """van Heel & Schatz (2005) 1/2-bit threshold curve.

    T(q) = (0.2071 + 1.9102/sqrt(n_q)) / (1.2071 + 0.9102/sqrt(n_q))

    ``n_q`` is the number of voxels in the ring/shell. The threshold is high
    where rings are sparse (low frequency, few voxels) and asymptotes to
    ~0.1716 where rings are well populated -- the spatial frequency at which the
    correlation drops below it carries ~1/2 bit of information.
    """
    n = np.asarray(n_q, dtype=np.float64)
    sq = np.sqrt(np.maximum(n, 1.0))
    return (0.2071 + 1.9102 / sq) / (1.2071 + 0.9102 / sq)


def _smooth3(x: np.ndarray) -> np.ndarray:
    """3-point moving average (edge-preserving) for crossing detection only."""
    x = np.asarray(x, dtype=np.float64)
    if x.size < 3:
        return x
    out = x.copy()
    out[1:-1] = (x[:-2] + x[1:-1] + x[2:]) / 3.0
    return out


def _resolution(freqs: np.ndarray, frc: np.ndarray, threshold: np.ndarray, start: int = 1) -> float:
    """First (low->high) frequency where FRC crosses below ``threshold``.

    Returns resolution_mm = 1 / q_cross. The DC ring (index 0) is skipped:
    DC always correlates (+1) because both half-maps carry positive total mass,
    so it is meaningless for resolution. Conventions:
      * crosses at a finite ring  -> interpolated 1/q_cross (the useful case)
      * never crosses (stays above to Nyquist) -> Nyquist-limited 1/freqs[-1]
        (finest the grid can measure)
      * already below at the first usable ring -> inf (no resolved correlation)
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    thr = np.asarray(threshold, dtype=np.float64)
    s = _smooth3(np.asarray(frc, dtype=np.float64))
    n = len(freqs)
    if n <= start:
        return float("inf")
    g = s - thr
    for i in range(start, n):
        if g[i] < 0:
            if i == start:
                return float("inf")
            g0, g1 = g[i - 1], g[i]
            f0, f1 = freqs[i - 1], freqs[i]
            if g0 <= g1:
                qc = f0
            else:
                qc = f0 + (g0 / (g0 - g1)) * (f1 - f0)
            return float(1.0 / qc) if qc > 0 else float("inf")
    return float(1.0 / freqs[-1]) if freqs[-1] > 0 else float("inf")


def _null_curve() -> dict:
    empty = np.zeros(0, dtype=np.float64)
    return {
        "freqs": empty,
        "frc": empty,
        "threshold": empty,
        "n_q": np.zeros(0, dtype=np.int64),
        "resolution_mm_half_bit": float("inf"),
        "resolution_mm_0143": float("inf"),
    }


def frc_curve(map_a: np.ndarray, map_b: np.ndarray, voxel: float, *, apodize: bool = True) -> dict:
    """Ring/shell FRC of two equal-shape ND maps (2D -> ring, 3D -> shell).

    For each radial frequency bin q:
        FRC(q) = sum_ring Re(F_A . conj(F_B)) / sqrt(sum_ring|F_A|^2 . sum_ring|F_B|^2)

    Rings span 0..Nyquist (0.5/voxel) in ``min(shape)//2`` bins (so rings stay
    populated even for non-square maps). Returns freqs (cycles/mm), frc,
    half-bit threshold, per-ring voxel count n_q, and the resolution_mm at both
    the 1/2-bit and the fixed-0.143 crossings.
    """
    a = np.asarray(map_a, dtype=np.float64)
    b = np.asarray(map_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"map shape mismatch: {a.shape} vs {b.shape}")
    if a.size == 0 or not np.any(a) or not np.any(b):
        return _null_curve()

    if apodize:
        w = _apodize(a.shape)
        a = a * w
        b = b * w

    Fa = np.fft.fftn(a).ravel()
    Fb = np.fft.fftn(b).ravel()

    q = _freq_magnitude(a.shape, voxel)
    q_nyq = 0.5 / voxel
    n_rings = max(1, min(a.shape) // 2)
    dq = q_nyq / n_rings
    ring = np.floor(q.ravel() / dq).astype(np.int64)
    ring[(q.ravel() > q_nyq) | (ring >= n_rings)] = -1
    sel = ring >= 0
    ring = ring[sel]

    num = np.real(Fa[sel] * np.conj(Fb[sel]))
    aa = (Fa[sel].real ** 2 + Fa[sel].imag ** 2)
    bb = (Fb[sel].real ** 2 + Fb[sel].imag ** 2)

    sum_num = np.bincount(ring, weights=num, minlength=n_rings)[:n_rings]
    sum_aa = np.bincount(ring, weights=aa, minlength=n_rings)[:n_rings]
    sum_bb = np.bincount(ring, weights=bb, minlength=n_rings)[:n_rings]
    n_q = np.bincount(ring, minlength=n_rings)[:n_rings].astype(np.int64)

    denom = np.sqrt(sum_aa * sum_bb)
    frc = np.where(denom > 0, sum_num / np.where(denom > 0, denom, 1.0), 0.0)
    freqs = (np.arange(n_rings) + 0.5) * dq
    threshold = _half_bit_threshold(n_q)

    return {
        "freqs": freqs,
        "frc": frc,
        "threshold": threshold,
        "n_q": n_q,
        "resolution_mm_half_bit": _resolution(freqs, frc, threshold),
        "resolution_mm_0143": _resolution(freqs, frc, np.full_like(freqs, 0.143)),
    }


def _band_mean(curve: dict, voxel: float) -> float:
    """Mean FRC over the low-to-mid band (DC excluded, up to half-Nyquist).

    DC (ring 0) is excluded because it always correlates at +1 (both half-maps
    have positive total mass) and would mask a spurious method. The band runs to
    half-Nyquist (0.25/voxel) -- the range where REAL structure still reproduces
    but noise has already decorrelated.
    """
    freqs = np.asarray(curve.get("freqs", []), dtype=np.float64)
    frc = np.asarray(curve.get("frc", []), dtype=np.float64)
    if freqs.size <= 1:
        return float("nan")
    half_nyq = 0.25 / voxel
    band = (np.arange(freqs.size) >= 1) & (freqs <= half_nyq)
    if not band.any():
        band = np.arange(freqs.size) >= 1
    if not band.any():
        return float("nan")
    return float(np.mean(frc[band]))


# ----------------------------------------------------------------------------
# Track splitting + split-half FRC
# ----------------------------------------------------------------------------

_PLANES = {
    "xy": 2,  # sum/project over z (axis 2)
    "xz": 1,  # sum/project over y (axis 1)
    "yz": 0,  # sum/project over x (axis 0)
}


def _track_acq(track: dict, frames_per_acq: int) -> int:
    frames = np.asarray(track.get("frames", []))
    if frames.size == 0:
        return 0
    return int(frames.reshape(-1)[0]) // int(frames_per_acq)


def split_tracks(tracks: list[dict], frames_per_acq: int, split: str, seed: int = 0):
    """Partition tracks into two independent halves (A, B).

    ``split='acq_parity'`` (default): a track's source acquisition is
    ``frames[0] // frames_per_acq``; even-acq tracks -> A, odd-acq -> B. Each
    700-frame acquisition is an independent imaging window of the SAME
    vasculature, so even/odd acqs are two independent samples -- the property
    FRC requires. ``split='random'`` assigns each track to A/B by a seeded coin
    flip (fallback when acq structure is unavailable / unreliable).
    """
    a: list[dict] = []
    b: list[dict] = []
    if split == "random":
        rng = np.random.default_rng(seed)
        for t in tracks:
            (a if rng.random() < 0.5 else b).append(t)
    elif split == "acq_parity":
        for t in tracks:
            (a if _track_acq(t, frames_per_acq) % 2 == 0 else b).append(t)
    else:
        raise ValueError(f"unknown split {split!r} (use 'acq_parity' or 'random')")
    return a, b


def _stack_points(tracks: list[dict]) -> np.ndarray:
    pts = [np.asarray(t["positions"], dtype=np.float64).reshape(-1, 3)
           for t in tracks if len(np.asarray(t.get("positions", [])))]
    if not pts:
        return np.empty((0, 3), dtype=np.float64)
    return np.concatenate(pts, axis=0)


def split_half_frc(
    tracks: list[dict],
    frames_per_acq: int,
    *,
    bounds_min=None,
    bounds_max=None,
    voxel_mm: float | None = None,
    split: str = "acq_parity",
    spacing: dict | None = None,
    seed: int = 0,
) -> dict:
    """Split-half FRC/FSC reproducibility report for one track set.

    Splits ``tracks`` into two independent halves (see ``split_tracks``),
    reconstructs a density map per half over a COMMON grid (shared bounds +
    voxel, so the two maps are comparable), then computes 2D FRC for each
    projection plane (xz, yz, xy via SUM projection) and a 3D FSC.

    Why SUM projection (not max): summing integrates the localization density
    through the slab, giving a linear projection of the reconstructed density
    whose Fourier transform is the central slice of the 3D transform -- so the
    plane FRC is a faithful 2D measurement and energy/correlation are preserved.
    A max projection is nonlinear, discards counts, and biases toward bright
    voxels, distorting the frequency content the correlation depends on.

    Args:
        tracks: list of track dicts ({'positions': (N,3) mm xyz, 'frames': (N,)}).
        frames_per_acq: frames per acquisition window (e.g. 700) for acq_parity.
        bounds_min/max: optional shared mm corners. If None, the union bounds of
            BOTH halves are used (so A and B share one grid).
        voxel_mm: super-res voxel (mm). If None, defaults to spacing['dx']/2 when
            ``spacing`` is given, else 0.07 mm.
        split: 'acq_parity' (default) or 'random'.

    Returns a dict:
        'xz'/'yz'/'xz'  -> per-plane curve {freqs, frc, threshold, n_q,
                           resolution_mm_half_bit, resolution_mm_0143}
        'fsc3d'         -> the 3D shell-correlation curve (same keys)
        'repro_score'   -> scalar mean low-band FRC across the planes + 3D FSC
                           (DC excluded, up to half-Nyquist); HIGHER = more
                           reproducible; usable to RANK methods.
        'meta'          -> bounds, effective voxel, grid shape, counts, split.

    NOTE (anisotropic y): yz/xy resolutions are limited by the coarse synthesized
    elevation sampling and will read coarser than xz -- expected; that is exactly
    what makes the y-involving planes a useful test of whether a method helps or
    hurts the weak elevation axis.
    """
    a_tracks, b_tracks = split_tracks(tracks, frames_per_acq, split, seed)
    pts_a = _stack_points(a_tracks)
    pts_b = _stack_points(b_tracks)

    if voxel_mm is None:
        dx = (spacing or {}).get("dx") if spacing else None
        voxel_mm = float(dx) / 2.0 if dx else 0.07

    planes = ("xz", "yz", "xy")

    def _empty_result(reason: str) -> dict:
        out = {p: _null_curve() for p in planes}
        out["fsc3d"] = _null_curve()
        out["repro_score"] = 0.0
        out["meta"] = {
            "split": split,
            "voxel_mm": float(voxel_mm),
            "n_tracks_a": len(a_tracks),
            "n_tracks_b": len(b_tracks),
            "n_points_a": int(len(pts_a)),
            "n_points_b": int(len(pts_b)),
            "empty": reason,
        }
        return out

    if len(pts_a) == 0 or len(pts_b) == 0:
        return _empty_result("one or both halves empty")

    if bounds_min is None or bounds_max is None:
        allp = np.concatenate([pts_a, pts_b], axis=0)
        bmin = allp.min(axis=0)
        bmax = allp.max(axis=0)
        pad = np.maximum((bmax - bmin) * 0.02, (voxel_mm or 0.07) * 2.0)
        bmin = bmin - pad
        bmax = bmax + pad
    else:
        bmin = np.asarray(bounds_min, dtype=np.float64).reshape(-1)
        bmax = np.asarray(bounds_max, dtype=np.float64).reshape(-1)

    extent = bmax - bmin
    if not np.all(extent > 0):
        return _empty_result("degenerate bounds")

    v, n = _grid_params(bmin, bmax, voxel_mm)
    dens_a = reconstruct_density(pts_a, bmin, bmax, v)
    dens_b = reconstruct_density(pts_b, bmin, bmax, v)

    result: dict = {}
    band_means: list[float] = []
    for plane in planes:
        axis = _PLANES[plane]
        proj_a = dens_a.sum(axis=axis)
        proj_b = dens_b.sum(axis=axis)
        curve = frc_curve(proj_a, proj_b, v)
        result[plane] = curve
        bm = _band_mean(curve, v)
        if np.isfinite(bm):
            band_means.append(bm)

    fsc3d = frc_curve(dens_a, dens_b, v)
    result["fsc3d"] = fsc3d
    bm3 = _band_mean(fsc3d, v)
    if np.isfinite(bm3):
        band_means.append(bm3)

    result["repro_score"] = float(np.mean(band_means)) if band_means else 0.0
    result["meta"] = {
        "split": split,
        "voxel_mm": float(v),
        "voxel_mm_requested": float(voxel_mm),
        "grid_shape": tuple(int(x) for x in n),
        "bounds_min": [float(x) for x in bmin],
        "bounds_max": [float(x) for x in bmax],
        "n_tracks_a": len(a_tracks),
        "n_tracks_b": len(b_tracks),
        "n_points_a": int(len(pts_a)),
        "n_points_b": int(len(pts_b)),
    }
    return result


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def _load_tracks_and_fpa(path) -> tuple[list[dict], int, dict | None]:
    data = load_pickle(path)
    tracks = data.get("tracks") or []
    fpa = int(data.get("frames_per_acq") or data.get("n_frames") or 700)
    spacing = data.get("spacing")
    return list(tracks), fpa, spacing


def _union_bounds(track_sets: list[list[dict]]):
    mins = []
    maxs = []
    for tracks in track_sets:
        pts = _stack_points(tracks)
        if len(pts):
            mins.append(pts.min(axis=0))
            maxs.append(pts.max(axis=0))
    if not mins:
        return None, None
    bmin = np.min(np.stack(mins), axis=0)
    bmax = np.max(np.stack(maxs), axis=0)
    pad = np.maximum((bmax - bmin) * 0.02, 0.1)
    return bmin - pad, bmax + pad


def _fmt_res(v: float) -> str:
    if v is None or not np.isfinite(v):
        return "inf"
    return f"{v:.3f}"


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="python -m ultratrace_ulm.frc",
        description=(
            "Split-half FRC/FSC reproducibility arbiter for 3D ULM track pickles. "
            "Finer resolution_mm AND higher repro_score = a more reproducible "
            "(more real, less spurious) reconstruction. All inputs are scored on a "
            "COMMON union grid so resolutions are directly comparable."
        ),
    )
    p.add_argument("pkls", nargs="+", help="tracks pickle file(s)")
    p.add_argument("--split", choices=["acq_parity", "random"], default="acq_parity")
    p.add_argument("--voxel-mm", type=float, default=None,
                   help="super-res voxel (mm); default = first pickle's dx/2, else 0.07")
    p.add_argument("--seed", type=int, default=0, help="seed for --split random")
    args = p.parse_args(argv)

    loaded = []
    track_sets = []
    for path in args.pkls:
        tracks, fpa, spacing = _load_tracks_and_fpa(path)
        loaded.append((path, tracks, fpa, spacing))
        track_sets.append(tracks)

    voxel_mm = args.voxel_mm
    if voxel_mm is None:
        for _, _, _, spacing in loaded:
            dx = (spacing or {}).get("dx") if spacing else None
            if dx:
                voxel_mm = float(dx) / 2.0
                break
        if voxel_mm is None:
            voxel_mm = 0.07

    bmin, bmax = _union_bounds(track_sets)

    rows = []
    for path, tracks, fpa, spacing in loaded:
        res = split_half_frc(
            tracks, fpa, bounds_min=bmin, bounds_max=bmax,
            voxel_mm=voxel_mm, split=args.split, spacing=spacing, seed=args.seed,
        )
        rows.append((path, res))

    name_w = max(12, *(len(__import__("pathlib").Path(p).name) for p, _ in rows))
    header = (f"{'file'.ljust(name_w)} | {'xz_res_mm':>9} | {'yz_res_mm':>9} | "
              f"{'xy_res_mm':>9} | {'fsc3d_res':>9} | {'repro_score':>11}")
    print(f"# split-half FRC (split={args.split}, voxel_mm={voxel_mm:.4g})")
    print(f"# finer res_mm + higher repro_score = better (more reproducible)")
    print(header)
    print("-" * len(header))
    from pathlib import Path
    for path, res in rows:
        name = Path(path).name.ljust(name_w)
        xz = _fmt_res(res["xz"]["resolution_mm_half_bit"])
        yz = _fmt_res(res["yz"]["resolution_mm_half_bit"])
        xy = _fmt_res(res["xy"]["resolution_mm_half_bit"])
        f3 = _fmt_res(res["fsc3d"]["resolution_mm_half_bit"])
        rs = res["repro_score"]
        print(f"{name} | {xz:>9} | {yz:>9} | {xy:>9} | {f3:>9} | {rs:>11.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
