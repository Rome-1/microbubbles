"""Operating-point sweep machinery: normalization arms, matched-rate scoring (J3).

WHY THIS EXISTS
---------------
Two operating points in this pipeline are accidental and they interact:

* the SVD tissue cutoff is a **rank-24 fallback that never fires on purpose** --
  ``spectral_centroid_cutoff`` looks for the first temporal mode whose spectral
  centroid exceeds 100 Hz, no mode on this 222 Hz data gets past ~85-96 Hz, so it
  returns ``round(0.1 * 240) = 24`` every time;
* the detection z-score is computed against ``_slice_stats`` -- mean and std
  pooled **per elevation plane over all frames and all depths**. Attenuation and
  the beam profile vary strongly with z, so one denominator per elevation plane
  is wrong at both ends of the depth axis.

They interact through that denominator: change the rank and you change the
residual whose std normalizes every detection. So they must be swept jointly,
which is what this module and ``scripts/modal/j3_instrument_app.py`` do.

THE ONE RULE
------------
Every comparison is made at **matched detection density** on the un-injected real
volume, or at **matched false-alarm rate** on null data -- never at a matched
threshold. J2 produced a garbage result by comparing two arms at a fixed 2-sigma
cut when their noise floors differed; an arm that merely inflates its own
denominator would otherwise look like a win. :func:`threshold_for_count` and
:func:`threshold_for_fa` are the only sanctioned ways to pick an operating point
here.

That is cheap to enforce because non-maximum suppression is threshold-free: the
peak set is found once and the threshold only selects a suffix of it. One
detection pass per (filter, normalization, volume) therefore yields the entire
detection-rate-vs-threshold curve, and any matched-rate operating point is a
quantile of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

__all__ = [
    "NORM_MODES",
    "zscore_field",
    "find_peaks",
    "threshold_for_count",
    "threshold_for_fa",
    "match_peaks_to_truth",
    "recovery_table",
    "stratify",
]

NORM_MODES: tuple[str, ...] = ("per_elev", "per_elev_zband", "spatial_tgc")


def _ndimage(xp: Any):
    if xp.__name__ == "cupy":
        import cupyx.scipy.ndimage as nd
        return nd
    import scipy.ndimage as nd
    return nd


# --------------------------------------------------------------------------- #
# Normalization arms
# --------------------------------------------------------------------------- #
def zscore_field(
    magnitude: Any,
    *,
    mode: str = "per_elev",
    smoothing_sigma: float = 1.0,
    n_z_bands: int = 8,
    voxel_mm: Sequence[float] = (0.5547, 0.2, 0.2),
    tgc_sigma_lambda: float = 9.0,
    wavelength_mm: float = 0.8,
    xp: Any = np,
) -> Any:
    """z-score a filtered magnitude volume under one of three normalizations.

    ``magnitude`` is ``(frames, elev, z, x)``. Returns a same-shaped float32
    z-score field. All three arms share the in-plane Gaussian pre-smoothing the
    shipped detector applies, so they differ only in the denominator.

    ``per_elev``
        The shipped behaviour (``tracking._slice_stats``): one mean and one std
        per elevation plane, pooled over all frames, depths and lateral
        positions, taken over positive voxels only. This is the arm whose
        depth-pooling is under suspicion.

    ``per_elev_zband``
        The same statistic computed in ``n_z_bands`` depth bands and then
        **linearly interpolated along z**, so the denominator tracks depth
        without introducing band-edge discontinuities that would themselves
        create spurious local maxima. This is the minimal change that tests the
        pooled-statistics concern: if recovery under ``per_elev`` is
        depth-non-uniform and this arm flattens it, the pooling was the cause.

    ``spatial_tgc``
        The full spatially varying normalization: divide by the square root of a
        heavily smoothed time-mean power map, then z-score the result per
        elevation plane. This mirrors ``beamform_core.compute_global_tgc``
        (Gaussian sigma = ``tgc_sigma_lambda * lambda``, per-axis in voxels) but
        is applied to the **post-clutter-filter** residual rather than the raw
        compound. The difference matters: the beamform-time map is dominated by
        tissue, this one by the noise-plus-blood the detector actually competes
        with. Note that at 9 lambda the elevation sigma is ~13 planes of a
        25-plane axis, i.e. nearly flat on that axis -- faithful to the shipped
        formula, and the reason this arm is mostly an in-plane correction.
    """
    nd = _ndimage(xp)
    data = xp.asarray(magnitude, dtype=xp.float32)
    if data.ndim != 4:
        raise ValueError(f"Expected volume (frames,elev,z,x), got {data.shape}")
    if smoothing_sigma > 0:
        data = nd.gaussian_filter(data, sigma=(0, 0, smoothing_sigma, smoothing_sigma))
    n_f, n_e, n_z, n_x = (int(v) for v in data.shape)

    if mode == "spatial_tgc":
        power = xp.mean(data * data, axis=0)  # (elev, z, x)
        sigma_mm = float(tgc_sigma_lambda) * float(wavelength_mm)
        sig = tuple(sigma_mm / float(v) for v in voxel_mm)
        gain = xp.sqrt(xp.maximum(nd.gaussian_filter(power, sigma=sig), 1e-20))
        data = data / gain[None]
        mode_for_stats = "per_elev"
    else:
        mode_for_stats = mode

    if mode_for_stats == "per_elev":
        means = xp.zeros((n_e,), dtype=xp.float32)
        stds = xp.ones((n_e,), dtype=xp.float32)
        for e in range(n_e):
            v = data[:, e]
            m = v > 0
            if bool(xp.any(m)):
                sel = v[m]
                means[e] = sel.mean()
                s = sel.std()
                stds[e] = s if float(s) > 1e-10 else 1.0
        return ((data - means[None, :, None, None]) / (stds[None, :, None, None] + 1e-10)).astype(
            xp.float32
        )

    if mode_for_stats != "per_elev_zband":
        raise ValueError(f"unknown normalization mode {mode!r}; expected one of {NORM_MODES}")

    nb = int(n_z_bands)
    edges = np.linspace(0, n_z, nb + 1).astype(int)
    centers = 0.5 * (edges[:-1] + edges[1:] - 1)
    bm = np.zeros((n_e, nb), dtype=np.float32)
    bs = np.ones((n_e, nb), dtype=np.float32)
    for e in range(n_e):
        for b in range(nb):
            v = data[:, e, edges[b] : edges[b + 1], :]
            m = v > 0
            if bool(xp.any(m)):
                sel = v[m]
                bm[e, b] = float(sel.mean())
                s = float(sel.std())
                bs[e, b] = s if s > 1e-10 else 1.0
    zi = np.arange(n_z, dtype=np.float64)
    mean_z = np.stack([np.interp(zi, centers, bm[e]) for e in range(n_e)], axis=0)
    std_z = np.stack([np.interp(zi, centers, bs[e]) for e in range(n_e)], axis=0)
    mean_z = xp.asarray(mean_z, dtype=xp.float32)[None, :, :, None]
    std_z = xp.asarray(std_z, dtype=xp.float32)[None, :, :, None]
    return ((data - mean_z) / (std_z + 1e-10)).astype(xp.float32)


# --------------------------------------------------------------------------- #
# Threshold-free peak extraction
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Peaks:
    """All local maxima of a z-score field above a permissive floor.

    Because non-maximum suppression is applied before thresholding, this single
    object encodes the detector's behaviour at *every* threshold above ``floor``:
    a threshold ``t`` selects exactly ``zscore >= t``. That is what makes matched
    -density and matched-false-alarm comparisons cheap enough to be the default.
    """

    frame: np.ndarray
    e: np.ndarray
    z: np.ndarray
    x: np.ndarray
    zscore: np.ndarray
    n_frames: int
    n_voxels: int
    floor: float

    def at(self, threshold: float) -> "Peaks":
        m = self.zscore >= float(threshold)
        return Peaks(self.frame[m], self.e[m], self.z[m], self.x[m], self.zscore[m],
                     self.n_frames, self.n_voxels, float(threshold))

    def density_per_frame(self, threshold: float) -> float:
        return float(np.count_nonzero(self.zscore >= threshold)) / max(self.n_frames, 1)


def find_peaks(
    zscore: Any,
    *,
    min_distance: int = 2,
    floor: float = 2.0,
    elev_radius: int | None = None,
    xp: Any = np,
) -> Peaks:
    """Local maxima of a z-score field, kept above a permissive ``floor``.

    NMS geometry follows the shipped detector (``tracking.detect_batch``): an
    isotropic ``(2*min_distance+1)`` cube in voxel units. That is anisotropic in
    *physical* units -- 2 voxels is 0.4 mm in-plane but 1.1 mm in elevation --
    but changing it is J1's business, not this instrument's, so we keep the
    shipped geometry and expose ``elev_radius`` for anyone who wants to vary it.
    """
    nd = _ndimage(xp)
    z = xp.asarray(zscore, dtype=xp.float32)
    r = int(min_distance)
    er = r if elev_radius is None else int(elev_radius)
    size = (1, 2 * er + 1, 2 * r + 1, 2 * r + 1)
    mx = nd.maximum_filter(z, size=size, mode="nearest")
    hit = (z == mx) & (z > float(floor))
    idx = xp.where(hit)
    to_np = (lambda a: xp.asnumpy(a)) if xp.__name__ == "cupy" else (lambda a: np.asarray(a))
    f, e, zz, xx = (to_np(i).astype(np.int32) for i in idx)
    vals = to_np(z[hit]).astype(np.float32)
    order = np.argsort(f, kind="stable")
    return Peaks(
        frame=f[order], e=e[order], z=zz[order], x=xx[order], zscore=vals[order],
        n_frames=int(z.shape[0]), n_voxels=int(np.prod([int(v) for v in z.shape[1:]])),
        floor=float(floor),
    )


# --------------------------------------------------------------------------- #
# Matched-rate operating points
# --------------------------------------------------------------------------- #
def threshold_for_count(peaks: Peaks, target_total: int) -> float:
    """Threshold giving exactly ``target_total`` detections -- matched DENSITY.

    Applied to the peak set of the **un-injected real volume**, this is the
    primary operating-point match: it makes no assumption about what is signal
    and what is not, only that every arm is allowed the same detection budget.
    An arm that wins here won by finding *better* detections within a fixed
    budget, which is the only kind of win that survives the noise-floor
    confound.

    Returns ``+inf`` if the arm cannot even reach the target above its floor --
    a real failure that must not be silently rescaled away.
    """
    n = int(target_total)
    zs = np.sort(peaks.zscore)[::-1]
    if n <= 0:
        return float("inf")
    if n > zs.size:
        return float("inf")
    return float(zs[n - 1])


def threshold_for_fa(peaks: Peaks, fa_per_frame_per_mm3: float, voxel_mm3: float) -> float:
    """Threshold giving a target false-alarm density -- matched FA RATE.

    Applied to a null volume's peak set. Normalizing by volume lets nulls of
    different spatial extent (e.g. a quiet crop) be compared with full volumes.
    """
    vol_mm3 = float(peaks.n_voxels) * float(voxel_mm3)
    target = float(fa_per_frame_per_mm3) * vol_mm3 * float(peaks.n_frames)
    return threshold_for_count(peaks, int(round(target)))


# --------------------------------------------------------------------------- #
# Scoring against injected truth
# --------------------------------------------------------------------------- #
def match_peaks_to_truth(
    truth: dict[str, np.ndarray],
    peaks: Peaks,
    *,
    tol_voxels: tuple[float, float, float] = (1.0, 3.0, 3.0),
) -> np.ndarray:
    """Boolean per truth row: was this injected bubble-frame detected?

    The tolerance is anisotropic in voxels and near-isotropic in millimetres
    (1 x 0.5547 mm elevation vs 3 x 0.2 mm in-plane = 0.55 vs 0.60 mm), which is
    the right convention here: a detector should not be rewarded for elevation
    precision the 0.5547 mm sampling cannot deliver, nor charged for it.

    Matching is one-directional (each truth row asks "is there a peak near me")
    and does not enforce a one-to-one assignment. A peak may therefore satisfy
    two injected bubbles that landed within tolerance of each other. At the
    designed injection density (~30/frame in ~1.06 M voxels) that is rare, and
    the alternative -- a global assignment -- would make recovery depend on
    unrelated bubbles' positions, which is worse for a stratified read-out.
    """
    n = int(truth["frame"].size)
    out = np.zeros(n, dtype=bool)
    if n == 0 or peaks.zscore.size == 0:
        return out
    te, tz, tx = (float(v) for v in tol_voxels)
    bounds = np.searchsorted(peaks.frame, np.arange(peaks.n_frames + 1))
    order = np.argsort(truth["frame"], kind="stable")
    tf = truth["frame"][order]
    fb = np.searchsorted(tf, np.arange(peaks.n_frames + 1))
    for f in range(peaks.n_frames):
        t0, t1 = int(fb[f]), int(fb[f + 1])
        if t0 == t1:
            continue
        p0, p1 = int(bounds[f]), int(bounds[f + 1])
        if p0 == p1:
            continue
        rows = order[t0:t1]
        de = np.abs(truth["e"][rows][:, None] - peaks.e[p0:p1][None, :])
        dz = np.abs(truth["z"][rows][:, None] - peaks.z[p0:p1][None, :])
        dx = np.abs(truth["x"][rows][:, None] - peaks.x[p0:p1][None, :])
        out[rows] = np.any((de <= te) & (dz <= tz) & (dx <= tx), axis=1)
    return out


def recovery_table(
    truth: dict[str, np.ndarray],
    hit: np.ndarray,
    *,
    z_index_to_mm: float = 0.2,
) -> dict[str, Any]:
    """Overall and marginal recovery, plus the depth-uniformity read-out.

    Reports recovery stratified by speed, depth band and SNR (the three the bead
    asks for) and, separately, by ``axial_frac`` -- because the compounding null
    is **axial only**, a speed-only stratification averages a bubble moving into
    the null with one moving across it and would hide a ~15 dB hole.

    ``depth_nonuniformity`` is the max-minus-min recovery across depth bands at
    fixed SNR, which is the specific quantity the pooled-``_slice_stats``
    concern predicts should be large for ``per_elev`` and small for
    ``per_elev_zband``.
    """
    out: dict[str, Any] = {
        "n_truth": int(hit.size),
        "recovery": float(hit.mean()) if hit.size else 0.0,
    }
    for key in ("speed_mms", "snr_db", "z_band", "axial_frac"):
        if key not in truth:
            continue
        out[f"by_{key}"] = stratify(truth[key], hit)
    if "z_band" in truth and "snr_db" in truth:
        per_snr = {}
        for s in np.unique(truth["snr_db"]):
            m = truth["snr_db"] == s
            byband = stratify(truth["z_band"][m], hit[m])
            vals = [v["recovery"] for v in byband.values() if v["n"] >= 20]
            per_snr[f"{float(s):g}"] = (max(vals) - min(vals)) if len(vals) >= 2 else float("nan")
        out["depth_nonuniformity_by_snr"] = per_snr
        finite = [v for v in per_snr.values() if np.isfinite(v)]
        out["depth_nonuniformity"] = float(np.mean(finite)) if finite else float("nan")
    return out


def stratify(values: np.ndarray, hit: np.ndarray) -> dict[str, dict[str, float]]:
    """Recovery rate per unique level of ``values``, with a Wilson 95% interval."""
    out: dict[str, dict[str, float]] = {}
    for v in np.unique(values):
        m = values == v
        n = int(np.count_nonzero(m))
        k = int(np.count_nonzero(hit[m]))
        p = k / n if n else 0.0
        if n:
            zc = 1.959963985
            den = 1.0 + zc * zc / n
            ctr = (p + zc * zc / (2 * n)) / den
            half = zc * np.sqrt(p * (1 - p) / n + zc * zc / (4 * n * n)) / den
            lo, hi = max(0.0, ctr - half), min(1.0, ctr + half)
        else:
            lo = hi = 0.0
        out[f"{float(v):g}"] = {"n": n, "k": k, "recovery": p, "lo95": lo, "hi95": hi}
    return out
