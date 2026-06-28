"""Data-driven SVD cutoff selection from the singular-value spectrum (mb-3k4).

WHY THIS EXISTS
---------------
The shipped "adaptive" cutoff (``svd.spectral_centroid_cutoff``) picks the first
temporal mode whose power-weighted mean frequency exceeds ``tissue_freq_hz``
(default 100 Hz). On THIS acquisition that test never fires: the frame rate is
222 Hz, so the Nyquist frequency is 111 Hz, and the highest centroid any mode
attains is ~60 Hz. The function therefore ALWAYS hits its fallback,
``round(0.1 * n_frames)`` -> **70 modes removed** regardless of the data. The
"adaptive" filter is, in practice, a constant 70-mode cut.

The clutter literature is unanimous that the tissue subspace is **small** -- a
few to low-tens of modes on several-hundred-frame ensembles (Demené 2015 "first
turning point"; Baranger 2018; Lok/Song 2020 BCR plateau ~rank 8, usable 10-24;
McCall 2023 fixed 15-20 %). Removing 70 of 700 modes over-removes by ~10x in
quiet regions and throws away weak blood signal -> the coverage loss we measured.
See ``docs/literature/lit-detection-svd.md`` §1.

WHAT THIS DOES
--------------
- ``singular_value_knee``: the **low (tissue) cutoff** as the first turning point
  of the singular-value curve, via the parameter-light Kneedle / max-distance-to-
  chord rule in log space (D15/L20 "gradient turning-point"). The old 10 % value
  is recast as a **ceiling/guard** (``max_rank``), not a floor.
- ``mp_noise_cutoff``: an optional **high (noise) cutoff** from the Marchenko-
  Pastur upper edge of the noise eigenvalue bulk -- a principled borrow from RMT
  denoising (Veraart 2016). Off by default; not yet a standard ULM default, so we
  isolate the low-cutoff lever first.
- ``select_svd_cutoffs``: combine the two into ``(low, high_remove)`` for the
  projection ``u[:, low : n_frames - high_remove]``.

All functions operate on the descending singular-value vector ``s`` (or the Gram
eigenvalues), so they are nearly free wherever the Gram/covariance eigen-
decomposition is already computed (the GPU and CPU SVD paths both have it).
"""

from __future__ import annotations

import numpy as np


def singular_value_knee(
    singular_values: np.ndarray,
    *,
    min_rank: int = 1,
    max_rank: int | None = None,
    smooth: int = 0,
    log: bool = True,
) -> int:
    """First turning point ("knee") of a descending singular-value curve.

    Uses the Kneedle rule (Satopää 2011): normalise the curve to the unit
    square, draw the chord between the first and last points, and take the index
    of maximum perpendicular departure from that chord. For an L-shaped spectrum
    (steep tissue decay -> gentle blood/noise plateau) this is the corner of the
    L -- the tissue/blood boundary that D15 calls the "first turning point" and
    L20 finds via the singular-value gradient.

    Working in ``log`` space (default) is the right scale for a spectrum that
    decays over orders of magnitude; it makes the knee count-independent and far
    less sensitive to the absolute energy of the tissue modes.

    Returns the number of leading (tissue) modes to remove, clamped to
    ``[min_rank, max_rank]``. ``max_rank`` is the literature-grounded *ceiling*
    (e.g. 10 % of the frames) that guards against a degenerate pick on a soft
    spectrum; ``min_rank`` (>=1) guarantees the dominant tissue mode is removed.
    """
    s = np.asarray(singular_values, dtype=np.float64)
    s = s[np.isfinite(s) & (s > 0)]
    n = s.size
    if n == 0:
        return int(max(0, min_rank))
    # Enforce descending order (callers may pass either order).
    s = np.sort(s)[::-1]

    max_allowed = int(n - 1 if max_rank is None else min(int(max_rank), n - 1))
    min_allowed = int(max(0, min(int(min_rank), max_allowed)))
    if max_allowed <= min_allowed:
        return min_allowed
    if n < 3:
        return min_allowed

    y = np.log(s) if log else s.astype(np.float64)
    if smooth and int(smooth) > 1 and n >= int(smooth):
        k = int(smooth)
        kernel = np.ones(k, dtype=np.float64) / k
        y = np.convolve(y, kernel, mode="same")

    x = np.arange(n, dtype=np.float64)
    xn = x / (n - 1)
    span = y[0] - y[-1]
    if not np.isfinite(span) or abs(span) < 1e-300:
        return min_allowed
    yn = (y - y[-1]) / span  # decreasing 1 -> 0 for a descending curve
    # Distance below the chord (chord is the line yn = 1 - xn). For a convex,
    # L-shaped decreasing curve this departure is positive and peaks at the knee.
    departure = (1.0 - xn) - yn
    k = int(np.argmax(departure))
    return int(np.clip(k, min_allowed, max_allowed))


def _mp_median(gamma: float) -> float:
    """Median of the Marchenko-Pastur distribution (eigenvalues of (1/N)XXᴴ for
    unit-variance noise), in units of σ². For γ→0 this tends to 1; for moderate γ
    it sits below 1 because the MP bulk is left-skewed. Used to debias a sample
    median into a σ² estimate (Gavish & Donoho 2014, "optimal hard threshold")."""
    sg = np.sqrt(gamma)
    a, b = (1.0 - sg) ** 2, (1.0 + sg) ** 2
    lam = np.linspace(a, b, 4096)
    dens = np.sqrt(np.maximum((b - lam) * (lam - a), 0.0)) / (2.0 * np.pi * gamma * lam)
    cdf = np.cumsum(dens)
    if cdf[-1] <= 0:
        return 1.0
    cdf /= cdf[-1]
    return float(lam[int(np.searchsorted(cdf, 0.5))])


def mp_noise_cutoff(
    eigenvalues: np.ndarray,
    n_frames: int,
    n_voxels: int,
    *,
    safety: float = 1.0,
) -> int:
    """High-order (noise) cutoff from the Marchenko-Pastur upper edge.

    The Gram ``G = X Xᴴ`` (frames x frames) has eigenvalues equal to the squared
    singular values of the ``(frames, voxels)`` matrix ``X``. For a pure-noise
    ``X`` with per-entry variance ``σ²`` the eigenvalues of ``(1/N) G`` fall in
    the MP support with upper edge ``σ²(1+√γ)²``, ``γ = F/N``. Any eigenvalue of
    ``G`` above ``N·σ²(1+√γ)²`` is signal-bearing; everything below is noise.

    ``σ²N`` is estimated robustly via the **eigenvalue median**, debiased by the
    MP-median factor (``median ≈ N·σ²·μ_γ``), so the few signal outliers do not
    perturb it. Returns ``high_remove`` = the number of trailing noise modes to
    drop (0 if every eigenvalue sits above the edge).
    """
    ev = np.asarray(eigenvalues, dtype=np.float64)
    ev = ev[np.isfinite(ev)]
    ev = np.sort(np.maximum(ev, 0.0))[::-1]  # descending
    n = ev.size
    if n == 0 or n_voxels <= 0:
        return 0
    gamma = float(n_frames) / float(n_voxels)
    if not (0.0 < gamma < 1.0):
        return 0
    mu = _mp_median(gamma)
    median_ev = float(np.median(ev))            # ≈ N·σ²·μ_γ (median is on the bulk)
    sigma2_N = median_ev / max(mu, 1e-12)        # debiased N·σ²
    edge = sigma2_N * (1.0 + np.sqrt(gamma)) ** 2 * float(safety)
    if not np.isfinite(edge) or edge <= 0:
        return 0
    signal_end = max(int(np.count_nonzero(ev > edge)), 1)  # modes above the edge
    return int(max(0, n - signal_end))


def select_svd_cutoffs(
    eigenvalues: np.ndarray,
    n_frames: int,
    n_voxels: int,
    *,
    low_min: int = 1,
    low_max: int | None = None,
    smooth: int = 0,
    high: bool = False,
    high_safety: float = 1.0,
) -> tuple[int, int]:
    """Return ``(low, high_remove)`` for an SVD projection from Gram eigenvalues.

    ``low`` = data-driven tissue knee (clamped to ``[low_min, low_max]``).
    ``high_remove`` = MP noise cutoff when ``high`` is set, else 0. The two are
    de-conflicted so the kept window ``[low, n_frames - high_remove)`` keeps at
    least one mode.
    """
    ev = np.asarray(eigenvalues, dtype=np.float64)
    svals = np.sqrt(np.maximum(ev, 0.0))
    low = singular_value_knee(svals, min_rank=low_min, max_rank=low_max, smooth=smooth)
    high_remove = mp_noise_cutoff(ev, n_frames, n_voxels, safety=high_safety) if high else 0
    if low + high_remove >= n_frames:
        high_remove = max(0, n_frames - low - 1)
    return int(low), int(high_remove)
