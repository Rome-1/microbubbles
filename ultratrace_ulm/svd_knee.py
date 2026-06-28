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


def _temporal_basis(
    matrix: np.ndarray, *, voxel_chunk: int = 300_000
) -> tuple[np.ndarray, np.ndarray]:
    """Temporal singular vectors ``U`` (descending) and singular values ``s`` from
    the **raw** Gram ``G = X Xᴴ`` of the ``(frames, voxels)`` matrix ``X``.

    This matches the basis the ``filter_svd_3d`` "fast"/"knee" projection uses
    (raw, not mean-subtracted), so the modes scored here are the modes actually
    removed. The Gram is accumulated in voxel chunks in ``complex128`` so we never
    materialise a full conjugate copy of the (potentially many-GB) matrix -- the
    same memory discipline the GPU port follows.
    """
    x = np.asarray(matrix)
    n_frames = int(x.shape[0])
    n_vox = int(x.shape[1])
    g = np.zeros((n_frames, n_frames), dtype=np.complex128)
    if n_vox > 0:
        step = max(1, int(voxel_chunk))
        for c0 in range(0, n_vox, step):
            xc = x[:, c0:c0 + step]
            g += (xc @ xc.conj().T).astype(np.complex128)
    evals, u = np.linalg.eigh(g)
    order = np.argsort(evals)[::-1]
    u = u[:, order]
    s = np.sqrt(np.maximum(evals[order].real, 0.0))
    return u, s


def _neighbor_coherence(
    rows: np.ndarray, spatial_shape: tuple[int, ...]
) -> np.ndarray | None:
    """Normalised lag-1 spatial autocorrelation of each spatial singular vector.

    ``rows`` is ``(n_modes, n_voxels)`` -- one (possibly conjugated/scaled)
    spatial singular vector per row. Each row is reshaped to ``spatial_shape`` and
    the metric, for every spatial axis ``a`` with length >= 2, is the
    phase-invariant normalised inner product between the field and its one-voxel
    shift along ``a``::

        rho_a = |Σ conj(v_k)·v_{k+1}| / sqrt(Σ|v_k|² · Σ|v_{k+1}|²)   ∈ [0, 1]

    averaged over the available axes. It is a *cosine similarity* between adjacent
    voxels, not a mean-centred Pearson correlation: the non-centred form is stable
    for near-constant (DC-like) tissue modes (a constant field -> 1) and avoids the
    division blow-up a centred form suffers when the spatial mean dominates. It is
    invariant to the singular vector's arbitrary global phase and to its scale, so
    ``V_i ∝ Xᴴ U_i`` may be used directly (no need to divide by ``s_i``, which is
    ill-conditioned for the near-zero high-order modes).

    A smooth, low-spatial-frequency *tissue* mode has adjacent voxels nearly equal
    -> rho ~ 1; a spatially incoherent *blood/noise* mode has adjacent voxels
    decorrelated -> rho ~ 1/sqrt(pairs) ~ 0. Returns ``None`` if no spatial axis
    has length >= 2 (coherence is undefined).
    """
    n_modes = int(rows.shape[0])
    grid = rows.reshape((n_modes,) + tuple(int(d) for d in spatial_shape))
    ndim = len(spatial_shape)
    reduce_axes = tuple(range(1, ndim + 1))
    rho_sum = np.zeros(n_modes, dtype=np.float64)
    n_axes = 0
    for ax in range(ndim):
        axis = ax + 1
        if grid.shape[axis] < 2:
            continue
        sl_a = [slice(None)] * (ndim + 1)
        sl_b = [slice(None)] * (ndim + 1)
        sl_a[axis] = slice(0, -1)
        sl_b[axis] = slice(1, None)
        a = grid[tuple(sl_a)]
        b = grid[tuple(sl_b)]
        num = np.abs(np.sum(np.conj(a) * b, axis=reduce_axes))
        denom = np.sqrt(
            np.sum(np.abs(a) ** 2, axis=reduce_axes)
            * np.sum(np.abs(b) ** 2, axis=reduce_axes)
        )
        rho_sum += np.where(denom > 0, num / np.where(denom > 0, denom, 1.0), 0.0)
        n_axes += 1
    if n_axes == 0:
        return None
    return rho_sum / n_axes


def _collapse_index(
    coherence: np.ndarray,
    min_allowed: int,
    max_allowed: int,
    rel_threshold: float,
) -> int:
    """Index of the first collapse of a per-mode spatial-coherence curve.

    The leading (tissue) modes are spatially coherent; the curve falls off where
    the subspace turns to blood/noise. We locate the **first** crossing of a level
    set between a robust high reference (``max`` over the candidate window) and a
    robust floor (10th percentile over the evaluated modes)::

        threshold = floor + rel_threshold · (reference - floor)

    The returned cutoff is the first mode index whose coherence drops below
    ``threshold`` (clamped to ``[min_allowed, max_allowed]``). If coherence never
    drops within the window the tissue subspace extends past the ceiling ->
    ``max_allowed``. If there is **no contrast** (a flat curve: all-tissue or
    all-noise, no separable boundary) we cannot localise a boundary and fall back
    to the conservative ``min_allowed`` -- documented behaviour, important for the
    soft 5-angle regime where the collapse can be shallow.
    """
    coh = np.asarray(coherence, dtype=np.float64)
    m = coh.size
    if m == 0 or not np.any(np.isfinite(coh)):
        return min_allowed
    hi = min(m, max_allowed + 1)
    reference = float(np.max(coh[:hi]))
    floor = float(np.percentile(coh, 10))
    contrast = reference - floor
    if not np.isfinite(contrast) or contrast <= max(1e-6, 0.05 * abs(reference)):
        return min_allowed
    threshold = floor + float(rel_threshold) * contrast
    below = np.where(coh[:hi] < threshold)[0]
    cut = int(below[0]) if below.size else max_allowed
    return int(np.clip(cut, min_allowed, max_allowed))


def spatial_correlation_cutoff(
    matrix: np.ndarray,
    spatial_shape: tuple[int, ...] | None = None,
    *,
    u: np.ndarray | None = None,
    s: np.ndarray | None = None,
    min_rank: int = 1,
    max_rank: int | None = None,
    rel_threshold: float = 0.5,
    smooth: int = 0,
    n_eval: int | None = None,
    mode_chunk: int = 16,
    voxel_chunk: int = 300_000,
) -> int:
    """B18 spatial-singular-vector cutoff for the LOW (tissue) boundary.

    Baranger 2018 (B18) found the most robust automatic tissue/blood cutoff is
    based on the **spatial** singular vectors, not the singular values: low-order
    (tissue) modes are spatially smooth/coherent across the field while blood and
    noise modes are not, so the cutoff is the index where spatial coherence
    *collapses*. This implements that idea with a per-mode normalised lag-1 spatial
    autocorrelation (see ``_neighbor_coherence``) and a first-collapse rule (see
    ``_collapse_index``).

    The spatial singular vectors come from ``V_i = Xᴴ U_i / s_i`` where ``U`` are
    the temporal singular vectors of the **raw** Gram ``G = X Xᴴ`` (so the basis
    matches the projection ``filter_svd_3d`` actually applies) and ``s`` the
    singular values. Because the coherence metric is invariant to each vector's
    global phase and scale, we score ``Xᴴ U_i`` directly and never divide by the
    ill-conditioned small ``s_i`` of the high-order modes.

    Parameters
    ----------
    matrix : ``(frames, voxels)`` complex array, OR a ``(frames, *spatial)``
        volume (then ``spatial_shape`` is inferred and the trailing axes flattened).
        Always required -- the spatial vectors are ``Xᴴ U``.
    spatial_shape : the voxel grid, required when ``matrix`` is 2-D so the flat
        voxel axis can be reshaped for the neighbour metric.
    u, s : optionally precomputed temporal basis (descending, as produced by
        ``u[:, argsort(evals)[::-1]]``); pass these to avoid recomputing the Gram
        eigendecomposition when the pipeline already has it. ``matrix`` is still
        required (for ``Xᴴ U``); ``s`` is used only to skip null-space modes.
    min_rank, max_rank : the same guards as :func:`singular_value_knee` -- a floor
        that guarantees the dominant tissue mode is removed and a literature
        ceiling (e.g. 10 % of frames) guarding a degenerate pick on a soft spectrum.
    rel_threshold : level-set fraction between the coherence floor and reference
        at which the collapse is declared (0.5 = halfway). Lower -> fewer modes
        called tissue.
    n_eval : number of leading modes to score (default a few × ``max_rank``,
        enough to see the collapse and estimate the floor). Capped at the numeric
        rank (``s > 0``) and the frame count.
    mode_chunk, voxel_chunk : memory knobs -- modes are scored ``mode_chunk`` at a
        time (each block materialises only ``(mode_chunk, voxels)``), and the Gram,
        when computed here, is accumulated ``voxel_chunk`` voxels at a time.

    Returns the number of leading (tissue) modes to remove, clamped to
    ``[min_rank, max_rank]``.

    ROBUSTNESS / HONESTY (our 5-angle "soft knee" regime). With only 5 transmit
    angles the tissue/blood separation in the SVD is weaker than the 16-42-angle
    works B18/L20 validated on, so the coherence collapse can be shallow. Two
    documented fallbacks keep this safe rather than wrong: a flat (no-contrast)
    curve yields ``min_rank`` (minimal, conservative removal), and a curve that
    never collapses within the window yields ``max_rank`` (the ceiling guard).
    Prefer :func:`combined_low_cutoff` (the L20 ``min`` rule), and validate the
    chosen rank against the split-half render proxy -- there is no ground truth.

    FUTURE WORK. B18 derives a *second*, high-order (noise) threshold from the same
    machinery -- the index where coherence collapses again from the blood regime to
    the spatially-decorrelated noise floor. That needs the coherence curve scored
    over the full spectrum (every mode), not just the leading ``n_eval``; it is left
    as a follow-up so the low-cutoff lever stays isolated and cheap. The
    Marchenko-Pastur :func:`mp_noise_cutoff` already covers the high cutoff from the
    singular-value side.
    """
    x = np.asarray(matrix)
    if x.ndim > 2:
        if spatial_shape is None:
            spatial_shape = tuple(int(d) for d in x.shape[1:])
        x = x.reshape(x.shape[0], -1)
    if x.ndim != 2:
        raise ValueError(f"matrix must be (frames, voxels) or a volume, got ndim={x.ndim}")
    if spatial_shape is None:
        raise ValueError("spatial_shape is required when matrix is a 2-D (frames, voxels) array")
    spatial_shape = tuple(int(d) for d in spatial_shape)
    n_frames, n_vox = int(x.shape[0]), int(x.shape[1])
    if int(np.prod(spatial_shape)) != n_vox:
        raise ValueError(
            f"spatial_shape {spatial_shape} (prod={int(np.prod(spatial_shape))}) "
            f"does not match the voxel count {n_vox}"
        )

    max_allowed = int(n_frames - 1 if max_rank is None else min(int(max_rank), n_frames - 1))
    max_allowed = max(0, max_allowed)
    min_allowed = int(max(0, min(int(min_rank), max_allowed)))
    if n_frames == 0 or n_vox == 0 or max_allowed <= min_allowed:
        return min_allowed
    # No spatial adjacency on any axis -> the coherence metric is undefined.
    if all(d < 2 for d in spatial_shape):
        return min_allowed

    x = x.astype(np.complex64, copy=False)
    if u is None or s is None:
        u, s = _temporal_basis(x, voxel_chunk=voxel_chunk)
    else:
        u = np.asarray(u)
        s = np.asarray(s).real.astype(np.float64)

    n_modes = int(u.shape[1])
    rank = int(np.count_nonzero(s > (float(np.max(s)) * 1e-12))) if s.size else n_modes
    rank = max(1, min(rank, n_modes))
    if n_eval is None:
        n_eval = max(3 * (max_allowed + 1), 24)
    n_eval = int(min(n_eval, rank, n_modes))

    coherence = np.empty(n_eval, dtype=np.float64)
    step = max(1, int(mode_chunk))
    for b0 in range(0, n_eval, step):
        b1 = min(b0 + step, n_eval)
        ub = u[:, b0:b1].astype(np.complex64)
        # rows = (Xᴴ U_block)ᴴ = U_blockᴴ X : (kb, voxels). Computed as small @ big
        # so the many-GB matrix is never transposed/copied. Coherence is invariant
        # to the global conjugation this introduces.
        rows = ub.conj().T @ x
        block = _neighbor_coherence(rows, spatial_shape)
        if block is None:
            return min_allowed
        coherence[b0:b1] = block

    if smooth and int(smooth) > 1 and coherence.size >= int(smooth):
        k = int(smooth)
        coherence = np.convolve(coherence, np.ones(k) / k, mode="same")

    return _collapse_index(coherence, min_allowed, max_allowed, float(rel_threshold))


def combined_low_cutoff(
    matrix: np.ndarray,
    spatial_shape: tuple[int, ...] | None = None,
    *,
    u: np.ndarray | None = None,
    s: np.ndarray | None = None,
    singular_values: np.ndarray | None = None,
    min_rank: int = 1,
    max_rank: int | None = None,
    smooth: int = 0,
    rel_threshold: float = 0.5,
    n_eval: int | None = None,
    mode_chunk: int = 16,
    voxel_chunk: int = 300_000,
) -> int:
    """L20 combined LOW (tissue) cutoff: ``min(knee, spatial-correlation)``.

    Lok/Song 2020 take the final tissue rank as the **minimum** of two automatic
    estimators -- the singular-value gradient/turning point and the spatial-vector
    correlation -- which biases toward removing *fewer* tissue modes (keeping more
    weak blood signal), the conservative choice this pipeline wants. This composes
    :func:`singular_value_knee` (the Kneedle turning point) with
    :func:`spatial_correlation_cutoff` (B18), under the same ``min_rank`` /
    ``max_rank`` guards.

    The temporal basis ``(u, s)`` is computed once (from the raw Gram) and shared
    by both estimators, so this costs a single eigendecomposition. Pass precomputed
    ``u, s`` (and/or ``singular_values``) when the caller already has them.

    Returns the combined tissue rank, clamped to ``[min_rank, max_rank]``.
    """
    x = np.asarray(matrix)
    if x.ndim > 2:
        if spatial_shape is None:
            spatial_shape = tuple(int(d) for d in x.shape[1:])
        x = x.reshape(x.shape[0], -1)
    if x.ndim != 2:
        raise ValueError(f"matrix must be (frames, voxels) or a volume, got ndim={x.ndim}")
    n_frames = int(x.shape[0])

    x = x.astype(np.complex64, copy=False)
    if u is None or s is None:
        u, s = _temporal_basis(x, voxel_chunk=voxel_chunk)

    svals = s if singular_values is None else np.asarray(singular_values)
    knee = singular_value_knee(svals, min_rank=min_rank, max_rank=max_rank, smooth=smooth)
    spatial = spatial_correlation_cutoff(
        x,
        spatial_shape,
        u=u,
        s=s,
        min_rank=min_rank,
        max_rank=max_rank,
        rel_threshold=rel_threshold,
        smooth=smooth,
        n_eval=n_eval,
        mode_chunk=mode_chunk,
        voxel_chunk=voxel_chunk,
    )

    max_allowed = int(n_frames - 1 if max_rank is None else min(int(max_rank), n_frames - 1))
    max_allowed = max(0, max_allowed)
    min_allowed = int(max(0, min(int(min_rank), max_allowed)))
    return int(np.clip(min(int(knee), int(spatial)), min_allowed, max_allowed))


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
