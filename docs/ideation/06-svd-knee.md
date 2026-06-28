# SVD low-cutoff knee — the coverage-loss lever (mb-3k4)

> Status: **built + unit-tested + bake-off-ready (CPU)**. GPU bake-off pending Rome's
> Modal-spend OK. Bead `mb-3k4` (discovered-from `mb-crr.19`).

This is the "TOP NEXT STEP" from the handoff: per the detection/SVD literature
review (`docs/literature/lit-detection-svd.md` §1), our SVD tissue cutoff
over-removes, and that — not the per-block-rank machinery — is the likely real
coverage-loss lever.

## The finding (empirically confirmed)

The shipped `svd_method="adaptive"` path (`svd.spectral_centroid_cutoff`) is, on
**our** acquisition, a **constant 70-mode cut** — it never adapts:

- It returns the first temporal mode whose power-weighted mean frequency exceeds
  `tissue_freq_hz = 100 Hz`, else falls back to `round(0.1 · n_frames) = 70`.
- Our frame rate is **222 Hz → Nyquist 111 Hz**. The highest centroid any mode
  attains is **~60 Hz** (verified on synthetic *and* the math: even white noise
  has centroid ≈ Nyquist/2 ≈ 55 Hz). **No mode ever crosses 100 Hz**, so the test
  never fires and the fallback (70) is returned every time.

So "adaptive" removes **70 of 700 modes** regardless of the data. The clutter
literature is unanimous the tissue subspace is **small** — a few to low-tens of
modes (Demené 2015 first-turning-point; Baranger 2018; Lok/Song 2020 BCR plateau
~rank 8, usable 10–24; McCall 2023 fixed 15–20 %). Removing 70 over-removes by
~10× in quiet regions and discards weak blood signal → the measured coverage loss.

Reproduce the finding (no GPU):
```
python3 -m pytest tests/test_svd_knee.py::test_knee_beats_broken_adaptive_floor_on_our_framerate -q
```

## The fix

New `svd_method="knee"` (`ultratrace_ulm/svd_knee.py`), wired through the CPU
(`svd.filter_svd_3d`) and GPU (`gpu_svd.filter_svd_3d_gpu`) filters and exposed in
`detect_acqs(svd_method="knee")` and `validate_svd`:

- **Low (tissue) cutoff = data-driven knee.** First turning point of the
  singular-value curve via the parameter-light Kneedle / max-distance-to-chord
  rule in log space (D15/L20 gradient method). Computed from the Gram eigenvalues
  the SVD path *already* decomposes → nearly free.
- **The old 10 % is recast as a CEILING/guard, not a floor.** `low_cutoff=0.1` now
  caps the knee at 70. **Safety property: `knee ≤ 70 = current behaviour`, always.**
  The change is monotone in the "keep more signal" direction — it can recover
  coverage but can never over-filter worse than today.
- **Optional high (noise) cutoff** (`knee_high=True`): Marchenko–Pastur upper edge
  of the noise eigenvalue bulk, σ²N debiased via the MP-median rule
  (Gavish–Donoho 2014). **Off by default** so the first bake-off isolates the
  single low-cutoff lever (MP is a principled borrow, not yet a ULM standard).

`min_rank = 1` guarantees the dominant tissue/DC mode is always removed.

## Validation done (CPU, no spend)

- `tests/test_svd_knee.py` — 8 tests: knee finds the L-corner, respects guards, is
  frame-count-independent, handles degenerate input; MP separates signal from a
  real Gaussian-noise bulk; `select_svd_cutoffs` keeps a valid window; the CPU
  `method="knee"` filter retains strictly more energy than the fixed floor; and a
  regression guard that pins the adaptive-floor bug. Full suite: 53 passed, 1 skip.
- End-to-end on a synthetic 700-frame acq (6 tissue + 3 blood modes): adaptive=70,
  knee=9 — a ~10× reduction in modes removed, in the literature's expected range.

## Bake-off plan (needs Modal-spend OK)

1. **One logged validation acq (cheap, ~1 GPU-min):**
   ```
   modal run scripts/modal/app.py::validate_svd --knee-high False
   ```
   Confirms on real data: `adaptive_low` (expect 70), `knee_low` (expect single
   digits to low-tens), the singular-value spectrum head, and adaptive-vs-knee
   detection counts on one acq.

2. **60-acq bake-off vs `base60`** (the agreed comparison set; ~$2.5/run):
   ```
   modal run scripts/modal/app.py::detect_acqs --svd-method knee --tag knee60 --num-acqs 60
   modal run scripts/modal/app.py::track_acqs --tag knee60
   python3 scripts/bench.py tracks/base60 tracks/knee60 --labels base,knee
   ```
   Arbiters (bench.py): `occupied_fraction`/density + `mean_curvilinear_len_mm` +
   `frac_len_ge_35` (coverage recovery) traded against `frag_short_over_long` and
   `straightness` (residual-clutter false tracks). Optional follow-up: split-half
   FRC reproducibility (mb-crr.19.6) as the over-merge arbiter.

3. If the knee wins, optionally test `--knee-high True` (adds the MP noise cutoff),
   then fold the knee into the winning region+Kalman composite before the 223 scale-up.
