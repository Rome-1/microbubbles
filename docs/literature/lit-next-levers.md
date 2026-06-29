# Next highest-value algorithmic levers (ranked, literature-grounded)

> Companion to `lit-detection-svd.md` (SVD+detection deep-dive) and the agreed-10
> (`docs/ideation/03-agreed-10.md`). Scope: what to bake off NEXT, given the SVD-knee
> lever is built (`svd_knee.py`, `06-svd-knee.md`) and the cutoff sweep is running.
> Goal: move **coverage + track continuity WITHOUT manufacturing false tracks**, and
> be ruthless about GPU spend.

## Our weak axes (what these levers must serve)
1. **Elevation** — 1 physical receive row → 25 *synthesized* planes. Worst-resolved axis.
   The Kalman tracker is currently **isotropic** (`R = eye(3)·measurement_noise`,
   `tracking.py:262`) and the gate is an isotropic-mm box — the geometry mismatch is
   un-exploited on the tracking side (the stitcher already goes anisotropic, the Kalman does not).
2. **Soft SVD knee** — only 5 transmit angles → poorer per-frame SNR → the tissue/blood
   separation in the singular-value spectrum is *less clean* than the 16–42-angle works our
   defaults come from. A sharp gradient/Kneedle knee may be the wrong (least-robust) estimator here.
3. **No ground truth** — every metric is a proxy; we have **no over-merge arbiter** yet.
   `bench.py` measures length/frag/straightness/density on the coronal (x,z) plane only;
   there is no split-half reproducibility, no FRC, no saturation, no elevation-occupancy audit.

## What is already built (so we don't re-bake it)
SVD knee (low) + MP high (`svd_knee.py`, off by default) · region/per-block SVD · adaptive
per-block rank (`svd_rank.py`) · CFAR/MAD detect with debias + anisotropic-NMS + guard + global
floor + confidence tiers (`detect_cfar.py`) · empirical-PSF matched filter (`psf.py`) · post-hoc
stitching (`track_stitch.py`, already anisotropic) · Kalman predicted-gate (`gate_on_prediction`).
**Not built:** elevation-anisotropic Kalman (#4), intensity cost in `pair_costs` (#9), tracker
consumption of confidence (#5), split-half/FRC/saturation metrics (#10b), B18 spatial-correlation
SVD cutoff. These are where the next value is.

---

## Ranked levers

| # | Lever | Knob / method | Values to try | Proxy it moves & WHY | Effort | GPU? | Citation (verify) |
|---|-------|---------------|---------------|----------------------|--------|------|-------------------|
| 1 | **Elevation-anisotropic Kalman + Mahalanobis gate** (agreed #4, unbuilt) | Make `R` and the gate anisotropic: inflate `R[1,1]` (elevation=y); anisotropic gate (tight z/x, loose elev); seed `R` from sub-voxel curvature | `R_yy` ×5, ×10, ×15, ×25 vs lateral; gate elev ×2–4 of z/x | `mean_curv_len`↑, `frac_ge_*`↑, `straightness`↑ — elevation jitter currently breaks/curls otherwise-good tracks because the filter trusts y as much as x/z | M | No (Phase-B on checkpoints) | W24 RCA regime + `lit-detection-svd.md` §4 (verified-fulltext); vc-Kalman (verified-fulltext) |
| 2 | **No-GT arbiter: split-half FRC + saturation + elevation-occupancy** (agreed #10b — split-half FRC ✅ `frc.py`, saturation ✅ `bench.py:saturation_metrics`, elevation-occupancy audit STILL UNBUILT) | Odd/even-acq FRC (½-bit & 2σ) ✅; exponential saturation `C₀(1−e^{−κt})` on occupied-area-vs-acq ✅ (`sat_acqs_to_90pct`, `sat_fraction_of_ceiling`, `sat_late_early_slope_ratio`, `sat_r2` vs `sat_linear_r2`); per-elev-plane occupancy + midplane-attraction audit ← still TODO | n/a — adds metrics | **De-risks every other lever**: the only proxy that separates *recovered vessel* from *manufactured/over-merged track* (the stitch straightness regression and any coverage lever need this). Validated: base223 saturates (late/early slope 0.28, frac_of_ceiling ~1.0) where base60 is coverage-limited (0.49, 0.85) | M | No (CPU on `.bin`) | Hingot 2021 FRC (verified-abstract); Hingot 2019 saturation (verified-abstract); G24 (verified-fulltext) |
| 3 | **Intensity/appearance-aware association cost** (agreed #9, unbuilt) | Add `λ·\|log I_det − log Ī_track\|` to `pair_costs` (Mahalanobis+momentum). Intensities are *already carried, unused* (`tracking.py:438`) | `λ` ∈ {0.25, 0.5, 1.0}; clip log-ratio | Fewer ID switches in dense regions → `frac_ge_*`↑, `straightness`↑, fewer false merges. Near-zero cost | S | No | vc-Kalman adds brightness to state→CNR 4.92→8.84 (verified-fulltext); LB22 image-feature pairing (verified-abstract) |
| 4 | **B18 spatial-singular-vector cutoff → `min(knee, spatial-corr)`** | Build the spatial-similarity estimator; take `min(gradient-knee, spatial-corr)` for low cut; spatial-decorrelation tail for high cut | low = min(knee, spat); compare vs knee-only and MP-high | `occupied_fraction`/density recovery that is **robust on our soft 5-angle spectrum**, where a sharp knee is the *least-robust* of the methods Baranger tested | M | Folds into SVD pass | Baranger 2018 (verified-fulltext); L20 min-rule (verified-fulltext) |
| 5 | **VD physical-plausibility constraint + larger `max_gap`** (agreed #8, partial) | Raise `max_gap` 3→6/8 with predicted-gate ON; add velocity-difference reject (summed adjacent-speed within k≥3 frames must not deviate >100% from vector-sum) | `max_gap` ∈ {4,6,8}; VD on/off | `mean_curv_len`↑, `frac_ge_*`↑ **without** false tracks — VD rejects implausible bridges that a looser gap would admit | S | No | vc-Kalman VD (verified-fulltext): trace-length loss 3.6% vs 13.1% at low FR |
| 6 | **Two-tier confidence consumption in tracker** (agreed #5, half-built) | Detector already emits high/low conf; make low-conf **continuation-only** (match yes, spawn no) in `kalman_tracking_3d` | spawn k≈4–5, continue k≈2.5–3.5 | `occupied_fraction`↑ + length↑ from dim periphery, no one-frame-false-track explosion | M | Yes (conf detector re-run) | G24 FN≫FP, region-adaptive threshold (verified-fulltext) |
| 7 | **Widen elevation guard/background/NMS in CFAR** (lit §2–3) | `guard_radius (1,2,2)→(2–3,2,2)`, `local_radius (3,9,9)→(5,9,9)`, NMS elev 2×→3–4× lateral | sweep elev radii | `contrast`↑, `frag`↓ — kill elevation-smeared replicas + midplane bias contaminating the local background | S | Yes (fold into combo8) | LB22, M23, `lit-detection-svd.md` §2–4 (verified-fulltext) |
| 8 | **2D-first / elevation-optional linking** (agreed #4 variant) | Link in (z,x) first, assign elevation after; or drop y from the gate when ambiguous | 2D-first on/off; elev-gate-relax | Largest potential continuity win on the worst axis — **but largest over-merge risk** → must be gated by lever #2 | M | No | W24 separates row/col, effectively 2D-first (verified-fulltext) |
| 9 | **MP / SVHT high-order noise cutoff ON** (`knee_high=True`) | Flip `knee_high`; compare MP-edge vs Gavish–Donoho 4/√3 SVHT | knee_high on/off after knee wins | `contrast`↑ by dropping the noise tail; small, additive | S | Folds into SVD pass | Baranger invokes MP for noise floor (verified-fulltext); Gavish–Donoho 2014 (in code) |
| 10 | **Matched-filter PSF detection scale-up** (agreed #7, built, un-baked) | Run `psf.py` empirical-PSF cross-correlation before NMS at 60-acq | disjoint-half PSF self-check | weak-bubble sensitivity → `occupied_fraction`↑, continuity↑. Refinement, not headline | M/L | Yes | LB22, M23 (verified-fulltext) |

**GPU discipline:** levers 1, 2, 3, 5, 8 are **CPU-only** — they run Phase-B tracking from the
existing detection checkpoints or operate on saved `.bin`/pickles, so they are infinitely
sweepable for ~$0. Spend GPU only on the detection/SVD levers (4, 6, 7, 9, 10), and **bundle**
them into one combo detection run rather than one run per knob.

---

## Top 3 — algorithm + our-setup caveat

### 1. Elevation-anisotropic Kalman + Mahalanobis gate
The tracker today builds `R_template = eye(3)·measurement_noise` and `Q_template = diag(...)`
isotropically, and gates with an axis-independent mm box (`_tracking_gate` → `dx·2, dy·2, dz·2`).
The Mahalanobis cost (`tracking.py:421-423`) therefore trusts the synthesized elevation coordinate
exactly as much as the well-resolved lateral/axial ones, so a bubble whose y wanders by a plane or
two is either rejected from its own track (fragmentation) or pulls the smoothed path into a curl
(straightness loss).
```
R = diag([r_xx, alpha*r_xx, r_zz])          # alpha = 5..25 (elevation = index 1 = y)
# gate: anisotropic box, elev half-width = beta * lateral, beta = 2..4
# optional, principled: seed R_yy per-detection from the parabola/Gaussian curvature
#   of the elevation peak (broad peak -> large variance falls out, no hand-tuning)
```
Mahalanobis with an elevation-inflated `R`/`P` naturally widens the gate in y while keeping z/x
tight — exactly the anisotropy the stitcher already uses but the linker does not.
**OUR-SETUP caveat:** there is no GT to calibrate `alpha`; sweep it and pick by lever #2 (split-half
+ elevation-occupancy), not by a single asserted value. Start `alpha≈10`, `beta≈3` (lit §4 midpoint),
test up to ×25 because a 1-row synthesized aperture is *more* elevation-limited than the RCA the
×5–10 numbers came from.

### 2. No-GT arbiter (split-half FRC + saturation + elevation audit)
Every coverage/continuity lever above can win the current proxies by *manufacturing* tracks; we
cannot currently tell that apart from real recovery. Add three CPU metrics to `bench.py`:
(a) **split-half FRC** — render odd vs even acquisitions to two density maps, Fourier-ring-correlate,
read resolution at the ½-bit and 2σ thresholds; a *real* improvement makes the halves agree MORE
(resolution improves or holds) while a false-merge degrades it; (b) **saturation** — fit
`A(t)=C₀(1−e^{−κt})` to occupied-area-vs-acquisition and report the 90%-fill time (coverage that
plateaus early = efficient, coverage that never saturates = likely clutter); (c) **elevation audit** —
per-plane occupancy histogram + midplane-attraction ratio + per-track z-jitter, since our worst axis
is invisible in the coronal-only density metric.
**OUR-SETUP caveat:** FRC assumes the two halves are independent draws of the same vasculature — true
across acquisitions, but our 700-frame/acq, 222 Hz, single-subject data means the halves share slow
physiology; treat FRC as a *relative* arbiter between runs, not an absolute resolution number. This
is the single highest-leverage unbuilt item because it converts every other lever from "looks better"
to "verifiably better."

### 3. Intensity / appearance-aware association cost
`pair_costs = maha + momentum` (`tracking.py:438`) ignores detection brightness even though
`intensities` are threaded all the way into the cost loop. A bubble's echo amplitude is roughly
continuous frame-to-frame, so an appearance term cheaply disambiguates crossing tracks:
```
pair_costs = maha + momentum + lam * |log(I_det) - log(I_track_running_median)|
```
clip the log-ratio (e.g. to the stitcher's `intensity_log_tol=1.5`) so a single bright frame can't
veto a good match. This is the cheapest real win on the board — a few lines, CPU, sweep `lam`.
**OUR-SETUP caveat:** the bright-midplane elevation artifact biases raw intensity by plane; apply the
existing `debias_elevation` (already in the CFAR path) so the appearance term compares de-biased
amplitudes, otherwise it will spuriously penalize off-midplane continuations.

---

## Critical assessment: is the gradient/Kneedle knee the right SVD low cut?

**Honest answer: the Kneedle knee is a reasonable, safe *default* but it is the literature's
*least-robust* low-cut estimator, and our 5-angle geometry is exactly the soft-spectrum regime where
that matters.** Baranger 2018 (now **verified-fulltext**) compared **14** automatic estimators built
on singular values, temporal singular vectors, and spatial singular vectors across varying tissue-motion
and flow speeds. Two facts from their methods bear directly on us:

- The **gradient/turning-point** rule — "the clutter-rejecting threshold is chosen as the first turning
  point of the singular value curve … the first local minimum of the curvature radius" — is precisely
  what our log-space Kneedle approximates. **It was one of the estimators that LOST.**
- The winner, robust across all conditions, was the **spatial-singular-vector similarity** estimator:
  build the symmetric similarity matrix `C` (their Eq. 5) of the spatial singular vectors; tissue and
  blood appear as **two juxtaposed high-correlation square blocks**, the noise tail as a
  weak-correlation region; the tissue/blood and blood/noise boundaries are found by **fitting two
  juxtaposed squares to `C` via normalized correlation**. The *same* matrix yields the high-order
  (noise) cutoff.

So the literature-best automatic rule is **not** the knee alone. Lok/Song 2020 (verified-fulltext)
operationalize this as **`rank = min(gradient-turning-point, spatial-correlation)`**, computing the
low threshold globally for speed. Recommendation: **keep the Kneedle knee as the cheap, count-independent,
monotone-safe default (it can only keep more signal than the broken 70-floor), but build the B18
spatial-correlation estimator and switch the production low cut to `min(knee, spatial-corr)`** — this
is both the most-defensible rule and the one that helps *most* precisely where our spectrum is softest.

**High-order (noise) cutoff:** prefer **B18 spatial-decorrelation** (the method they actually
validated). The **Marchenko–Pastur edge is now better-grounded than `lit-detection-svd.md` stated** —
Baranger explicitly invokes MP ("Gaussian image noise, the singular values … following a
Marchenko–Pastur distribution") for the noise subspace; they use the spatial-similarity fit for the
*threshold*, but MP is not foreign to this literature. Our built `knee_high` (MP edge, debiased via
Gavish–Donoho median) is therefore a legitimate candidate; bake it against B18-decorrelation and let
the FRC arbiter decide.

**The honest caveat for a 1-row synthesized-elevation aperture:** B18's robustness comes from genuine
*spatial coherence* of tissue modes across a real field. One of our three spatial axes (elevation) is
**synthesized from a single physical row**, so its modes are not spatially coherent the way a
fully-sampled aperture's are — the spatial-similarity matrix may itself be soft along that axis. The
defensible plan is therefore: (i) compute the spatial-similarity estimator on the well-sampled (z,x)
modes, or per-elevation-plane, not naively over the flattened 25-plane field; (ii) take
`min(knee, spatial-corr)`; (iii) **validate the chosen rank against split-half FRC**, because no SVD
cutoff estimator is provably correct for our acquisition. Do not assert a calibrated rank.

## Levers the literature suggests we are MISSING for our weak axes
- **A no-GT over-merge arbiter (lever #2)** — the biggest gap. Without FRC/saturation we are flying
  blind on exactly the failure (false merges) our coverage levers risk. Build first.
- **Elevation-down-weighting on the TRACKING side (lever #1)** — we built anisotropy into the
  *stitcher* and the *detector NMS* but left the *Kalman* isotropic. The single most-mismatched
  component vs our geometry.
- **`min(knee, spatial-corr)` SVD rule (lever #4)** — the literature-best low cut we haven't built.
- **VD physical-plausibility track filter (lever #5)** — a cheap, principled way to *raise* length while
  *rejecting* implausible tracks, i.e. coverage without false tracks. Directly on-thesis and unbuilt.
- **Velocity-filtering / flow-direction separation** (arXiv 2101.09470, verified-abstract) and
  **tracking-prior-to-localization** (Leconte, seen in W24 refs, unverified) are larger architectural
  bets — note for a later round, not this bake-off.

---

### Verification ledger
- **verified-fulltext (methods parsed this session):** Baranger 2018 (spatial-similarity matrix, 14
  estimators, MP invocation, curvature turning-point as a losing estimator); vc-Kalman / Velocity-
  Constraint Kalman 2025 (VD rule, CNR 4.92→8.84, nRMSE 0.37→0.27, 3.6% vs 13.1% trace loss); W24 RCA
  (750 Hz, rows/cols filtered separately by SVD, PSF-correlation+Gaussian localization, Hungarian via
  simpletracker, keep >15-frame tracks). L20/M23/G24/BU25 carried verified-fulltext from `lit-detection-svd.md`.
- **verified-abstract:** Hingot 2021 FRC (½-bit & 2σ; resolution depends on localization precision AND
  MB count; 11–34 µm); Hingot 2019 saturation (`C₀(1−e^{−κt})`, acquisition-time lower bound vs
  concentration/flow); LB22 image-feature bipartite pairing; velocity-filtering (arXiv 2101.09470).
- **unverified:** tracking-prior-to-localization (Leconte) — title-level only.

Sources: [Baranger 2018 (unige full text)](https://access.archive-ouverte.unige.ch/access/metadata/88011ea7-bad6-4e52-bee4-2fdc8137efc3/download) · [Velocity-Constraint Kalman 2025 (PMC12136335)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12136335/) · [Hingot 2021 FRC (PubMed 34280094)](https://pubmed.ncbi.nlm.nih.gov/34280094/) · [Hingot 2019 saturation (Sci Rep)](https://www.nature.com/articles/s41598-020-62898-9) · [W24 RCA (arXiv 2406.01746)](https://arxiv.org/pdf/2406.01746) · [Velocity filtering (arXiv 2101.09470)](https://arxiv.org/pdf/2101.09470)
