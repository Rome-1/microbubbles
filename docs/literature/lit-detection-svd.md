# Literature review: detection + SVD clutter filtering for 3D ULM

Scientific basis for the tuning knobs on the **detection** and **SVD-clutter-filter**
side of our 3D ULM pipeline. Organized by knob. Each section gives what the published
literature says, a **recommended setting / range and WHY**, and an explicit
**OUR SETUP** note where the literature's defaults do not transfer to our acquisition
(5 transmit angles, 1 physical receive row → 25 *synthesized* elevation planes, a single
human dataset, no ground truth).

Pipeline under review: `beamform → temporal SVD clutter filter → 3D z-score/CFAR peak
detect + sub-voxel localize → Kalman+Hungarian tracking → smooth`. Per-acq beamformed
volume is `(frames=700, elev=25, z=225, x=378)` complex64; λ ≈ 1600/2.75e6 ≈ **0.58 mm**
at the stated 2.75 MHz / 1600 m·s⁻¹.

**Confidence tags** below: `[verified-fulltext]` = I read the relevant methods text;
`[verified-abstract]` = bibliographic record + abstract/secondary summary confirmed, full
methods not parsed; `[unverified-number]` = a specific figure I could not trace to primary
text and that should be re-checked before relying on it.

---

## 0. The corpus (all citations, with verification status)

| # | Citation | Verify |
|---|----------|--------|
| D15 | Demené C, Deffieux T, Pernot M, Osmanski B-F, Biran V, Gennisson J-L, et al. "Spatiotemporal Clutter Filtering of Ultrafast Ultrasound Data Highly Increases Doppler and fUltrasound Sensitivity." *IEEE Trans Med Imaging* 34(11):2271–2285, 2015. DOI [10.1109/TMI.2015.2428634](https://doi.org/10.1109/TMI.2015.2428634) · [PubMed 25955583](https://pubmed.ncbi.nlm.nih.gov/25955583/) | author list + biblio `[verified-abstract]` |
| B18 | Baranger J, Arnal B, Perren F, Baud O, Tanter M, Demené C. "Adaptive Spatiotemporal SVD Clutter Filtering for Ultrafast Doppler Imaging Using Similarity of Spatial Singular Vectors." *IEEE Trans Med Imaging* 37(7):1574–1586, 2018. DOI [10.1109/TMI.2018.2789499](https://doi.org/10.1109/TMI.2018.2789499) · [Semantic Scholar](https://www.semanticscholar.org/paper/e5ee8e9962ea9f6f44929aa325ceda0197b55a2b) | biblio `[verified-abstract]`; DOI `[unverified-number]` |
| S17 | Song P, Manduca A, Trzasko JD, Chen S. "Ultrasound Small Vessel Imaging With Block-Wise Adaptive Local Clutter Filtering." *IEEE Trans Med Imaging* 36(1):251–262, 2017. DOI [10.1109/TMI.2016.2605819](https://doi.org/10.1109/TMI.2016.2605819) · [IEEE Xplore](https://ieeexplore.ieee.org/document/7559732/) | biblio `[verified-abstract]` |
| L20 | Lok U-W, Song P, Trzasko JD, et al. "Real time SVD-based clutter filtering using randomized SVD and spatial downsampling for micro-vessel imaging on a Verasonics system." *Ultrasonics*, 2020. [PMC7293562](https://pmc.ncbi.nlm.nih.gov/articles/PMC7293562/) | methods `[verified-fulltext]` |
| HB22 | Heiles B, Chavignon A, Hingot V, Lopez P, Teston E, Couture O. "Performance benchmarking of microbubble-localization algorithms for ultrasound localization microscopy." *Nat Biomed Eng* 6:605–616, 2022. DOI [10.1038/s41551-021-00824-8](https://doi.org/10.1038/s41551-021-00824-8) · [PubMed 35177778](https://pubmed.ncbi.nlm.nih.gov/35177778/) | biblio `[verified-abstract]` |
| HV22 | Heiles B, Chavignon A, et al. "Volumetric ultrasound localization microscopy of the whole rat brain microvasculature." *IEEE Open J UFFC* 2:261–282, 2022. [Semantic Scholar](https://www.semanticscholar.org/paper/37334dd552520c44557b06b93c40d61123f44a6c) | biblio `[verified-abstract]` |
| C22 | Chavignon A, Heiles B, Hingot V, Orset C, Vivien D, Couture O. "3D Transcranial Ultrasound Localization Microscopy in the Rat Brain With a Multiplexed Matrix Probe." *IEEE Trans Biomed Eng* 69(7):2132–2142, 2022. [EMBS/TBME](https://www.embs.org/tbme/articles/3d-transcranial-ultrasound-localization-microscopy-in-the-rat-brain-with-a-multiplexed-matrix-probe/) · [PubMed 36028542](https://pubmed.ncbi.nlm.nih.gov/36028542/) | biblio `[verified-abstract]` |
| M23 | McCall JR, Santibanez F, Belgharbi H, Pinton GF, Dayton PA. "Non-invasive transcranial volumetric ULM of the rat brain with continuous, high volume-rate acquisition." *Theranostics* 13(4):1235, 2023. [thno.org/v13p1235](https://www.thno.org/v13p1235.htm) | methods `[verified-fulltext]` |
| LB22 | Lok U-W, Huang C, Trzasko JD, et al. "Three-Dimensional ULM with Bipartite-Graph MB Pairing and Kalman Tracking on a 256-Channel Verasonics with a 32×32 Matrix Array." *J Med Biol Eng*, 2022. [PMC9881453](https://pmc.ncbi.nlm.nih.gov/articles/PMC9881453/) | methods `[verified-fulltext]` |
| W24 | Wu A, Porée J, Ramos-Palacios G, Bourquin C, Ghigo N, Leconte A, Xing P, Sadikot AF, Chassé M, Provost J. "3D transcranial Dynamic ULM in the mouse brain using a Row-Column Array." *IEEE Trans Biomed Eng*, 2024/2025. [arXiv 2406.01746](https://arxiv.org/abs/2406.01746) | abstract `[verified-fulltext]`, methods `[verified-abstract]` |
| BU25 | Bureau F, Denis L, Coudert A, Fink M, Couture O, Aubry A. "Ultrasound matrix imaging for 3D transcranial in vivo localization microscopy." *Science Advances*, 2025. [arXiv 2410.14499](https://arxiv.org/abs/2410.14499) · [PMC12309695](https://pmc.ncbi.nlm.nih.gov/articles/PMC12309695/) | methods `[verified-fulltext]` |
| G24 | Gharamaleki SK, et al. "Evaluating Detection Thresholds: The Impact of False Positives and Negatives on Super-Resolution ULM." arXiv:2411.07426, 2024. [arXiv 2411.07426](https://arxiv.org/abs/2411.07426) | methods `[verified-fulltext]` |
| HBM25 | "Human brain hemodynamics for 3D ULM benchmarking." *Ultrasonics*, 2025. [ScienceDirect S001048252501724X](https://www.sciencedirect.com/science/article/pii/S001048252501724X) | abstract only `[verified-abstract]` (paywalled, methods not parsed) |

Supporting / context: Demeulenaere et al., whole-brain 3D ULM in mice (*EBioMedicine*, 2022);
3D ULM of the non-human-primate brain ([PMC11730257](https://pmc.ncbi.nlm.nih.gov/articles/PMC11730257/), 2024);
Marchenko–Pastur noise floor as used in diffusion-MRI RMT denoising (Veraart et al., *NeuroImage*, 2016, [PMC5159209](https://pmc.ncbi.nlm.nih.gov/articles/PMC5159209/)) — borrowed concept, not yet standard in ULM.

---

## 1. SVD clutter-filter cutoff (rank selection)

**Our knob:** `svd_low_cutoff = 0.1` (tissue/low-order cutoff, currently *floored at 10 %*
of the 700 temporal modes ⇒ ~70 modes), optional `svd_high_cutoff` (noise/high-order),
`cutoff_mode ∈ {global_rank, per_block}`, and a phase-invariant spectral-centroid cutoff
(`_spectral_centroid_cutoff_gpu`). Region-adaptive per-block subspace with a global rank
and partition-of-unity blend already exists.

### What the literature says

- **D15 (the founding method).** SVD of ultrafast data is a 2D space×time decomposition;
  tissue clutter, blood and noise occupy different singular-value regimes because they
  differ in **spatiotemporal coherence** (tissue = high energy + high spatial coherence +
  low temporal frequency; blood = lower energy, lower coherence; noise = lowest, flat).
  The clutter (tissue) cutoff is taken as the **first turning point** of the singular-value
  curve — the knee separating the steep high-energy tissue decay from the gentler blood
  regime. The singular-value spectrum "usually exhibits 3 different regimes." `[verified-abstract]`

- **B18 (the reference for *automatic* thresholding).** Compares **13 estimators** of the
  tissue/blood boundary built on singular values, temporal singular vectors, and spatial
  singular vectors. The winner — most robust across organs — is based on the **similarity
  (correlation) of *spatial* singular vectors**: low-order (tissue) modes are spatially
  smooth/correlated across the field, blood modes are not; the boundary is where spatial
  correlation collapses. The same machinery yields a **second, high-order noise threshold**
  (where modes become spatially decorrelated/noise-like). Energetic ("retained-variance")
  criteria were among the estimators but were *not* the most robust. `[verified-abstract]`

- **S17 (block-wise).** Tissue motion is only *locally* coherent; one global cutoff is wrong
  when tissue dynamics vary across the field. Splitting into **overlapping local blocks**,
  each with its own locally-coherent tissue subspace and its own adaptive cutoff, robustly
  rejects clutter that a global rank misses (and avoids over-filtering quiet regions).
  This is exactly the mechanism behind our region-adaptive SVD. `[verified-abstract]`

- **L20 (concrete numbers).** Two automatic rank methods: (i) **gradient/turning-point** of
  the singular-value curve (where it flattens) for the low-order cutoff, and (ii) the
  **spatial-correlation** method; the final rank is the **minimum of the two**. Empirically
  blood-to-clutter ratio rose steeply up to **rank ≈ 8**, then plateaued; usable in-vivo
  ranks were **~10–24** (sweet spot 14–26 for >20 dB BCR) on ~250-frame ensembles. Notably
  they computed the low-order threshold **globally and applied it to all blocks** for speed. `[verified-fulltext]`

- **M23 (3D, transcranial).** Volumetric ULM removed "**15–20 % of the largest singular
  values**" to reconstruct the MB signal — an empirical fixed fraction, with no high-order
  (noise) cutoff stated. `[verified-fulltext]`

- **Marchenko–Pastur noise floor.** Pure-noise singular values asymptotically follow the
  MP law; its **upper edge is a universal noise threshold** that separates signal-bearing
  from noise modes. This is standard in diffusion-MRI denoising (Veraart 2016) but I found
  **no ULM paper that adopts MP for the cutoff** — in ULM the high-order/noise cutoff is set
  empirically (B18 spatial decorrelation, or a fixed top-N). Treat MP as a *principled
  candidate for `svd_high_cutoff`*, not an established ULM default. `[verified-abstract]`

### Recommended setting / range and WHY

- **Method:** Do **not** hard-floor the tissue cutoff at a fixed fraction. Use a **data-driven
  knee** (first turning point of the singular-value curve, à la D15/L20 gradient method) for
  the low-order cutoff, and add a **separate high-order noise cutoff** (`svd_high_cutoff`)
  from B18-style spatial-vector decorrelation or the MP upper edge. Taking the
  **min(knee, spatial-correlation)** rank (L20) is the most defensible automatic rule.
- **Range:** Literature tissue cutoffs are **small** — single digits to low tens of modes,
  i.e. **a few %** of a several-hundred-frame ensemble (D15 first-knee; L20 BCR plateau ~8,
  usable 10–24 on 250 frames; M23 fixed 15–20 % on 3D). On our **700-frame** ensemble, the
  current **10 % floor = ~70 modes is on the high side** relative to the knee for quiet
  regions and likely throws away weak blood signal there. Recommend letting the knee select
  **as few as ~2–30 tissue modes** where justified, with the 10 % value re-cast as a *ceiling/
  guard*, not a floor.
- **Block-wise:** Keep `per_block` and choose each block's cutoff from its **own** knee, but
  **regularize the k-map** (median-clamp ±Δ across blocks, or blend toward the global knee) so
  isolated blocks don't over/under-filter — this is the standard fix and directly targets the
  −27 %/−79 % coverage/density loss you saw (which is a **rank artifact, not a boundary
  artifact**, since the PoU blend is already seam-free). Sweep Δ from `global_rank`→fully
  `per_block`. Blocks must be ≥ a few PSF widths so each has enough samples for a stable
  spectrum (S17 local-coherence assumption).

### OUR SETUP caveats

- The **10 % "floor" is not literature-grounded** as a fixed value; it is an artifact of our
  defaults. Every cited method picks the cutoff from the *spectrum*, not a constant fraction.
- Our 700-frame ensemble is **longer** than most cited works (250–512 frames), so a
  fixed-fraction cutoff scales the rank with frame count for no physical reason — another
  argument for a knee-based, count-independent rule.
- With **5 transmit angles** the per-frame SNR and clutter statistics are poorer than the
  16–42-angle compounding in the cited 3D works (BU25 uses ≥5 waves at 209 Hz; W24 uses 42
  plane waves) → the tissue/blood separation in the SVD spectrum is **less clean**, so the
  knee may be soft. Prefer the **more robust spatial-correlation estimator** over a sharp
  derivative, and validate the chosen k against the split-half render reproducibility proxy
  (no ground truth available to validate directly).

---

## 2. Microbubble detection threshold (CFAR / MAD multiplier, guard band, global floor)

**Our knob:** `detect_cfar.py` already does a **local median background + `1.4826·MAD`
scale** (so the scale is σ-normalized), a **guard band** (`guard_radius=(1,2,2)` in
elev,z,x), a **global floor** under the scale, a local background window
(`local_radius=(3,9,9)`), and a `sigma_threshold` z-score cut. A **confidence tier** exists
(high-conf may *spawn* tracks, low-conf may only *continue*).

### What the literature says

- **Fixed thresholds dominate the published 3D pipelines**, and they are *low*: M23 thresholds
  at **1–7 % of the maximum image intensity**; BU25 keeps MBs with **SNR ≥ 10.5 dB**
  (center intensity / mean of neighbouring voxels over a 5-mm patch) — i.e. an intensity
  contrast of ~3.4× over local background. LB22 simply uses an intensity threshold "to
  suppress noisy background." None of these are CFAR. `[verified-fulltext]`
- **CFAR rationale (radar).** When background level varies across the image, a *fixed*
  threshold gives a spatially-varying false-alarm rate. Cell-averaging CFAR estimates the
  local background from a ring of cells around each test cell, separated by a **guard band**
  that excludes the target's own PSF energy from the estimate, then sets
  `T = bg + k·scale`. A **robust** estimator (median + `k·MAD`, with `MAD·1.4826 ≈ σ`) is
  preferred over mean/σ in clutter with outliers — exactly what `detect_cfar.py` implements. `[verified-abstract]`
- **G24 (the only ULM paper to study the threshold itself).** Adding controlled detection
  errors to simulated ULM: **false negatives hurt far more than false positives** — at a 20 %
  error rate, FN dropped SSIM by **~45 %** vs only **~7 %** for FP. Recommendation: **bias
  toward sensitivity (lower threshold)**, accept more FPs, and use **region-adaptive
  thresholds** because **sparse regions are much more sensitive to error** than dense ones. `[verified-fulltext]`

### Recommended setting / range and WHY

- **Estimator:** Keep **local median + `1.4826·MAD`** (robust, σ-calibrated) with a guard band
  and a **global floor** under the scale (prevents the z-score from exploding and
  manufacturing detections in near-empty/noise-only regions — the classic CFAR failure mode
  in low-clutter cells). This is already correct in the code.
- **`sigma_threshold` (the k after σ-normalization):** Because the MAD is already scaled to σ,
  `sigma_threshold` is a **number of robust standard deviations**. A **~3σ** cut corresponds
  to a ~0.1 % per-cell false-alarm rate under Gaussian noise; with millions of voxel-cells
  per frame this still yields many raw peaks, which NMS + track-level rejection then prune.
  Given G24's "FN is costlier," run the detector **sensitive (k ≈ 2.5–3.5)** and let the
  **two-tier confidence** gate (high-conf spawns at k≈4–5, low-conf continues at k≈2.5–3)
  control false tracks rather than a single high threshold. Recommended starting point:
  **spawn ≈ 4σ, continue ≈ 2.5–3σ**. WHY: this turns G24's asymmetry into a concrete policy —
  recover coverage from dim periphery without a one-frame-false-track explosion.
- **Guard band:** must be **≥ the PSF radius** so the bright target does not contaminate its
  own background estimate. Current `(1,2,2)` (elev,z,x) under-guards **elevation** — the
  elevation PSF is the *widest* axis (§4), so the guard radius in elevation should be the
  *largest*, not the smallest. Recommend guard `(2–3, 2, 2)` or tie each guard radius to the
  per-axis PSF FWHM in voxels.
- **Background window (`local_radius`):** should be **several × PSF** so it estimates clutter,
  not the target. `(3,9,9)` is reasonable for z/x (~PSF×several) but again **too tight in
  elevation** relative to the wide elevation PSF; widen the elevation background radius to
  match.

### OUR SETUP caveats

- We have a **bright-midplane elevation artifact**; a per-elevation-plane background or the
  existing **`debias_elevation`** profile must be applied *before* the z-score, else the
  midplane sheet biases the local background and suppresses real off-midplane bubbles. The
  code's `elevation_bias_profile` is the right tool — use it.
- **No ground truth ⇒ no ROC.** You cannot pick `k` from a measured false-alarm rate. Pick it
  by the **track-level proxies** (mean curvilinear length, frac ≥35, fragmentation,
  occupied_fraction) and split-half reproducibility, sweeping `k` rather than asserting a
  value. The literature gives the *shape* of the recommendation (sensitive + two-tier), not a
  transferable number for our SNR.

---

## 3. NMS radius vs PSF size; anisotropic / matched-filter detection

**Our knob:** `anisotropic_nms_size(min_distance, …)` builds a `maximum_filter` with
**elevation radius defaulting to 2× the lateral/axial base**; optional matched-filter /
empirical-PSF cross-correlation upstream.

### What the literature says

- **One detection per resolution cell.** Two MBs closer than ~λ/2 are unresolvable (Wikipedia
  ULM; the diffraction limit); NMS/regional-maxima therefore operate at the **PSF / resolution-
  cell scale**. LB22 takes "**regional peaks of the 3D normalized-cross-correlation map**" as
  MB locations — i.e. matched-filter (template = system PSF) **before** peak-picking, which
  both denoises and enforces PSF-sized separation. BU25 takes regional maxima then
  sub-localizes. `[verified-fulltext]`
- **Matched filtering / PSF templates.** M23 convolves with a **calibrated 3D Gaussian kernel
  matched to the system PSF** before thresholding; LB22 uses **3D Gaussian-fit PSF + normalized
  cross-correlation**. The PSF is the natural detection template and the natural NMS radius. `[verified-fulltext]`
- **Localization method (HB22 benchmark).** Of 7 algorithms, **radial symmetry and Gaussian
  fitting** were rated highly on every index; **Gaussian fitting was ~50× slower** than radial
  symmetry for essentially equal accuracy; weighted-average/centroid and parabolic
  interpolation are faster but slightly less accurate. **No single winner** — the choice is an
  accuracy/speed trade. Radial symmetry is the common default (BU25, M23 use radial-symmetry /
  weighted-centroid on PSF-convolved data). `[verified-abstract]` Typical ULM localization
  precision is **sub-resolution (~λ/10)**, and the reconstruction grid is correspondingly λ/10. `[verified-fulltext]`

### Recommended setting / range and WHY

- **NMS radius ≈ PSF FWHM per axis** (≈ one resolution cell). Lateral/axial PSF ≈ λ·F# ≈
  0.58·0.5 ≈ **0.29 mm** ⇒ ~1 voxel-cell at λ/2 sampling, so a small lateral/axial NMS radius.
  **Elevation radius must be larger** because the elevation PSF is much wider (§4); the code's
  **default of 2× lateral is the right direction**, and may need to be **3–4×** for our
  row-limited aperture. WHY: an isotropic NMS on an anisotropic PSF either (a) leaves
  **elevation-smeared replicas** of one bubble as multiple detections, or (b) if enlarged
  isotropically, **merges genuinely distinct lateral bubbles**. Anisotropic NMS is the only
  correct option on anisotropic voxels.
- **Matched filter before NMS:** Adopt LB22/M23 — cross-correlate each frame with an
  (anisotropic, wide-in-elevation) **empirical 3D PSF** mined from bright isolated detections,
  then NMS on the correlation map. This both raises weak-bubble sensitivity (helps coverage)
  and makes the NMS radius physically meaningful (template width). Use a **disjoint-half PSF**
  as a free self-consistency check.
- **Sub-voxel localizer:** **radial symmetry** (fast, accuracy ≈ Gaussian per HB22) is the
  recommended default; reserve Gaussian fitting for a quality pass. Carry the **fit
  width/curvature as a localization covariance** into tracking (feeds §4 and the Mahalanobis
  gate).

### OUR SETUP caveats

- Our detection currently smooths in z/x but **not elevation** and uses isotropic NMS on
  anisotropic voxels — both fight the wide elevation PSF. Anisotropic, PSF-tied NMS +
  elevation smoothing is well-motivated by every 3D paper above.
- Our voxels are anisotropic (`elev=25` planes spanning the slice thickness vs `z=225, x=378`
  fine grids); the NMS radii **must be set in physical units / PSF-FWHM**, not a single voxel
  count, or elevation will be mis-handled.

---

## 4. Elevation / out-of-plane localization with a row-limited aperture (down-weight?)

**Our knob:** whether to **down-weight elevation** — inflate Kalman `R_yy` 5–10×, anisotropic
Mahalanobis gating, 2D-first linking, and treat the elevation coordinate as low-confidence.

### What the literature says

- **Fully-populated matrix arrays get a near-isotropic 3D PSF and do NOT down-weight
  elevation.** HV22 (whole rat brain, fully populated matrix), C22 (multiplexed 32×32),
  LB22 (32×32, **isotropic 24.7 µm voxels, uniform 3D treatment, no differential elevation
  handling**) and M23 (32×32 Vermon, "simultaneous elevation and lateral acquisition without
  special elevation handling") all treat the three axes symmetrically — *because their
  aperture is symmetric in azimuth and elevation*. `[verified-fulltext]` **This is the regime
  our literature defaults come from, and it does not apply to us.**
- **Row-column and limited/sparse apertures have a much wider, depth-degrading elevation
  (secondary) PSF.** W24 (RCA) must transmit in one direction and receive in the orthogonal
  to recover a usable second-direction focus; RCA imaging is characterized by a **spatially-
  varying, anisotropic PSF, edge-wave "ghost" lobes, and a secondary-axis −6 dB width that
  *worsens with depth*** (RCA literature reports elevation/secondary −6 dB spot growing from
  ~0.5 mm to ~0.9 mm over 5→20 mm `[unverified-number]` — directionally reliable even if the
  exact figures are probe-specific). The **secondary axis is the weak axis** for any
  row/column-limited aperture. `[verified-abstract]`
- **Aberration makes the out-of-plane problem worse** (BU25): an uncorrected, degraded PSF
  with high sidelobes causes **"vessel doubling"** artifacts; correcting it (reflection-matrix
  / matrix imaging) gave **+8 dB contrast, ×2–3 resolution, and +40–250 % detected tracks**.
  Our **single speed of sound (1600 m/s) and only 5 angles** mean we *cannot* do full UMI
  aberration correction (that needs a far richer reflection matrix), so we live with a
  degraded, elevation-heavy PSF. `[verified-fulltext]`

### Recommended setting / range and WHY

- **DOWN-WEIGHT elevation — yes, strongly, for our geometry.** Our elevation is **synthesized
  from a single physical receive row with 5 transmit angles** — an aperture *even more
  elevation-limited than an RCA*. The matrix-array papers' symmetric 3D treatment is therefore
  **inapplicable**; the RCA/limited-aperture regime applies, and that regime says the
  out-of-plane coordinate is the **least-reliable axis**.
- **Concrete:** inflate the elevation localization variance / Kalman `R_yy` by **~5–10×**
  relative to lateral (matches your idea #4), switch box-gating to an **anisotropic
  Mahalanobis gate** (wide in elevation, tight in z/x), and offer a **z-optional / 2D-first
  linking** mode so elevation jitter does not break otherwise-good tracks. **Seed the
  covariance from the parabola/Gaussian curvature** of each detection (the elevation peak is
  intrinsically broad ⇒ large variance falls out naturally — principled, not hand-tuned).
- **De-bias the bright midplane** before detection (§2) — a row-limited synthetic elevation
  focus tends to pile energy on the central plane, creating a "midplane-attraction" bias that
  fakes elevation localization; the existing `debias_elevation` profile addresses this.

### OUR SETUP caveats

- **No ground truth ⇒ elevation error cannot be measured.** Validate the down-weighting by
  proxies only: split-half elevation-plane occupancy, midplane-attraction audit, z-jitter of
  tracks, and track-level metrics (idea #10 harness). Do **not** claim a calibrated elevation
  precision.
- Every quantitative elevation-localization number in the matrix-array literature was obtained
  with a **fully-populated 2D aperture**; **none of those precisions transfer** to our 1-row
  synthesized elevation. The only transferable lesson from the *matched* regime (RCA / sparse)
  is qualitative: **the secondary axis is worse and should be trusted less.**

---

## 5. Aberration / single speed-of-sound (context for the above)

`[verified-fulltext]` BU25 is the state of the art: quantify aberration via the **reflection
point-spread function / Strehl ratio** (they measured S = 0.03 at 50 mm — severe), correct
with a **distortion-matrix + iterative-phase-reversal** scheme, gaining +8 dB contrast, ×2–3
resolution, +40–250 % tracks. **Relevance to us:** our angle×element data is "reflection-
matrix-like" but with **only 5 angles**, far too few to estimate per-pixel focusing laws;
and we use a **single global SoS (1600 m/s)**. So full UMI correction is out of reach. The
practical consequence is that our PSF is **broad and elevation-dominated and partially
aberrated**, which *reinforces* every recommendation in §2–§4 (sensitive detection,
matched-filter, anisotropic NMS, elevation down-weighting) rather than offering a separate fix.

---

## 6. Summary table — recommended settings for our knobs

| Knob | Lit-grounded recommendation | Source | Our-setup caveat |
|------|----------------------------|--------|------------------|
| **SVD tissue cutoff** | Data-driven **knee / min(gradient-turning-point, spatial-correlation)**, not a fixed 10 % floor; expect **few→low-tens of modes** | D15, B18, L20 | 10 % of 700 frames (~70) is too high for quiet blocks; floor → ceiling/guard |
| **SVD noise cutoff** (`high_cutoff`) | Add a high-order cutoff from **B18 spatial decorrelation** or **MP upper edge** | B18, Veraart16 | MP not yet standard in ULM — candidate, verify empirically |
| **Per-block SVD** | Keep `per_block`, **regularize/clamp the k-map**, blocks ≥ few PSF widths; sweep Δ global→per-block | S17, L20 | coverage loss is a **rank** artifact (PoU blend already seamless) |
| **Detection estimator** | local **median + 1.4826·MAD**, guard band ≥ PSF, **global floor** | CFAR theory, code | already implemented correctly |
| **Threshold k** | **Sensitive + two-tier**: spawn ≈ **4σ**, continue ≈ **2.5–3σ** | G24 (FN ≫ FP), M23, BU25 | no ROC → sweep k on track-level proxies |
| **Guard / background radius** | guard ≥ PSF FWHM **largest in elevation**; background several × PSF, wide in elevation | CFAR theory + §4 | current `(1,2,2)`/`(3,9,9)` **under-guard elevation** |
| **NMS radius** | **anisotropic, ≈ PSF FWHM per axis**; elevation **2–4× lateral** | LB22, M23, §4 | isotropic NMS on anisotropic voxels is wrong |
| **Matched filter** | cross-correlate with **empirical anisotropic 3D PSF** before NMS; disjoint-half PSF self-check | LB22, M23 | raises weak-bubble sensitivity (coverage) |
| **Sub-voxel localizer** | **radial symmetry** default (≈ Gaussian accuracy, ~50× faster); carry fit covariance | HB22 | feeds elevation down-weighting |
| **Elevation localization** | **Down-weight**: `R_yy`×5–10, anisotropic Mahalanobis gate, 2D-first option, seed R from curvature; **de-bias midplane** | W24/RCA regime, §4 | matrix-array symmetric treatment **does not apply** to our 1-row synthesized elevation |
| **Aberration** | full UMI correction infeasible (5 angles, single SoS) → rely on §2–§4 instead | BU25 | reinforces sensitive/anisotropic detection |

---

## 7. Honesty / verification notes

- **Read in full text:** L20, M23, LB22, BU25, G24 (methods parsed directly) and the W24
  abstract. Recommendations resting on these are highest-confidence.
- **Verified bibliographically (abstract/secondary only):** D15 (author list confirmed via
  PubMed), B18, S17, HB22, HV22, C22, HBM25. Their *headline methods* are well-attested in
  the secondary literature, but I did not parse their full methods text — the **B18 DOI
  `10.1109/TMI.2018.2789499` and the exact 13-estimator list should be re-checked** against
  the primary PDF before quoting verbatim.
- **`[unverified-number]`** the RCA elevation −6 dB figures (0.5→0.9 mm) come from a secondary
  RCA-imaging summary, not a primary methods section I read; the **direction** (elevation
  wider, worsening with depth) is robust, the **exact numbers are probe-specific** — do not
  cite them as ours.
- **No ULM paper found that uses Marchenko–Pastur for the SVD cutoff.** It is presented here
  as a principled borrow from MRI/RMT, not an established ULM default.
- **HBM25** (human-data 3D ULM benchmark) is the most on-point for our *human* dataset but is
  paywalled; only the abstract was available. Worth a full read if access can be obtained — it
  benchmarks probe configurations, MB concentration, sub-pixel motion and skull aberration on
  a realistic human-brain vasculature simulation, which is exactly our validation gap.
