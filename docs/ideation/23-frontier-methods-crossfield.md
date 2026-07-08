# Frontier methods — cross-field + deep-math round (fresh, novel)

**Date:** 2026-07-08. Three fresh fable agents, math-deep, instructed to EXTEND (not repeat)
docs 19/21/22 and the two detection-quality consults. Sources: (A) MRI/CT/PET/SPECT
reconstruction math; (B) pure/applied math; (C) wild cross-field. The value is less any single
idea than the **convergence** — independent fields deriving the *same* fixes for our named
problems. Raw agent deliverables are dense and equation-rich; this is the synthesis + the cheap-
experiment shortlist.

## The convergences (independent fields → same fix)

### 1. Detection MULTIPLICITY (one bubble → 2–3 detections) — FIVE principled routes, not NMS
- **PET list-mode MLEM** (A): the sensitivity term `sⱼ = Σₑ Hₑⱼ` divides the multiplicity out
  of the reconstructed density *analytically* — one bubble firing 3 detections gives an unbiased
  flux. Also gives the honest PSF-aware super-res render. Runs on the acq-0 list we hold.
- **Neyman–Scott / Thomas cluster process** (B): the same deconvolution identity
  `ρ̂_p = μ⁻¹·h⁻¹∗λ_d`, and it *measures* the multiplicity as GT-free numbers — `μ` (detections
  per bubble), `σ` (the correct dedup radius, replacing NMS's guess), `ρ_p` (true concentration)
  — from the pair-correlation function `g(r)=1+(1/ρ_p)(4πσ²)^{-3/2}e^{-r²/4σ²}`.
- **Determinantal point process** (B): a repulsion kernel `L_ij=q_i·k(y_i,y_j)·q_j` with an
  anisotropic PSF kernel — a global, order-independent, score+geometry NMS with an MDL stopping
  rule.
- **Shake-the-Box / IPR** (C, experimental fluids): our multiplicity is very plausibly TomoPIV
  **ghost particles** (5 transmit angles = the tomographic cameras), a phenomenon fluid dynamics
  already has full ghost-suppression theory for (temporal-coherence test).
- **Sheaf consistency-radius** (C, category theory): `C(s)=√Σ‖r_i(s_i)−r_j(s_j)‖²` across the 5
  angle-views — merge when the views agree below the noise floor; a geometry-aware dedup that
  uses the known angle restriction maps (distinct from distance-merging).
- (Also: profile-HMM insert states absorb multiplicity, B/C.)
→ **The multiplicity is a deconvolution / point-process problem with a principled fix. All routes
  propose the SAME cheapest experiment: measure μ,σ on acq-0 → de-dup → re-track.**

### 2. GT-free VALIDATION (repairs the "inflated coverage / wrong denominator" credibility hit)
- **Optimal transport / Sinkhorn divergence** (B, C): `S_ε(ρ_odd,ρ_even)` on odd/even acq splits
  — a debiased, geometry-aware metric (a vessel shifted 1 voxel: small W₂ but tanks pixel-FRC),
  a true metric with a transport plan localizing disagreement. Replaces the ad-hoc
  Coverage@equal-FRC the evaluator caught inflating.
- **Persistent homology** (B): β₀ at physiological scale = a *fragmentation number* no other
  metric produces (broken vessels = excess components); bottleneck stability = GT-free
  connectivity reproducibility.
- **Capture–recapture** (C, ecology): two stochastic detection runs → Lincoln–Petersen/Chapman →
  **the project's first defensible GT-free RECALL number with a CI** (`N̂=(n_A+1)(n_B+1)/(m+1)−1`).
- **Percolation exponents** (C): a true cerebral tree has fractal statistics; fit `τ`/`D_f` to
  test whether a map has the right connectivity, referenceless. Also sets thresholds at the
  connectivity phase transition (coverage-vs-continuity as a critical phenomenon).

### 3. TRANSCRANIAL aberration (the gate on every clinical moonshot in doc 19)
- **VLBI closure phase** (C): image *through* the skull with NO speed-of-sound. Closure phase
  `Ψ=arg[V(u,v)V(v,w)V(w,u)]` is invariant to per-element phase — ~17k aberration-immune
  constraints *per transmit* (N=134). Marginalizes the skull out; needs no guide stars (unlike
  doc-19's distortion matrix), no CT, no training. The EHT maneuver.
- **HBT intensity interferometry** (C): `g^{(2)}=1+|g^{(1)}|²` (Siegert) gives visibility
  *amplitudes* phase-blindly — completes closure phase's aberration-immune data set. NOT SOFI
  (spatial cross-detector, not temporal cumulants).
- **Passive seismic interferometry** (C): recover `c(z)` from the clutter we currently discard
  (cross-correlate the reverberant residual) — an independent measurement of the aberration.
→ **Closure + HBT + passive-SoS = one convergent, CT-free, GT-free program for transcranial ULM.**

### 4. TRACKING-FREE vessel building (sidesteps the dataset-wide re-track blocker, doc 22)
- **Diffusion-MRI tractography** (A): integrate streamlines through the validated 0.96 velocity-
  direction field (`dr/ds=v̂`), RK4 + curvature gate. Signed vectors → arteries/veins for free;
  bifurcations = vessel-ODF (spherical deconvolution). Builds long vessels *without linking a
  single noisy detection*.
- **Optimal transport / Benamou–Brenier** (A/B/C): `W₂²=min∫ρ|v|²` s.t. `∂_tρ+∇·(ρv)=0` — the
  velocity field as the geodesic between frame densities; association-free, mass-conserving.
- **4D-Var / EnKF** (C, weather): assimilate detections into `∂_tρ+∇·(ρv)=0` via the adjoint;
  association never happens; EnKF gives flow-shaped uncertainty (inflates on the weak elevation).

### 5. Other high-value singletons
- **RTS backward smoother** (B) — nearly FREE add to our forward-only Kalman; lower-variance
  positions → cleaner streamlines, directly on the crispness/jitter gap.
- **MR-fingerprinting dictionary** (A) — replace SVD with matched-filter to a transit-fingerprint
  dictionary; deterministic detection (kills the non-determinism) + velocity for free.
- **SENSE beamforming g-factor** (A) — 134 channels = coils; the elevation g-factor map IS the
  principled elevation-uncertainty σ_elev that threads MLEM / MEDI / Mahalanobis association.
- **QSM/MEDI dipole inversion** (A) — the elevation MTF null ≈ the QSM cone; use in-plane vessel
  edges as a morphological prior to recover elevation structure the array can't measure.
- **PHD/CPHD/GLMB (RFS filters)** (B) — Bayes-optimal unknown-count tracking in clutter with
  multiplicity; the machinery literally built for our regime.
- **PDE-constrained variational flow** (A/B) — incompressibility `∂_z v_z=−(∂_x v_x+∂_y v_y)`
  *reconstructs* the weak elevation velocity; ships pressure + wall-shear-stress.
- **Graph signal processing** (B) — `v̂=(I+γL)⁻¹v_obs` denoises flow ALONG vessels; spectral A/V.
- **Profile-HMM / affine-gap alignment** (C) — track stitching with native insert(multiplicity)/
  delete(dropout) states; Baum–Welch learns detection efficiency (dovetails capture-recapture).

## The cheap-experiment shortlist (all acq-0, GT-free, ~afternoon, single niced process)
Run when the box frees up. Ranked by (leverage × cheapness × convergence-backing):
1. **Multiplicity measure → de-dup → re-track** — Ripley-K/PCF fit for (μ,σ,ρ_p), then DPP or
   Thomas-EM dedup, re-run the tracker, measure median/max track length. *3-way convergent
   (MLEM/Thomas/DPP); confirms-and-fixes the sustain problem in one.*
2. **Capture–recapture recall** — two stochastic detection runs → Chapman N̂ + recall(z) with CI.
   *The single most valuable missing number in the project; trivial.*
3. **MLEM density on the acq-0 list** — honest de-duplicated super-res render; ratio-map vs drizzle
   shows where multiplicity bias lived. *No re-track, no training.*
4. **Tractography on the validated field** — seed streamlines, RK4 integrate, count ≥35-arclength
   centerlines vs reference 1,421. *Vessels without re-tracking; sidesteps the blocker.*
5. **RTS backward smoother** on existing acq-0 tracks — split-half streamline reproducibility
   before/after. *Nearly free.*
6. **OT Sinkhorn divergence** on odd/even splits — the honest GT-free metric. *Milliseconds.*
7. **Closure-phase invariance test** — is `Var(Ψ)` across acqs ≪ `Var` of raw per-baseline phase?
   *Falsifiable one-afternoon test of the whole transcranial program.*

## Highest-leverage bets
- **Near-term, can't-lose:** the **multiplicity measure → de-dup → re-track** (#1) — three
  independent derivations, runs on data we hold, confirms-and-fixes; plus **capture-recapture**
  (#2) for the recall number the project has never had.
- **The render that ships:** **MLEM density (#3) + tractography streamlines (#4) + PC-Doppler
  signed velocity** — each from validated inputs, none needing the dataset-wide re-track.
- **The moonshot (bet-the-lab):** **closure-phase + HBT + passive-SoS aberration-immune
  interferometric ULM** — the CT-free route to transcranial adult ULM, i.e. the unlock that turns
  this into a magnet-free neuroimaging modality. Falsifiable in one afternoon (closure-phase
  stability test).
- **The unifying moonshot:** a single **continuity-equation-constrained joint density+velocity
  reconstruction** (Benamou–Brenier / 4D-Var) into which detection, tracking, and flow dissolve —
  realized as an interpretable convex/variational + random-finite-set object (with OT and
  persistence-diagram GT-free objectives), not a black-box neural field.

Codex is separately implementing/testing the empirical de-dup sustain fix (detq/); its
before/after track numbers will anchor experiment #1. Beads: mb-bfl (sustain), mb-k25 (epic).
