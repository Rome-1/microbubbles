# Frontier brainstorm — where does Ultratrace 3D ULM go next?

**Method:** 5 Fable agents, each given the same grounding (the real pipeline, the fixed
data, the hard constraints, the non-determinism finding) but a distinct "beyond the
horizon" lens: **acquisition/physics · learning-based reconstruction · tracking/flow-
modeling · science & applications · moonshots/cross-disciplinary**. Ran in parallel,
zero compute, zero Modal spend (produced while WAITing on upstream issue #2).

This doc is the **synthesis** — the raw per-lens deliverables are dense and worth reading,
but the real value is in the *convergences*: ideas that ≥2–4 independent lenses arrived at
from different directions. Those are the high-conviction bets.

---

## The one big reframe (emergent — 4 of 5 agents landed here independently)

> **The detection non-determinism we filed as upstream bug #2 is not (only) a bug — it may be the signal.**

We root-caused that identical code on identical input yields ~85% different detections
run-to-run, because ~99.9% of detections sit on a reshuffling ~2σ noise floor over a
small stable core (~48/acq). Four agents, from four different lenses, independently
concluded the same thing: **the information that separates a real bubble from a noise
blip is not in any single frame — it is in temporal/motion coherence and run-to-run
statistics.** That reframe spawns a whole family of methods that get *stronger* the noisier
the front end is:

- **Fluctuation-imaging (SOFI / SPARCOM)** — moving bubbles make a voxel's intensity
  fluctuate; higher-order temporal cumulants of the SVD-filtered IQ sharpen the effective
  PSF by √n and suppress uncorrelated background **without ever thresholding a detection**.
  "ULM without localization," deterministic *by construction* because statistics don't
  reshuffle. *(moonshot lens #1)*
- **Noise2Noise self-supervision** — two stochastic detection runs are two independent
  noisy views of one true bubble field: exactly the pairing Noise2Noise needs. Train a
  net to predict run-B from run-A; what survives is provably the invariant vasculature.
  *(ML #2, moonshot #3)*
- **Track-before-detect** — integrate likelihood *along candidate trajectories* through the
  raw volume; a bubble is confirmed by motion coherence, not by brightness. *(tracking #1)*
- **Bootstrap uncertainty maps** — each stochastic run is a bootstrap sample; per-voxel
  variance gives calibrated confidence ("this vessel p>0.99; this one speculative"). The
  first ULM maps a clinician can regionally trust. *(ML #9, moonshot #8)*

**If Rome wants one headline out of this whole exercise, it's this:** our upstream bug
investigation may have surfaced a *method*, not just a defect. The cheapest experiments
below test it this week.

---

## Tier 1 — ship-this-week / near-term (A10G-cheap, no-GT, high-conviction)

Ranked by (conviction × cheapness × how directly it hits our named tensions: coverage-vs-
continuity, short tracks, crude detection, overlapping bubbles, synthetic elevation).

| # | Bet | Lens(es) | What it hits | Cost |
|---|-----|----------|--------------|------|
| 1 | **Two-frame persistence gate (TBD-lite)** — keep a 2σ candidate only if a plausible partner exists ±4.5 ms along an admissible velocity vector | tracking #1 | guts the reshuffling floor at its root | trivial |
| 2 | **Velocity-space Hough/Radon TBD** — shift-and-add the space-time volume over a bounded velocity grid (0.09–0.9 mm/frame); peaks = motion-confirmed detections *with velocity attached* | tracking #1 | coverage+continuity jointly; free velocity | batched conv, <24GB |
| 3 | **Coherence-factor PSF sharpening** — reweight each DAS voxel by receive-aperture coherence (CF = \|Σxᵢ\|²/(N·Σ\|xᵢ\|²)) before detection | physics #1 | sharper PSF → fewer false peaks, better localization | arithmetic in beamform kernel |
| 4 | **Matched-filter localization** — auto-harvest an empirical 3D PSF from isolated stable-core bubbles, detect by normalized cross-correlation instead of raw z-score | physics #4 | crude detection; pulls low-SNR bubbles | cheap, local |
| 5 | **Global min-cost-flow association + covariance-aware costs** — retire greedy Hungarian; network-flow with skip-k-frame edges + anisotropic Mahalanobis costs | tracking #2, #5 | stitches through SVD dropouts; short tracks | ms-scale CPU graph solve |
| 6 | **SOFI temporal-cumulant smell test** — 2nd/4th-order cumulant volume of one acq's SVD-filtered IQ; does vessel structure appear sub-diffraction with no detector? | moonshot #1 | reproducibility crisis; whole new modality | afternoon of code |
| 7 | **Noise2Noise consensus detector** — train a small 3D U-Net run-A → run-B; recover the invariant field | ML #2, moonshot #3 | non-determinism → clean detector | ~1 day |
| 8 | **Doppler/Kasai velocity prior** — lag-one autocorrelation of SVD-filtered IQ → axial velocity field; feed as Kalman predict prior + link-consistency gate | physics #7, tracking #3 | short tracks; seeds velocimetry | standard, cheap |
| 9 | **Anisotropic-elevation covariance** — measure elevation PSF width; set R_z ≫ R_x,R_y; Mahalanobis (not Euclidean) association cost | physics #6, tracking #5 | synthetic-elevation honesty; fragmentation | pure track-stage change |
| 10 | **Self-gated pulsatility + artery/vein split** — band-pass the aggregate mean-speed at the heart rate (240 cycles), bin displacements by cardiac phase, threshold pulsatility index | tracking #6 | *new science output*, additive | post-processing |

**The near-term flagship bet (composed):** #1→#2 (TBD-confirmed detections) → #5 (global
flow with covariance costs) → closed in an iterative **shared flow-field loop** (below).
Each piece independently shippable; together they should push median track length toward
the *physical* transit ceiling (~20 frames) and make the fused map far denser and more
trustworthy than any single-track tweak.

---

## Tier 2 — bigger builds, mid-term (feasible on our constraints)

- **Physics-sim digital twin → manufactured ground truth → DECODE learned detector**
  *(physics #9 + ML #1 + moonshot #2 — 3-lens convergence).* Simulate point scatterers on
  known 3D tracks through our *exact* forward model (5 angles, 1×134, 25-planes-from-1-row,
  f#0.5, 2.75 MHz) + our MACH beamformer; inject into **real harvested clutter residual**
  (bubble-free SVD-filtered voxels) to close the realism gap. Now every proxy metric
  becomes a real number, and we can train a probabilistic sub-voxel detector with
  calibrated uncertainty. **This is the infrastructure that multiplies everything else** —
  it's the "MNIST of 3D ULM." First experiment: simulate ONE acq, run our existing detector
  on it, measure the FP/FN curve against truth we control → tells us whether the 2σ floor
  is recoverable signal or genuinely gone.
- **Shared flow-field prior (iterative EM)** *(tracking #3).* Blood velocity is a spatially-
  coherent, pulsatile field; every bubble in a voxel obeys it. EM: track → re-estimate
  v(x,t) by robust local regression + ∇·v≈0 → re-track. Rescues coverage AND continuity
  together, and **pins the weak elevation axis by spatial coherence** rather than poor
  per-frame localization. Cross-acq registration makes the field much stronger.
- **Vessel-graph structural prior** *(tracking #4).* Fit a centerline graph from accumulated
  localizations, then track *along arclength* — collapses 3D association to nearly-1D,
  makes bifurcations explicit graph choices, supplies the elevation coordinate the data
  can't. Bootstraps: rough map → skeletonize → track in s → denser map → repeat.
- **Sparse deconvolution for overlapping bubbles** *(physics #4 + ML #5 + tracking #8 —
  3-lens).* Recover two emitters inside one PSF via CLEAN / unrolled sparse recovery against
  the known anisotropic PSF. Lifts the concentration ceiling → map fills faster. Reach:
  un-mix by *motion* — two overlapping bubbles with different velocities separate across
  frames even when inseparable in one.
- **Deep-unrolled robust-PCA clutter filter** *(ML #4).* Replace the global SVD cutoff with
  learned, spatially-adaptive L+S soft-thresholds; self-supervised (L+S≈data + temporal
  sparsity on S). Aims for region-SVD's −47% fragmentation *without* its −27% coverage hit.
- **Distortion-matrix / CLEAN aberration correction** *(physics #2 + moonshot #5).* Use the
  bubbles themselves as guide stars: extract the phase aberration from the 5-angle reflection
  matrix, correct the single-speed-of-sound smear **without CT**. Even a coarse per-slab
  correction should tighten the PSF everywhere.
- **GNN track-finding (HEP-style)** *(moonshot #6).* Build a space-time graph over *raw
  stochastic* detections; a GNN edge-classifier learns that real tracks persist across runs
  while noise edges don't. Weak-labeled from the stable core or distilled from min-cost-flow.
- **Uncertainty/confidence maps + differentiable proxy-loss meta-optimization** *(ML #6, #9).*
  Ship error bars; and let ES/gradients jointly optimize the SVD-cutoff / z-threshold /
  Kalman-gate knobs against a label-free track-coherence loss instead of human coordinate-
  descent.
- **Richer motion models (IMM / flow-tangent Kalman)** *(tracking #9).* Constant-velocity
  gating breaks at curves/bifurcations — exactly where vessels carry the most information.
  Cheap surgical fix that composes with the +43% predicted-state gate we already have.

---

## Tier 3 — moonshots (break a constraint, but reachable-ish on one dataset)

- **Neural field / 4D flow PINN as *the* reconstruction** *(ML #3 + tracking #7 + moonshot
  #7/#10 — the single biggest cross-lens attractor).* One continuous, differentiable field
  F(x,y,z,t) → (occupancy, velocity, confidence), supervised by all 223 acqs' localizations
  as noisy samples, regularized by **divergence-free flow + tubularity + Womersley pulsatile
  profiles**. Discrete detection and tracking *vanish as intermediates*; coverage-vs-
  continuity vanishes with them because continuity becomes a prior, not a data-association
  gamble. Validated by **held-out-acquisition prediction** — a real generalization metric on
  a dataset with no ground truth. PINNs fit a single scene, which is exactly our regime, so
  this is a *reachable* moonshot. Reach: invert for **pressure & wall-shear-stress** →
  CFD-from-ULM, a patient-specific microvascular hemodynamic digital twin from 4 minutes.
- **Differentiable forward-model inversion (seismic FWI analogue)** *(ML #6 + moonshot #4).*
  Make the whole chain (DAS, SVD, PSF, elevation synthesis) differentiable; ask gradient
  descent "what bubble field + speed-of-sound best explains the raw IQ?" Detection and
  beamforming stop being separate lossy stages. Keystone first experiment: differentiable
  DAS, backprop from a beamformed voxel to raw channel data on one frame.
- **Cryo-EM-style iterative refinement over the 223 acqs** *(moonshot #9).* Treat acqs as a
  particle stack; EM against a shared model while solving per-acq motion/SoS nuisance params
  → 4-min-scan motion becomes a *solved latent*, enabling pulsatility-resolved imaging.
- **Diffusion vascular prior for gap-filling** *(ML #7).* Train a generative prior on vessel
  morphology (public angiography + synthetic vascular-tree generators), apply as a plug-and-
  play regularizer with data-consistency projection. Hallucination confronted directly via
  held-out-acq inpainting recovery rate.
- **Closed-loop adaptive acquisition** *(moonshot #11).* An uncertainty-driven policy decides
  where the next transmit looks. Offline first experiment on the digital twin: does non-
  uniform angle allocation reconstruct the tree with fewer transmits than uniform? (A 30s
  adaptive scan matching today's 4 min = the step that makes ULM clinically routine.)

---

## The "so what" — application north stars (science lens)

Two hard constraints govern all of this: **(1) the adult skull** (transcranial ULM is
aberrated/attenuated — possible but compromised), and **(2) synthetic elevation + no ground
truth** (prerequisite for any biomarker claim vs. a pretty render). The reachable clinical
wedges route *around* the skull:

- **FLAGSHIP (near-term, plant the flag here): bedside neonatal 3D functional ULM through
  the fontanelle — healthy vs HIE, quantitative flow-deficit map.** No skull. Proven acoustic
  window. Built-in ground-truth path (follow-up MRI; beats the coarse Doppler it replaces).
  Lands in a neuroprotection decision window measured in *hours*. One vivid figure — normal
  baby beside injured baby, deficit rendered in flow — recruits clinical partners.
- **Intra-operative neurovascular guidance** (skull open, OR pays, concurrent DSA = built-in
  validation): live bypass-graft patency, AVM nidus/feeders, aneurysm-clip perforator check.
- **Flow-native biomarkers only ULM gives** (each track is a velocity waveform): per-vessel
  pulsatility index, artery/vein separation, cerebral autoregulation monitoring (ICU/TBI).
- **The platform play:** a validated human cerebral angioarchitecture *atlas* (ULM vs
  DSA/CTA/micro-CT) — the "what's normal?" denominator every downstream biomarker needs, and
  the credibility gate that converts "beautiful render" into "trusted measurement."
- **MOONSHOT: transcranial, whole-brain, awake-adult functional ULM** — solve skull-aberration
  correction (Tier-2 distortion-matrix) + true matrix-array 3D, and Ultratrace stops being an
  ultrasound technique and becomes a **magnet-free, radiation-free, portable functional
  neuroimaging modality** below the fMRI voxel. Stroke triage, dementia screening, adult fULM
  all fall out of that one unlock.

---

## Convergence map (which lenses agreed — the meta-signal)

| Idea | acq/physics | ML/DL | tracking | science | moonshot |
|------|:-:|:-:|:-:|:-:|:-:|
| Non-determinism as signal (SOFI / N2N / TBD / bootstrap) | | ✓ | ✓ | | ✓✓✓ |
| Neural field / 4D flow PINN reconstruction | | ✓ | ✓ | | ✓✓ |
| Digital twin → manufactured GT → learned detector (DECODE) | ✓ | ✓ | | | ✓ |
| PSF sharpening / aberration correction (CF, CLEAN, distortion-matrix) | ✓✓ | | | | ✓ |
| Flow-physics priors (∇·v=0, incompressibility, Womersley) | | ✓ | ✓ | | ✓ |
| Global/graph association replacing greedy Hungarian | | | ✓ | | ✓ |
| Sparse deconvolution for overlapping bubbles | ✓ | ✓ | ✓ | | |
| Doppler/velocimetry → flow-native biomarkers | ✓ | | ✓ | ✓ | |

The rows with the most ✓ are where independent reasoning converged — start there.

---

## What I'd tell Rome in one paragraph

The frontier splits cleanly into three horizons. **Now:** a cluster of cheap, no-GT,
A10G-friendly experiments — TBD-lite motion gating, coherence-factor PSF sharpening,
matched-filter localization, global min-cost-flow tracking, and (the surprise) turning our
own detection non-determinism into signal via SOFI temporal cumulants + Noise2Noise. Several
are an afternoon of code and directly attack coverage-vs-continuity and short tracks at the
root. **Next:** build the physics-sim digital twin — it manufactures the ground truth this
whole project has lacked and becomes the benchmark every other idea is scored against. **The
horizon:** a physics-informed 4D neural flow field that dissolves detection-and-tracking into
one continuous inverse problem, and — the reason any of it matters — a bedside neonatal
functional-ULM flagship that routes around the adult skull and turns a beautiful render into
a clinical decision. The single most important insight from the exercise: **the reproducibility
problem we filed upstream may not be a bug to fix but a measurement we weren't taking.**
