# Ideation round 25 — codex + fable iterate on "make it even better"

Date 2026-07-08. Requested by Rome ("get codex and fable to iterate on a bunch of ideas").
Three grounded idea streams (each read the docs + code): **render/viz** (fable),
**signal/insight+validation** (fable), **reconstruction/algorithm** (codex). A fourth
(reconstruction, fable) failed to engage (0 tool-uses) and was covered by codex.

State entering this round (all GT-free validated): graph-Laplacian along-vessel field
regularization → split-half Dice 0.61→0.71 / cos 0.84→0.95 (mb-ki9); 4-panel composite
(mb-4k2); coverage-vs-reference ~90% within 1 vox at ~1.8× null (mb-ska); dataset-wide
over all 216 acqs (mb-9ay). Persistent honesty rails: the **elevation (y) axis is 2.77×
coarser & weakly synthesized** — anything smearing across it inflates metrics; the
resampling unit is the **acquisition** (216), not voxels/samples.

---

## Synthesis — top picks to pursue (impact × feasibility), cross-cutting the three lists

1. **Cardiac pulsatility index + phase map** (signal #2) — recover heart rate from the
   *untouched* 222 Hz time axis (240 frames/acq = 1.079 s ≈ 5–9 mouse cardiac cycles),
   build per-voxel PI = (v_max−v_min)/v_mean over cardiac phase. The biggest genuinely-new
   signal; arteries pulsate/veins don't → an **independent** A/V discriminator. Beaded → **mb-pi (P1)**.
2. **Bootstrap CIs over acquisitions** (signal #1) — resample the 216 acqs (B≈1000), put a
   95% CI on every headline number (Dice, cos, A/V counts, coverage-z). Cheapest credibility
   multiplier; almost no compute. Beaded → **mb-boot (P1)**.
3. **Fixed-volume split-half Pareto search** (recon codex #2) — tune σ/γ/step/seeding for
   Dice/cos *at fixed occupied volume* (the volume clamp IS the anti-smear guard). Directly
   pushes the current win further and defensibly. Beaded → **mb-parev (P1)**.
4. **Directional FSC/FRC resolution** (signal #3) — odd/even Fourier-shell correlation →
   an objective resolution (µm) at 0.143/0.5, reported *per-axis* (elevation honestly worse).
   Replaces "coarsen-2 grid" hand-waving with a field-standard number. Beaded → **mb-fsc (P2)**.
5. **Coherence-enhancing PDE / tensor-Laplacian regularizer** (recon codex #1,#10) — the
   next-gen regularizer beyond the current scalar graph-Laplacian: structure-tensor-steered
   anisotropic diffusion that preserves cross-vessel separation. Beaded → **mb-tens (P2)**.
6. **Living-angiogram honest render** (render #1+#2+#7) — confidence/split-half-modulated
   opacity (fade the smoothing-filled/uncertain segments) + flow-comet advection along
   streamlines + split-half "core vs ghost". Folds into the render beads mb-y8i/mb-hi8 and
   the diff-viewer being built this session.
7. **Cross-metric A/V agreement** (signal #4) — Cohen's κ between antiparallel-adjacency and
   pulsatility-based A/V (independent estimators agreeing ≈ GT-grade). Depends on #1(pulsatility).
   Beaded → **mb-avx (P2)**.

Runners-up worth keeping: held-out-acquisition predictive cosine (signal #5), physical-
plausibility gate battery (signal #6, in-plane ∇·v only), temporal cross-correlation
velocimetry (signal #9, timing-independent velocity check), multi-vector field at crossings
(recon #3), grid-free MLS/RBF field (recon #5), bidirectional-consistency pruning (recon #15),
tubes + HDR bloom render (render #3,#5), elevation-honesty overlay (render #13).

---

## Source list A — RENDER / VISUALIZATION (fable, 15 ideas)
Tier-1: (1) confidence-modulated opacity/brightness [honesty flagship]; (2) flow-comet
advection [living-angiogram signature]; (3) HDR additive cores + restrained bloom; (4) A/V
two-tone with opposed flow; (5) tube/ribbon geometry with flow-aligned shading. Tier-2:
(6) pulsatility brightness wave [label as illustrative]; (7) split-half core-vs-ghost dual
render; (8) aerial-perspective depth cueing; (9) direction-hue (DTI) live toggle; (10)
flux-modulated comet density. Tier-3: (11) clip-plane/slab scrubber; (12) perfusion
volumetric backdrop; (13) elevation-honesty overlay; (14) matplotlib fleet-safe hero (Agg +
faux-bloom, chrome-free); (15) antiparallel "railroad" twin-track motif. Every aesthetic
lever is paired with the honesty rail it threatens; confidence-opacity governs all.

## Source list B — RECONSTRUCTION / ALGORITHM (codex, 15 ideas)
(1) confidence-weighted tensor-Laplacian regularization; (2) fixed-volume split-half Pareto
search; (3) multi-vector field at crossings/bifurcations; (4) uncertainty-aware streamline
integration (RK4 / particle-filter, confidence-hysteresis stop); (5) grid-free MLS/RBF vector
field; (6) elevation-axis super-resolution with anti-smearing prior [highest gaming-risk];
(7) bootstrap-ensemble probabilistic tractography; (8) global min-cost gap-closing between
fragments; (9) curvature-regularized variational streamlines; (10) coherence-enhancing PDE
diffusion on velocity tensors; (11) soft mass-conservation only where resolution supports it;
(12) optimal-transport barycenter of odd/even vessel measures; (13) self-supervised
masked-acquisition denoiser; (14) bifurcation-aware adaptive seeding; (15) bidirectional-
consistency pruning. Recurring guard: fixed occupied volume + elevation-FWHM check.

## Source list C — SIGNAL / INSIGHT + VALIDATION (fable, 16 ideas)
Tier-1: (1) bootstrap CIs over acquisitions; (2) cardiac pulsatility index + phase map; (3)
directional FSC resolution; (4) cross-metric A/V agreement (κ); (5) held-out-acq predictive
cosine. Tier-2: (6) physical-plausibility gate battery (in-plane ∇·v); (7) flow-conservation
(Kirchhoff) flux atlas; (8) elevation-smear audit (2D-vs-3D of every metric); (9) temporal
cross-correlation velocimetry (timing-independent velocity). Tier-3: (10) disturbed-flow /
laminar index; (11) vessel-diameter/caliber map (in-plane); (12) tortuosity & branching
morphometrics; (13) wall-shear-rate proxy [most inflatable — in-plane only]; (14) vascular-
territory parcellation; (15) harder null-model suite; (16) split-half residual anomaly finder.
Two honesty rules throughout: report in-plane (x–z) or 2D-vs-3D-ablate every transverse/3D
metric; resample acquisitions (not voxels/samples).

Provenance: this round's raw agent outputs summarized here; builds on docs/ideation/19–24.
