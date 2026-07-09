# The flow-diversity gap: why the reference render looks better, and how to measure it

> **⚠️ REFRAMED (2026-07-09).** Rome corrected the objective after this investigation: the goal is
> **accurate tracked bubble trajectories** (GT-free-validated), not visual match and not the
> orientation-diversity `cl` used throughout below. Matching the reference's `cl≈0.53` was an
> **unjustified prior** — within a single acquisition the data is *coherent* (`cl≈0.72`); the 0.53
> "diversity" is mostly *inter-acquisition* pooling, and our low-`cl` tracks were partly tracking
> error. So the "diversity gap" this doc chases is largely an artifact. Current direction:
> **`docs/bubble-tracking-acq0.md`**. This doc and its `cl` metric are kept for the record.


**Status: OPEN investigation (2026-07-09).** Prompted by Rome's visual read: *none of our
renders look like what Aleph shipped; theirs strikes a better balance across many different
flows, ours prefers lots of parallel long-track work; theirs looks better and it's unclear how
to even quantitatively evaluate it.* This doc states the diagnosis, an evaluation framework
(the part that was missing), and the fix directions. Numbers filled by
`scripts/wf_render_signal/flow_diversity.py` + `render_flow_diversity.py`.

## 1. The observation, made concrete

Side-by-side (reference screenshot `renders/references/Screenshot 2026-06-29 …png` vs our
`renders/composite_science.png` / diff-viewer):

| | Aleph reference | Ours (graph-field tractography) |
|---|---|---|
| density | **sparse**, delicate, lots of black negative space | **dense**, space-filling |
| structure | **resolved individual vessels**, fanning/branching/crossing | **combed parallel** streamlines |
| direction | **many directions coexisting** locally | locally near-uniform |
| object | tracked bubble paths (a length-filtered subset) | integrated field streamlines |

Their render reads as real angiography; ours reads as a smoothed vector field.

## 2. Root cause (a design consequence, not a bug)

Two choices in our pipeline manufacture the parallel/dense look:
1. **`build_field` averages velocity to ONE vector per voxel.** Where two vessels cross or a
   vessel branches, the reference keeps both bubble paths; we collapse them to their mean
   direction. Crossings/branches — the source of "diverse flows" — are averaged away *before*
   tractography ever runs.
2. **Heavy along-vessel smoothing (γ=1.5, 200 iters) + dense seeding + gap-filling** straightens
   streamlines into long parallels and fills space, because that maximizes split-half length and
   Dice — the very metric we optimized. **The compass was misaligned:** Dice/cosine reward smooth
   parallel fields, which is orthogonal to (often opposed to) structural/visual fidelity.

So our headline scientific claim (long, reproducible vessels + extra channels) and the render's
beauty are *different objectives we conflated*. G1 (defensible science) is met; G2 (a render that
beats the blog aesthetically) is not — and Dice can't see the difference.

## 3. How to quantitatively evaluate it (the missing piece)

Reproducibility (Dice/cos) is necessary but blind to diversity. Add a **structure axis** and,
crucially, **measure distance-to-the-reference-structure** — the reference tracks are the target,
and we have them (all 216 acqs), so "how close does our render's structure come to theirs" is
directly computable.

**(A) Orientation diversity — per-voxel orientation tensor.** For all segment directions in a
voxel, `T = mean(d⊗d)`, eigenvalues λ1≥λ2≥λ3, coherence `cl=(λ1−λ2)/Σλ`. `cl≈1` = one dominant
direction (parallel); low `cl` = crossings/dispersion. Report **mean cl**, **crossing-fraction**
(cl<0.4), and **in-plane orientation entropy** for: reference tracks, our field streamlines, our
raw velocity samples. Hypothesis: reference low-cl / high-entropy (diverse); our field high-cl
(parallel). The **raw-samples** number is the key diagnostic — if our raw per-localization
velocities are already dispersed like the reference, the diversity *is in our data* and we threw
it away by averaging (→ fixable by a multi-vector field, not by needing more data). Split-half the
cl map so real crossings are separated from noise.

**(B) Distance-to-reference structure.** KS-distance between our and the reference's cl
distributions; agreement of the two in-plane orientation histograms; occupancy overlap (already
mb-ska: ~90% within 1 vox). Minimizing this distance (while keeping our extra channels) is a
concrete, optimizable render objective.

**(C) Sparsity / resolvedness.** Largely a *render* parameter (how many/thick streamlines we draw),
but report drawn-primitive count and occupied fraction so "sparse & resolved" is tunable, not
accidental. Honest counter-axis: our density buys **coverage** (mb-ska ~90% of reference vessels);
theirs buys **resolvedness** at the cost of the ~69% of detections their tracker discards.

The full scorecard: **diversity (A) · structural distance to reference (B) · sparsity (C) ·
coverage (mb-ska) · reproducibility (Dice/cos)** — the reference wins A & C; we win coverage; a
render that *beats* the blog must match A/C while keeping our coverage + extra validated channels.

## 4. Fix directions (to test, ranked)

1. **Multi-vector field at crossings** (preserve 2–3 flow modes per voxel instead of the mean) →
   tractography that crosses/branches → restores diversity at the root. The principled fix; directly
   targets cause #1. (bead mb-32d-adjacent / codex idea #3.)
2. **Render short coherent track-fragments** rather than long integrated streamlines — keeps local
   direction diversity, reads more track-like/sparse. Cheap.
3. **Sparser + thinner + lower-smoothing render** — fewer high-confidence seeds, less gap-filling,
   more transparency → resolved not filled. Render-parameter sweep against the (A)/(B) metrics.
4. **Recover the discarded detections** (mb-aro) for a denser-yet-resolved reconstruction; would
   benefit from the other 215 acqs' raw detections (re-requested upstream).

The evaluation in §3 is what makes 1–4 *optimizable* instead of eyeballed.

## 5. Results (measured 2026-07-09)

Per-voxel orientation coherence `cl` (1 = locally parallel; low = crossing/diverse), on the
shared coarsen-2 grid:

| source | mean cl (3D / xz) | crossing-frac (cl<0.4) | in-plane entropy | split-half cl r |
|---|---|---|---|---|
| **Aleph reference tracks** | 0.531 / 0.591 | 0.35 / 0.27 | 0.997 | 0.33 / 0.30 |
| **our RAW velocity samples** | **0.510 / 0.569** | 0.36 / 0.28 | 0.997 | 0.38 / 0.34 |
| **our graph-field streamlines** | **0.989 / 0.990** | **0.000 / 0.002** | 0.965 | — |

**Headline: our raw data is exactly as diverse as the reference (cl 0.51 vs 0.53, crossing-frac
0.36 vs 0.35) — our field-building averages it away to cl 0.99 (zero crossings).** We did not lack
the diversity; `build_field`'s per-voxel mean destroyed it. The diversity is **real structure**, not
noise: the crossing pattern reproduces across independent acquisition halves (raw cl r≈0.36, ref
r≈0.32 — modest but clearly non-zero). Global orientation *entropy* barely separates the three
(0.965–0.997) because curved parallel streamlines still span many global angles — the gap is
**local**, which is exactly what `cl` captures.

**Visual proof** (`renders/flow_diversity_ref_vs_ours.png`, hue = in-plane orientation): the
reference is finely *intermixed* hues everywhere (many directions coexisting locally); ours is large
*uniform* color blocks (each region combed to one direction). The picture is the metric.

**Implication for the fix:** the win is not more data — it's to STOP averaging. A **multi-vector
field** (cluster each voxel's velocity samples into 1–3 modes, tract through modes so streamlines
cross/branch) should pull our render's cl from 0.99 toward the reference's ~0.53. Target metric:
match the reference cl distribution (KS-distance → 0) while keeping our coverage (mb-ska ~90%) and
extra channels. Fragment-rendering the raw samples (which already sit at cl 0.51) is the quick
sparse-and-diverse demonstration.

Scripts: `scripts/wf_render_signal/flow_diversity.py` (metric), `render_flow_diversity.py` (figure).

## 6. Fix v1 — multi-vector field (measured 2026-07-09)

`scripts/wf_render_signal/tractography_multivector.py`: per voxel, greedy angular
clustering of the velocity samples into up to 3 flow MODES (samples pooled over an
in-plane neighbourhood to denoise); a crossing-preserving render primitive
(`mv_streamlets`) seeds one short bidirectional streamlet **per mode per voxel** — a
2-mode voxel emits two crossing streamlets, like dMRI multi-fibre glyphs.

Clustering finds real multi-modality: of ~14k occupied voxels, **26% carry ≥2 modes**
(3.1k two-mode, 0.5k three-mode). Result:

| reconstruction | mean cl | crossing-frac |
|---|---|---|
| single-vector field streamlines (before) | 0.99 | 0.00 |
| **multi-vector short streamlets (fix v1)** | **0.83** | 0.05 |
| Aleph reference (target) | 0.53 | 0.35 |

**Outcome: a real, visible improvement — cl 0.99 → 0.83, and the render
(`renders/multivector_vs_reference.png`) goes from uniform color blocks (combed) to
many intermixed diverse short primitives that recover the reference's fans/arcs/
crossings — but it does NOT fully close the gap (0.83 vs 0.53) and is still denser.**

Why the residual gap, honestly: our spatial mode-clustering has to *denoise*, and
denoising merges near-crossings (the second mode is often a minority, so few streamlets
follow it → the voxel stays coherent). The reference reaches cl 0.53 by *temporal*
denoising — linking each bubble into a track over frames — which preserves diversity
because a track follows one bubble; we can't track well at our concentration (the whole
reason we went field-based). So there's a genuine tension: **long+reproducible (our
field-tractography science claim) vs short+diverse (the reference-like render)** — they
are different objects optimizing different axes.

Paths to close further (next): (a) looser/ODF-style multi-modal estimation (more modes,
weighted proportionally) — risks noise, gate by split-half; (b) sparser+thinner render to
match reference density; (c) revisit temporal tracking on the recovered dense detections
(mb-aro) — the reference's actual advantage. The `cl`-vs-reference distance in §3 is now
the objective to optimize any of these against.

## 7. Resolution — three experiments, gap closed (measured 2026-07-09)

Ran three experiments in parallel against the §3 objective (plus a cross-method overlap
analysis). Decision doc: `docs/reconstruction-scorecard.md`. Headline: **path (a) works** —
an ODF-style multi-peak field closes the gap, and the recovered crossings are real.

| reconstruction | mean `cl` | crossing-frac | KS to ref | split-half | verdict |
|---|---|---|---|---|---|
| single-vector field (combed) | 0.99 | 0.00 | large | — | reproducible, not diverse |
| multi-vector fix v1 (§6) | 0.83 | 0.05 | 0.488 | — | partial |
| **ODF spatial fix** | **0.565** | **0.273** | **0.080** | `cl r=0.324` (real) | **closes gap (89%)** |
| temporal tracking (acq-0) | 0.33 | 0.71 | — | — | viable, under-supplied |
| Aleph reference (target) | 0.531 | 0.349 | 0 | `cl r≈0.32` | the ruler |

1. **ODF spatial fix (`scripts/wf_render_signal/odf_fix.py`).** Estimate a per-voxel orientation
   distribution and keep *every split-half-reproduced peak* (not the mean); seed streamlines
   proportional to peak dispersion. `cl` 0.83→**0.565** (reference 0.531), KS-distance
   0.488→**0.080** (89% of the distributional gap). The **1,129 extra crossing/branch peaks
   reproduce split-half** (`cl r=0.324` vs reference ~0.32) — the crossings are structure, not
   denoising artifacts. This is the fix: stop averaging, keep the *reproducible* modes. Because it
   decomposes the same occupied voxels, it keeps the field's coverage (~90%); its measured
   **occupancy split-half Dice is 0.65** (field 0.71 — adding diversity costs ~8%, still strongly
   reproducible vs an isotropic control's 0.07). So it is diverse **and** defensible.
   `renders/odf_vs_reference.png`.

2. **Overlap / consensus (`overlap_consensus.py`).** Cross-method validation: densely rasterize
   both reconstructions on the shared grid. The **shared** set (10.5k vox, 64% of the reference)
   is where both methods independently agree; the fair per-voxel test shows agreement flags real
   high-confidence voxels (shared reproduces 0.69 vs 0.44 for ref-only, for the reference). It
   also attributes the two methods cleanly: **ref-only = diverse fine detail** (cl 0.61, what the
   ODF fix recovers) vs **ours-only = extra coverage but combed** (cl 0.96). Consensus is a
   *validation/attribution* tool — its Dice (0.48) does not beat the field (0.71), so it is not
   itself the answer. `renders/overlap_consensus.png`.

3. **Temporal tracking (`temporal_tracking.py`).** Refutes the §6 assertion that we "can't track
   at our concentration": on acq-0, each detection has **0.00 competitors within one linking gate**
   → linking is unambiguous. Tracks are diverse (`cl 0.33`, partly tracker noise), linking 26% of
   detections (reference discards a comparable ~69%). But one acquisition gives only 362 short
   tracks — reference density/length need the pooled 216 acqs. **Viable, under-supplied** — the
   strongest argument for re-requesting the other 215 acqs' raw detections (issue #2).
   `renders/tracking_vs_reference.png`.

**Net:** the two-deliverable hedge from §6 is retired. Ship **one object — the ODF multi-peak
field**, now validated as diverse *and* reproducible (`cl 0.565`/KS 0.080; diversity split-half
`r 0.324`; occupancy split-half Dice 0.65 vs the combed field's 0.71 — the measured cost of
diversity), with the single-vector graph-field as the conservative core it collapses to, and
temporal tracking as the data-gated upside. See the scorecard for the full decision.
