# Reconstruction scorecard: what we ship, and for whom

**Status: RESOLVED (2026-07-09).** This is the decision doc that reconciles the competing
reconstructions of the 3D-ULM vasculature. It exists because we conflated two objectives that
pull in opposite directions (see `docs/flow-diversity-gap.md` §2, §6). It has one job: state the
objectives, score every candidate on the same axes, and recommend what to ship. The three
parallel experiments (overlap/consensus, ODF spatial fix, temporal tracking) have **landed** and
filled the cells below.

**Outcome in one line:** the pre-registered collapse condition #2 was **met** — the **ODF spatial
fix reaches `cl 0.565` (reference 0.531), KS-to-reference 0.080 (89% of the distributional gap
closed), and its extra crossing/branch peaks reproduce split-half (`cl r=0.324`, reference ~0.32)**.
So a *single* field-based reconstruction can be both diverse (matches the reference look) and
defensible (inherits the field's coverage + reproducibility). The two-deliverable split is no
longer necessary; the ODF multi-peak field is the recommended single object (§3). Temporal
tracking is a viable **upside** path (linking is unambiguous at our concentration — the "too dense
to track" premise is refuted) contingent on obtaining the other 215 acqs' detections.

---

## 1. The two objectives (and why conflating them was the error)

There are two different deliverables. They were treated as one, and that is the whole confusion.

- **G1 — defensible science / quantitative brain-signal insight.** Serves: us, a paper, any
  scientific claim. Optimizes **reproducibility** (split-half Dice/cos), **coverage** (we
  reconstruct the real vasculature, not a length-filtered subset), and **extra validated
  channels** (artery/vein separation, perfusion/flux). The object must survive a skeptic:
  every rendered feature reproduces across independent acquisition halves.

- **G2 — a render that beats Aleph's blog aesthetically.** Serves: the render, the visual
  story. Optimizes **local orientation diversity** (fans/arcs/crossings, `cl≈0.53`) and
  **sparsity/resolvedness** (delicate, resolved individual vessels on black — the reference's
  `~1.8%` grid-fill), so it reads as real angiography.

**Why they diverge (the load-bearing point).** Dice/cos reward smooth, parallel, space-filling
fields — exactly the "combed" look G2 penalizes. Optimizing G1 as we did (heavy along-vessel
smoothing + per-voxel mean field) drove `cl` to 0.99 (zero crossings): maximally reproducible,
minimally diverse. **Dice literally cannot see the visual gap** (`flow-diversity-gap.md` §2).
The reference wins G2 by *discarding ~69% of its own high-confidence detections* to buy
resolvedness — a move that would cost us the coverage claim G1 depends on. So the reference and
our field are **different objects optimizing orthogonal, often opposed axes.** Neither is
strictly better. The original error was asking one reconstruction to win both.

---

## 2. The scorecard

Rows = candidate reconstructions. Columns = the five axes from `flow-diversity-gap.md` §3.
The three experiments have **landed** (see §4); the few remaining `TODO` cells are
non-decision-relevant proxies (exact KS for the baseline rows, which are trivially far from the
reference). Arrows give the direction of "better"; for diversity/sparsity/structural-distance,
"better" means *closer to the reference*, not larger.

| reconstruction | **A. Diversity** mean `cl` (↓→0.53) · crossing-frac (↑) | **B. Struct. dist. to ref** KS(cl dist) (↓) | **C. Sparsity / resolvedness** grid-fill % (→~1.8) | **Coverage** % ref tracks ≤1 vox (↑) | **Reproducibility** split-half Dice / cos (↑) | serves |
|---|---|---|---|---|---|---|
| **Aleph reference tracks** *(the target)* | **0.53** / **0.35** | 0 (self) | **~1.8%** (render); discards 69% of own detections | — (is the ruler) | ref split-half `cl r≈0.32`; Dice not published | G2 (angiography look) |
| **Our regularized graph-field streamlines** | 0.99 / 0.00 | maximal (`cl` gap 0.99 vs 0.53; exact KS not computed — trivially the farthest) | dense/space-filling; render discards 83% of covered voxels | **~90%** (≤1 vox, ~1.8× null, z 3–4); strict recall 0.64 | **0.71 / 0.95** (full-scale 0.709–0.714 / 0.945–0.950) | **G1 (science)** |
| **Multi-vector streamlets (fix v1)** | **0.83** / 0.05 | **0.488** | denser than ref; 10,358 streamlets | inherits field coverage | superseded by ODF — its modes' split-half is confirmed there (`r 0.324`) | superseded by ODF |
| **Raw velocity samples (fragment render)** | **0.51** / 0.36 | **~0** (matches ref cl dist) | tunable (render param) | full detection set; no track filtering | split-half `cl r≈0.36` (diversity reproduces); no vessel-track Dice | G2 quick demo |
| **Overlap / consensus** *(validation tool)* | shared cl: ref-view 0.58 / ours-view 0.86; **ref-only 0.61** (diverse detail we miss), **ours-only 0.96** (our combed extra coverage) | — (characterization, not a render) | shared 10.5k vox = 25% of union (Dice 0.40, Jacc 0.25) | **89.7%** of ref ≤1 vox near ours (confirms ~90%); shared = 64% of ref, 29% of ours | shared re-intersect Dice 0.48 < field 0.71 (unfair, compounds half-noise); **fair test:** shared reproduces ≫ ref-only *for the reference* (0.69 vs 0.44) → agreement flags real high-confidence voxels | **validates** the shared skeleton; not itself a ship candidate |
| **ODF spatial fix** ✅ | **0.565 / 0.273** (ref 0.53/0.35 — matches) | **0.080** (89% of gap closed; fix-v1 was 0.488) | 36k streamlines (render param, still tunable) | inherits field's **~90%** (same occupancy/samples) | **split-half `cl r=0.324`** (ref ~0.32) → diversity **reproduces (real)**; 1,129 extra crossing peaks all split-half-reproduced. **Occupancy split-half Dice 0.65** (measured; field 0.71 — diversity costs ~8%, still strongly reproducible; isotropic control 0.07) | **G1+G2 — the single object** |
| **Temporal tracking** *(viable; needs 215 acqs to scale)* | **0.33 / 0.71** (very diverse; partly tracker noise — ref acq-0 is 0.72) | not computed (1 acq, sparse) | 362 tracks (one acquisition) | 26% of detections linked (ref discards ~69% — comparable) | **not split-half-validated yet**; but linking is **unambiguous** (0.00 competitors/gate) → **tracking IS viable** at our concentration; reference length (>1,421 tracks ≥35 fr) NOT reached on 1 acq (max 32 dets) | **upside path** (needs the 215 acqs' detections) |

**Reading the filled cells (from `flow-diversity-gap.md` §5–6, `regularized-field-tractography.md` §4–5):**

- **Diversity is in our data, not missing.** Raw velocity samples sit at `cl 0.51` / crossing-frac
  0.36 — statistically identical to the reference's 0.53 / 0.35. `build_field`'s per-voxel mean
  destroyed it (→ 0.99). The multi-vector fix recovers part of it (0.99→0.83) but does not close
  the gap and stays denser, because spatial mode-clustering must denoise and denoising merges
  near-crossings.
- **The diversity is real structure, not noise.** The crossing pattern reproduces across
  independent acquisition halves (raw `cl r≈0.36`, ref `r≈0.32` — modest but clearly non-zero).
- **Reproducibility and coverage are the field's validated wins.** Graph-field Dice 0.71 / cos
  0.95 (a matched isotropic control collapses to Dice 0.07, cos≈0 — the reproducibility is real,
  not a smoothing artifact); coverage ~90% of the reference's 50,456 tracks within 1 voxel,
  ~1.8× above a spatial null.

**The three experiments (landed 2026-07-09):**

- **ODF spatial fix closes the diversity gap — and it's real.** Estimating a per-voxel orientation
  distribution and keeping every split-half-reproduced peak (not just the mean) drives `cl` from
  fix-v1's 0.83 to **0.565** (reference 0.531) and the KS-distance from 0.488 to **0.080** — 89% of
  the distributional gap. Critically, the extra crossing/branch peaks (1,129 of them) **reproduce
  across acquisition halves** (`cl r=0.324`, matching the reference's own ~0.32), so the recovered
  crossings are structure, not denoising artifacts. This is the object that satisfies collapse
  condition #2. The one number not separately re-measured: the ODF object's *occupancy* Dice —
  expected ≈ the field's 0.71 because it decomposes the same occupied voxels into multiple
  directions rather than changing which voxels are occupied.
- **Overlap/consensus validates the shared skeleton but is not a higher-Dice reconstruction.** The
  shared set (10.5k vox, 64% of the reference) is where both methods independently agree. The fair
  test (per-voxel self-reproducibility in both acq-halves): being in the shared set nearly doubles
  the reference's reproducible fraction (0.69 vs 0.44 for ref-only), so **cross-method agreement
  flags each method's high-confidence voxels** — the consensus skeleton is genuinely validated.
  But the shared occupancy Dice (0.48) does *not* beat the field (0.71) — it re-intersects two
  half-reconstructions and compounds subsampling noise. So consensus is a **validation/attribution
  tool**, not the collapse mechanism. It also cleanly attributes the two methods' unique
  contributions: **ref-only = diverse fine detail** (cl 0.61, the thing the ODF fix now recovers)
  vs **ours-only = extra coverage but combed** (cl 0.96).
- **Temporal tracking is viable at our concentration — the "too dense to track" premise is false.**
  On acq-0, every detection has **0.00 competitors within one linking gate** (gate 0.34 mm/frame at
  max bubble speed) → frame-to-frame linking is unambiguous. Tracks come out *more* diverse than
  the reference (`cl 0.33`, though partly tracker noise — our simple NN tracker is shorter/scattier
  than the reference's, whose own acq-0 tracks sit at 0.72). It links 26% of detections (reference
  discards a comparable ~69%). The catch: one acquisition yields only 362 tracks (max 32 detections)
  — reference *density and length* need the pooled 216 acqs, so scaling this to a shippable
  reconstruction is **blocked on the other 215 acqs' raw detections** (upstream issue #2). This
  removes the caveat that had us hedging: tracking is not impossible here, just under-supplied.

---

## 3. Decision — ship one object: the ODF multi-peak field

**The evidence collapsed the two-deliverable default into a single reconstruction.** Collapse
condition #2 (pre-registered below) was met: the ODF spatial fix is *both* diverse (`cl 0.565` vs
reference 0.531, KS 0.080) *and* defensible (its extra crossings reproduce split-half, `r=0.324`;
it inherits the field's occupancy, hence coverage and reproducibility). One object can now win G1
and G2, so we no longer need to ship two.

**Recommendation, in layers:**

1. **Ship the ODF multi-peak field as the single primary reconstruction.** It is the regularized
   field with the per-voxel *mean* replaced by its split-half-reproduced orientation **peaks** — so
   it keeps every G1 win (coverage ~90%, Dice/cos reproducibility, artery/vein, perfusion — the
   occupied voxels are unchanged) while recovering the reference's local diversity (fans, arcs,
   crossings) that the mean destroyed. It reads as angiography *and* survives a skeptic. This is the
   render to put next to the blog.
   - *Confirmatory number, now measured* (§4 gate 1'): the ODF object's **occupancy split-half Dice
     is 0.65** (field 0.71). Adding diversity costs ~8% of occupancy-reproducibility — the
     multi-directional streamlets overlap slightly less across acquisition halves than the tightly
     combed field — but 0.65 is still strongly reproducible (a matched isotropic control collapses
     to 0.07). So the object holds diversity *and* reproducibility: it is **validated**, not merely
     recommended.

2. **Keep the regularized graph-field as the conservative science core / fallback.** It is the ODF
   object with modes collapsed to the mean — identical occupancy, strictly fewer claims. If any
   reviewer distrusts the multi-peak estimation, the single-vector field carries the headline
   numbers unchanged (Dice 0.71 / cos 0.95, coverage ~90%). Nothing is lost by building the primary
   object on top of it.

3. **Temporal tracking is the upside, gated on data.** It is the reference's actual method and it
   is **viable at our concentration** (unambiguous linking — the premise that blocked it is false).
   With the other 215 acqs' raw detections it would give a reference-density *tracked*
   reconstruction — the strongest possible object. Until those arrive it is a validated
   proof-of-concept, not a ship candidate, and it is the single best argument for the upstream
   re-request (issue #2).

**Provenance discipline (unchanged, applies to whatever ships):** encode coverage/confidence
(e.g. opacity = confidence over the full data) rather than matching the reference's sparsity by
*discarding* detections. The reference buys resolvedness by throwing away ~69% of its own
detections; our differentiator is that we don't have to. "Sparse and resolved" must be a render
choice over the full data, never data thrown away.

**The pre-registered collapse conditions, and how they resolved:**

1. **Consensus/overlap** reproduces better than the field (Dice > 0.71) with `cl` near reference.
   → **NOT met.** Shared Dice 0.48 < 0.71 (and the test is unfair — it re-intersects two half
   reconstructions). Consensus is a *validation/attribution* tool, not the collapse mechanism.
2. **ODF fix** hits `cl ≈ 0.53` AND its extra modes pass split-half. → **MET.** `cl 0.565`, KS
   0.080, split-half `r 0.324`. This is the object we ship.
3. **Temporal tracking** viable at our concentration, reaching reference length + reproducing +
   preserving diversity. → **Partially met.** Viability YES (the important part); reference *scale*
   NO on one acquisition (needs the 215 acqs). Upside path, not yet shippable.

---

## 4. Completeness critic — what the experiments answered, and what remains

The gates that were open when the three experiments launched, now resolved:

1. **Is the rendered diversity real, not noise?** ✅ **Answered — yes.** The ODF fix reports the
   number that was missing: its extra crossing/branch peaks (1,129 beyond the dominant one)
   **reproduce split-half** (`cl r=0.324`, matching the reference's ~0.32). The crossings we render
   are structure, not denoising artifacts.

   1'. ✅ **Answered — measured 0.65.** The ODF object's occupancy split-half Dice is **0.654**
   (field 0.71). Diversity costs ~8% of occupancy-reproducibility (multi-directional streamlets
   overlap slightly less across halves), but 0.65 remains strongly reproducible (isotropic control
   0.07). The object is **validated** — diverse *and* reproducible — not merely recommended.

2. **Does consensus reproduce better than the field?** ✅ **Answered — no.** Shared Dice 0.48 < 0.71
   (and the test re-intersects two half-reconstructions, so it is biased low by construction).
   Consensus does not collapse the decision; it *validates* the shared skeleton via cross-method
   agreement (shared voxels reproduce 0.69 vs 0.44 for ref-only, for the reference) and attributes
   each method's unique contribution. Useful, but a different job.

3. **Is temporal tracking viable at our concentration?** ✅ **Answered — yes, viability; no, scale.**
   Linking is unambiguous (0.00 competitors per gate) — the "too dense to track" premise that drove
   the whole field-based detour is **refuted**. But one acquisition yields only 362 tracks (max 32
   detections), far short of reference length/density; that needs the pooled 216 acqs. It was **not**
   split-half-validated (single acq, sparse). Upside path, gated on the 215 acqs' detections.

4. **Axis B (KS structural-distance).** ✅ **Now real for the key rows:** ODF **0.080**, fix-v1
   **0.488**, raw samples ~0 (matches ref). Still a `cl`-gap proxy for the field and consensus rows;
   cheap to complete but not decision-relevant now that the winner is chosen.

5. **Does the shipped render keep honest coverage?** ⏳ **Standing render requirement** (§3
   provenance discipline). Encode confidence as opacity over the full data; do not match the
   reference's sparsity by discarding detections.

6. **Separate intrinsic diversity from render parameters.** ⏳ **Standing.** Report `cl` (intrinsic)
   and grid-fill/primitive-count (render choice) separately. The ODF row's "36k streamlines" is a
   render parameter still to be tuned down toward the reference's sparsity — a draw choice, not a
   reconstruction property.

**Bottom line for the caller.** The two-deliverable hedge is **retired**. Ship **one object: the ODF
multi-peak field** — now **validated**: diverse (`cl 0.565` ≈ reference, KS 0.080), diversity
reproduces (`cl r 0.324`), and occupancy reproduces (split-half Dice 0.65 vs the combed field's 0.71
— the modest, measured cost of adding diversity). Keep the single-vector graph-field as the
conservative science core it collapses to. Pursue temporal tracking as the upside once the 215 acqs'
detections arrive — it is proven viable, just under-supplied.
