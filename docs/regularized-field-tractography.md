# Regularized-field tractography: a validated improvement on 3D-ULM vessel reconstruction

**Result in one line.** Reconstruct cerebral vasculature from ultrasound-localization-microscopy
(ULM) data by integrating streamlines through a *regularized* microbubble-velocity field, instead
of linking individual bubble detections into tracks. This produces long (~50 mm), coherent,
concentration-robust vessels that reproduce across independent halves of the data (occupancy
Dice 0.60, direction cosine 0.85) — matching the reference pipeline's vessel length while
sidestepping the tracking failure that caps detection-based tracks at ~17 frames on densely
seeded data, and it renders as continuous velocity-colored flow rather than a discrete point
cloud.

Author: microbubbles/crew (Gas Town). Date: 2026-07-08. Data: the public Aleph Neuro
sanitized ultratrace (acq-0 slice used throughout). Reference method: `alephneuro/microbubbles`.

---

## 1. Background and goal

3D ULM images vasculature below the ultrasound diffraction limit by localizing individual
microbubble contrast agents over many frames and accumulating their positions. Aleph Neuro's
open pipeline is the reference: **beamform (5-angle plane-wave compounding) → temporal-SVD clutter
filter → 3D peak-detect + sub-voxel localize → Kalman + Hungarian track linking → render**. Its
published render is a beautiful jet-colored point cloud of ~2,300 long tracks (median track only
7 frames; the render is an aggressively length-filtered subset).

**Goal:** a measured, validated improvement on that method — not a re-implementation, a better
result on the same fixed data (216 acquisitions × 240 frames @ 222.43 Hz; no ground truth).

---

## 2. The problem, established by measurement

Our fork's detection→tracking produced **max 17-frame tracks**, versus the reference's max 143.
We root-caused this empirically, and the honest path matters because two attractive hypotheses
were falsified:

- **Detection quality is not the cause.** Our per-frame detection density is *lower* than the
  reference's (32 vs 41 detections/frame); frame-to-frame persistence is *higher* (46.6% vs
  37.1% of detections have an in-gate neighbor next frame).
- **Detection multiplicity ("one bubble → 2–3 detections") is REFUTED.** A direct intra-frame
  nearest-neighbor test found **zero** same-frame sub-PSF detection pairs, for both our data and
  the reference — our detector already applies non-maximum suppression. An elegant multi-field
  convergence (PET MLEM sensitivity-normalization, determinantal point processes, Neyman–Scott
  cluster deconvolution, tomographic-PIV ghost suppression, sheaf consistency) had all pointed
  at a principled de-dup fix; the measurement showed there was nothing to de-dup.
- **The real signal: concentration/association ambiguity.** Our detections are ~2.4× more
  tightly packed in-frame (median in-frame neighbor 1.99 vs the reference's 4.79 gate-units).
  Densely-packed bubbles give any frame-to-frame tracker ambiguous choices → identity swaps →
  short tracks. The reference's 240-frame `selected_sequence` window is, in effect, a
  lower-concentration regime where tracking is unambiguous. This is a *tracking-paradigm* limit,
  not a detector defect.

---

## 3. The method: tractography through a regularized velocity field

If dense bubbles make *tracking* ambiguous, stop tracking. We already hold a **validated
velocity-direction field**: pooling all localizations' frame-to-frame displacements into a
per-voxel mean flow direction gives a field whose direction reproduces at split-half cosine
**0.96** across independent acquisition halves. Build vessels by integrating streamlines through
that field (the diffusion-MRI tractography idea), never linking a single noisy detection —
immune to the concentration/ambiguity that breaks tracking.

**The essential lever — field regularization.** Naive tractography on the raw field is
coverage-limited: streamlines break wherever the field goes unconfident, capping at ~17 mm (no
better than tracking). The validated *direction* is confident only over short segments; the
confident *connectivity* is short-range. Confidence-weighted Gaussian smoothing of the field
fills those gaps so streamlines integrate across them:

```
V_smooth = gaussian(V · C, σ) / gaussian(C, σ)     # C = per-voxel confidence (count)
```

integrate `dr/ds = v̂(r)` (RK, anisotropic mm geometry), terminating on low confidence, sharp
direction reversal (>~72°), or domain exit.

| field | streamline median | max | ≥10 mm | ≥20 mm | ≥40 mm |
|---|---|---|---|---|---|
| raw | 2.2 mm | 16.9 mm | 9% | 0% | 0% |
| smoothed σ≈1.5 | 9.0 mm | 70.6 mm | 46% | 16% | 3% |
| smoothed σ≈2.5 | 14.3 mm | ~50 mm | 67% | 31% | 8% |

Method: `scripts/wf_render_signal/tractography_field.py`.

---

## 4. Validation (GT-free)

There is no ground truth, so we validate by **reproducibility across independent halves of the
data** — the test that separates real structure from a smoothing artifact. Build and integrate
the field on odd-parity acquisitions and, independently, on even-parity acquisitions; compare
the two streamline maps:

| smoothing σ | occupancy Dice(odd,even) | direction cosine (shared) | max vessel |
|---|---|---|---|
| (1.2,0.7,1.2) | 0.47 | 0.73 | 36 mm |
| (1.8,1.0,1.8) | 0.55 | 0.79 | 53 mm |
| **(2.5,1.4,2.5)** | **0.60** | **0.85** | ~53 mm |

Two independent halves agree on 60% of occupied voxels and point the same way (cos 0.85), and
the long vessels persist. **The ~50 mm vessels are reproducible structure, not smoothing
artifacts.** The σ knob trades vessel length vs. detail and is chosen to *maximize
reproducibility*, not length (over-smoothing inflates length without reproducibility — the
σ2.5 "222 mm" streamline outlier is such an artifact and is excluded).

---

## 5. The render

`renders/tractography_line_viewer.png` (viewer: `scripts/wf_render_signal/tractography_line_viewer.html`).
5,731 streamlines drawn as **connected, per-vertex jet-colored polylines** (flow speed in mm/s,
0–38 scale), pure-black background, near-coronal camera, calibrated colorbar + 10 mm scalebar,
depth-fog cueing. It reads as continuous flowing vasculature with resolved high-flow cores —
showing the coherent flow *structure* that the reference's discrete point cloud cannot.

---

## 6. Why this is an improvement on the reference method

- **Concentration-robust.** No frame-to-frame association, so it does not fail on dense data —
  the exact failure that capped tracking at 17 frames.
- **Long + coherent.** ~50 mm reproducible vessels vs the reference's 43 mm max track.
- **Defensibly validated.** A real split-half reproducibility number (Dice 0.60 / cos 0.85),
  not a proxy metric prone to inflation.
- **Richer output.** The same field carries velocity *direction* (the reference renders speed
  only), artery/vein separation (antiparallel-adjacency, split-half 0.917), and a perfusion/flux
  channel — readouts the reference lacks.

---

## 7. Honest caveats

- The render is denser than the reference (5,731 vs their filtered ~2,300 streamlines) — richer,
  but a stylistic choice to tune.
- Streamlines are flow-field integration lines (the coherent flow), not individual bubble
  trajectories — arguably *more* informative, but a different object; state it plainly.
- Dice 0.60 means ~40% of occupied voxels are half-specific; reproducibility is real but not
  perfect, and it depends on the smoothing σ.
- The imaged data is a single anatomical slice; the velocity field pools all 216
  acquisition-repeats of it (504,090 reference localizations). *(Correction 2026-07-08: an
  earlier draft framed this as an "acq-0 slice, dataset-wide not yet run" — the field is in fact
  built from all 216 acquisitions, confirmed at full scale in mb-9ay/§8.4.)*
- Gaussian smoothing is a crude regularizer — a physically-principled one (below) should do
  better. *(Done: mb-ki9, §8.1 — the along-vessel graph-Laplacian raises Dice 0.61→0.71.)*

---

## 8. Directions to go further (beaded, not yet started)

1. **PDE / graph-Laplacian field regularization** (replace Gaussian): smooth *along* vessels via
   a graph-Laplacian low-pass, and enforce incompressibility `∂_z v_z = −(∂_x v_x + ∂_y v_y)` to
   reconstruct the weak (synthesized) elevation velocity — longer, higher-reproducibility,
   physically-constrained vessels.
   **✓ DONE (2026-07-08, mb-ki9)** — the along-vessel graph-Laplacian regularizer raises
   split-half reproducibility from Dice 0.61 / cos 0.84 to **Dice 0.71 / cos 0.95** (+17% / +13%,
   seed-robust); a matched-diffusivity isotropic control collapses (Dice 0.07, cos ≈ 0), proving
   the gain is anisotropy, not more smoothing. Incompressibility was neutral on this coarse
   elevation axis (honest finding). See `docs/pde-field-regularization.md`;
   method `scripts/wf_render_signal/tractography_pde.py`.
2. **Composite render**: tractography vessels + flow-*direction* hue + artery/vein two-tone +
   perfusion/flux channel — structure *and* the science readouts the blog lacks.
   **✓ DONE (2026-07-08, mb-4k2)** — 4-panel science composite on the graph-regularized field:
   (A) structure+speed, (B) DTI-style flow-direction RGB, (C) validated antiparallel artery/vein
   pairs, (D) perfusion/flux throughput. Fleet-safe matplotlib (no chrome).
   `renders/composite_science.png`; method `scripts/wf_render_signal/composite_render.py`.
   (The polished *hero* aesthetic render remains mb-hi8.)
3. **Hero render capture + selection**: validate the untested ribbon-quad "living-angiogram"
   shader on a cool box; pick line-viewer vs hero; tune streamline density and (Line2) vessel
   width to match/beat the blog's crispness.
4. **Dataset-wide tractography** over all 216 acquisitions (field built at full scale) — and,
   if the maintainer releases the other 215 acqs' detections, an apples-to-apples comparison.
   **✓ RESOLVED (2026-07-08, mb-9ay)** — the velocity field already pools all 216 acquisitions
   (the reference pkl holds acq_index 0–215; the "acq-0 only / need the other 215" premise was
   mistaken). The graph result holds at *full scale* (all 453,634 velocity samples, no 250k
   subsample): Dice 0.709–0.714 / cos 0.945–0.950. `scripts/wf_render_signal/fullscale_validate.py`.
5. **Quantitative coverage-vs-reference** on a shared grid (the render's coverage claim vs the
   reference tracks), reported honestly against a random-split null.
   **✓ DONE (2026-07-08, mb-ska)** — our field-tractography covers ~90% of the reference's 50,456
   tracks (within 1 voxel), ~1.8× above a circular-shift spatial null (z 3–4): we reconstruct the
   reference vasculature, not noise. The mb-ki9 graph field has higher *strict* recall (0.64 vs
   0.60) but lower precision (fills beyond the reference's filtered set) — coverage matches; its
   validated win is on the reproducibility axis (§4). See `docs/pde-field-regularization.md` §3b;
   `scripts/wf_render_signal/coverage_vs_reference.py`.

Provenance / research trail: `docs/ideation/19–24` (frontier brainstorms, the reference-gap
root-cause, the multiplicity refutation, the tractography result). Beads: epic `mb-k25`.
