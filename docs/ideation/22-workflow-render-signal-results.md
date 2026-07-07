# Workflow results — render + brain signal (2-round multi-model), honest scorecard

**Date:** 2026-07-07. Workflow `ulm-render-signal` (11 agents: opus + sonnet + 2× fable + codex
gpt-5.5, 2 iterate rounds, ~908k tokens). Design→synthesize→[implement→evaluate]×2, scored on
GT-free yardsticks vs the Aleph reference. The evaluator (opus) was adversarial and caught every
inflated headline; numbers below are the **honest, re-derived** ones. Epic mb-k25.

## The real win: the brain SIGNAL (validated, honest)
- **Velocity-direction field** — split-half cosine **0.96 (n≥10) / 0.92 (n≥5)** on a *true
  odd-vs-even acquisition split*, vs ~0.006 random-direction null. Independently reproduced by
  the evaluator (0.96/0.91). A genuine, held-out-acquisition-reproducible cerebral flow field.
- **Artery/vein separation** — lateral-gated antiparallel vessel pairs (687 artery / 608 vein,
  1,295 voxels), direction split-half cosine **0.917**. Real A/V signal.
- **Motion-confirmed MCF tracker** (codex-built: two-stage Zhang-2008 successive-shortest-paths
  with anti-teleport physical gate + Kalman tracklets) — **gate-clean** (0 box violations, max
  Euclidean step 0.758 ≤ 0.761 ceiling) and **beats the reference tracker per-acq on acq-0**:
  8 vs 6 tracks ≥35, 4 vs 2 ≥50, 31.5% vs 30.9% detection recovery. Frame-scramble null: L≥20
  and L≥35 FDR = **0.000** (scrambling frames yields zero long tracks — the tracks are real).
- **Downstream hemodynamics** (measured, physiological): wall-shear-stress 0.3–0.8 Pa; capillary
  transit heterogeneity correct-sign at ~2 mm; tortuosity clean right-skew.
- **Honest negative:** pulsatility/PWV deprioritized — the 240-frame (1.08 s) acquisition windows
  give only 0.93 Hz frequency resolution and the cardiac band is contaminated by acq-boundary
  harmonics (fs/240 = 0.9268 Hz) + a per-acq warm-up transient. No clean whole-brain cardiac
  clock from the pickle; true self-gated 4D needs dense per-frame localizations (upstream re-run).

## The RENDER: honest, teleport-free, but not yet beating the blog
- The "Living Angiogram" viewer ships: drizzle density (all 1.77M localizations, not the blog's
  filtered subset), flow-direction color wheel, depth cueing, trust/confidence scaffolding,
  data-true legend + mm cube + gnomon. Teleport-kill is real: **74.4 → 16.9 cm/s** max speed
  (the elev_gate_factor=3 clutter is gone). The flow-direction z-buffer/color bug was fixed.
- **But it does NOT beat the reference on any hard yardstick, and it doesn't yet look better:**
  - ≥35-frame tracks **170 vs reference 1,421** (worse even than the raw composite's 242 — the
    post-hoc gate-fix *splits* tracks rather than re-tracking).
  - ref-recall **46%** exact-voxel (94–97% at 1-voxel tolerance) — below the 95% target.
  - Coverage headline "21.8×" was inflated (wrong denominator). **Honest Coverage@equal-FRC =
    3.9×** vs the reference's *full-localization* coverage — and that 3.9× is mostly a
    **frame-count artifact** (we process ~3× more frames), not an algorithmic gain. The velocity
    field's honest coverage is **0.92×** (a non-win).
  - Visually the r1 render is a dim, sparse point spray (cyclic hue doesn't survive channel-wise
    MAX blending at this depth complexity). The r2 render improved instrumentation but the
    side-by-side-vs-blog frame was flagged not-done.

## The one lever that unblocks everything: dataset-wide MCF
Both deliverables are capped by the same blocker. The MCF tracker already **beats the reference
per-acquisition** — but the released pickle contains detections for **acq 0 only**
(`detections['acq_indices']` unique == [0]). Running MCF across all 216 acqs is the ONLY measured
path to (a) beat the reference's 1,421 ≥35-frame tracks, (b) lift ref-recall toward 95%, and (c)
feed the render real long, gate-clean tracks instead of the split composite. Two ways to unblock:
1. **Ask upstream** to release the per-acq detections for all 216 acqs (they already gave acq-0
   in the pickle) — free, added to the issue-#2 reply draft.
2. **Re-run the detector on Modal** over all 216 acqs (needs Rome's spend approval), then MCF.

Beads filed: **mb-4yw**, **mb-65b** (dataset-wide MCF unblock). Epic **mb-k25**.

## Verdict
- **Signal > Render** on defensibility. The velocity-direction field + A/V separation is a
  genuine, reproducible, GT-free-validated scientific result — the "insightful brain signal" the
  brief asked for, and it comes with a (dense) render.
- The **render is honest but not yet exceptional**; making it beat the blog needs the dataset-wide
  motion-confirmed tracks (the blocker above), then wiring the *validated* direction field +
  confidence into the render (instead of the unvalidated composite hue).
- Process win: the multi-model loop with an adversarial evaluator **prevented overclaiming** —
  every inflated coverage headline was caught and corrected. Codex/gpt-5.5 was the algorithm
  workhorse (the MCF tracker).

Artifacts (gitignored, local): `outputs/reference/wf/r{1,2}/{render,signal}/`. Durable code +
workflow preserved under `scripts/workflows/` and `scripts/wf_render_signal/`.
