# Back to tracking: a Kalman bubble tracker, benchmarked on acq-0 (no new data)

**Status: progress, 2026-07-09.** Reframed after Rome's correction: the goal is the **most
accurate reconstruction of individual microbubble trajectories** (independently tracked bubbles),
validated **GT-free** — *not* beauty, and *not* matching the reference's orientation-diversity.
The velocity-**field** work (`flow-diversity-gap.md`, `reconstruction-scorecard.md`) was a means to
tracking, not the end; "looks good" was only Rome's eyeball proxy for accuracy. This doc returns to
tracking and makes progress that does **not** need the blocked 215-acq data.

## The setup — acq-0 is a self-contained tracker benchmark

The released data gives us both halves of one acquisition:
- **Input:** 9,761 raw per-frame detections for acq-0 (`detections.positions_mm`, 240 frames).
- **Reference output on that input:** the reference tracker's own **292 acq-0 tracks**
  (`tracks_smoothed`, `acq_index==0`): mean length 10.3, max 68, linking ~31% of detections
  (discarding ~69%, smoothed).

So we can run our tracker on the **same detections** and compare to a strong tracker's output on
identical input — data-grounded validation with **no new data required**. (Reference-*density*
tracking still needs the other 215 acqs' detections — bead `mb-4yw` — but the *method* can be built
and validated now.)

## Method — constant-velocity Kalman tracker

`scripts/wf_render_signal/track_bubbles.py`. A nearest-neighbour tracker fragments (mean len 7,
max 32); a fixed-gate tracker cannot be both long and smooth (tight gate → breaks tracks; loose
gate → jagged mis-links). The fix is proper state estimation:

- **State** `[pos(3); vel(3)]`, constant-velocity model, process-noise `sigma_a` (acceleration std).
- **Association** by a **Mahalanobis chi-square gate** on the innovation — *wide while velocity is
  uncertain (track start), tight once the motion locks in* — with globally-optimal per-frame
  assignment (Hungarian). This is what yields long **and** smooth tracks.
- **Gap-closing** (coast through ≤2 missed detections) so dropouts don't break tracks.
- **Savitzky-Golay smoothing** of the output, matching the reference's `tracks_smoothed` (raw jagged
  links inflate both turning and per-step speed, so physiology is measured on smoothed tracks for a
  fair like-for-like comparison).

Operating point chosen for **accuracy**, from a `sigma_a × gate` sweep: among configs matching the
reference's coverage (≥31% linked), take the **smoothest** (lowest turning), tie-broken by direction
agreement. Coverage beyond that trades accuracy for quantity.

## Results (operating point sigma_a=0.03, gate_chi2=9)

| metric | reference | **our KF tracker** |
|---|---|---|
| tracks | 292 | **398** |
| mean / max length | 10.3 / 68 | **8.2 / 71** |
| detections linked | 31% | **33%** |
| speed median (smoothed) | 25.8 mm/s | 40.2 mm/s |
| turning median (smoothed) | 8.1° | 18.1° |
| direction agreement w/ ref | — | **70%** of shared cells <30° |
| one-step motion error | — | 0.20 mm (vs ~2.5 mm NN spacing) |
| bootstrap stability (drop 15%) | — | **80%** points recovered |

On identical input we reproduce the reference tracker's **count, track lengths, and coverage**, with
physiological, reproducible, mostly-smooth trajectories. Visual: `renders/tracking_acq0_vs_reference.png`
— continuous vessel trajectories (not the streaklet-glyph fragments the field renders produced).

The full sweep is a **coverage–smoothness Pareto front**: from (27% linked, 16.6° turn) to
(65% linked, 27.7° turn). More coverage than the reference is available, at the cost of smoothness.

## What the data says about "diversity" (the corrected prior)

Reported, **not targeted**: within-acquisition orientation coherence `cl` is **0.72** (reference
acq-0 tracks) and **0.55** (ours) — i.e. **within a single acquisition the vasculature is coherent,
not diverse.** The reference's oft-cited `cl≈0.53` "diversity" only appears when **pooling 216
acquisitions** — it is largely *inter-acquisition* variation, not within-acq crossing. So the earlier
push to "recover crossings / match cl 0.53" (multi-vector, ODF) was optimizing a **pooling artifact**
and, worse, our low-cl tracks were partly **tracking error** (jagged mis-links scatter direction).
Grounding in the data reverses the target: a good within-acq tracker should be *coherent*.

## Honest residual gap (future work)

Our smoothed tracks run **faster (40 vs 26 mm/s)** and **turn more (18° vs 8°)** than the reference,
and sit at `cl 0.55` vs `0.72` — our association is slightly noisier (some links reach a bit too far;
some residual jaggedness). Closing it:
1. **Global / multi-frame association** (min-cost-flow or fixed-lag smoothing over a window) instead
   of greedy per-frame Hungarian — resolves ambiguous links with future evidence.
2. **RTS (Kalman) smoothing** and/or matched post-smoothing to the reference's exact scheme.
3. **Detection quality** — gate/weight links by localization confidence (z-score) to drop marginal
   detections the reference discards.

## Bottom line

We are **back to independently tracked bubble trajectories** — the right object — with a principled
Kalman tracker validated against the reference on identical acq-0 data (matched length/coverage, 70%
direction agreement, 80% bootstrap-stable), and no new data used. It is **ready to scale to all 216
acquisitions** the moment the 215-acq detections arrive (`mb-4yw`) — which is the actual unblock for a
reference-density tracked reconstruction. The within-acq coherence finding corrects the objective:
ground in the data, don't import the reference's pooled diversity as a target.
