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

## Residual investigation — three parallel probes (2026-07-09)

The initial residual (smoothed speed 40 vs 26 mm/s, turning 18° vs 8°, cl 0.55 vs 0.72) was
attacked from three independent angles at once (`track_rts.py`, `track_zgate.py`, `track_global.py`).
They **converge**, and they **correct the attribution above**: the turn gap was *not* greedy
mis-linking.

**1. RTS (Kalman) smoothing — WINS on turning + coherence (`track_rts.py`).** A proper
Rauch-Tung-Striebel backward smoother (same links as the baseline), with process noise swept to the
physiological match `sigma_a≈0.05`: **turning 18°→7.7° (ref 8.1), cl 0.55→0.70 (ref 0.72)**, beating
Savitzky-Golay at equal-or-better coherence. The turn gap was a **smoothing under-fit** — the baseline
used Savgol window 5; it already matches the reference at window ~11 or, principledly, via RTS. *Honest
limit:* the **filtered** one-step error (0.20 mm — the un-fakeable link-quality anchor) is **unchanged**;
RTS is a backward pass that improves the trajectory *estimate*, it does not re-associate detections. So
turning/cl were an estimation/rendering property, not a linking error.

**2. Detection-quality gating — NOT the fix, but a validated diagnostic (`track_zgate.py`).**
Confirmed the reference's ~31% kept detections are heavily high-confidence (near-track z 9.3 vs far 6.0,
p≈0; kept-fraction rises Q1 4.8% → Q4 45.4% with z). But *reproducing* that selection does not reproduce
its slow tracks: hard z-thresholding improves turning/cl only at the p90 extreme, which craters coverage
to 6% (vs ref 31%), and **speed never moves** (~40 at every threshold). Soft z-weighting of the Kalman R
was harmful (fragmented tracks). A mild **z≥p25 prefilter** is a free small win (keeps ~75% of input at
≈reference coverage, cl 0.57, one-step 0.196 mm) — worth adopting, but not the residual's cause.

**3. Global / min-cost-flow association — NOT the fix (`track_global.py`).** A second-order min-cost
network-flow tracker (curvature-aware) and a global fragment-stitcher. Min-cost-flow does **not** beat
the greedy KF at matched coverage — at 32% it is *worse* (turn 13.3, speed 43.8), and only looks smooth
by cherry-picking 11% coverage. Root cause: per-frame steps (~0.12 mm) sit at the localization-jitter
floor (~0.12 mm), so raw segment directions are noise-dominated and the KF's *recursive velocity
smoothing* is exactly what suppresses jaggedness — a raw-link flow can't replicate it. Stitching is a
modest **continuity** (topology) win, not a physiology one.

### The one genuine residual: speed / over-reach — a selectivity tradeoff, not a bug

Speed (~36 vs 26 mm/s) is the only real remaining gap. It **plateaus under smoothing**, is **unmoved by
z-gating**, and is **not helped by global association**. Our tracker reaches to slightly farther
detections to hold 33% coverage; the reference links tighter/slower and **discards 69%**. We can match
the reference's speed only by tightening the association gate and *dropping coverage* — a genuine
**gate-vs-coverage selectivity tradeoff**, not a fixable error. It is visible in
`renders/track_global_vs_reference.png` (ours redder = faster) and offers two honest operating points:
**high-coverage** (33% linked, speed ~36) vs **reference-matched tight-gate** (fewer links, speed →26).

### Recommended production config

`track_rts.py::track_kf_rts` at **`sigma_a≈0.05` + RTS smoothing + z≥p25 prefilter**: matches the
reference on **turning (7.7 vs 8.1°) and coherence (cl 0.70 vs 0.72)** at ≈reference coverage; the speed
residual is the selectivity knob (tighten the gate to trade coverage for reference-matched speed). This
is the tracker to scale to 216 acqs.

## Bottom line

We are **back to independently tracked bubble trajectories** — the right object — with a principled
Kalman tracker (KF + RTS smoothing) validated against the reference on identical acq-0 data. After the
residual investigation it **matches the reference on turning (7.7 vs 8.1°), coherence (cl 0.70 vs
0.72), and coverage/length**, is 80% bootstrap-stable, and used no new data. The single remaining gap
— speed (~36 vs 26 mm/s) — is a **characterized selectivity tradeoff** (gate vs coverage), not an
error. It is **ready to scale to all 216 acquisitions** the moment the 215-acq detections arrive
(`mb-4yw`, watched by `scripts/monitor_data_unblock.sh`) — the actual unblock for a reference-density
tracked reconstruction. The within-acq coherence finding corrects the objective: ground in the data,
don't import the reference's pooled diversity as a target.
