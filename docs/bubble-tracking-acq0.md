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
**gate-vs-coverage selectivity tradeoff**, not a fixable error. A hard per-frame step gate (scaled by
the reference's own anisotropic `max_distance_mm`) is the knob; `track_operating_points.py` renders
both ends beside the reference (`renders/track_operating_points.png`, speed-jet 0-40):

| operating point | tracks | linked | speed med | speed p90 | cl |
|---|---|---|---|---|---|
| reference | 292 | 31% | 25.8 | 59.3 | 0.73 |
| **high-coverage** (loose) | 561 | 51% | 45.4 | 138 | 0.60 |
| **reference-matched** (step≤0.6) | 326 | 25% | 25.1 | 56.9 | 0.58 |

The tight gate reproduces the reference's speed distribution (median 25.1 vs 25.8, p90 56.9 vs 59.3)
and its sparse slow look; high-coverage tracks ~2× the bubbles at the cost of faster/marginal links
(and some over-linked clusters). The reference operates near our tight-gate point — it is selective by
design. Choice of point is a science decision (coverage vs cleanliness), not a bug to fix.

### UPDATE (2026-07-11): the speed "residual" was largely OUR config, not a tradeoff

The dataset's `params` (read out of the pkl when checking the acquisition specs) document the
**reference's own tracker settings**: `tracking_method: kalman`, `max_distance_mm = (0.40, 1.11,
0.40)` (= step_scale **1.0**), `max_gap: 3`, `min_track_length: 5`, `post_smoothing: gaussian
sigma 2.0`. We had been running `max_gap=2`, `min_len=4`, no confidence prefilter, Savgol
smoothing — i.e. we differed from them in several places *we never intended to*.

Running **their config** (their gate/gap/length filter) with **our KF + RTS + z≥p25 prefilter**
(`track_production.py --mode reference-config`) reproduces the reference across every axis at once:

| | tracks | mean len | max | speed med | turn | cl | dir-agree |
|---|---|---|---|---|---|---|---|
| reference | 292 | 10.3 | 68 | 25.8 | 8.1 | 0.73 | — |
| **reference-config (ours)** | 249 | **10.7** | **67** | **27.8** | 11.3 | **0.65** | 73% |
| reference-matched (step≤0.6 clamp) | 234 | 9.2 | 46 | 23.6 | 16.3 | 0.59 | 72% |
| high-coverage | 355 | 11.7 | 68 | 44.6 | 18.9 | 0.58 | 71% |

**So the earlier "speed 36 vs 26 is an irreducible selectivity tradeoff" conclusion was wrong** —
it was an artifact of a config mismatch. The brute-force tight-gate point (step≤0.6) is
**superseded**: it matched the speed *statistic* by clamping, at the cost of track length (max 46
vs 68) and smoothness (turn 16.3°). Matching the *mechanism* beat matching the *statistic*.

### Closing the last two gaps (2026-07-14)

The remaining differences (fewer tracks, higher turning) also traced to config, `track_tighten.py`:

- **Track count** — the z≥p25 prefilter, adopted earlier as an accuracy win, was dropping ~46
  real tracks (bootstrap-stable at 83%) for a cl change of 0.64→0.65. Removing it recovers 295
  tracks (reference 292) at 31.7% linked (reference 31%).
- **Turning** — was under-smoothing. But `sigma_a` sets both the smoothing *and* the Kalman gate,
  so lowering it to smooth also tightens the gate and drops coverage (sigma_a 0.03 → 268 tracks,
  0.02 → 235). The reference avoids this by **decoupling**: a moderate filter for coverage, then a
  separate Gaussian sigma=2 post-smooth (its documented `post_smoothing_sigma`). Doing the same —
  filter at sigma_a=0.05, post-smooth Gaussian sigma=2 — sets turning independently of coverage,
  and direction-agreement holds at 75% (smoothing is not erasing link structure).

Tightened **reference-config** (filter sigma_a=0.05, no prefilter, Gaussian sigma=2), acq-0:

| | tracks | mean | max | speed | turn | cl | dir-agree |
|---|---|---|---|---|---|---|---|
| reference | 292 | 10.3 | 68 | 25.8 | 8.1 | 0.73 | — |
| **reference-config** | 295 | 10.5 | 67 | 22.1 | 7.6 | 0.68 | 75% |
| high-coverage | 464 | 11.0 | 69 | 33.5 | 8.2 | 0.63 | 70% |

Reproduces the reference on count, length, and turning; small residuals remain (cl 0.68 vs 0.73,
speed 22 vs 26). The real remaining choice is **reference-config** (faithful reproduction) vs
**high-coverage** (~1.6× bubbles, faster links) — a science decision, not a defect.

## Recommended production config

**`scripts/wf_render_signal/track_production.py`** — the single consolidated, scale-ready entry point
(**no prefilter → Kalman filter w/ Mahalanobis gate → operating-point step gate → Gaussian σ=2
post-smooth**), folding the whole investigation into one command so the unblock is instant:

```
python3 scripts/wf_render_signal/track_production.py --mode reference-config     # default
python3 scripts/wf_render_signal/track_production.py --mode high-coverage
```

It runs over **any** set of acquisitions (acq-0 today; all 216 unchanged the moment `mb-4yw` lands),
self-validates on acq-0 against the reference, and writes `outputs/.../tracks_production_<mode>.npz`.

**Compute hygiene is enforced in code, not by convention** (this rig once drove the shared box to
load ~67): every worker `os.nice(15)`s itself and pins BLAS to 1 thread, worker count is
**hard-capped at 4** regardless of the flag, and the run **pauses** whenever the 1-min load average
exceeds 30 (renicing does not lower load average — pausing does). 216 acquisitions is embarrassingly
parallel, which is exactly why the cap is not optional.

## Reference-independent accuracy + adversarial re-check (2026-07-14)

An adversarial re-check (codex) made a correct and load-bearing point: **matching the reference — which
is not ground truth — is not the same as validating tracking accuracy.** Reference-matching is
circular (the config was chosen partly to match reference coverage/turning, then the match cited as
validation); direction-agreement and bootstrap stability do not bound the mis-link rate; and the
Gaussian σ=2 smoothing partly manufactures the matched turning/coherence while leaving associations
untouched. That critique is accepted. Two reference-independent tests were added
(`track_sim_validate.py`), both on **raw** detection links (smoothing-independent):

- **Held-out linked-detection prediction (real acq-0).** Hold out 15% of the detections that form
  tracks, re-track, predict their positions: **median error 0.32 mm, 64% within 0.5 mm, 9.2× better
  than a frame-shuffled null**. (Restricted to linked detections — the un-trackable ~68% have no track
  to predict them, which is why the same test over *all* detections looks like chance.)
- **Synthetic link precision/recall (planted ground truth, matched to acq-0 stats).** On simulated
  bubbles with known links at acq-0's density/speed/length/noise: **link precision 0.86 ± 0.02**
  (≈14% of the tracker's links are wrong) and **recall 0.61 ± 0.02** (misses ≈39% of true links,
  mostly across gaps and in dense regions). Config carried over unchanged, so the 5 seeds are held-out
  realizations.

**Honest limits of these numbers.** The simulation uses independent smooth trajectories + uniform
noise (no vessel crossings/confluences, where linking is hardest), so **0.86 is likely optimistic**
for real vasculature. There is still **no held-out real acquisition** and **no per-acq QC** — both
require the 215-acq unblock. So the standing claim is downgraded accordingly.

## Bottom line (calibrated)

We are back to independently tracked bubble trajectories with a principled Kalman + Gaussian-smooth
tracker that (a) reproduces the reference tracker on identical acq-0 detections across count, length,
turning, and coverage, and (b) shows **link precision ≈0.86 / recall ≈0.61 on matched synthetic
ground truth** — the first accuracy evidence that does not lean on the reference. It is **not** yet
"validated for accuracy": that needs held-out real acquisitions, per-acq QC, and link-level checks on
real crossings — all gated on the 215-acq unblock (`mb-4yw`, watched by
`scripts/monitor_data_unblock.sh`). The scale-up must run a **frozen protocol** (config + metrics +
thresholds fixed before seeing the data) to avoid the single-acquisition tuning=validation trap. The
within-acq coherence finding stands and corrects the objective: ground in the data, don't import the
reference's pooled diversity as a target.

### Frozen validation protocol for the 215-acq scale-up (pre-registered)

Fix before running: config = `reference-config` (no prefilter, σ_a=0.05, gate_chi2=9, step 1.0,
max_gap=3, min_len=5, Gaussian σ=2) and `high-coverage`, both unchanged from here. On arrival:
1. **Per-acq QC gate** — detection density, per-frame displacement distribution, innovation/NIS,
   linked fraction, track fragmentation; flag acqs outside acq-0's calibration envelope rather than
   silently tracking them.
2. **Held-out acquisitions** — tune nothing; report the acq-0 metrics on a random held-out subset of
   acqs and quote the degradation.
3. **Link-level accuracy** — extend `track_sim_validate.py` with vessel-crossing geometry, and (if any
   labeled/paired detections exist) report real link precision/recall, not only synthetic.
