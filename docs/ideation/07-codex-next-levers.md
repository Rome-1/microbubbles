# 07 - Independent next-lever opinion

This is a deliberately skeptical second opinion. I am not assuming the current
plan is wrong, but I would not let the new SVD knee become the center of gravity
until it survives a few falsification tests.

The key acquisition fact is still the constraint that should dominate the next
experiments: this is not a matrix-array 3D ULM dataset. The 25 elevation planes
are synthesized from 5 transmit angles and 1 physical receive row. Elevation is
therefore a weakly observed coordinate, not a peer of x and z. Any method that
lets y/elevation break tracks, duplicate detections, or create apparent
curvature can win the current length proxies while making the vascular map less
real.

## Decision rules before the next bake-off

Use the same 60 acquisitions for all comparisons unless a test explicitly needs
odd/even or first/second splits. A run should not be called better just because
mean length rises. I would require:

- `mean_curvilinear_len_mm`, `frac_len_ge_20`, and `frac_len_ge_35` improve.
- `frag_short_over_long` decreases or stays close to baseline.
- `straightness` does not drop by more than about 10 percent unless split-half
  reproducibility clearly improves.
- `n_tracks` and `peak_density` do not explode without a matching improvement in
  split-half FRC/FSC or density-map correlation.
- Elevation occupancy does not collapse into the bright midplane or spread into a
  uniform elevation haze.

The last two checks matter because the current metrics can reward either
over-linking or residual clutter.

## 1. Tracking-first factorial: make association elevation-aware

### Method

Run a tracking-only factorial on the existing detection checkpoints before
spending more GPU time on new SVD/detection variants. The current tracker already
has Kalman prediction and a Mahalanobis residual, but the hard pre-gate is still
axis-wise and elevation can reject an otherwise plausible x/z continuation.

Test a "2D-first, y-soft" association policy:

- Gate on prediction: `gate_on_prediction=True`.
- `max_gap`: `4, 6, 8` frames.
- Hard gate in x/z: `0.35, 0.50, 0.70 mm`.
- Hard gate in y: `1.0, 1.5, 2.0, inf mm`; include an explicit y-ignored pre-gate.
- Measurement covariance: keep x/z at the current base, inflate y by `5x, 10x,
  15x, 25x`.
- Cost: Mahalanobis residual plus reversal penalty, with a small appearance term
  `lambda_I * abs(log I_det - log I_track)` for `lambda_I = 0, 0.2, 0.5`.
- Keep `min_track_length` fixed for the first sweep so improved results cannot
  hide behind changed filtering.

This can be run on saved detections. It should be cheaper and more diagnostic
than re-beamforming.

### Why it should move the proxies

If the bottleneck is association rather than SVD, this should increase
`frac_len_ge_20/35/50` and reduce `frag_short_over_long` without materially
changing `occupied_fraction`. That pattern would be strong evidence that many
short tracks are the same bubbles being broken by gating, not missing
detections.

The 1-row geometry makes this especially plausible: a real bubble should have a
coherent x/z path, while the y coordinate may jitter or snap toward the midplane.
Letting y dominate the gate is a false precision problem.

### Validation

Reject settings that get long tracks by merging unrelated bubbles:

- `straightness` must stay within about 10 percent of baseline or improve.
- Mean and p90 speed should not jump sharply relative to baseline.
- Split-half density maps should become more similar, not merely denser.
- Inspect a small set of longest new tracks in x/z/y separately. A good change
  should look like clean x/z continuation with noisy y, not y-driven zig-zags.

This is my highest-priority experiment because it can improve continuity without
creating new detections.

## 2. Continuation-only low-confidence detections

### Method

Use the existing CFAR confidence output as intended: high-confidence detections
may spawn tracks, low-confidence detections may only continue active tracks near
their predicted position.

Try:

- Spawn threshold: `4.5, 5.0, 5.5 sigma`.
- Continuation threshold: `3.0, 3.5, 4.0 sigma`.
- `global_floor_fraction`: `0.35, 0.50`.
- CFAR `guard_radius`: `(2,2,2)` and `(3,2,2)` for `(elev,z,x)`, not the current
  elevation-light `(1,2,2)`.
- NMS radius: `(4,2,2)`, `(6,2,2)`, `(8,2,2)` for `(elev,z,x)`.
- Pair with the y-soft association from experiment 1, not the baseline gate.

Do not allow low-confidence detections to start tracks. Do not count interpolated
or low-confidence points as equivalent to high-confidence points in audit output.

### Why it should move the proxies

This targets intermittent dropouts, which should increase track length and reduce
fragmentation while avoiding the usual false-positive failure mode of lowering
the global threshold. It is a better match to no-GT ULM than a single aggressive
detection threshold: the tracker supplies the prior, and the detector supplies
candidate continuations.

In this dataset, it also gives the SVD knee a fairer test. If knee filtering
recovers weak bubbles but they are still below the spawn threshold in some
frames, a continuation tier should turn recovered signal into longer tracks
instead of just more one-frame peaks.

### Validation

Use three counters in addition to `bench.py`:

- fraction of accepted points that are low-confidence continuations;
- number of tracks spawned from low-confidence detections, which should be zero;
- number of low-confidence candidates rejected by the gate.

Good behavior: moderate rise in total points, longer tracks, stable or improved
straightness, and better split-half reproducibility. Bad behavior: large `n_tracks`
increase, density entropy rising, or long tracks dominated by low-confidence
points.

## 3. Treat the elevation artifact as a first-class failure mode

### Method

Before more SVD variants, run an elevation audit and a small debias/association
sweep. The bright-midplane artifact may be a bigger lever than the number of SVD
modes removed.

Compute, per run:

- detection count by elevation plane;
- track point count by elevation plane;
- mean z-score/intensity by elevation plane;
- per-track y standard deviation and y range;
- correlation between y occupancy and the pre-detection elevation bias profile.

Then test:

- Debias on/off.
- `debias_smooth_sigma`: `1, 2, 4`.
- `floor_fraction`: `0.25, 0.40, 0.60`.
- "Track in x/z, carry y" mode: y is used for reporting and smoothing, but not
  for the initial association gate.
- Optional y clamp for association only: y residual clipped to `1.0, 1.5, 2.0 mm`
  before cost calculation.

### Why it should move the proxies

If the midplane artifact suppresses off-midplane detections or creates duplicate
elevation peaks, it can simultaneously reduce coverage and fragment tracks.
Correcting it should improve `occupied_fraction`, reduce short fragments, and
raise straightness. It should also make the SVD knee less risky, because retaining
more post-SVD energy will otherwise retain more artifact energy too.

### Validation

This experiment should be vetoed by elevation-specific diagnostics, not only by
length:

- The elevation occupancy profile should become less midplane-dominated but not
  flat.
- Split-half maps should agree in x/z and in coarse y slabs.
- Long-track gains should not come from tracks that oscillate across many
  synthesized elevation planes.
- Compare coronal x/z density and elevation-resolved density. A method that
  improves x/z while making y meaningless may still be acceptable, but it should
  be labeled as 2.5D ULM rather than trusted volumetric ULM.

This is where I am most concerned the current plan is underweighting the
instrument geometry.

## 4. Stress-test the SVD knee instead of assuming knee=17 is portable

### Method

The knee fix is worth testing, but the first result should be treated as a
hypothesis, not as the new default. Run a deliberately boring cutoff sweep and
stability analysis:

- Fixed low cutoffs: `8, 12, 17, 24, 35, 50, 70`.
- Kneedle per acquisition with `knee_min=1`, ceiling `70`.
- Kneedle with a floor: `knee_min=4` and `knee_min=8`.
- Optional high cutoff off for the main sweep; MP high cutoff only as a second
  pass after the low-cutoff question is settled.
- Record the selected knee for every acquisition and summarize median, IQR, min,
  max, and correlation with detection counts.
- Freeze one global cutoff from the first 30 acquisitions and score it on the
  next 30; then reverse the split.

### Why it should move the proxies

If the old adaptive path really over-removed blood signal, low cutoffs should
increase `occupied_fraction`, `total_points`, and long-track fractions. But the
failure mode is equally obvious: retaining residual tissue/clutter can create
trackable false structure. "Knee <= 70" is not false-track-safe; it is only less
aggressive filtering.

A fixed sweep is important because a single Kneedle value can look principled
while merely fitting one acquisition's spectrum. If `k=17` beats fixed `12` and
`24` only on one acquisition, I would not ship it.

### Validation

Require a monotonicity sanity check and a reproducibility check:

- As k decreases from 70 to 8, detection count should rise. If track length rises
  while split-half correlation falls, the run is probably tracking clutter.
- The best k should not be a razor-thin optimum. Prefer a plateau such as
  `12-24` over a single lucky value.
- A knee-selected run should beat or match a fixed cutoff near the median knee.
  If fixed `17` performs the same as Kneedle, use fixed `17` for the next composite
  until there are more acquisitions or subjects.
- Compare first-half vs second-half selected knees. If the knee distribution
  shifts materially, it is acquisition-specific and needs regularization.

This is the contrarian SVD take: the bug was real, but the bottleneck may still
be downstream.

## 5. Empirical PSF matched filter, but train and test it disjointly

### Method

Use the empirical PSF work, but make it a conservative validation experiment
rather than a blind sensitivity boost.

Build PSF templates from bright isolated detections:

- candidate percentile: top `0.5, 1, 2 percent` by z-score after debiasing;
- isolation: no other peak within `(8,4,4)` voxels in `(elev,z,x)`;
- patch radius: `(8,4,4)` and `(10,5,5)`;
- normalize patches by peak or local energy;
- build separate odd-acq and even-acq PSFs;
- cross-correlate with the opposite split's data, then run CFAR/NMS.

Sweep NMS radii tied to the measured PSF width:

- elevation radius: `ceil(0.75*FWHM_y)`, `ceil(1.0*FWHM_y)`, `ceil(1.25*FWHM_y)`;
- z/x radius: `ceil(0.75*FWHM)`, `ceil(1.0*FWHM)`.

### Why it should move the proxies

The 1-row synthesized elevation PSF is likely broad and structured. A matched
filter can recover weak bubbles and suppress non-PSF-shaped clutter, which should
increase coverage and continuity without simply lowering threshold everywhere.
It also gives physical numbers for elevation NMS and tracking covariance instead
of guessing "2x" or "4x" radii.

### Validation

The PSF must generalize across splits:

- odd-built PSF and even-built PSF should have similar FWHM and sidelobe shape;
- applying odd PSF to even data should improve split-half reproducibility, not
  just same-split metrics;
- matched filtering should not narrow the elevation profile artificially or pull
  peaks toward the midplane;
- reject if peak density rises but contrast/FRC do not.

This is high leverage, but I would run it after the tracking/elevation sweeps
because it changes the detection distribution and can mask association problems.

## Contrarian challenges to the current plan

### Challenge 1: SVD may not be the bottleneck anymore

The strongest validated gains so far are tracking-side: region+Kalman gave about
+50 percent track length, and stitching gave much larger length gains with a
straightness warning. That pattern is hard to reconcile with "SVD cutoff is the
dominant limiter" as a complete explanation. It suggests many bubbles are already
detected often enough to be linked better, but the association policy breaks
them.

If a tracking-only sweep on the same detections can recover most of the knee
gain, the next dollar should go to association and validation, not more SVD.

### Challenge 2: The knee at 17 may be an acquisition-specific artifact

Kneedle on a soft singular-value curve is a reasonable heuristic, not an oracle.
With 5 angles and a single receive row, the tissue/blood/noise regimes may be
less clean than in the matrix-array literature. A single real-data knee of 17 is
not enough evidence that 17 is stable across 223 acquisitions, let alone across
subjects.

The old cutoff of 70 was clearly broken. That does not imply the replacement is
fully specified. The right next result is a knee distribution and fixed-rank
sweep, not just "knee beats adaptive."

### Challenge 3: The bright-midplane elevation artifact may dominate both SVD
and detection thresholds

If an elevation brightness profile is biasing z-scores and local maxima, changing
SVD rank can look like a coverage fix while actually changing how much artifact
survives. In a 1-row synthesized-elevation setup, this is not a minor nuisance:
it can create duplicate peaks, suppress off-plane bubbles, and break tracks by
injecting y jitter.

I would not trust a coverage improvement until it passes an elevation occupancy
and y-jitter audit.

### Challenge 4: Current proxies can be gamed by false tracks

Mean curvilinear length, `frac_len_ge_*`, and fragmentation are necessary but not
sufficient. A permissive stitcher, a low SVD cutoff, or a low detector threshold
can create longer paths through residual clutter. Even straightness can be gamed
by linking along coherent tissue streaks.

Split-half reproducibility is not optional. It is the closest thing to a no-GT
truth serum here: real vasculature should recur; random clutter and over-merged
tracks should not.

### Challenge 5: Re-beamforming every SVD/detection variant may be the wrong
compute allocation

Given the cost and the lack of ground truth, exhaustive GPU bake-offs can create
a false sense of progress. Tracking-only experiments on saved detections are
cheap and directly test the leading alternative hypothesis. The next compute
should prioritize designs that discriminate between "missing bubbles" and
"broken association" rather than just producing another improved aggregate
metric.

## If I had to bet on a single change

I would bet on replacing the current association policy with a y-soft,
prediction-gated tracker: x/z-first hard gating, elevation-downweighted
Mahalanobis cost, `max_gap=6`, and a small intensity-continuity term. In concrete
terms, start with:

- `gate_on_prediction=True`;
- x/z gate `0.5 mm`;
- y gate `1.5 mm` or y ignored in the pre-gate;
- y measurement covariance `15x` x/z;
- `max_gap=6`;
- intensity cost `lambda_I=0.2`;
- stitch only after this, with `vel_cos_min >= 0.8`.

Reasoning: it improves continuity without adding detections, so it has the
lowest risk of manufacturing new false tracks. It is also the change most
specific to this dataset's geometry. A matrix array would not justify such
aggressive y skepticism; this 1-row synthesized-elevation setup does.

The SVD knee should still be run, but I would treat it as the second bet. If the
knee wins only when paired with the y-soft tracker and continuation-only
thresholding, the interpretation should be "we recovered weak detections and
fixed association," not "SVD was the bottleneck."
