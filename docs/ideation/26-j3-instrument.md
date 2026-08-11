# J3 — the injection + null instrument, and the joint SVD-rank x normalization sweep

> Bead `mb-fmj`, child of the J1–J5 epic `mb-e1o`. Code: `ultratrace_ulm/inject.py`,
> `ultratrace_ulm/opsweep.py`, `scripts/modal/j3_instrument_app.py`,
> `scripts/j3_report.py`. Tests: `tests/test_inject.py` (19). Modal spend: ~$2.

## Why the instrument exists

This dataset has no ground truth and both of our validation habits have failed.
"Matches the reference" is circular — we reproduce it (1,451 vs 1,421 tracks ≥ 35)
and it is not ground truth. Held-out position prediction is beaten by
tracker-free interpolation, so it does not measure the tracker.

So we make ground truth: bubble echoes we authored, with known trajectories,
added to the beamformed **complex** volume before clutter filtering, so they
traverse the same SVD, the same normalization and the same detector as real ones.

The instrument's second job is to make J2's failure mode impossible. J2 compared
two arms at a fixed 2σ threshold when their noise floors differed, which measures
the denominator rather than the detector. Here **every** comparison is at matched
detection density on the un-injected volume or at matched false-alarm rate on
null data. That is cheap to enforce: non-maximum suppression runs before
thresholding, so one detection pass yields the whole rate-vs-threshold curve and
any matched-rate point is a quantile of it.

## What the instrument does, and what it deliberately does not model

The signal model is `s(r) = A·h(r − p)` with `h(u) = env(u)·exp(i·k_z·u_z)`. The
beamformer applies focusing phase `exp(+i·2k·z)` while the echo carries
`exp(−i·2k·z0)`, so the residual phase depends only on `z − z0`. Two things fall
out rather than being bolted on:

- **Doppler is automatic.** At a fixed voxel the phase rotates at `k_z·v_z`, which
  is what decides where a bubble sits in the SVD's temporal spectrum — i.e.
  whether the clutter filter eats it. A slow bubble is correctly hard. Tested
  against the algebra (`test_injected_axial_motion_produces_the_right_doppler`).
- **The PSF is measured, not assumed.** Elevation is the weak axis and no
  analytic Gaussian would get the anisotropy right. Measured on real data from
  400 isolated bright detections: **FWHM 1.66 mm elevation vs 1.00 mm in-plane**,
  and an axial carrier of **3.096 rad/voxel against the predicted π = 3.142** —
  an independent confirmation of the model, since λ/4 axial sampling of a 2k
  carrier is exactly π per voxel.

Because injection happens *after* the coherent 4-angle sum, the compounding gain
`|sin(2φ)/(4 sin(φ/2))|` is applied by hand; otherwise synthetic bubbles would be
detectable at precisely the 88.97 mm/s where real ones vanish. Not modelled, and
stated in the code: grating lobes (J5 came back REFUTED), intra-frame spatial
smear (so fast-bubble recovery is a mild upper bound), and bubble destruction.

## Result 1 — the nulls: one of the three the bead asked for is useless

| null | detections/frame at the matched operating point | verdict |
|---|---|---|
| `block_shuffle` (block 1 and block 20) | **34.00** for all 18 arms | **carries zero information** |
| `phase_surrogate` (per-voxel Fourier) | 57–86 | valid, but pessimistic |
| `quiet_crop` (real, quietest region) | 4.0–8.8 | most informative, conservative |

Frame-shuffling is not a false-alarm null for a per-frame detector. Permuting
frames leaves every frame — and every real bubble in it — exactly intact; it
destroys trajectories, not echoes. The measurement is unambiguous: both block
sizes returned the target density to two decimals for every operating point. It
remains the right null for *tracker*-level claims, where destroying trajectories
is the intervention you want.

The phase surrogate works, and its cost is now stated exactly rather than
hand-waved: randomizing each voxel's Fourier phase leaves the array supported on
the same frequency bins, so clutter rank rises from "number of spatial modes"
(a few) to "bandwidth in bins" (tens). A fixed-rank cut then removes much less of
it — which is why it returns 1.7–2.5x the clean density and is a pessimistic
bound. (Corollary: a *global* per-bin phase would leave the Gram spectrum exactly
unchanged while failing to remove bubbles, which is why the draw is per-voxel.)

## Result 2 — the joint sweep

Resolved cutoffs on acq 0 (240 frames, 25 × 154 × 275):

| arm | modes removed (low) | MP high_remove | kept |
|---|---|---|---|
| rank24 (status quo) | 24 | 0 | 216 |
| knee | 21 | 0 | 219 |
| knee_spatial | 2 | 0 | 238 |
| any arm `+mp` | unchanged | 122 | — |

Recovery at matched density 34 detections/frame, 6 realizations × 300 bubbles
(~42,000 bubble-frames), SNR 6–21 dB, speeds 5–130 mm/s, four directions:

| filter | normalization | recovery | vs status quo | depth non-unif. |
|---|---|---|---|---|
| rank24 | **spatial_tgc** | **0.396** | **+0.030** | 0.633 |
| knee | spatial_tgc | 0.390 | +0.024 | 0.770 |
| knee | per_elev_zband | 0.385 | +0.019 | 0.872 |
| rank24 | per_elev_zband | 0.385 | +0.018 | 0.860 |
| knee | per_elev | 0.369 | +0.003 | 0.729 |
| rank24 | per_elev (STATUS QUO) | 0.366 | — | 0.903 |
| …`+mp` arms | any | 0.245–0.272 | −0.09 to −0.12 | — |
| knee_spatial | any | 0.002–0.003 | −0.36 | — |

Standard error on every difference is ±0.003 (and that is the *conservative*
independent-arms figure; the arms share the same injected bubbles).

**Which operating point wins, and by how much.** `rank24 + spatial_tgc`, on both
acquisitions:

| | acq 0 | acq 1 (replication) |
|---|---|---|
| status quo `rank24 \| per_elev` | 0.366 | 0.551 |
| winner `rank24 \| spatial_tgc` | **0.396 (+0.030)** | **0.564 (+0.013)** |
| relative gain | +8.2 % | +2.4 % |
| depth non-uniformity | 0.903 → **0.633** | 0.621 → **0.452** |

The direction replicates and both margins clear the ±0.003 standard error
comfortably, but **the magnitude varies by 2.3x between two acquisitions of the
same animal**, so "+0.03" is not a number to carry forward. The defensible claim
is *a small, consistent, single-digit-percent gain, larger where the baseline is
weaker* — which is what a depth-and-lateral gain correction should do.

The honest framing is that this is a **normalization win, not an SVD-rank win**.
The rank axis is flat: on acq 0 the knee resolves to 21 against rank24's 24 and
buys +0.003 ± 0.003, i.e. nothing; on acq 1 the knee resolves to **24 exactly**,
so the two arms are literally the same filter. The bead's premise was that rank
24 is an accidental, never-fired fallback — it is, and it is also *fine*. Leave
it alone.

**One arm does not replicate and should not be adopted.** `per_elev_zband`, the
minimal fix of one denominator per (elevation, depth band), gave +0.018 on acq 0
and **−0.014 on acq 1**. It changes sign. Had acq 0 been run alone it would have
looked like a modest win; it is not one.

Two arms actively hurt and should not be used:

- **The MP high cutoff is a trap.** At this aspect ratio (γ = 240/1.06M) the
  Marchenko–Pastur bulk collapses to a point, so its upper edge sits above nearly
  every noise eigenvalue and the rule removes **122 of 240 modes**. Recovery falls
  by 0.09–0.12 in every combination. It is doing what the mathematics says; the
  mathematics is simply not applicable at γ this small.
- **The B18 spatial-correlation cutoff fails outright here**, returning **2**
  modes and recovery 0.003 — a 100x collapse. It fell back to `min_rank` because
  the coherence curve has no contrast, exactly the "soft knee" degenerate case its
  own docstring warns about for low angle counts. Leaving 238 of 240 modes leaves
  the tissue clutter in, the z-score denominator explodes, and nothing is found.

This settles the "VOID empirical backing" question the bead raised: of the three
idle mechanisms, the knee is harmless-but-pointless, and the MP and spatial
-correlation cutoffs are actively harmful on this data. None should be enabled.

## Result 3 — depth non-uniformity, the specific prediction

The concern was that `_slice_stats` pools mean and std per elevation plane over
all frames **and all depths**, so the denominator is wrong at both ends of the
depth axis. It is, and the effect is largest exactly where predicted — shallow:

| depth band | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| status quo (`per_elev`) | 0.306 | 0.283 | 0.391 | 0.375 | 0.349 | 0.403 | 0.447 | 0.250 |
| winner (`spatial_tgc`) | 0.388 | 0.377 | 0.438 | 0.391 | 0.347 | 0.402 | 0.465 | 0.270 |
| gain | **+0.082** | **+0.094** | +0.047 | +0.016 | −0.002 | −0.001 | +0.018 | +0.020 |

The spatially varying normalization buys **+0.08 to +0.09 in the two shallowest
bands and essentially nothing in the middle**. That is the pooled-statistics
concern, confirmed and localized. Relative depth non-uniformity (spread ÷ mean,
per SNR, averaged) falls from 0.903 to 0.633.

Worth noting what did *not* work: `per_elev_zband` — the minimal fix, one
denominator per (elevation, depth band) — recovers only +0.018 of the +0.030 on
acq 0, barely moves non-uniformity there (0.860 vs 0.903), and **reverses to
−0.014 on acq 1**. The residual is lateral, not axial: the beam profile varies
across x as well as z, and a purely depth-indexed denominator cannot see that.
Only the full spatial map helps, and only it replicates.

Pooling over SNR understates how bad this gets. At a fixed 12 dB — right in the
detector's transition region, where a depth-dependent denominator does the most
damage — the status quo's recovery by depth band is:

| band | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| recovery @ 12 dB | **0.042** | 0.135 | 0.237 | 0.369 | 0.301 | **0.440** | 0.384 | 0.212 |

**A 10x swing across depth at fixed injected SNR** — 4.2 % recovery in the
shallowest band against 44 % five bands deeper, for bubbles of identical
contrast. Recovery is not merely depth-non-uniform; at the operating point the
pipeline actually uses, depth is a larger effect on detectability than a 3 dB
change in SNR.

Two caveats on the depth read-out. The metric is reported *relative* to mean
recovery, because a raw max-minus-min rewards an arm for recovering less. And the
extreme bands (0 and 7) have ~2,000 samples against ~6,500 in the middle, because
the injection margin keeps tracks off the volume faces.

## Result 4 — two complementary blind spots in velocity space

The speed marginal alone shows only a mild dip at 88.97 mm/s, because it averages
axial and in-plane motion. Split by direction (status quo arm, matched density):

| direction | 5 | 20 | 45 | 70 | 88.97 | 110 | 130 mm/s |
|---|---|---|---|---|---|---|---|
| **axial** | 0.591 | 0.709 | 0.511 | 0.107 | **0.002** | 0.020 | 0.044 |
| **lateral** | **0.077** | 0.376 | 0.535 | 0.666 | 0.630 | 0.624 | 0.653 |
| elevation | 0.051 | 0.097 | 0.203 | 0.383 | 0.392 | 0.596 | 0.487 |
| oblique | 0.209 | 0.561 | 0.575 | 0.489 | 0.362 | 0.256 | 0.148 |
| *predicted 4-angle gain* | −0.0 | −0.7 | −3.8 | −11.4 | −92 | −13.8 | −11.3 dB |

**Read this carefully — the axial row is half a consistency check.** We *apply*
the compounding gain to injected bubbles, so the axial collapse at 88.97 mm/s is
partly built in by construction; it confirms the injection did what it was told,
not that the null exists. What is *not* built in, and is the useful output, is the
**exchange rate between the dB axis and the detection axis**: −3.8 dB costs
0.511 → nothing much, −11.4 dB costs 0.511 → 0.107, a 4.8x loss. That is the
number J2 needs to predict its own payoff, and it could not be guessed.

The rows that carry no imposed gain are the genuinely new finding, and they run
the *other* way — **recovery rises with speed for in-plane motion**. Lateral at
5 mm/s is 0.077; at 70 mm/s it is 0.666, an 8.6x difference with no gain term
anywhere. The mechanism is emergent from the signal model rather than programmed:
a slow in-plane bubble sits at nearly constant amplitude and constant phase in
its own voxel, which is exactly what tissue looks like, so the clutter filter
removes it. Axial motion escapes this even when slow, because axial displacement
rotates the carrier — 5 mm/s axial is 12.5 Hz of Doppler, well clear of the
tissue band, hence axial 0.591 against lateral 0.077 at the same speed.

So the pipeline has **two complementary blind spots**, and between them they
cover a large part of velocity space:

- **slow in-plane motion** — the clutter filter eats it (lateral 5 mm/s: 0.077);
- **fast axial motion** — the coherent compound annihilates it (axial 89 mm/s:
  0.002, and still only 0.02–0.04 at 110–130).

A vessel's orientation is fixed, so this is not averaged away in practice: an
axially-running vessel at ~89 mm/s and a laterally-running vessel at ~5 mm/s are
both close to invisible, for entirely different reasons. Elevation is the third
axis of the same story and quantifies the weak-axis cost directly: at 45 mm/s,
elevation-moving bubbles are recovered at 0.203 against lateral's 0.535 — **under
40 % of the in-plane rate at matched speed**, which is what 0.5547 mm sampling
against a 1.66 mm PSF buys you.

## What this instrument is for next

It is built to be reused, and the reuse is the point:

- **J2** turns the compounding gain off (`compounding_gain_on=False`) and measures
  how much of the axial deficit its hypothesis bank recovers, against injected
  ground truth rather than against track counts.
- **J4** gets a real detection-rate-below-the-knee number instead of a guess.
- Any future arm gets scored at matched rate by construction.

## Honest limits

- Two acquisitions, one animal, one session. Every *qualitative* finding
  replicates on acq 1 — spatial_tgc wins and flattens depth, MP and the B18
  spatial-correlation cutoff are catastrophic, the axial null and the slow
  in-plane hole both reappear — but the winner's margin moves 2.3x, and
  `per_elev_zband` changes sign. Two acquisitions is enough to reject an arm,
  not to quote a coefficient.
- Recovery is per *bubble-frame*, not per track. A tracker-level score needs the
  linking stage, and `block_shuffle` is the right null for that.
- Injected bubbles are straight constant-velocity tracks at constant amplitude.
  Real bubbles turn, and vessel geometry is not modelled — this bounds detection,
  not morphology.
- Absolute recovery (~0.40) is a property of the chosen SNR grid, not of the
  pipeline. Only differences between arms are meaningful.
