# The operating point, measured rather than inherited

Result of the J1–J5 program. Two settings changed, both because a measurement said so, applied
to all 216 acquisitions of the corrected dataset on identical detections (2,167,465 of them).

## The numbers

| | baseline (shipped) | proposed | reference |
|---|---|---|---|
| gate | 2 voxels/frame = 89 / 247 / 89 mm/s | **130 / 130 / 130 mm/s** | — |
| `min_track_length` | 15 | **8** | 5 |
| tracks | 7,296 | **22,685** (3.1×) | 50,456 (at floor 5) |
| tracks ≥35 frames | 1,451 | **1,531** (+80, +5.5%) | 1,421 |
| tracks ≥50 frames | 582 | **601** (+19) | — |
| linked detections | 9.16% | **16.7%** (+7.54 pts) | — |
| purity vs permutation null | 99.5% | **95.2%** | not measurable (no null published) |

The baseline arm reproduces our full recreation run exactly (7,296 / 1,451 / 582), which is the
check that this harness is production-faithful — see the `max_cost` note below.

**Not one chance chain reaches 35 frames.** Across 2.17M frame-permuted detections the null
produces 35 tracks at the baseline operating point and 1,088 at the proposed one, and **zero**
of length ≥35 at either. So the headline ≥35 population is null-free, and the +80 is not
purchased with contamination. The purity cost of the change lands entirely on short and
medium tracks, which is where a length floor was always doing its work.

## Why these two settings

**Gate.** The shipped default is a box gate of 2 voxels/frame, which in physical units is 89
mm/s in-plane and 247 mm/s in elevation — 2.77× looser on the axis with the worst localization
(0.5547 mm voxels synthesized from 8 physical rows, against 0.2 mm in-plane). Per-step
velocities across the 216-acquisition recreation hard-wall at exactly that default: observed
89.2 / 246.7 / 89.3 mm/s against 89.15 / 246.76 / 89.33 predicted, three significant figures on
all three axes, with zero steps above 92 mm/s in-plane. The measured speed distribution was an
artifact of our own gate, and every bubble faster than ~89 mm/s in-plane was structurally
unlinkable.

A note on a coincidence that cost real time: the gate wall and the 4-angle coherent-compounding
null are the same speed by algebra (voxel = λ/4 so 2 voxels/frame = λ·FR/2; PRF = 4·FR so
λ·PRF/8 = λ·FR/2 = 88.97 mm/s). Gate-censoring and compounding-annihilation are therefore
confounded in every track-level statistic and separate only by axis. That is why the asymmetry
this program chased could not be attributed from track data alone.

**Floor.** `min_track_length` is applied only at emission (`tracking.py:343`, `:722`, `:729`) —
it never enters the assignment cost, the Kalman update, the gate, or spawn logic. Lowering it
recovers no associations; it decides what gets published. So it is chosen against a purity
curve rather than argued about: with the null above, purity is 36.5% at length 2, clears 95% at
**8**, and reaches 99.6% at 15. Going 8 → 15 costs 3× the yield to buy 3.4 points, and the
shipped 15 buys no planted-crossing precision (flat 0.91–0.96 across the whole range).

**The two interact.** At floor 15 only 1.2% of the gate's link gain is null-reproducible; at 8
it is 11.8%; at 5 it is 28.6%. A gate gain quoted without its floor is not a meaningful number.

## What did not change, and why that is a result

`rank 24 + spatial_tgc` survived the joint sweep. The adaptive SVD cutoff never fires on this
data (no mode centroid reaches the 100 Hz boundary; max 85–96 Hz), so rank 24 is the
10%-of-frames fallback — accidental, and also fine. The win in that sweep was normalization,
not rank. Two mechanisms with prior code support turned out to be actively harmful: the
Marchenko-Pastur high cutoff removes 122 of 240 modes (correct maths in a regime where
γ = 240/1.06M collapses the bulk to a point), and the B18 spatial-correlation cutoff returns 2
modes. `per_elev_zband` changes sign between acquisitions (+0.018 / −0.014).

## Harness note (a trap worth documenting)

`kalman_tracking_3d`'s own `max_cost` default is **1e5**; the production `TrackingOptions` uses
**10.0**. Calling the function directly with its defaults runs a 10,000× more permissive
assignment than the pipeline ships. An earlier version of this comparison did exactly that and
produced a baseline of 1,890 tracks ≥35 against the recreation's 1,451, which would have been
reported as beating the reference by 35% had the discrepancy not been chased down. The script
now passes `max_cost` explicitly.

## Honest scope

This is a publication-and-association improvement: more of the trajectories the pipeline already
builds, with the chance-chain fraction measured rather than assumed. It is **not** more signal
extracted from the raw data. Every upstream idea in the program that promised that — the
motion-phase hypothesis bank, the interleaved 888 Hz per-transmit movie, grating-lobe
correction, MP/B18 rank selection — is refuted or measured harmful. The one upstream lever still
standing is a ~2.3 dB velocity-dependent decline in per-angle coherence, and it is not yet
established: the curve's binning variable is derived from the same phasors as the quantity being
binned, so it needs re-binning against independently tracked step velocity before it means
anything.
