# From the Aleph baseline to our pipeline: every discrete change

We start from the shipped `alephneuro/microbubbles` pipeline run with its own settings, on the
corrected 216-acquisition public dataset. **Two settings differ in what we ship** — and only one axis of one of them. This document
explains each one, what forced it, and what it cost — plus the things we tried and rejected, so
the ledger is not just a list of wins.

All numbers are from all 216 acquisitions, on identical detections, with our baseline arm
reproducing the reference recreation exactly (7,296 tracks / 1,451 ≥35 frames / 582 ≥50).

---

## Not a difference: deterministic, phase-invariant clutter filtering

We fixed this and so did they, independently, within days of each other (their PR #4 and #5).
Their measurements agree with ours — a phase re-draw moves the old score by 15.6 Hz on their
measurement and 17.5 Hz on ours; the invariant score by ~0. Run-to-run detection agreement goes
from 93.8% to 99.998%.

It is listed only so the record is complete: **as of the merge in `619eed6` this is no longer a
difference between our fork and upstream.** The one piece we still carry is a chunked
double-precision Gram, which computes bit-identical eigenvectors without materialising a ~4.1 GB
complex128 copy of a 2.0 GB matrix. That is a memory optimization, not a result.

---

## Change 1 — the in-plane association gate, in physical units

![the velocity wall](figures/change_gate_wall.png)

### What the gate is

After detecting bubbles in each frame, the tracker has to decide which detection in frame *t+1*
is the same bubble as one in frame *t*. It only considers candidates inside a box around the
prediction. That box is the **gate**, and anything that would have to move further than the gate
allows simply cannot be linked.

### The voxels are not cubes, and that is the whole problem

| axis | voxel size | 2 voxels/frame at 222.43 Hz |
|---|---|---|
| lateral (x) | 0.2004 mm | 89.2 mm/s |
| axial (z) | 0.2008 mm | 89.3 mm/s |
| **elevation (y)** | **0.5547 mm** | **246.8 mm/s** |

Lateral and axial are sampled at λ/4 (λ = 0.800 mm at 2.0 MHz in 1600 m/s tissue). Elevation is
**2.77× coarser**, because the 25 elevation planes are synthesized from only 8 physical receive
rows — it is the axis we localize *worst*.

The shipped default gate is **2 voxels per frame on each axis**. Because the voxels are
anisotropic, that one rule means two very different physical limits: a bubble may move 89 mm/s
in-plane, but 247 mm/s in elevation. The most permissive gate sits on the least trustworthy
axis.

### It was binding, and we can prove it

Per-step velocities across the whole 216-acquisition run stop dead at exactly the default:

| axis | observed maximum | 2 voxels/frame |
|---|---|---|
| lateral | 89.2 mm/s | 89.15 |
| axial | 89.3 mm/s | 89.33 |
| elevation | 246.7 mm/s | 246.76 |

Three significant figures on all three axes, with zero steps above 92 mm/s in-plane. The "speed
distribution" of the published tracks was a picture of our own gate. Any bubble faster than
~89 mm/s in-plane could not be linked at all, no matter how cleanly it was detected.

**Reading the figure.** The lobes at 0, ~45 and ~90 mm/s are integer voxel displacements per
frame (1 voxel/frame = 44.6 mm/s in-plane), not physiological flow modes — sub-voxel localization
is not smoothing them out. Blue (the shipped gate) stops dead at the 2-voxel line in both in-plane panels; orange reaches
the third lobe. The elevation panel is the control: we change nothing there, so the two curves
end at the same place and orange simply sits above blue because more tracks are published.

### Why the elevation gate is *not* a mistake — and why our first recommendation was

A gate has to cover two different things at once: how far a bubble can really move in a frame,
**and** how far our estimate of its position can be wrong. Voxel size is a reasonable proxy for
the second. Elevation voxels are 2.77× larger because elevation is resolved 2.77× worse, so a
gate specified in voxels automatically opens wider on the axis where localization is least
certain. That is the gate doing its second job, and it is defensible.

Our first version of this document called the anisotropy "backwards" and tightened elevation to
130 mm/s to match the in-plane axes. That was wrong, and the evidence against it was already in
`outputs/gate_sweep/` when the claim was made. Over 60 acquisitions, tracks ≥35 frames:

| gate (in-plane / elevation) | tracks ≥35 |
|---|---|
| 89 / 247 — Aleph default | 372 |
| 130 / 130 — our first recommendation | 397 |
| **130 / 247 — in-plane only** | **477** |

Tightening elevation gave away 80 of the 105 available long tracks. On the full dataset the same
choice costs 253 (1,786 against 1,531). Velocity units are the right way to express a *motion*
limit; they are the wrong way to express a *precision* limit, and elevation is dominated by the
latter.

### Why did Aleph use voxels?

The honest answer is the one above: **a voxel gate scales with per-axis localization
uncertainty, and on the elevation axis that is the dominant term.** Measured on this data, that
scaling is worth 253 long tracks over the physical-units alternative we first proposed. It is a
better default than it looks.

Two supporting facts, which are context rather than explanation. They built the physical-units
path themselves — `max_dist_mms` is in `_tracking_gate` from the earliest commit in the shared
history, simply not as the default. And a voxel rule always works, because grid spacing comes
from the beamformed file whereas a mm/s rule needs a frame rate; their commit `ae169ea` has now
made the frame rate available by carrying it through `/config`. But neither is a *justification*
for their own default — they wrote the pipeline and have always known their own frame rate. The
justification is the uncertainty scaling.

They reached the same conclusion in reverse for their *tissue* threshold in `ae169ea`, replacing
a 100 Hz boundary with `--tissue-velocity` in mm/s because "a frequency boundary only means
something alongside its carrier". Their new helper confirms the arithmetic —
`doppler_velocity_to_freq(40, 2e6, 1600)` is exactly 100.0 Hz, so their boundary was 40 mm/s all
along. Physical units are right for a threshold on *motion*. The gate is partly a threshold on
*precision*, which is why only its in-plane half should move.

### What we ship

**Raise the in-plane gate to 130 mm/s. Leave elevation exactly as Aleph had it.** One axis, not
three, via their own `max_dist_mms` — no new code.

**Effect: +335 tracks of ≥35 frames** (1,451 → 1,786, +23%) and +135 of ≥50 frames (582 → 716).

130 mm/s is not a physiological number; it is the knee of a yield-versus-false-link trade, past
which the gain flattens while chance links climb.

---

## Change 2 — publishing shorter tracks, with the error rate measured

![the purity curve](figures/change_length_purity.png)

### What the setting is

`min_track_length` is the last step before results are written: **any track shorter than N frames
is deleted from the output.** Aleph ships N = 15. We ship N = 10.

The single most important thing to understand about it: **it does not change tracking at all.**
It is applied only where finished tracks are written out (`tracking.py:343`, `:722`, `:729`) and
never enters the gate, the assignment cost, or the motion model. A 10-frame track exists inside
the tracker either way. At N = 15 it is built and then thrown away; at N = 10 it is built and
kept. Lowering it recovers no new associations — it only stops deleting.

So the question is not "does this find more bubbles" (it cannot). The question is: **of the
tracks we would newly publish, how many are real?**

### How we measured the error rate

A short track could be a real bubble seen briefly, or a coincidence — a few unrelated detections
that happened to line up. To tell them apart we built a control where *every* track is by
definition a coincidence:

> Take the same detections. Shuffle which frame each one belongs to. Run the identical tracker.
> Any track it now finds is pure chance, because the trajectories have been destroyed.

Run both, and at each candidate length N compare how many tracks the real data yields against how
many the shuffled data yields. That ratio is the false-track rate at that length. (For scale:
2.17M detections go into this.)

| if we publish tracks of ≥ N frames | tracks published | of those, coincidences |
|---|---|---|
| N = 2 | 333,722 | **76%** |
| N = 5 | 61,861 | 21% |
| N = 8 | 26,889 | 7.3% |
| **N = 10 (ours)** | **18,390** | **3.9%** |
| N = 15 (Aleph's) | 9,339 | 0.9% |
| N = 35 (the headline metric) | 1,786 | **0%** |

Two-frame tracks are mostly noise, as expected. By 10 frames the false rate has dropped below 4%
and the curve has flattened — going from 10 to 15 throws away half the remaining tracks to buy
back 3 percentage points. Note the last row: at 35 frames the chance count is *zero* out of
2.17M shuffled detections, which is why tracks ≥35 is the metric used against the reference.

This table is computed **at the gate we ship**. The floor has to be re-derived whenever the gate
changes: at the tighter gate we first proposed, the 95%-purity knee sat at 8, and carrying that
8 across to the looser gate would have shipped 92.7% purity while citing a curve that justified
96%.

We also checked that the extra tracks are not junk by two other routes: planted-crossing link
precision is flat (0.91–0.96) across the whole range, so the stricter floor buys no accuracy; and
the added short tracks lie along the same vessels as the long ones, where the shuffled control's
do not.

### What it costs and what it buys

- **2.5× more published trajectories** (7,296 → 18,390) and detections linked into tracks rising
  from 9.2% to 16.5%.
- **False-track rate 0.5% → 3.9%.** In absolute terms, about 712 of the 18,390 published tracks
  are expected to be coincidences, against 35 today.
- **The long-track count is untouched**, because a floor at 8 or 15 cannot affect a 35-frame
  track. That is why tracks ≥35 is the metric used for comparison against the reference.

### The two changes are not independent

A wider gate makes longer chains, so the floor's knee moves with it — measured, not assumed: at
the tight 130/130 gate the 95% knee was at 8; at the gate we actually ship it is at **10**. The
converse holds too: at floor 15 only 1.2% of the gate's link gain is reproduced by the shuffled
control, at 8 it is 11.8%, at 5 it is 28.6%. Neither setting means anything quoted alone.

---

## Confirmed rather than changed: clutter rank and normalization

`rank 24 + spatial TGC` survived a sweep against synthetic injected bubbles scored at matched
false-alarm rate, so we ship Aleph's setting. Worth recording *why* rank 24 is defensible: the
adaptive cutoff never fires on this data (no mode's spectral centroid reaches the 100 Hz
boundary; the maximum is 85–96 Hz), so it always falls back to "10% of frames" — accidental, and
also correct. The sweep's win was normalization, not rank; on one acquisition the data-driven
knee resolves to exactly 24, the status quo filter.

---

## What we tried and rejected

| candidate | verdict | why |
|---|---|---|
| Elevation grating lobes (row pitch is 2.08λ) | **refuted** | 1.93× excess at the predicted offset, but 6 of 27 *arbitrary* offsets beat it; the apex wanders with depth where a real lobe is fixed |
| Motion-phase hypothesis bank (compounding null) | **refuted at modelled strength** | the model predicts a 169:1 lateral:axial ratio in the top speed band where real data shows 1.53 — it would leave the data axially extinct |
| 888 Hz interleaved per-transmit movie | **abandoned** | the four steered volumes are mutually decorrelated in the raw field |
| Marchenko-Pastur high cutoff | **harmful** | strips 122 of 240 modes (right maths, wrong regime); costs 0.09–0.14 recovery |
| B18 spatial-correlation cutoff | **harmful** | returns 2 modes |
| `per_elev_zband` normalization | **rejected** | changes sign between acquisitions (+0.018 / −0.014) |
| Injector's slow-lateral blind spot, as a real-data claim | **rejected** | propagated through tracking it predicts a ratio of 0.01–0.04 where real data shows 0.98 |

Two of our own results were withdrawn as well: a coherence "refutation" that turned out to be
measuring tissue clutter rather than bubbles, and a "1.348× vs reference" figure produced by a
harness that silently inherited a 10,000× looser assignment cost than production uses.

---

## Net effect

![what the changes bought](figures/change_summary.png)

| | Aleph reference | our recreation | with our two changes |
|---|---|---|---|
| tracks ≥35 frames | 1,421 | 1,451 | **1,786** |
| tracks ≥50 frames | — | 582 | **716** |
| tracks at native floor | 50,456 (floor 5) | 7,296 (floor 15) | 18,390 (floor 10) |
| detections linked into tracks | — | 9.2% | **16.5%** |
| false-track rate vs shuffled control | never measured | 0.5% | 3.9% |

**No coincidence track anywhere reaches 35 frames** across 2.17M shuffled detections, at either
setting. The headline population is free of chance chains, so the +80 is not bought with
contamination.

## Honest scope

This is an improvement in **association and reporting**: more of the trajectories the pipeline
already builds, with the error rate measured rather than assumed. It is **not** more signal
extracted from the raw data — every upstream idea that promised that is in the rejected table.

One upstream lever remains live and unproven: per-angle coherence declines ~2.3 dB across the
measured speed range (post-clutter-filter median 0.85 against 0.50 for random phase — so our
detections are *not* speckle, and the coherent compound is buying real gain). That curve
currently bins by a velocity derived from the same phases it measures, so it needs re-binning
against independently tracked velocity before it can be claimed.
