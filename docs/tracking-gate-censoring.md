# The step-speed distribution is censored by our own tracking gate (J1, bead mb-5pd)

> **CORRECTION (2026-08-18).** This report's verdict recommended tightening elevation to
> 130 mm/s alongside the in-plane widening. That was wrong, and this document's own Tables
> section contains the evidence: at 130/247 (elevation left at the Aleph default) the sweep
> gives **477** tracks ≥35 against **397** at 130/130. Tightening elevation gave away 80 of the
> 105 available long tracks; on the full 216-acquisition dataset it costs 255. A voxel-sized
> gate scales with per-axis localization uncertainty, and elevation — synthesized from 8
> physical rows — is dominated by that term, so its wider gate is doing real work. The
> measurement below stands; only the elevation half of the recommendation is retracted. Shipped
> configuration and reasoning: `docs/from-baseline-to-improved.md`.


Reproduce with `python3 scripts/wf_render_signal/gate_sweep.py` (local, CPU, ~5 min at 4 workers).
Tables below are its output over batches 0000/0012/0024/0036/0048 of `outputs/corrected_full` —
**60 acquisitions, 603,254 cached detections**, retracked through the production path
(`ultratrace_ulm.tracking._track_detections`) with **nothing changed but the gate**.

## The measurement

The 216-acquisition run's per-step velocities wall off at exactly the default box gate. At
`(dx, dy, dz) = (0.20041, 0.55467, 0.20078)` mm and 222.43 Hz, two voxels per frame is

| axis | gate (mm/frame) | gate (mm/s) | observed max step |
|---|---|---|---|
| lateral (x) | 0.400818 | 89.15 | 89 |
| elevation (y) | 1.109334 | 246.75 | 246 |
| axial (z) | 0.401568 | 89.32 | 89 |

The distribution does not approach the gate, it *ends* at it. And the gate is **2.77x looser on
elevation** — the axis whose voxels are 2.77x coarser and whose localization is worst — which is
backwards.

**Baseline reproduction (checked before anything else, and by the script on every run):**
retracking the cached detections at the default gate reproduces `tracks_0000.pkl` track-for-track
(432 tracks, identical first-frame/first-x/length fingerprints), and the `max_dist_mms` path at
the equivalent mm/s numbers reproduces the spacing-derived gate identically. Both are PASS/abort
gates in `gate_sweep.py`.

## The confound, stated up front

The gate wall and the 4-angle coherent-compounding null are the same speed by algebra:
gate `= 2·(λ/4)·FR = λ·FR/2`; null `= λ·PRF/8` with `PRF = 4·FR`, also `λ·FR/2` — 88.97 mm/s.
They differ only in which axis they act on: the compounding null suppresses **axial** motion;
the box gate censors **all three** axes. So the axis composition of the recovered mass, not the
total, is the readout.

## Verdict

1. **The wall is the gate, and it lifts.** Widening in-plane to 180 mm/s puts 8.2% of all steps
   beyond the old wall, in 2,126 tracks whose median length (21) equals the population median —
   these are ordinary persistent tracks, not fragments. Linked detections rise 55,454 → 82,088
   (9.2% → 13.6%) and tracks ≥35 points rise 372 → 523.
2. **The recovery is lateral-dominant, and the axial deficit starts *below* the wall.** Below
   half the old gate, lateral:axial step counts are 0.95–0.98 (isotropic). Approaching the wall
   they climb to 1.20 (0.5–0.75) and 1.53 (0.75–1.0); above it they sit at 2.3. Speed-dependent,
   axis-specific, and not explicable by the gate itself (the x and z gates differ by 0.2%). This
   is what axial suppression by the compounding null looks like; it is consistent with it, not
   proof of it. Roughly 1,600 axial steps are "missing" against an isotropic expectation in the
   0.5–1.5-gate range — those the tracker can never recover, because detection never produced them.
3. **The elevation gate should come in, not stay out.** Tightening elevation 247 → 130 mm/s
   (~1 voxel/frame) costs ~11% of links while preserving the in-plane recovery; 90 mm/s is below
   one elevation voxel per frame, cuts 47% of baseline links, and is not interpretable as motion.
4. **How much of the unlinked pool is gate-stranded:** about **a third of it, and no more.** At
   the frozen gate 42.2% of detections never join even a 2-point chain; widening in-plane to 180
   drops that to 28.4%. The remaining 48–58% of detections *do* get chained and are then thrown
   away by `min_track_length=15`. The dominant term in the unlinked pool is the length filter,
   not the gate.
5. **Where it turns over:** not in link precision — planted-crossing precision is *higher* at
   110–180 (0.95–0.96 under a 50-crossing load) than at the frozen gate (0.93), with recall
   tripling — but in link stability on real data. Between 130/247 and 180/247 the fraction of
   baseline links retained falls 95.0% → 92.0% and lost links nearly double (2,772 → 4,402) for
   a 1.8-point linked gain. **130 in-plane / 130 elevation** is the defensible operating point:
   +6.6% links over baseline, 85% of baseline links kept, 1,542 recovered supra-wall steps in
   892 tracks, and the elevation gate finally tighter than a voxel of jitter.

Sim caveat: absolute recall in the planted benchmark is low (0.12 at the frozen gate) because the
simulated speed regime is read from the *linked* baseline tracks and so is itself gate-censored —
the precision *trend* across gates is the usable signal, not the level.

## Tables

### Sweep — pooled over 60 acquisitions

| gate in-plane/elev (mm/s) | tracks | linked | linked% | med len | >=35 | med step (mm/s) | p99 | max lat | max ax | max elev | supra-gate steps | in tracks | med len of those | in tracks >=35 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| default (2 vox) | 2118 | 55454 | 9.2% | 21 | 372 | 47.5 | 167 | 89 | 89 | 246 | 0 (0.0%) | 0 | 0 | 0% |
| 89 / 247 | 2106 | 55193 | 9.1% | 21 | 370 | 47.5 | 167 | 89 | 89 | 247 | 2 (0.0%) | 2 | 44 | 50% |
| 89 / 130 | 1855 | 49105 | 8.1% | 22 | 341 | 45.3 | 135 | 89 | 89 | 130 | 0 (0.0%) | 0 | 0 | 0% |
| 89 / 90 | 1191 | 30403 | 5.0% | 21 | 208 | 38.9 | 106 | 89 | 89 | 90 | 0 (0.0%) | 0 | 0 | 0% |
| 110 / 247 | 2514 | 65588 | 10.9% | 21 | 441 | 50.8 | 189 | 110 | 109 | 247 | 1468 (2.3%) | 937 | 21 | 17% |
| 110 / 130 | 2113 | 55981 | 9.3% | 22 | 381 | 47.3 | 141 | 108 | 105 | 130 | 911 (1.7%) | 623 | 22 | 18% |
| 110 / 90 | 1310 | 33198 | 5.5% | 20 | 222 | 40.0 | 113 | 104 | 103 | 90 | 319 (1.0%) | 230 | 20 | 12% |
| 130 / 247 | 2707 | 70943 | 11.8% | 21 | 477 | 53.1 | 207 | 130 | 130 | 247 | 2747 (4.0%) | 1368 | 21 | 16% |
| 130 / 130 | 2225 | 59130 | 9.8% | 22 | 397 | 48.5 | 149 | 130 | 130 | 130 | 1542 (2.7%) | 892 | 22 | 16% |
| 130 / 90 | 1371 | 34530 | 5.7% | 21 | 222 | 40.7 | 121 | 130 | 129 | 90 | 511 (1.5%) | 329 | 20 | 11% |
| 180 / 247 | 3182 | 82088 | 13.6% | 21 | 523 | 58.2 | 236 | 180 | 180 | 247 | 6454 (8.2%) | 2126 | 21 | 14% |
| 180 / 130 | 2514 | 65755 | 10.9% | 21 | 423 | 51.1 | 179 | 180 | 180 | 130 | 3451 (5.5%) | 1374 | 21 | 14% |
| 180 / 90 | 1463 | 36744 | 6.1% | 20 | 235 | 42.0 | 146 | 180 | 180 | 90 | 1062 (3.0%) | 520 | 20 | 12% |

`supra-gate steps` = steps whose per-frame displacement exceeds what the OLD gate allowed on some
axis, i.e. steps that could not have existed in the 216-acq run. The elevation column of the next
table is structurally ~0 whenever the elevation gate is tightened — nothing can exceed a bound
that only moved inward.

### Axis composition of the recovered mass

| gate | supra steps | lateral | axial | elevation | lateral:axial |
|---|---|---|---|---|---|
| 110 / 247 | 1468 | 1047 | 453 | 2 | 2.31 |
| 110 / 130 | 911 | 669 | 259 | 0 | 2.58 |
| 110 / 90 | 319 | 256 | 68 | 0 | 3.76 |
| 130 / 247 | 2747 | 1913 | 968 | 3 | 1.98 |
| 130 / 130 | 1542 | 1127 | 487 | 0 | 2.31 |
| 130 / 90 | 511 | 411 | 120 | 0 | 3.42 |
| 180 / 247 | 6454 | 4671 | 2560 | 3 | 1.82 |
| 180 / 130 | 3451 | 2628 | 1133 | 0 | 2.32 |
| 180 / 90 | 1062 | 855 | 306 | 0 | 2.79 |

### Approach to the wall (the control that makes the above interpretable)

Bands are fractions of the OLD gate. If in-plane motion were already lateral-dominant below the
wall, a lateral-dominant recovery above it would mean nothing.

| band (x old gate) | lateral | axial | lateral:axial | (gate) |
|---|---|---|---|---|
| 0.00–0.25 | 30143 | 30861 | 0.98 | default |
| 0.25–0.50 | 16285 | 17149 | 0.95 | default |
| 0.50–0.75 | 4557 | 3792 | 1.20 | default |
| 0.75–1.00 | 2351 | 1534 | 1.53 | default |
| 0.00–0.25 | 32715 | 34419 | 0.95 | 180/130 |
| 0.25–0.50 | 18164 | 20026 | 0.91 | 180/130 |
| 0.50–0.75 | 6295 | 5307 | 1.19 | 180/130 |
| 0.75–1.00 | 3439 | 2356 | 1.46 | 180/130 |
| 1.00–1.50 | 1981 | 848 | 2.34 | 180/130 |
| 1.50–2.00 | 608 | 271 | 2.24 | 180/130 |

### Link attribution vs the frozen baseline

| gate | newly linked dets | lost | baseline links kept | unlinked pool |
|---|---|---|---|---|
| 89 / 247 | 139 | 400 | 99.3% | 90.9% |
| 89 / 130 | 1402 | 7751 | 86.0% | 91.9% |
| 89 / 90 | 1426 | 26477 | 52.3% | 95.0% |
| 110 / 247 | 12135 | 2001 | 96.4% | 89.1% |
| 110 / 130 | 8531 | 8004 | 85.5% | 90.7% |
| 110 / 90 | 3685 | 25941 | 53.2% | 94.5% |
| 130 / 247 | 18261 | 2772 | 95.0% | 88.2% |
| 130 / 130 | 11923 | 8247 | 85.1% | 90.2% |
| 130 / 90 | 4821 | 25745 | 53.6% | 94.3% |
| 180 / 247 | 31036 | 4402 | 92.0% | 86.4% |
| 180 / 130 | 18831 | 8530 | 84.6% | 89.1% |
| 180 / 90 | 7159 | 25869 | 53.3% | 93.9% |

### What the unlinked pool is made of

`min_track_length=2` keeps every chain the association step formed, so the split is between
detections the gate/cost never linked at all and detections linked into chains shorter than the
production minimum of 15.

| gate | detections | never chained (gate-limited) | chained but pruned (<15) | kept |
|---|---|---|---|---|
| default | 603254 | 42.2% | 48.6% | 9.2% |
| 130 / 247 | 603254 | 35.3% | 52.9% | 11.8% |
| 180 / 247 | 603254 | 28.4% | 58.0% | 13.6% |
| 180 / 130 | 603254 | 34.4% | 54.7% | 10.9% |

### Planted-crossing link precision (simulation, production tracker)

Harness from `track_sim_validate.py` (identity-restricted edge scoring, clutter held fixed so
crossings add rather than displace), driven through the production Kalman/Hungarian tracker and
calibrated to the corrected_full regime rather than the reference pickle.

| gate | crossings | precision | recall |
|---|---|---|---|
| default | 0 | 0.949 | 0.122 |
| default | 50 | 0.933 | 0.131 |
| 110 / 130 | 0 | 0.988 | 0.226 |
| 110 / 130 | 50 | 0.951 | 0.244 |
| 130 / 130 | 0 | 0.988 | 0.288 |
| 130 / 130 | 50 | 0.956 | 0.312 |
| 180 / 130 | 0 | 0.985 | 0.386 |
| 180 / 130 | 50 | 0.956 | 0.412 |

## What this does not settle

No ground truth exists for these acquisitions. Reference-matching is not validation, and held-out
position prediction is beaten by tracker-free interpolation, so neither is used here. The claims
above are: the wall is our gate (proved by construction — the retrack is byte-identical and the
wall moves with the parameter); the recovered mass is lateral-dominant with a speed-dependent
axial deficit (measured); and the pool split between gate and length filter (measured). Whether
the recovered links are *correct* is bounded only by the planted-crossing simulation, whose
speed regime is inherited from gate-censored tracks — which is exactly the circularity that
motivates re-running it once the gate moves.
