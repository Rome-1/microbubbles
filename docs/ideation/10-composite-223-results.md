# Composite scale-up to 223 — results (session 3, 2026-06-29)

Goal: scale the session-2 recipe (detection `svd_method=knee` + tracking `g3r10+max_gap6`)
to the full 223 acquisitions, FRC/saturation-arbitrate, and build the baseline-vs-improved
render. Budget: Rome approved up to $100; ~$28 spent this session (mostly the GPU re-detect).

## HEADLINE (clean, validated, shippable): the elevation-anisotropic Kalman tracking win scales 60→223
`base223_best` = base60's 223 **adaptive** detection checkpoints, tracked with
`elev_gate_factor=3, elev_meas_factor=10, max_gap=6` (prediction OFF, intensity OFF), vs the
`base223` baseline (same detections, default isotropic tracking, max_gap=3). **Same detections —
a clean tracking-only isolation.** Cross-validated by TWO independent no-GT arbiters:

| arbiter | metric | base223 | base223_best | verdict |
|---|---|---|---|---|
| **FRC** (split-half acq-parity) | repro_score | 0.7295 | **0.7732** | +6.0% more reproducible |
| | xz_res_mm | 0.792 | 0.780 | finer |
| | fsc3d_res_mm | 0.335 | 0.321 | +4.2% finer |
| **saturation** (Hingot-2019, `bench.py`) | late/early_slope | 0.284 | **0.188** | more plateaued = more real |
| | acqs_to_90% | 136 | 108 | fills the bed faster |
| | linear_r2 | 0.878 | 0.820 | less clutter-like |
| | ceiling_frac | 0.565 | 0.657 | higher asymptotic coverage |
| track quality | frag_short/long | 162 | 99 | −39% |
| | straightness | 0.857 | 0.898 | +4.8% |
| | mean_curv_len_mm | 0.585 | **1.891** | +223% |
| | frac_len≥50 | 7.1e-5 | 1.7e-4 | +142% |

n_tracks 168k→254k, but BOTH arbiters confirm the extra tracks are **real recovery, not clutter**
(FRC repro up + saturation plateaus better). The +223% curvilinear length confirms the elevation
anisotropy applied (gap6 alone can't produce that). **This is the baseline-vs-headline result.**
Renders built: `outputs/render/baseline` (168k tr) and `outputs/render/composite_tracking` (254k tr).

## NEGATIVE RESULT: the knee *detection* is clutter-flooded — abandon as-is
Re-detected the knee SVD cutoff on all 223 acqs (`swp_knee`, ~$13 GPU). Tracking it ran away
(~3.4 h, killed) because the knee detection emits **~490k localizations/acq (~700/frame)** vs
adaptive's **23.6k/acq (~33/frame)** — ~21× denser, clutter-saturated (real ULM is tens/frame).

The session-2 "knee beats the floor (+4.3% repro)" FRC result was the **static-tissue confound the
literature flagged**: keeping low-order tissue modes adds detections that reproduce across odd/even
acq halves (tissue is static) and inflate FRC repro without being microvasculature.

### Key sub-finding: density is INSENSITIVE to SVD mode count (it's a detection-path bug, not a cutoff)
Sampled the existing fixed-cutoff sweep (all 41-acq):

| tag | modes removed | mean loc/acq | loc/frame |
|---|---|---|---|
| swp_fix8 | 8 | 453k | 647 |
| swp_fix17 | 17 | 483k | 689 |
| swp_fix30 | 30 | 510k | 728 |
| swp_fix70 | 70 | 487k | 695 |
| swp_kneeH | knee+MP | 551k | 787 |
| swp_knee | knee~17 | 490k | 699 |
| **base60 (adaptive)** | **70** | **23.6k** | **33** |

`swp_fix70` removes the **same 70 modes** as `adaptive` yet has **20× the density**. So "re-detect
the knee with more modes removed" CANNOT fix the flood — the lever isn't the SVD cutoff. Every
`detect_acqs_multi` variant (`fast`/`knee` filter, identical 2σ z-score detector) floods; the
`detect_acqs` `adaptive` path stays sane. This is a **magnitude-scaling / threshold mismatch between
the two detection code paths** (`filtered_magnitude_gpu` method `fast`/`knee` vs `adaptive`, feeding
the same 2σ detector → 20× different counts). Beaded separately. Until fixed, the whole `swp_*`
detection sweep is in a flooded regime and its inter-variant FRC rankings are tissue-confounded.

## Recommendation
Ship `base223_best` (the tracking win) as the headline composite — it is clean, dual-arbiter
validated, and rendered. Treat the knee/coverage *detection* lever as blocked on the detection-path
scaling bug (worth fixing, but it's an investigation, not a knob-turn). Detect long-tail perf issue:
see `09-detection-longtail.md` (bead mb-d17).
