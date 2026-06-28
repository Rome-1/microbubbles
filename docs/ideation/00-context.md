# Ideation brief — improving the microbubbles 3D ULM pipeline

Shared context for a cross-model (Opus + Codex) ideation round. Goal: converge on
**10 improvement ideas both models agree are worth trying**, then build + run them.
Everything is measured against a **baseline** (what Aleph shipped) on a single dataset.

## The pipeline (what exists)
`beamform (MACH delay-and-sum) → temporal SVD clutter filter → 3D z-score peak
detect + sub-voxel localize → Kalman+Hungarian track linking → smooth → 3D viewers`.
Repo: `ultratrace_ulm/`. Pipeline runs on Modal (GPU A10G 24GB). Reads a single
sanitized HDF5 (demodulated complex IQ).

## The data (FIXED — we cannot get more)
- One ~4-minute human scan: **223 acquisitions × ~700 frames @ 222 Hz** (demodulated complex IQ).
- Per-acq beamformed volume: **(frames=700, elev=25, z=225, x=378) complex64 ≈ 11.9 GB.**
- Acquisition raw IQ: `(702 loops, 5 transmit angles, 1 row, 134 cols, 256 time)`.
  **Only 5 transmit angles; a 1×134 receive aperture; 25 elevation planes are SYNTHESIZED
  (not physical rows).** Single speed of sound (1600 m/s). f# 0.5, tx 2.75 MHz.

## Hard constraints
- **No ground truth.** All metrics are PROXIES (see below). Cannot claim "correct," only "better/worse on a proxy."
- **One dataset.** Ideas needing more data / other modalities / CT are out of scope.
- **GPU memory:** A10G 24GB. The (700, 2.13M-voxel) matrix is 11.9 GB; never hold two copies.
  Patterns that work: chunk over voxels/frames, accumulate Gram in complex128 (chunks stay complex64),
  `free_all_blocks()` between iterations, `del` loop-views. (These cost real failed runs.)
- **Network:** R2 drops large transfers — data now lives on the Modal volume (local reads). Fixed.
- **Budget:** generous but finite. Bake-off ideas at 60-acq scale (~$2.5 each); winners at full 223.

## Evaluation = track-level metrics (the real arbiter)
`scripts/bench.py` on the track `.bin`. KEY metrics (no ground truth):
- **fragmentation** (short/long track ratio) — LOWER better
- **mean curvilinear track length (mm)** — HIGHER better
- **frac of tracks ≥10/20/35/50 frames** — HIGHER better (continuity)
- **straightness** (end-to-end / curvilinear) — higher ~better per short bubble pass
- **occupied_fraction / peak_density / contrast (CNR proxy)** — coverage & render quality
- For "beauty": longer, less-fragmented, well-distributed tracks render as cleaner vessels in
  the Three.js `track-viewer` / `volume-viewer` (not yet built for full data).

**LESSON LEARNED: power-volume diffs are MISLEADING for clutter changes — track-level metrics
are the arbiter.** (Motion correction looked great in the power volume but fragmented tracks.)

## What we've already TRIED + RESULTS (60-acq, vs baseline)
| Change | Verdict | Evidence |
|---|---|---|
| Phase-invariant adaptive SVD cutoff | ✅ correctness | Aleph's `|rfft(eigvec.real)|²` cutoff was phase-dependent (nondeterministic, CPU≠GPU 15×). Fixed to phase-invariant `|fft(complex eigvec)|²`. |
| Predicted-state Kalman gate | ✅ | gate on Kalman prediction not last-obs: mean curv-length **+43%**, less fragmentation |
| Region-adaptive SVD (per-block local subspace, global rank, PoU blend) | ✅ continuity | fragmentation **−47%**, tracks≥35 **+95%**, tracks≥20 +79%. TRADE-OFF: occupied_fraction **−27%**, peak_density −79% (removes peripheral + hot-spot clutter; may also remove weak real bubbles) |
| Motion correction (per-acq rigid 3D from tissue B-mode) | ❌ | de-clutters power volume but **fragments tracks +199%**, loses all ≥50-frame tracks. Excluded. |
| **Composite = region-SVD + Kalman gate** | 🏆 best so far | mean track length **+50% (0.59→0.89 mm)**, tracks≥35 +79%, fragmentation −39%, straighter; trade-off: coverage/density lower |

## Open problems / promising directions (rank + extend these)
1. **Coverage vs continuity trade-off** — region-SVD improves continuity but loses ~27% coverage.
   How to keep continuity AND recover coverage? (adaptive block sizing, gentler/region-specific cutoffs,
   robust PCA tissue+sparse-bubble, learned clutter filter, depth-varying cutoff without hard blocks…)
2. **Tracks are SHORT** (median ~6 frames; few ≥35). Bubbles cross the FOV fast or links break.
   Better association (multi-frame min-cost-flow, motion priors, intensity/PSF similarity, track
   stitching), or tracking-before-localization, or longer temporal modeling.
3. **Detection is crude** — z-score peaks + centroid/parabola sub-voxel; Gaussian smoothing in z/x
   NOT elevation; isotropic NMS on anisotropic voxels; per-elev-slice noise stats. PSF-aware /
   matched-filter / CFAR / 3D ML localization; localization covariance carried into tracking.
4. **Overlapping bubbles** — peak detector assumes sparsity. Sparse deconvolution, multi-emitter
   fitting, tracking-before-localization → could shorten required acquisition / increase yield.
5. **Elevation is synthesized from 1 physical row** (25 planes). Is elevation localization reliable?
   Anisotropic handling; maybe down-weight elevation; the bright-midplane artifact we saw.
6. **Aberration / single speed-of-sound** — ultrasound-derived SoS, coherence-based local
   phase-aberration (no CT). Our data IS angle×element (reflection-matrix-like) but only 5 angles.
7. **Beamforming** — only delay-and-sum; coherence factor / MV weighting / apodization tuning could
   sharpen the PSF cheaply.
8. **TGC / normalization**, **SVD rank selection** (the cutoff floors to 10% here), **temporal
   filtering**, **knee filter** tuning.
9. **Vessel-graph / flow priors**, **velocity-field regularization** (longer-term).

## What we want from ideation
Rank ideas by **(expected gain on the track-level metrics, especially recovering COVERAGE without
losing CONTINUITY, and increasing track LENGTH) × (feasibility on ONE dataset, no ground truth, A10G,
budget)**. Favor cheap-to-try, high-signal changes. Be concrete: which stage, what to implement, what
metric should move, how to validate without ground truth, and any GPU-memory risk. Critique the
existing results too (e.g. is region-SVD's coverage loss acceptable? is the short-track problem
detection or tracking?).
