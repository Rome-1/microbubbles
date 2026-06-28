# Opus round-1 ideation — improving the 3D ULM pipeline

Author: Opus 4.8. Scope: single fixed dataset, no ground truth, A10G 24 GB,
track-level proxy metrics in `ultratrace_ulm/bench.py`. Ranked by
`(expected gain on track metrics — coverage-without-losing-continuity and track
length) × (cheapness / single-dataset feasibility)`.

I read the brief plus `svd.py`, `svd_region.py`, `gpu_svd.py`,
`gpu_svd_region.py`, `tracking.py`, `gpu_detect.py`, `beamform_core.py`, and
`bench.py`. Findings that shape the ranking:

- **Detection z-scoring is per-elevation-plane but GLOBAL within a plane**
  (`_slice_stats` / `detect_batch_gpu`): one mean/std per elev plane over all
  positive voxels across all 700 frames. A single threshold over the whole plane
  systematically under-detects in dim/peripheral regions and over-detects in
  bright ones. This is a prime coverage lever and it is independent of the SVD.
- **NMS is isotropic** (`maximum_filter size=(1, fs, fs, fs)`) on a volume whose
  elevation PSF is enormously wider than z/x (1 physical receive row — see the
  elevation critique). Isotropic NMS both double-counts elevation-smeared lobes
  and over-suppresses true neighbours in z/x.
- **Localization (`subpixel_localize_3d`) treats elevation like z/x** and feeds a
  full-confidence (x, y, z) to the tracker; the Kalman R is isotropic. Elevation
  is the least trustworthy axis yet it gets equal vote → curls/breaks tracks.
- **Association is greedy frame-to-frame Hungarian with no appearance term, no
  post-hoc stitching, `max_gap=3`.** Intensities are carried but unused in cost.
  This is where most of the short-track problem lives (see critique 2).
- **Spatial TGC is already applied at beamform** (`compute_global_tgc`,
  `spatial_tgc=True` in `scripts/modal/app.py`), so global depth gain is handled;
  the remaining coverage problem is *local* (within-plane) equalization, not
  global TGC.

## Cross-cutting no-ground-truth validators (used by the ideas below)

Two proxies beyond `bench.py` that every idea should report, because "more
tracks" is ambiguous (clutter vs signal):

1. **Split-half render reproducibility (FRC-like).** Build the coronal track-
   density render from odd vs even acquisitions (the 223 acqs are independent
   ~4 s windows of the same vasculature). Compute their correlation / SSIM.
   A *real* improvement makes the two independent renders agree MORE (the same
   vessels reappear); clutter makes them agree less. This is the single most
   trustworthy no-GT arbiter we have and it costs nothing extra to compute.
2. **Detections-per-frame and inter-detection-persistence histograms.** Cheap to
   log in `detect_localize_acq`; lets us see whether a change adds *persistent*
   detections (good) or one-frame flickers (clutter).

Report these alongside `frac_len_ge_{20,35,50}`, `frag_short_over_long`,
`straightness`, `mean_curvilinear_len_mm`, `occupied_fraction`,
`contrast_cnr_proxy` for every bake-off.

---

## Critiques of the existing results (asked for explicitly)

### Is region-SVD's −27% coverage loss recoverable? — YES, mostly.
The loss is a *parameterization* artifact, not fundamental. `global_rank` removes
the **same k everywhere**, where k is the global spectral-centroid cutoff
dominated by the bright central tissue. Peripheral/skull-shadowed blocks have
**lower-rank tissue**, so applying the central (larger) k over-projects them and
takes weak bubble energy with it → peripheral coverage dies (occupied_fraction
−27%, peak_density −79%). `per_block` recovers that coverage (smaller k where
tissue rank is smaller) but reintroduced **intensity banding** (different residual
energy per block) which is why it was abandoned. The banding is an *intensity*
problem, not a subspace problem, and it is separately fixable (renormalize block
energy, or — better — make detection locally background-relative so absolute
block scale stops mattering; idea #3). So the sweet spot exists: **local k +
remove the banding by other means.** Ideas #2 and #6 target this directly.
Verdict: recover ~half-to-most of the coverage while keeping ≥80–90% of the
continuity gain is realistic.

### Is the short-track problem (median ~6 frames) detection or tracking? — BOTH, but tracking dominates the *recoverable* part.
Evidence it's tracking-limited: the two biggest wins so far were both
association/clutter, not detection (Kalman-prediction gate +43% length; region-
SVD continuity). Greedy frame-to-frame Hungarian with `max_gap=3`, no appearance
cue, and **no post-hoc stitching** guarantees that any 4-frame detection dropout
permanently splits a real path into two sub-min-length fragments that get
discarded. That is cheap to fix (ideas #1, #5, #7, #8) and should be tried first.
Evidence it's *also* detection-limited: independent per-frame z-score peaks with
a global-within-plane threshold miss weak bubbles intermittently, and isotropic
NMS on a wide-elevation PSF flickers detections — both inject the dropouts that
break tracks (ideas #3, #4, #9). There is **also a hard physical floor**: the
imaged elevation slab is thin (25 synthesized planes over a few-mm y-span), so a
bubble with any out-of-plane velocity genuinely transits the FOV in few frames.
You cannot beat that floor, which is the strongest argument for treating
elevation as a low-confidence axis rather than chasing length there.

### Is elevation (25 planes from 1 physical receive row) trustworthy? — NO, and the bright midplane is a real bias to manage.
With `num_rows == 1`, the receive aperture has **zero elevation extent**
(`y_rows = [0]` in `beamform_iq`), so there is **no receive-side elevation
focusing at all**. Elevation discrimination comes only from the transmit
elevation delays and the f-number apodization → the elevation PSF is vastly wider
than z/x and the 25 planes are heavily correlated (synthesized, not resolved). A
point scatterer's energy therefore smears across elevation and **piles up near
the steered midplane (y≈0)** — exactly the bright-midplane artifact observed.
Consequences to act on: (a) elevation sub-voxel position is near-meaningless →
down-weight it in tracking (idea #5); (b) NMS must have a large elevation radius
or it counts one bubble several times along elevation (idea #4); (c) the midplane
bias should be normalized out before detection so it doesn't manufacture a sheet
of midplane detections (idea #4/#12). The per-plane z-score in `_slice_stats`
already removes the midplane *mean*, which helps, but not the wider PSF / variance
structure. Net: trust z and x; treat y as a coarse, ~±1-plane attribute.

---

## Ranked ideas

### 1. Post-hoc track stitching (fragment gap-bridging pass) — Tracking
**(a) Stage:** tracking, after `kalman_tracking_3d`, before `_smooth_tracks`.
**(b) Implementation:** add `stitch_tracks(tracks, opts)` operating on the
returned track list (so it iterates with zero GPU cost on the saved pickle).
For every track build (start frame/pos, end frame/pos, end velocity, mean
log-intensity). Form candidate links end_i → start_j where
`0 < frame_j_start − frame_i_end ≤ max_stitch_gap` (try 8–15, i.e. ~2–4× the
tracking `max_gap`), the gap-extrapolated endpoint
`pos_end_i + v_i·Δframe` lands within an anisotropic tolerance of `start_j`
(tight in z/x, loose in elevation), velocity directions agree
(cos > 0.5), and intensities match (|Δlog I| small). Solve one global
`linear_sum_assignment` over the link-cost matrix; merge chains transitively;
re-run the existing smoothing on merged tracks.
**(c) Metric & why:** `mean_curvilinear_len_mm` ↑, `frac_len_ge_{35,50}` ↑,
`frag_short_over_long` ↓. Directly reconnects the fragments that detection
dropouts and the `max_gap=3` cap currently sever — the cheapest attack on the
median-6-frame problem.
**(d) No-GT validation:** bench deltas + split-half reproducibility must hold or
improve (stitching real fragments should keep renders consistent; spurious
merges will *lower* straightness and split-half agreement — a built-in guard).
Sanity-cap: a correct stitch should rarely increase total track *points*, only
re-bin them into longer tracks.
**(e) GPU/feasibility:** none — pure CPU on the track list; trivially cheap to
sweep `max_stitch_gap`/tolerances. No re-beamform.
**(f) Effort:** S.

### 2. Spatially-smoothed / clamped per-block SVD rank — SVD (clutter)
**(a) Stage:** clutter filter (`svd_region.filter_svd_3d_region` + GPU mirror).
**(b) Implementation:** new `cutoff_mode="smoothed_rank"`. Compute the per-block
adaptive spectral-centroid k (the existing `per_block` path / `region_block_cutoffs`),
then **regularize the (n_z_blocks × n_x_blocks) k-map** before applying: clamp
each block to `median_k ± Δ` (Δ=1–2) and/or Gaussian-smooth-and-round the map.
Apply each block's regularized k as a fixed `n_components` (local subspace, local-
but-smooth count). This keeps continuity (near-uniform removal → no harsh banding)
while letting low-rank peripheral blocks keep their weak bubbles (recovered
coverage). Sweep Δ from 0 (=global_rank) to ∞ (=per_block) — it's a single knob
spanning both prior runs.
**(c) Metric & why:** `occupied_fraction` / `peak_density` recover toward baseline
while `frac_len_ge_35` and `frag_short_over_long` stay near the region-SVD
composite. This is the direct dial for the coverage-vs-continuity trade.
**(d) No-GT validation:** bench table at several Δ; pick the knee where coverage
recovers without `frac_len_ge_35` regressing. Split-half reproducibility should
*rise* if the recovered peripheral detections are real vessels.
**(e) GPU/feasibility:** identical memory profile to existing region-SVD (one
block on-card at a time); only adds a tiny per-block cutoff computation. Safe.
**(f) Effort:** S–M.

### 3. CFAR / local-background z-score detection — Detection
**(a) Stage:** detection (`detect_batch`, `gpu_detect.detect_batch_gpu`).
**(b) Implementation:** replace the global-within-plane mean/std (`_slice_stats`)
with a **local background**: estimate per-voxel local mean/std via a large-kernel
spatial filter on the time-averaged magnitude (e.g. `uniform_filter` /
`gaussian_filter` with σ ≈ 8–15 voxels in z/x, plus a guard band), then
`z = (smoothed − local_mean) / (local_std + ε)`. This is a 2-parameter CFAR. It
equalizes detectability across the FOV so one global `sigma_threshold` finds
weak peripheral bubbles AND stops over-firing in bright cores.
**(c) Metric & why:** `occupied_fraction` ↑ (peripheral coverage) without
lowering the global threshold; should also lengthen tracks by making detection of
a moving bubble *uniform* along its path (fewer dropouts where it passes through a
dimmer region). **Synergy: makes per_block region-SVD viable again** because a
local-relative detector is immune to per-block intensity banding — so #2 can use
larger Δ / true per_block and recover even more coverage.
**(d) No-GT validation:** detections-per-frame map should fill peripheral vessels
that were previously empty; split-half reproducibility must hold (CFAR can also
amplify noise in truly empty regions — watch `contrast_cnr_proxy` and the
straightness of the resulting tracks).
**(e) GPU/feasibility:** `cupyx.scipy.ndimage.uniform_filter` over the smoothed
volume — same memory class as the existing GPU smoothing (~6 GB), well within
24 GB. Compute the background on the time-mean (one volume) to keep it cheap.
**(f) Effort:** M.

### 4. Anisotropic, elevation-aware NMS + midplane de-biasing — Detection
**(a) Stage:** detection (`detect_batch*`).
**(b) Implementation:** two cheap changes. (i) Make NMS anisotropic:
`maximum_filter size=(1, fe, fz, fx)` with `fe` (elevation) set to roughly the
elevation PSF width (start ~5–7 planes) and `fz=fx=2·min_distance+1`. This
collapses the elevation-smeared replicas of one bubble into a single detection
and stops over-suppressing true z/x neighbours. (ii) Before z-scoring, divide
each frame by a smooth **elevation-bias profile** (the per-plane time-mean of the
magnitude, lightly smoothed across planes) to flatten the bright midplane so it
stops generating a sheet of midplane peaks.
**(c) Metric & why:** fewer duplicate/flicker detections → `frag_short_over_long`
↓, longer tracks; removing the midplane sheet raises `contrast_cnr_proxy` and
density `entropy_ratio` (less smeared). Frees real z/x neighbours → coverage.
**(d) No-GT validation:** count of detections clustered along elevation at fixed
(z,x,frame) should drop sharply; split-half render of the *coronal* (z,x) plane
should sharpen. Confirm `mean_speed` doesn't collapse (a sign of merged distinct
bubbles).
**(e) GPU/feasibility:** trivial — just kernel-size and a per-plane normalization
vector. No memory change.
**(f) Effort:** S.

### 5. Anisotropic, elevation-inflated localization covariance → Kalman R + gate — Localization/Tracking
**(a) Stage:** localization → tracking handoff (`detect_localize_acq`,
`kalman_tracking_3d`).
**(b) Implementation:** stop treating (x, y=elev, z) as equal-confidence. (i) In
the gate, widen elevation tolerance and tighten z/x (`max_dist` already a 3-vector
— set y much larger). (ii) In the Kalman update, make `R` anisotropic with
`R_yy` inflated 5–10× (elevation measurement is coarse). Optionally derive a
per-detection z/x covariance from the local peak curvature (parabolic fit already
computed) and pass it through, but the constant elevation inflation captures
90% of the benefit. Net effect: tracks follow the trustworthy z/x motion and
treat elevation as a soft attribute, so elevation jitter stops curling/breaking
paths.
**(c) Metric & why:** `straightness` ↑, `mean_curvilinear_len_mm` ↑,
`frac_len_ge_*` ↑ — directly removes elevation-noise-driven track breaks and the
spurious 3D wander that shortens curvilinear length.
**(d) No-GT validation:** bench deltas; the elevation-marginal track wander
(std of y residual about a smoothed track) should drop; split-half stability of
the coronal render should hold (z/x is what we trust, so this shouldn't move
coverage much — purely a continuity/quality play).
**(e) GPU/feasibility:** none — small CPU changes in the Kalman matrices.
**(f) Effort:** S–M.

### 6. Soft / shrinkage SVD subspace removal (robust-PCA-lite) — SVD (clutter)
**(a) Stage:** clutter filter (`filter_svd_3d*`).
**(b) Implementation:** replace the hard "project out components 0..k" with a
**soft per-component weight** `w_i ∈ [0,1]`. Instead of zeroing the leading k
temporal modes, attenuate them by a shrinkage factor that depends on each mode's
"tissue-ness" — e.g. the spectral centroid of mode i relative to `tissue_freq_hz`
(modes straddling the tissue/blood boundary get partially retained, not deleted).
Concretely: `filtered = U diag(w) Uᴴ M`-style, with `w_i` a sigmoid on
`(centroid_i − tissue_freq)`. Boundary modes that carry mixed tissue+weak-bubble
energy are no longer thrown away wholesale → weak bubbles survive.
**(c) Metric & why:** recovers `occupied_fraction`/`peak_density` (the boundary
modes are exactly where peripheral weak-bubble energy hides) while keeping
continuity, because confidently-tissue modes are still fully removed.
**(d) No-GT validation:** sweep the sigmoid sharpness from hard (=current) to
soft; track the coverage-vs-`frac_len_ge_35` frontier and split-half
reproducibility, same protocol as #2. #2 and #6 are alternative routes to the
same goal — bake both, keep the better frontier.
**(e) GPU/feasibility:** same memory as `filter_svd_3d_gpu` (the projection is
still `U diag(w) Uᴴ M` in chunks); `diag(w)` is free. Safe.
**(f) Effort:** M.

### 7. Appearance (intensity / PSF-width) term in association cost — Tracking
**(a) Stage:** tracking (`kalman_tracking_3d` cost assembly).
**(b) Implementation:** intensities are already carried per detection but never
used in the cost. Add `λ·|log I_det − log Ī_track|` (and, if available, a PSF-
width-similarity term) to `pair_costs` next to Mahalanobis + momentum. A bubble
keeps a roughly consistent brightness frame-to-frame, so in crowded regions the
tracker stops swapping onto the wrong (brighter/dimmer) neighbour.
**(c) Metric & why:** fewer ID switches → longer continuous tracks
(`frac_len_ge_*` ↑) and straighter paths (`straightness` ↑); especially helps
where vessels are dense and the geometric gate alone is ambiguous.
**(d) No-GT validation:** bench deltas; ID-switch proxy = count of large
intensity jumps within a track should drop. Split-half stability holds.
**(e) GPU/feasibility:** none — adds one vectorized term to the existing cost.
**(f) Effort:** S.

### 8. Larger `max_gap` with strict predicted + velocity-consistent gating — Tracking
**(a) Stage:** tracking.
**(b) Implementation:** raise `max_gap` 3 → 6–8, but only because the
prediction-gate (mb-crr.10) is on: a bridged detection must lie near the
Kalman-*predicted* position (x + Δt·v) AND keep velocity direction (reuse the
momentum/reversal machinery). This re-acquires bubbles after short detection
dropouts that currently kill the track.
**(c) Metric & why:** `mean_curvilinear_len_mm` ↑, `frac_len_ge_*` ↑,
`frag_short_over_long` ↓.
**(d) No-GT validation:** sweep `max_gap`; watch for the failure mode where too-
large a gap *merges distinct bubbles* — that shows up as falling `straightness`,
falling `mean_speed`, and falling split-half reproducibility. Pick the largest
gap before those degrade. (Pairs naturally with #1; #8 is in-pass, #1 is post-hoc
— do both and compare.)
**(e) GPU/feasibility:** none.
**(f) Effort:** S.

### 9. PSF-matched-filter detection — Detection
**(a) Stage:** detection, before z-scoring.
**(b) Implementation:** estimate an empirical 3D PSF kernel from this dataset (no
GT needed): take the brightest, well-isolated detections from a baseline run,
align and average their local magnitude patches → an anisotropic kernel (wide in
elevation). Cross-correlate each frame's magnitude with this kernel, then run the
existing z-score + NMS on the matched-filter response. Matched filtering is the
optimal linear boost of point-target SNR against noise → recovers weak bubbles
the raw z-score misses.
**(c) Metric & why:** higher detection SNR → `occupied_fraction` ↑ (weak
peripheral bubbles) and longer tracks (uniform detection along a path). A
principled coverage+length lever.
**(d) No-GT validation:** detections-per-frame fills peripheral vessels; split-
half reproducibility must rise (matched filter to a *real* PSF should boost
repeatable structure, not noise). Compare kernels estimated from disjoint acq
halves — a stable PSF estimate is itself evidence.
**(e) GPU/feasibility:** a separable 3D convolution per frame on GPU
(`cupyx.scipy.ndimage`); same memory class as the smoothing already done (~6 GB).
Safe.
**(f) Effort:** M.

### 10. Angle-domain coherence-factor weighting at beamform — Beamforming
**(a) Stage:** beamforming (`beamform_iq`).
**(b) Implementation:** we coherently sum 5 transmit-angle images. Also accumulate
the incoherent sum, and form a per-voxel **coherence factor**
`CF = |Σ_a x_a|² / (N · Σ_a |x_a|²) ∈ [0,1]`; multiply the compounded volume by
CF (or CF^γ). Coherent point targets (bubbles) keep CF≈1; incoherent
clutter/noise is suppressed. This sharpens the effective PSF and removes clutter
*before* the SVD, letting the SVD be gentler (→ recovered coverage).
**(c) Metric & why:** `contrast_cnr_proxy` ↑, cleaner detection → coverage + length;
enables a smaller SVD low-cutoff (keep more weak-bubble components).
**(d) No-GT validation:** split-half reproducibility and `contrast_cnr_proxy`;
compare CF on vs off at matched detection count. **Caveat I'll flag for the
critique round:** with only 5 angles, CF is a 5-sample statistic → high variance
and a known bias toward suppressing low-SNR (possibly real, weak) targets. Sweep
γ and a CF floor; if it only buys contrast at the cost of coverage, drop it.
**(e) GPU/feasibility:** needs a second host accumulator (incoherent sum, float)
alongside the existing coherent one — fits the `stream_accumulate` path
(host-side, ~2 volumes). Device peak unchanged. Moderate care, not a blocker.
**(f) Effort:** M.

### 11. High-cutoff noise-floor removal to enable a lower low-cutoff — SVD (clutter)
**(a) Stage:** clutter filter (`high_cutoff` is already plumbed but defaults None).
**(b) Implementation:** turn on removal of the last few singular components (the
noise subspace) via `high_cutoff`, which lets you *lower* the low-cutoff and keep
more of the weak-bubble mid-spectrum without re-admitting noise. One added knob,
already supported end-to-end.
**(c) Metric & why:** recovers coverage (lower low-cutoff keeps weak bubbles)
while `contrast_cnr_proxy` stays up (noise floor removed). Cheap probe of the
coverage frontier.
**(d) No-GT validation:** 2D sweep (low_cutoff × high_cutoff) on 60-acq; pick the
cell that maximizes coverage at fixed `frac_len_ge_35` and split-half stability.
**(e) GPU/feasibility:** none — same kernel, different component window.
**(f) Effort:** S.

### 12. 2.5D detection: detect in (z,x), assign elevation by PSF-weighted centroid — Detection/Localization
**(a) Stage:** detection + localization.
**(b) Implementation:** given that elevation is unresolved, detect peaks on the
**elevation-collapsed** (z,x) map (max- or sum-projection across elevation),
which removes elevation-driven flicker entirely; then assign each detection an
elevation as the PSF-weighted centroid over the elevation column at that (z,x)
(coarse, soft). This is the detection-side complement to #5's tracking-side
down-weighting.
**(c) Metric & why:** kills the dominant source of one-frame detection dropouts
(elevation lobe wobble) → `frag_short_over_long` ↓, `frac_len_ge_*` ↑; cleaner
coronal render → coverage/contrast. Honest about the data's actual resolution.
**(d) No-GT validation:** compare against #4 (anisotropic NMS) — both attack the
same elevation problem; keep whichever gives the better length/coverage/split-half
frontier. Risk: genuinely elevation-separated vessels get merged — check whether
high-elevation-gradient regions lose structure.
**(e) GPU/feasibility:** cheaper than current (detect on a 2D map per frame).
Safe.
**(f) Effort:** M.

### 13. Global multi-frame association (min-cost flow over a sliding window) — Tracking
**(a) Stage:** tracking (replaces greedy per-frame Hungarian within a window).
**(b) Implementation:** over a sliding window of ~10–20 frames, build a flow
graph (detection nodes, transition edges gated by motion, enter/exit edges) and
solve min-cost flow / successive-shortest-paths, so a locally ambiguous link is
resolved using future evidence instead of being greedily committed. Keeps the
existing motion/appearance costs as edge weights.
**(c) Metric & why:** fewer wrong commits → longer tracks and fewer switches
(`frac_len_ge_50` ↑, `frag` ↓). The principled ceiling on the association gains
that #1/#7/#8 approximate cheaply.
**(d) No-GT validation:** bench + split-half vs the greedy baseline at matched
detections.
**(e) GPU/feasibility:** CPU; windowing keeps graph size bounded. Main cost is
engineering, not memory.
**(f) Effort:** L. (Lower confidence per unit effort than #1; do #1 first and only
escalate here if stitching plateaus.)

### 14. Beamform apodization sweep (f-number, Tukey alpha) — Beamforming
**(a) Stage:** beamforming (`beamform_iq`, `tukey_alpha`, `f_number`).
**(b) Implementation:** cheap grid-scan of `tukey_alpha` (currently 0.5) and the
receive f-number (currently a very open f/0.5) on 60 acqs. Tighter apodization /
f-number trades main-lobe width for sidelobe level; sharper z/x PSF → cleaner
detection and localization.
**(c) Metric & why:** sharper PSF → `contrast_cnr_proxy` ↑, fewer sidelobe false
detections, better localization → straighter/longer tracks.
**(d) No-GT validation:** bench + split-half; watch that tightening the aperture
doesn't *reduce* coverage (smaller aperture → lower sensitivity to weak bubbles).
**(e) GPU/feasibility:** none beyond re-beamforming (already the dominant cost) —
but it's a parameter scan, so budget for several 60-acq beamforms.
**(f) Effort:** S (low engineering, moderate compute $).

### 15. Full robust-PCA (low-rank tissue + sparse bubble) clutter filter — SVD (clutter) [stretch]
**(a) Stage:** clutter filter (replaces SVD projection).
**(b) Implementation:** solve `M = L + S` with L low-rank (tissue) and S sparse
(bubbles) per acquisition (e.g. a few iterations of inexact-ALM / IALM with
nuclear + L1 thresholds), take S as the bubble signal. This is the principled
version of #6 and is the standard "best" clutter filter in the ULM literature.
**(c) Metric & why:** could recover coverage AND continuity simultaneously (S
keeps weak sparse bubbles that hard-SVD removes) — the highest upside on the
coverage-vs-continuity trade.
**(d) No-GT validation:** same frontier protocol (coverage vs `frac_len_ge_35` vs
split-half) against the region-SVD composite.
**(e) GPU/feasibility:** **the risk item.** Iterative SVDs on the (700, ~2.13M)
matrix at 11.9 GB, multiple passes, with extra L and S copies — easy to blow
24 GB. Must be done block-wise (reuse the region machinery) with strict
`free_all_blocks()` discipline, and even then iteration cost is real. High
effort, real OOM risk.
**(f) Effort:** L. (Listed last on purpose: highest upside, lowest
feasibility-per-effort. Only after the cheap clutter ideas #2/#6/#11 are exhausted.)

---

## Suggested bake-off order (given the priorities)

1. **First wave (all S, no re-beamform, iterate on saved pickles/volumes):**
   #1 stitching, #4 anisotropic-NMS/midplane, #5 elevation-down-weighted R,
   #7 appearance cost, #8 larger gap, #11 high-cutoff. These are nearly free and
   several are likely additive — run them as a small factorial.
2. **Second wave (coverage recovery, reuse region machinery):** #2 smoothed-rank
   and #3 CFAR together (they unlock each other), then #6 soft-SVD as the
   alternative to #2; #9 matched filter.
3. **Third wave (re-beamform, $$):** #10 coherence factor, #14 apodization sweep.
4. **Escalate only if needed:** #12 2.5D detection, #13 min-cost-flow, #15
   robust-PCA.

The headline bet: **#1 + #2 + #3** should recover most of region-SVD's lost
coverage *and* push median track length well past 6 frames — the two stated
priorities — for a few S/M-effort changes and one 60-acq bake-off each.
