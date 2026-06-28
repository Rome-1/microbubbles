# Opus round-2 — critique of Codex + unified ranked top-10

Author: Opus 4.8. Inputs: `00-context.md`, `01-opus-r1.md` (mine), `01-codex-r1.md`
(Codex). I re-read the relevant source to ground the critique:
`svd_region.py`, `gpu_svd_region.py`, `gpu_detect.py`, `tracking.py`.

Three code facts that decide several disagreements below:

- **Region-SVD already does overlap + feather.** `svd_region.py:13-27`: blocks
  overlap (`overlap=0.25`) and are blended with a cos²/sin² **partition of unity**
  that sums to exactly 1 ("seam-free... correct even outside the ≤2-overlap
  regime"). So the −27% coverage loss occurred **with seam-free feathering already
  in place.**
- **`per_block` adaptive cutoff already exists** (`gpu_svd_region.py:123`,
  `region_block_cutoffs`). The missing piece is *regularizing* the per-block rank
  map, not building per-block machinery.
- **No track-breaking temporal seam.** `gpu_detect.py:68` `frame_chunk=150` is a
  memory loop over one volume (detections concatenated in order); tracking
  consumes the full per-acq frame sequence. There is no temporal chunk boundary
  that severs tracks.

---

## Part 1 — Critique of Codex's list

### The four Codex emphasized

**#1 Overlap-and-feather region SVD — DISAGREE (as the coverage fix); AGREE (as hygiene).**
This is Codex's headline coverage idea and it is **largely already in the codebase**:
overlapping blocks + a true partition-of-unity cos²/sin² blend (`svd_region.py`).
The −27% coverage loss happened *despite* seam-free feathering. That is decisive
evidence the loss is **not** a boundary/blend artifact — it is a **rank/intensity**
artifact: `global_rank` removes the same k (driven by the bright central tissue)
everywhere, over-projecting low-rank peripheral blocks and taking weak-bubble
energy with them (this is my R1 critique, and the code confirms the premise).
Building *more* overlap/feather will smooth any residual seams but will **not**
recover the lost coverage. So: AGREE it is worth adding the **boundary
diagnostics** Codex mentions (cheap, confirms there are no grid-dropouts left),
DISAGREE that it is the coverage lever. The lever is per-block **rank** (#2) plus
a detector that is immune to block intensity scale (#4). Codex mis-attributes the
mechanism.

**#2 Adaptive SVD rank per region (elbow / Marchenko-Pastur / energy) — AGREE, strongly. Top-tier.**
This is the correct diagnosis-matched fix and it aligns with my R1 #2. One
refinement Codex conflates: the **low** cutoff (tissue rank) and the **high**
cutoff (noise floor) are different jobs. The coverage loss lives in the *low*
cutoff being too large for low-rank peripheral blocks, so the per-block estimator
that matters is a **tissue-rank** estimator (singular-value elbow / retained-energy
/ temporal-autocorrelation of the modes). Marchenko–Pastur is naturally a
**noise-floor / high-cutoff** tool — useful, but it answers a different question
(where does signal end and noise begin), not "how many tissue modes." Use MP for
the high cutoff, elbow/energy for the low cutoff, and — critical, and missing from
Codex — **regularize the (n_z × n_x) k-map** (clamp to median ± Δ, or smooth it)
so you don't reintroduce the per-block intensity banding that got `per_block`
abandoned the first time. Adaptive rank *without* regularization is the exact knob
that already failed. This is consensus, with my regularization rider.

**#3 Track-aware detection threshold sweep — AGREE, strongly — and promote it to methodology.**
This is less an "idea" than the correct **evaluation protocol** for every
detection change: lower the threshold, judge by downstream track metrics, never by
raw detection count. It directly operationalizes the context's stated lesson
("track-level metrics are the arbiter"). The one risk — lowering the threshold
floods the linker with one-frame false positives — is *exactly* solved by Codex's
own #11 (low-confidence detections may only continue, not start, tracks). #3 + #11
are a single coherent package and I'd build them together. Strong consensus.

**#4 Local CFAR / MAD detection — AGREE, strongly. Near-identical to my R1 #3.**
The current `_slice_stats` is one mean/std per elevation plane over all 700 frames
— a single global-within-plane threshold that under-detects dim periphery and
over-detects bright cores. Local-background normalization is the single biggest
*coverage* lever that is independent of the SVD. Codex's **MAD** refinement is a
genuine improvement over my mean/std: the median/MAD is robust to the bubbles
themselves contaminating the local statistic. Two cautions both lists should
honor: (a) put a **global floor** under the local scale so CFAR doesn't amplify
pure noise to threshold in genuinely empty tissue; (b) it **unlocks aggressive
per-block region-SVD** (#2), because a background-relative detector is immune to
block intensity banding — so #2 and #4 are mutually enabling. Consensus lock.

### The rest of Codex's list

**#5 Gap-closing tracker (1–3 missed frames) — AGREE, but partly already shipped.**
`tracking.py` already bridges `max_gap=3` and already gates on the Kalman
*prediction* (the +43% win in the context table). So "allow 1–3 missed frames" is
the current state. The real, unbuilt lever is **raising** the cap (6–8) *guarded
by* strict predicted-position + velocity-direction gating (my R1 #8). Codex
under-credits what's already there; the increment is the larger gap, not the gap
mechanism.

**#6 Motion-model / Mahalanobis gating with anisotropic covariance — AGREE.**
Today's gate is box-gating on a `max_dist` 3-vector, not a true Mahalanobis gate.
Moving to Mahalanobis with an **anisotropic** covariance that inflates elevation is
the right call and it merges cleanly with my R1 #5 (elevation-inflated Kalman R).
Consensus.

**#7 Two-pass association (conservative then recovery) — AGREE.** This is the same
target as my R1 #1 (post-hoc stitching), framed as a second linking pass. Both are
worth it; I'd implement the recovery pass as explicit fragment **stitching** on the
saved track list (zero GPU cost, trivially sweepable) rather than a second in-loop
pass, but the intent is identical. Consensus.

**#8 PSF-aware 3D localization fit — AGREE on the goal, DISAGREE on the method's cost.**
Extracting a per-detection **covariance** to feed the tracker is exactly right and
synergizes with anisotropic R. But a full anisotropic-Gaussian nonlinear fit per
candidate, across ~700 frames × 60 acqs × many candidates, is the most expensive
thing on either list for an uncertain marginal gain over a cheap surrogate. The
**parabola curvature** already computed in `subpixel_localize_3d` gives a
per-axis covariance essentially for free; use that as the R-input. Reserve the
full PSF fit for a later round if curvature-covariance plateaus. Agree on
direction, prefer the cheap version.

**#9 Treat elevation as low-confidence — AGREE, strongly. Consensus lock.** This is
the same conclusion as my entire R1 elevation critique (1 physical receive row →
no receive-side elevation focus → wide PSF, bright midplane, untrustworthy y). Both
models independently converged here; it should be in the build set.

**#10 Empirical PSF calibration from isolated bubbles — AGREE, strongly.** Same as
my R1 #9. Mining bright isolated detections to estimate an anisotropic empirical
PSF (no GT needed) feeds three consumers: matched-filter detection, NMS radius, and
localization/tracker covariance. The PSF estimated from disjoint acq-halves is also
a free self-consistency check. Consensus.

**#11 Low-confidence continuation-only retention — AGREE, strongly. One of Codex's best.**
Two detection sets — high-confidence may *start* tracks, low-confidence may only
*continue* them — is the elegant fix that lets you lower the threshold (coverage,
fewer dropouts) without an explosion of one-frame false tracks. This is more
specific and better than anything equivalent on my list, and it is the natural
partner to #3. I want this.

**#12 Duplicate/merge handling, PSF-tied NMS radius — AGREE.** Same target as my R1
#4 (anisotropic NMS); Codex adds the good idea of tying the NMS radius to the
empirical PSF (#10) and preventing one detection from spawning competing fragments.
Merge with #4. Consensus.

**#13 Per-stage coverage audit (energy → post-SVD → detections → tracks → rejects, + tile boundaries) — AGREE, strongly. Build it first.**
This is the highest value-per-line item on either list. It is the instrument that
tells us whether coverage dies at SVD, detection, or tracking — and therefore which
of #2/#4 vs the tracking ideas to invest in. It is cheap and it de-risks everything
else. It pairs with my R1 cross-cutting **split-half render reproducibility**
(odd vs even acquisitions → correlation/SSIM) as the master no-GT arbiter Codex's
list otherwise lacks. Together they are the instrumentation layer (Item 0 below).

**#14 Temporal batching with overlap — DISAGREE (phantom seam).** I checked: there
is no track-breaking temporal seam to fix. `gpu_detect.py` `frame_chunk=150` is a
memory loop over a single volume with detections concatenated in frame order;
SVD is per full acquisition; tracking consumes the whole per-acq sequence; tracks
do not cross acquisition boundaries by design (different ~4s windows of the same
vasculature). Building temporal overlap solves a problem the pipeline does not
have, and it costs memory/compute. Drop unless the audit (#13) actually shows a
termination spike at a frame boundary.

**#15 Vessel-consistency priors to rescue fragments — AGREE in principle, defer.**
The circularity risk (build a vessel map from tracks, then use it to make more
tracks → confirmation bias that inflates the proxy metrics without adding real
signal) makes this the *last* thing to do, and only validated against split-half
reproducibility (a real prior should make independent halves agree *more*). L
effort, lower confidence per unit effort. Below the cut.

### Where Codex's framing is weakest (my sharpest disagreement)

Codex treats the coverage loss as a **boundary/blend** problem (#1) and leans
"short tracks are **detection**-dominated" (#3). The code says the blend is already
seam-free, so coverage loss is a **rank** problem, not a boundary one. And the
evidence (two biggest wins so far were both association/clutter, not detection)
says the *recoverable* part of the short-track problem is **tracking**-dominated.
So I rank the cheap tracking post-processing above the detection rework, and I
rank per-block **rank** above more feathering — the reverse of where Codex's
emphasis points.

---

## Part 2 — Unified ranked top-10

Ranked by **(expected track-metric gain) × (feasibility: one dataset, no GT, A10G
24 GB, cheap at 60 acq, prefer no re-beamform)**. Tag = CONSENSUS (I believe Codex
would also pick it — and I'd defend it) or CONTESTED.

> **Item 0 (prerequisite, build first — CONSENSUS): instrumentation.**
> Per-stage coverage audit (Codex #13: beamformed energy → post-SVD energy →
> detections → accepted/rejected tracks, with block-boundary overlay) **plus**
> split-half render reproducibility (Opus cross-cutting: coronal track-density
> render from odd vs even acqs → correlation/SSIM). Not a track-metric mover by
> itself, so off the ranked 10 — but it is the arbiter that every item below is
> validated against, and it tells us where coverage actually dies. ~1 day, no GPU
> risk. Do not start the bake-offs without it.

**1. Post-hoc track stitching / recovery pass — CONSENSUS.**
Build `stitch_tracks` on the saved track list: link end_i→start_j when the
gap-extrapolated endpoint lands within an anisotropic tolerance (tight z/x, loose
elevation), velocity directions agree, intensities match; one global
`linear_sum_assignment`, merge chains, re-smooth.
*Moves:* `mean_curvilinear_len_mm` ↑, `frac_len_ge_35/50` ↑, `frag_short_over_long` ↓.
*Why #1:* cheapest possible (pure CPU on the pickle, infinitely sweepable, no
re-beamform), directly attacks the median-6-frame problem, and the recoverable
short-track loss is tracking-dominated. (Opus #1 + Codex #7.)

**2. Local CFAR / MAD detection — CONSENSUS.**
Replace global-within-plane `_slice_stats` with a local background (median + k·MAD
via a large-kernel spatial filter on the time-mean magnitude, guard band, global
floor under the scale).
*Moves:* `occupied_fraction` ↑ (peripheral coverage at fixed global threshold),
track length ↑ (uniform detection along a path → fewer dropouts).
*Why #2:* the single biggest coverage lever independent of the SVD; reuses the
existing GPU smoothing (~6 GB, safe); and it **unlocks #3**. (Opus #3 + Codex #4.)

**3. Adaptive per-block SVD rank + regularized k-map — CONSENSUS.**
Per-block low cutoff from singular-value elbow / retained-energy (MP for the high
cutoff), then **clamp/smooth the (n_z × n_x) k-map** (median ± Δ) before applying.
Sweep Δ from 0 (=global_rank) to ∞ (=per_block).
*Moves:* `occupied_fraction` / `peak_density` recover toward baseline at fixed
`frac_len_ge_35`.
*Why #3:* the correct, mechanism-matched coverage fix; reuses existing region
machinery (same memory profile); regularization is what makes per_block viable.
Made fully effective by #2 (banding-immune detector). (Opus #2 + Codex #2.)

**4. Elevation-as-low-confidence: anisotropic Kalman R + Mahalanobis gate — CONSENSUS.**
Inflate `R_yy` 5–10×, widen the elevation gate and tighten z/x, switch box-gating
to a Mahalanobis gate; seed R from the cheap parabola curvature.
*Moves:* `straightness` ↑, `mean_curvilinear_len_mm` ↑, `frac_len_ge_*` ↑ — stops
elevation jitter from curling/breaking tracks.
*Why #4:* both models independently converged on elevation being untrustworthy;
pure CPU, near-free. (Opus #5 + Codex #6/#9.)

**5. Low-confidence continuation-only retention + track-aware threshold — CONSENSUS.**
Two detection sets: high-confidence may start tracks, low-confidence may only
extend existing tracks; lower the threshold and judge by track metrics.
*Moves:* track length ↑, `frag` ↓, `occupied_fraction` ↑ — coverage *without* a
one-frame-false-track explosion.
*Why #5:* elegant, cheap, and the natural partner to #2/#1; Codex's strongest
specific tracking idea. (Codex #11 + #3.)

**6. Anisotropic / PSF-tied NMS + midplane de-biasing — CONSENSUS.**
Make `maximum_filter` anisotropic (`size=(1, f_elev, f_z, f_x)` with a large
elevation radius ≈ PSF width); divide each frame by a smooth per-plane elevation
bias profile before z-scoring.
*Moves:* `frag_short_over_long` ↓, length ↑, `contrast_cnr_proxy` ↑ — collapses
elevation-smeared replicas and the bright-midplane sheet that inject dropouts.
*Why #6:* trivial (kernel size + a normalization vector), attacks the dominant
source of one-frame detection flicker. (Opus #4 + Codex #12.)

**7. Empirical PSF calibration → matched-filter detection — CONSENSUS.**
Mine bright isolated detections → average aligned patches → anisotropic empirical
3D PSF (wide in elevation); cross-correlate frames with it before z-score/NMS; feed
the same PSF to #6's NMS radius and #4's covariance.
*Moves:* `occupied_fraction` ↑ (weak peripheral bubbles), continuity ↑.
*Why #7:* principled SNR boost; separable GPU conv (~6 GB, safe); the
disjoint-half PSF is a free self-consistency check. (Opus #9 + Codex #10/#8 —
using cheap curvature-covariance, not full per-candidate Gaussian fits.)

**8. Larger `max_gap` with strict predicted + velocity-consistent gating — CONSENSUS.**
Raise `max_gap` 3 → 6–8, admitting a bridged detection only if it sits near the
Kalman-*predicted* position and preserves velocity direction.
*Moves:* `mean_curvilinear_len_mm` ↑, `frac_len_ge_*` ↑, `frag` ↓.
*Why #8:* near-free; the real unbuilt increment over the already-shipped gap=3
prediction gate. Sweep the cap; back off when `straightness`/`mean_speed`/split-half
fall (the merge-distinct-bubbles failure). (Opus #8 + Codex #5.)

**9. Overlap-and-feather boundary diagnostics (NOT more feathering) — CONSENSUS, downgraded.**
The feather/PoU blend already exists, so build only the **block-boundary
diagnostics** Codex proposed: confirm there are no grid-dropouts at block seams and
that detection density/velocity distributions are continuous across boundaries.
*Moves:* indirectly guards `occupied_fraction` (rules out a seam artifact so #3 gets
the credit/blame correctly).
*Why #9 not higher:* the coverage lever is rank (#3), not boundary; this is cheap
insurance, not a mover. Largely folds into Item 0. (Codex #1, re-scoped.)

**10. Appearance (intensity / PSF-width) term in association cost — CONTESTED.**
Intensities are already carried per detection but unused in the cost; add
`λ·|log I_det − log Ī_track|` (and PSF-width similarity if available) to the
Mahalanobis + momentum cost.
*Moves:* fewer ID switches → `frac_len_ge_*` ↑, `straightness` ↑ in dense regions.
*Why CONTESTED:* Codex did not list it (their tracking ideas were gating/two-pass),
so it is the one pick beyond explicit consensus — but it is near-free, standard MOT,
and I'd defend that Codex would adopt it once they saw intensities sit unused in
`pair_costs`. (Opus #7.)

### Deliberately below the cut
- **Coherence-factor / apodization beamforming sweeps** (Opus #10/#14): require
  re-beamform ($$ per 60-acq), and CF on 5 angles is a high-variance 5-sample
  statistic. Wave 3 only.
- **Soft-shrinkage / robust-PCA SVD** (Opus #6/#15): #6 is a fine alternative route
  to #3's frontier (bake if #3 plateaus); full robust-PCA is the real OOM risk on
  the (700 × 2.13M) matrix — last resort.
- **Min-cost-flow association** (Opus #13): the principled ceiling on #1/#5/#8;
  escalate only if cheap stitching plateaus.
- **Temporal-overlap batching** (Codex #14): phantom seam — dropped.
- **Vessel-consistency fragment rescue** (Codex #15): circularity risk; last, and
  only against split-half.

### Build-wave summary
- **Wave 0 (instrument):** Item 0 (audit + split-half).
- **Wave 1 (near-free, iterate on pickles/volumes):** #1, #4, #6, #8, #10 — run as
  a small factorial; several are additive.
- **Wave 2 (coverage, reuse GPU/region machinery):** #2 + #5 together (mutually
  enabling), then #3, then #7; #9 diagnostics alongside.
- The headline bet: **#1 + #2 + #3 + #5** recovers most of region-SVD's lost
  coverage *and* pushes median track length well past 6 frames — the two stated
  priorities — at S/M effort and one 60-acq bake-off each.
