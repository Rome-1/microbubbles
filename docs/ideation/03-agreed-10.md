# The agreed 10 — Opus ∩ Codex consensus build set

Synthesized by the orchestrator from `02-opus-r2.md` + `02-codex-r2.md` (each model's
unified top-10 after cross-critiquing the other's Round-1 list). Every item below
appeared in **both** models' reasoning and both would defend building it. Provenance
noted as (Opus Rx#, Codex Rx#).

**Prerequisite (build first, both models insist): Item 0 — instrumentation & no-GT
validation harness.** Per-stage coverage audit (beamformed energy → post-SVD energy →
detections → accepted/rejected tracks, with block-boundary overlay) + split-half render
reproducibility (odd vs even acquisitions → coronal track-density correlation/SSIM) +
vessel-map sharpness metric. This is the arbiter every item is scored against; it tells
us *where* coverage dies. (Opus Item0, Codex R2#8/#9.) Folded into idea #10 below for the
build list but gates everything.

All ideas baked off at **60 acquisitions vs base60**, scored by `scripts/bench.py`
track-level metrics + the Item-0 harness. Winners scaled to full 223.

| # | Idea | Stage | What to build | Metric it should move | Effort | Provenance |
|---|------|-------|---------------|----------------------|--------|------------|
| 1 | **Post-hoc track stitching** | tracking (post) | `stitch_tracks` on the saved track list: link end_i→start_j when gap-extrapolated endpoint is within an anisotropic tol (tight z/x, loose elev), velocity dirs agree, intensity matches; one `linear_sum_assignment`, merge chains, re-smooth. Pure CPU on the pickle, no re-detect. | mean_curv_len ↑, frac≥35/50 ↑, fragmentation ↓ | S | Opus#1, Codex#1 (both ranked #1) |
| 2 | **Local CFAR / MAD detection** | detect | Replace per-elev-plane mean/std (`_slice_stats`) with a local background (median + k·MAD via large-kernel spatial filter on time-mean magnitude), guard band, **global floor** under the scale. | occupied_fraction ↑ (dim periphery), track length ↑ (uniform detection → fewer dropouts) | M | Opus#2, Codex#6 |
| 3 | **Adaptive per-block SVD rank + regularized k-map** | SVD | Per-block low cutoff from singular-value elbow / retained-energy; **clamp/smooth the (n_z×n_x) k-map** (median ± Δ); sweep Δ from global_rank→per_block. Reuses `region_block_cutoffs`. | occupied_fraction / peak_density recover at fixed frac≥35 | M | Opus#3, Codex R1#2 |
| 4 | **Elevation-aware tracking** | tracking | Inflate Kalman `R_yy` 5–10×; switch box-gating → **Mahalanobis** gate with anisotropic cov (wide elev, tight z/x); offer a **z-optional / 2D-first** linking mode; seed R from parabola curvature. | straightness ↑, mean_curv_len ↑, frac≥* ↑ (stop elev jitter breaking tracks) | M | Opus#4, Codex#2/#4 |
| 5 | **Low-confidence continuation-only + track-aware threshold** | detect→track | Two detection sets: high-conf may **start** tracks, low-conf may only **continue** them; lower threshold, judge by track metrics. | track length ↑, fragmentation ↓, occupied ↑ — coverage without one-frame-false-track explosion | M | Opus#5, Codex#7 |
| 6 | **Anisotropic / PSF-tied NMS + midplane de-biasing** | detect | `maximum_filter` size `(1, f_elev, f_z, f_x)` with large elev radius ≈ PSF; divide each frame by a smooth per-plane elevation bias profile before z-score. | fragmentation ↓, length ↑, contrast ↑ (kill elev-smeared replicas + bright-midplane sheet) | S | Opus#6, Codex R1#12 |
| 7 | **Empirical PSF → matched-filter detection** | detect | Mine bright isolated detections → average aligned patches → anisotropic empirical 3D PSF (wide in elev); cross-correlate frames before z-score/NMS; feed PSF to #6 radius + #4 covariance. Disjoint-half PSF = free self-consistency check. | occupied_fraction ↑ (weak bubbles), continuity ↑ | M/L | Opus#7, Codex R1#10 |
| 8 | **Larger max_gap + association sweep** | tracking | Raise `max_gap` 3→6–8, admit a bridged detection only near the Kalman-**predicted** position + velocity-direction-consistent; sweep gap/gates/costs. Mark interpolated points as flagged (Codex). | mean_curv_len ↑, frac≥* ↑, fragmentation ↓ | S | Opus#8, Codex#3/#7 |
| 9 | **Intensity / appearance-aware association cost** | tracking | Add `λ·|log I_det − log Ī_track|` (+ PSF-width similarity if available) to the Mahalanobis+momentum cost (intensities already carried, currently unused in `pair_costs`). | fewer ID switches → frac≥* ↑, straightness ↑ in dense regions | S | Opus#10, Codex#10 (both listed, both flagged contested → mutual) |
| 10 | **Instrumentation & no-GT validation harness** (Item 0) | eval | Per-stage coverage audit + split-half (odd/even) render reproducibility (correlation/SSIM) + vessel-map sharpness metric + elevation bias audit (z-plane occupancy, midplane attraction, z-jitter). The arbiter for all of the above. | n/a (decision support) — but de-risks every other item | M | Opus Item0, Codex#5/#8/#9 |

## Where the two models disagreed (recorded for honesty)
- **Coverage mechanism:** Codex's headline was "overlap-and-feather region SVD"; Opus showed (from the code) the blend is **already** a seam-free partition of unity, so the −27% loss is a **rank** artifact, not a boundary one. → resolved toward **#3 (rank)**; the feather idea downgraded to boundary *diagnostics* (folded into #10).
- **Detection vs tracking priority:** Opus argued the recoverable short-track loss is **tracking-dominated** (rank #1 stitching); Codex warned against deprioritizing detection too far. → resolved by building **both** (tracking #1/#4/#8/#9 + detection #2/#5/#6/#7), bake-off independently.
- **Dropped by both:** temporal-overlap batching (phantom seam — no track-breaking temporal boundary exists); vessel-consistency fragment rescue (circularity risk — deferred, only validated against split-half).

## Build order (both models' waves)
- **Wave 0:** #10 instrumentation (gates everything).
- **Wave 1 (near-free, on saved tracks/volumes):** #1, #4, #8, #9, #6 — small factorial, several additive.
- **Wave 2 (coverage, reuse GPU/region machinery):** #2 + #5 together (mutually enabling: CFAR makes aggressive per-block SVD banding-immune), then #3, then #7.
- **Headline bet:** #1 + #2 + #3 + #5 should recover region-SVD's lost coverage *and* push median track length past 6 frames — the two stated priorities.
