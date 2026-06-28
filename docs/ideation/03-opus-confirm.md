# Opus confirmation — green-light gate on the agreed 10

Author: Opus 4.8. Input: `03-agreed-10.md` (orchestrator synthesis) checked against
my own `02-opus-r2.md` and the constraints in `00-context.md`.

**Verdict: AGREE (no swap).**

---

## (1) Are these 10 the right build set? — AGREE

Yes. The agreed-10 is, item-for-item, my own Round-2 ranked top-10. The orchestrator
made exactly one structural change versus my numbering, and it is an *improvement*:

- My downgraded #9 ("overlap-and-feather **boundary diagnostics**, not more feathering")
  was correctly folded into the instrumentation item (#10 / Item 0) — that is where it
  belongs, because it is a *diagnostic*, not a track-metric mover. Good call.
- My CONTESTED #10 (appearance/intensity term in the association cost) was promoted into
  the explicit ten. I flagged it CONTESTED only because I wasn't sure Codex would list it;
  the provenance in `03-agreed-10.md` shows Codex *did* (their R2 #10), both flagged it,
  and it resolved to mutual. So the one item I was least sure was consensus is now
  verified consensus. That removes my only reservation.

Everything else is a clean intersection. Each of the 10 is something I would defend
building, each maps to a stated priority in `00-context.md` (recover the −27% coverage
without losing continuity; push median track length past ~6 frames), and each is feasible
at the 60-acq bake-off scale on one dataset with no GT. Item 0 gating everything is
correct and non-negotiable — it is the arbiter that tells us *where* coverage dies, and
it pairs the per-stage coverage audit (Codex) with split-half render reproducibility
(mine) into the no-GT master metric. I have no objections to the set or the build waves.

## (2) One swap? — None.

I decline the swap. The 10 are my ranked priorities, and nothing below the cut beats the
marginal item:

- The principled ceilings I deferred — **min-cost-flow association**, **robust-PCA SVD**,
  **coherence-factor beamforming** — are all higher-effort and several need re-beamform
  ($$) or risk OOM on the (700 × 2.13M) matrix. They are explicitly "escalate only if the
  cheap version plateaus" items, not green-light items.
- **Vessel-consistency fragment rescue** carries a circularity risk (build a vessel map
  from tracks, then use it to make more tracks → inflates the proxy without adding signal).
  Last, and only validated against split-half.

If *forced* to name a swap, the only candidate I'd even consider removing is **#9
(appearance-cost)** — it's the lowest-confidence mover, since ID-switches matter mostly in
dense regions and the field is fairly sparse after region-SVD — in favor of min-cost-flow.
But I **reject my own swap**: #9 is near-free (intensities already sit unused in
`pair_costs`) and additive, whereas min-cost-flow is the expensive ceiling on #1/#5/#8 and
belongs to a later wave. Keep the set as-is.

## (3) Build phase — who builds what, and the cross-checks I want

Clean split by file/stage to parallelize with zero merge collision and play to where each
model's reasoning was load-bearing.

**Opus implements (the tracking cluster — all on the saved track list / `tracking.py`):**
- **#1 Post-hoc track stitching** — my #1; the stitching design (anisotropic tol, LSA over
  chains, re-smooth) is mine and it's the headline cheap win.
- **#4 Elevation-as-low-confidence** (anisotropic Kalman `R`, Mahalanobis gate, R seeded
  from parabola curvature) — the elevation-untrustworthy thread was the crux of my critique.
- **#8 Larger `max_gap` + strict predicted/velocity-consistent gating** — near-free
  increment over the already-shipped gap=3 prediction gate.
- **#9 Appearance/intensity term in the association cost** — near-free, same file.

These four are the Wave-1 near-free factorial, all touch the same file/data structure, so
Opus owning the whole cluster means no collision with Codex's GPU/detection work.

**Codex implements (the detection + SVD-rank + noise-stats cluster):**
- #2 Local CFAR/MAD detection (Codex's MAD refinement beat my mean/std).
- #3 Adaptive per-block SVD rank + regularized k-map.
- #5 Low-confidence continuation-only + track-aware threshold (Codex's strongest specific idea).
- #6 Anisotropic/PSF-tied NMS + midplane de-biasing.
- #7 Empirical PSF → matched-filter detection.

**Co-own #10 instrumentation:** Codex builds the per-stage coverage audit (their design);
Opus builds the split-half render reproducibility arbiter (my cross-cutting design). Build
this first; both bake-off tracks depend on it.

### What I want Codex to cross-check in *my* work
- **#1 stitching — merge-distinct-bubbles failure.** Confirm the anisotropic-tol +
  velocity-agreement gate is not fusing two real bubbles into one track (would inflate mean
  length while corrupting velocity). Acceptance: a real stitch makes odd/even split-half
  renders agree *more*; `straightness` and `mean_speed` must not degrade.
- **#4 elevation inflation — cross-vessel leakage.** Confirm widening the elevation gate /
  inflating `R_yy` doesn't let z/x jumps through the Mahalanobis gate, and that elevation
  occupancy doesn't collapse onto the bright midplane (the known attractor).
- **#8 larger gap — over-bridging.** Independently find the gap-cap sweep knee; back off
  the moment `straightness` / `mean_speed` / split-half fall.
- **#9 appearance cost — scintillation.** Confirm λ isn't high enough to fragment tracks
  through intensity blinking; it should help in dense regions and be neutral elsewhere.
- **Cross-cutting (the motion-correction lesson):** every tracking change judged ONLY on
  `bench.py` track-level metrics + split-half, never on the power volume. I'd like Codex to
  re-run `bench.py` independently on my output pickles and confirm the deltas.

### What I'll cross-check in Codex's work (so it's mutual)
- **#3:** verify the (n_z × n_x) k-map regularization (clamp to median ± Δ) is actually
  applied and swept — without it, #3 collapses to the bare `per_block` setting that already
  failed by reintroducing per-block intensity banding.
- **#2:** verify there is a **global floor** under the local CFAR scale so it doesn't
  amplify empty-tissue noise to threshold.

---

**One-line verdict: AGREE.**
