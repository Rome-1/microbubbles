# Downstream program: scaling with more data (ULMShare) — PLANNING ONLY

**Status: OUT OF SCOPE to act on now.** Rome: "A downstream and more ambitious version of this
will involve using a heck of a lot more data via https://arxiv.org/abs/2606.07851 — keep planning
and ideas docs for this, but that's downstream of our initial plan of improving this pipeline."

This file is a parking lot so the idea isn't lost. We do **not** download, train on, or build
against external datasets until the in-repo pipeline improvements (epic `mb-crr`) are landed and
validated against baseline.

## What ULMShare (arXiv 2606.07851) would unlock

A large-scale in-vivo ULM dataset is valuable to us in three ways:

1. **A real benchmark with reference reconstructions.** Our biggest handicap is *no ground truth*.
   ULMShare ships reconstructions, trajectories, saturation curves, Fourier-ring-correlation, and
   track-length statistics. That gives `BENCH` (mb-crr.17) an external calibration target instead
   of pure self-consistency — we could finally say "our localization error / FRC is X vs a
   community reference," not just "different from baseline."
2. **Training data for the learned components.** The learned pieces we're deferring —
   learned clutter filters (`ADAPT-SVD` stretch), spatiotemporal learned localization
   (`MULTIBUBBLE`), and especially the flow-matching model (`FLOWMATCH`) — are data-hungry.
   A single 4-minute scan can demonstrate them but can't robustly train them. Multi-subject,
   multi-scanner data is what makes a learned model generalize.
3. **Cross-subject priors.** A vessel-topology prior (`VESSEL-GRAPH`) or an aberration prior
   (`SKULL`) learned across many subjects could be transferred to a new wearable scan as a strong
   initialization.

## How it would slot in (later)

- Keep our pipeline's interfaces (beamformed HDF5 schema, tracks pickle, `.bin` exports) stable so
  an external dataset can be ingested through the *same* `track` / `viz` / `bench` path. Effectively
  treat ULMShare as "another set of beamformed acquisitions" once an adapter is written.
- The flow-matching residual-as-signal experiment (`flow-matching.md`) becomes far stronger with
  many subjects: does the "unpredictable = functional" map reproduce *across people*, not just
  across split-halves of one scan?

## Gate before any of this happens

1. Epic `mb-crr` improvements landed + composite diff vs baseline accepted by Rome.
2. Explicit go-ahead from Rome to bring in external data (storage, licensing, compute spend).
3. A written ingestion spec so external data flows through our existing validated path.

Until all three: this stays a planning doc. Generating *more ideas* here is encouraged; acting on
them is not.
