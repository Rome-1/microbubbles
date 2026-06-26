# Flow-matching on the vessel network — context (beads `VESSEL-GRAPH` mb-crr.15 → `FLOWMATCH` mb-crr.16)

This captures Rome's idea and turns it into a concrete, in-scope plan (single-scan data only).

## The idea (Rome)

> "Infer the vessel network jointly with trajectories. Could we build some flow matching — even
> going so far as to train on the first half to predict flow in the second half? Where these
> predictions are worst would presumably give the most signal on what is the actual thinking parts."

Two distinct claims, both attractive:

1. **Flow as a learned field, not independent bubbles.** Model the blood-flow velocity field on
   the vascular graph and let it *predict* — a generative/flow-matching model of "given the vessel
   geometry and recent flow, what does flow do next." This regularizes tracking (crossings, gaps,
   low-SNR segments) by replacing per-bubble nearest-neighbor heuristics with a learned prior over
   physically plausible flow.
2. **The residual is the signal.** Train on the first ~2 minutes, predict the second ~2 minutes.
   Where prediction is **worst** — where flow deviates most from its own learned baseline — is
   plausibly where something *changed*: neurovascular coupling, task-driven hyperemia, the
   "actively modulating" regions. That error map is itself a candidate readout for downstream
   decoding ("telepathy"). This is a genuinely novel framing: **use predictability as a saliency
   map for function.**

## Why it fits this data (and stays in scope)

We have a continuous ~4-minute scan: 223 acquisitions @ 222 Hz. That is enough to:
- Build a velocity field / vessel graph from the full track set (depends on `VESSEL-GRAPH`).
- **Temporal split** the scan (first half / second half) for a clean train→predict protocol with
  zero new data — exactly the in-scope experiment Rome described.

No new acquisition is required, so this is actionable within our current constraints. The bigger
multi-subject version belongs in `ulmshare-downstream.md`.

## Concrete formulation

**Substrate (from `VESSEL-GRAPH`):** from accumulated tracks, estimate vessel centerlines, a graph
`G=(nodes, edges)` with per-edge direction, and a time-resolved velocity field `v(x, t)` sampled
on the graph (or on a voxel grid masked to vessels).

**Model options, simplest-first:**
1. **Per-edge autoregressive baseline** (sanity floor): predict each edge's flow rate from its own
   recent history + graph-neighbor flows (mass conservation at bifurcations). Cheap, interpretable,
   establishes the residual map before any heavy ML.
2. **Flow matching / conditional generative field**: train a continuous-time model
   `dx/dτ = u_θ(x, τ | conditioning)` (flow matching) that transports a simple prior to the
   distribution of next-window velocity fields, conditioned on vessel geometry + recent flow. This
   is the literal "flow matching" reading and gives a *distribution* over futures, not a point
   estimate — so "prediction is worst" becomes "observed future has low likelihood / high
   transport cost under the model," a principled surprise measure.
3. **Graph neural operator** over `G` if (1)–(2) underfit the branching structure.

**Train→predict protocol (the residual-as-signal experiment):**
- Fit on window A (first half). Predict window B (second half). Compute a per-region
  **surprise/residual map** (prediction error or negative log-likelihood).
- **Controls that decide if the residual means anything:**
  - *Stationarity control:* also predict A from B (reverse) and predict held-out interior windows;
    a real functional signal should be more than drift/motion/SVD artifacts.
  - *Motion confound:* run after `MOTION` correction — otherwise residuals just track head motion.
  - *SNR confound:* check the residual map isn't merely the low-SNR / low-track-density map.
  - *Null model:* shuffle time within vessels; the residual map should collapse toward null.

**Validation without ground truth (consistent with the whole project):**
- 3D diff: residual/surprise volume overlaid on the vessel graph and the baseline ULM render.
- Split-half repeatability of the residual map across independent sub-windows (does the "salient"
  region reproduce?). Reproducibility is our stand-in for correctness.

## Dependencies & risks

- **Hard-depends on `VESSEL-GRAPH`** (need a clean graph + velocity field) and benefits enormously
  from `MOTION` and `KALMAN-GATE` (a residual map built on motion artifacts is worthless).
- Risk: the residual map may be dominated by motion / SNR / clutter rather than neurovascular
  function — the controls above are how we find out *honestly* rather than over-claim.
- This is research-tier (p3). It is documented and scoped now; we act on it only after the
  correctness + vessel-graph substrate exists.

## Why the "where prediction fails = thinking" hook is worth taking seriously

It inverts the usual decode target. Instead of predicting flow *well* everywhere, we treat the
model's *failure* as the informative quantity — the regions a flow model trained on baseline
behavior cannot predict are, by construction, the regions doing something non-baseline. If even a
weak version of this reproduces across split-halves, it's a strong lead for the downstream
telepathy program, and it costs only compute on data we already have.
