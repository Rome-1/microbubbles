# Roadmap — improving the microbubbles 3D ULM pipeline

**Epic:** `mb-crr`. **Baseline:** exactly what Aleph Neuro shipped on this repo's `main`.
**Fork-only:** all work on `origin` (Rome-1/microbubbles); never push to `upstream` (alephneuro).

We have **no ground truth**. Validation is therefore (a) visuals that look anatomically /
physically plausible and (b) **3D diffs vs the baseline**. Every insight ships with its own
3D volume + diff; a final composite bundles all the worthwhile improvements against baseline.
Heavy compute (download / beamform / track / volume generation) runs on **Modal GPU**; only
small artifacts come back to the local box for browser viewing.

## Constraints that shape sequencing

- **No local GPU**, **87 GB free local disk**. The ~98 GB sample and the beamformed output
  live on Modal's shared `research` volume. We never pull the raw or full beamformed data local.
- **Stuck with the data we have**: one ~4-minute human scan (223 acquisitions @ 222 Hz,
  demodulated complex IQ). Anything needing more data (e.g. ULMShare) is documented but
  **out of scope to act on now** — see `ulmshare-downstream.md`, `ideas-beyond-data.md`.
- **Modal spend requires Rome's explicit approval**, and every container must be verified torn
  down after each run.

## Prioritized sequence

The ordering is deliberate: engineering enablers first (they make every later experiment
cheaper and faster), then the validation backbone, then correctness, then algorithmic gains
ordered by (expected fidelity gain) / (risk).

| # | Bead | Why here |
|---|------|----------|
| 0 | `MODAL` (mb-crr.1) | Nothing runs without it. Gated on spend approval. |
| 1 | `STREAM-BF` (mb-crr.2) | Streaming beamformer — turns "load all, write at end" into bounded-memory, resumable, chunked. Makes everything after it cheaper. |
| 1 | `DIFFVIZ` (mb-crr.5) | The validation backbone. No insight is "done" until it has a 3D volume + diff vs baseline. |
| 1 | `BENCH` (mb-crr.17) | Objective metrics so we don't tune until the render merely looks good. |
| 2 | `BOUNDARY` (mb-crr.6) | Boundary-safe tracking by default (cumulative offsets, timestamps, 3 modes). Pure correctness. |
| 2 | `CLI-HONEST` (mb-crr.7) | Kill silent fallbacks (`gaussian_fit`, `patch_radius`). Scientific-integrity bug. |
| 2 | `SHARD` / `STREAM-TRACK` (mb-crr.3/.4) | Shards + miniature dataset + streaming track output; faster dev loops. |
| 2 | `TESTS` (mb-crr.8) | Unit + golden small-data regression. |
| 3 | `PSF-DETECT` (mb-crr.9) | Anisotropic, PSF-aware detection + per-detection localization covariance. |
| 3 | `KALMAN-GATE` (mb-crr.10) | Gate on predicted state, not last observation; richer association. |
| 3 | `MOTION` (mb-crr.11) | Motion correction from the tissue component before accumulation. |
| 4 | `ADAPT-SVD` (mb-crr.12) | Spatially adaptive SVD (overlapping blocks + region cutoffs). |
| 4 | `SKULL` (mb-crr.13) | Skull-aware beamforming — largest fidelity lever, highest risk. See `skull-aware.md`. |
| 5 | `MULTIBUBBLE` (mb-crr.14) | Overlapping-bubble handling → shorter required acquisitions. |
| 6 | `VESSEL-GRAPH` (mb-crr.15) → `FLOWMATCH` (mb-crr.16) | Joint vessel-network inference, then Rome's flow-matching idea. See `flow-matching.md`. |

## The deeper reframing

The baseline is a hard pipeline of independent stages:

```
beamform → SVD filter → detect → localize → track
```

The thesis behind the later beads is to make it a **partially joint inverse problem** where each
stage carries *uncertainty* forward instead of collapsing to a hard thresholded decision:

```
channel IQ
  → corrected acoustic field        (SKULL: local aberration / SoS)
  → tissue + moving scatterers       (ADAPT-SVD / MOTION / robust PCA)
  → temporally coherent trajectories (PSF-DETECT covariance → KALMAN-GATE)
  → vascular graph + flow field      (VESSEL-GRAPH → FLOWMATCH)
```

We don't solve it end-to-end on day one. We move each interface from "hard decision" to
"distribution," one bead at a time, and prove each step with a 3D diff vs baseline.
