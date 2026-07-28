# Upstream issue #2 (detection non-determinism): what the fix does and does not cover

Bead **mb-ivt**. Upstream closed
[alephneuro/microbubbles#2](https://github.com/alephneuro/microbubbles/issues/2) on
2026-07-28 by merging [PR #4](https://github.com/alephneuro/microbubbles/pull/4) — the
temporal SVD eigendecomposition now runs in **complex128** instead of complex64. Nobody
re-ran the reported repro against the merged fix, so this document does that, and then
asks whether complex128 is the whole story.

Three questions, three measurements:

1. **Does PR #4 remove the run-to-run variation?** Two runs of the pristine upstream CLI
   at each revision (pre-#4 and HEAD), same beamformed acquisition, same recipe.
2. **Where does the variation actually enter** — the adaptive cutoff, or the retained
   subspace at a fixed cutoff?
3. **Is anything left?** The mode score is still `|rfft(u.real)|²`, which is
   phase-dependent, while a Hermitian eigensolver fixes each eigenvector only up to an
   arbitrary unit phase.

_Numbers are filled in below as the runs land; every number is from a run on this
project's Modal volume, not an estimate._

## Setup

- Input: `beamformed/base60_refs/acq_0000.h5` (one acquisition, 700 frames,
  25×225×378 voxels) for the component-level probe, and `pristine_bf_1.h5` (the
  acquisition beamformed by the pristine upstream CLI) for the end-to-end check.
- Recipe from the bug report: `--svd-method adaptive --frame-rate 222
  --sigma-threshold 2.0 --knee-filter --temporal-sigma 0`.
- Perturbations, each modelling a real way two "identical" runs differ:
  - **threads** — BLAS thread count 16 vs 1 (changes the reduction order, which is the
    actual cross-machine/cross-run jitter);
  - **eps** — a 10⁻⁶ relative input perturbation (float32-epsilon proxy, the model PR #4
    used to argue its case);
  - **phase** — each eigenvector multiplied by an arbitrary unit phase (what a different
    LAPACK build, or cuSOLVER on the GPU path, is free to return).
- Probes: `scripts/modal/determinism_probe_app.py` (component level, three variants of
  the cutoff side by side) and `scripts/modal/pristine_determinism_app.py` (end-to-end,
  pristine package pinned at each revision).

Variants compared at the component level:

| variant | Gram precision | mode score |
|---|---|---|
| `v0_shipped` | complex64 | `\|rfft(u.real)\|²` (phase-dependent) |
| `v1_pr4` | complex128 | `\|rfft(u.real)\|²` (phase-dependent) |
| `v2_ours` | complex128 (chunked) | `\|fft(u)\|²` over `\|f\|` (phase-invariant) |

## Results

(pending)

## Bottom line

(pending)
