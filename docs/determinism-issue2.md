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

### 1. The adaptive cutoff never moves on this data — because it never fires

On the full acquisition (700 × 25 × 225 × 378), no temporal mode's spectral centroid comes
close to the 100 Hz tissue boundary:

| variant | max mode centroid (Hz) | modes above 100 Hz | cutoff |
|---|---|---|---|
| `v0_shipped` | 69.6 | 0 | 70 (fallback) |
| `v1_pr4` | 70.7 | 0 | 70 (fallback) |
| `v2_ours` | 70.4 | 0 | 70 (fallback) |

This is not an acq-0 quirk. Across the five acquisitions available as beamformed shards
(0, 30, 59, 111, 222) the highest mode centroid ranges **69.6–72.9 Hz** with the shipped
score and **68.8–73.2 Hz** with the phase-invariant one — never within 25 Hz of the
boundary, always the fallback.

`spectral_centroid_cutoff` therefore returns its `10% of frames` fallback — a constant 70 —
for every variant and every perturbation. **On this dataset `--svd-method adaptive` is a
fixed cutoff of 70, not an adaptive one**, and a cutoff that cannot move cannot be the
mechanism behind run-to-run detection changes. That rules out the mechanism proposed in the
issue *and* the one PR #4's description gives ("shifts the selected clutter cutoff").

### 2. complex64 → complex128 removes a small but real jitter

Component-level probe on the 3 GB crop (`[:, :, 60:180, 100:280]`), comparing raw detection
sets against the baseline run:

| variant | threads 16 vs 1 | 1e-6 input perturbation | eigenvector phase re-draw |
|---|---|---|---|
| `v0_shipped` (complex64) | 99.86% overlap | 99.44% | 100% |
| `v1_pr4` (complex128) | 100% | 100% | 99.997% |
| `v2_ours` (complex128 + invariant score) | 100% | 100% | 99.997% |

So PR #4 does what it claims — it makes the detections bit-stable under both reduction-order
and epsilon perturbation — but the effect it removes is on the order of **half a percent of
detections**, not the ~85% in the report. (The residual 0.003% under a phase re-draw is
floating-point rounding in the projection, not a cutoff change: the cutoff is 70 throughout.)

### 3. End-to-end, the pristine pipeline is ~94% reproducible even BEFORE the fix

Two runs of the pristine `ultratrace-ulm track` (bug-report recipe) on the same beamformed
acquisition, second run at a different BLAS thread count:

| revision | detections A / B | exact voxel overlap | median NN | p90 NN |
|---|---|---|---|---|
| `6ed4116` (pre-#4, complex64) | 38,282 / 38,268 | **93.76%** | 0.97 µm | 3.1 µm |
| `1939006` (HEAD, complex128) | 43,039 / 43,040 | **99.998%** | 0 | 0.004 µm |

PR #4 works: post-merge the two runs differ by a single detection out of 43,040. Note the
second column too — the fix does not only stabilise the detection field, it **changes** it,
from 38.3k to 43.0k detections (+12%). complex64 was over-suppressing the retained
blood/bubble subspace, which is the same effect measured in `docs/ideation/15` (mb-a0a).
Anyone comparing results across this commit is comparing two different detection fields, not
two runs of the same one.

The detection counts match the report (~38–39k), but the disagreement does not: the ~6% of
detections that differ are displaced by **~1 µm** — sub-voxel localization landing either
side of an integer boundary — whereas the report describes peaks moving **0.27 mm**, two
orders of magnitude further, with only 14% agreement.

**We could not reproduce the reported severity.** Starting from a fixed beamformed
acquisition, the pre-fix code is already ~94% reproducible at voxel resolution and
essentially identical in position. Two candidate explanations were then checked and both
are ruled out:

- **Beamforming is not the hidden variable.** Two runs of the pristine `beamform` command on
  the same raw input produce a **bit-identical** compound (700×25×225×378, max abs
  difference exactly 0.0) — the assumption the bug report made holds.
- **The randomized-SVD path upstream mentioned is not in the public code.** `svd.py` maps the
  `gpu`/`gpu_full`/`randomized` method names onto the same deterministic `fast` covariance
  projection, and the public package has no GPU SVD module at all. Whatever randomized path
  they removed lived in their private repo, so it cannot explain a run made from the public
  CLI.

So the gap stands: on the public code, from a fixed input, in this environment, the pre-fix
non-determinism is **~6% of detections displaced by ~1 µm**, not ~86% displaced by 0.27 mm.
Either the original measurement carried an environment-specific factor (a different
BLAS/threading build is the obvious candidate — the c64 path's sensitivity is exactly the
kind of thing that varies by library), or the two runs it compared differed in more than the
seed. Closing that would need the original run's artifacts; it does not change any
conclusion below.

### 4. The phase defect is real, and PR #4 does not address it

`|rfft(u.real)|²` scores the real part of an eigenvector that a Hermitian solver defines
only up to a unit phase. Re-drawing that phase and rescoring the same eigenvectors:

- on the real acquisition, `v1_pr4`'s max centroid moves 70.7 → 70.9 → 71.7 Hz across phase
  draws, while `v2_ours` stays at 70.4 Hz for every draw and every perturbation;
- on synthetic tissue+blood data (20 phase draws), the shipped score moves a mode's centroid
  by up to **17.5 Hz**; the phase-invariant score by **0.000 Hz**
  (`scratchpad/phase_sensitivity.py`).

Nothing here flips the cutoff, because on this data the cutoff is pinned to the fallback.
On data where a mode sits near 100 Hz it would — which is the mb-a0a failure we already hit,
where LAPACK and cuSOLVER chose different cutoffs and the track counts came out ~15× apart.

## Bottom line

- Issue #2 is closed for a real reason: PR #4 makes the decomposition double-precision, and
  that is measurably what removes the reduction-order jitter (99.44% → 100% at the component
  level). Nothing about the merge is wrong.
- But the *stated mechanism* is not the one operating on this data: the cutoff is a constant
  70 on every acquisition tested, so nothing about cutoff selection can move detections. What
  PR #4 actually fixed is eigenvector instability in the **retained subspace** at that fixed
  cutoff.
- The reported severity (14% overlap, 0.27 mm) does not reproduce: pre-fix is ~94%
  reproducible with ~1 µm displacements, beamforming is bit-identical across runs, and the
  randomized path upstream referred to is not in the public code. That gap is unexplained and
  is flagged as unexplained.
- One defect from the same code path survives the merge: the mode score is phase-dependent,
  so the cutoff is a function of the linear-algebra library's phase convention rather than of
  the data. That is a latent CPU-vs-GPU divergence, not a cross-run one — which is exactly
  how it bit us in mb-a0a. Draft fix (not pushed):
  `docs/upstream-pr-draft-phase-invariant-cutoff.md` + `docs/upstream/`.
- Also flagged for upstream, no correctness impact: PR #4's `np.asarray(matrix,
  complex128)` materialises a ~24 GB double copy of a single acquisition's matrix, one
  commit after the streaming change that removed exactly that kind of RAM spike.
