# DRAFT upstream PR — phase-invariant adaptive SVD cutoff (follow-up to #4)

> **SUPERSEDED 2026-07-31 — do not send this.** Upstream merged the same fix hours before we
> would have: PR #5 (commit `b8ba5b7`, "Make the spectral centroid invariant to eigenvector
> phase") replaces `|rfft(u.real)|²` with the two-sided power spectrum of the complex
> eigenvector folded onto `|f|` — the identical change, arrived at independently, with the
> same reasoning about the LAPACK representative and about folding powers rather than
> amplitudes so opposite-sign Doppler cannot cancel.
>
> Their measurements agree with ours where they overlap. They report a phase re-draw moving
> the old centroid by up to **15.6 Hz** and the new one by ~1e-11 Hz (we measured 17.5 Hz vs
> 0.000 Hz on synthetic data). They independently state that no centroid reaches the 100 Hz
> boundary on the released dataset — "max 85-96 Hz, so the cutoff comes from the
> 10%-of-frames fallback either way" — which is the same finding as ours (69.6-72.9 Hz on the
> retracted export, cutoff pinned to the fallback), now confirmed on the corrected data where
> the fallback is 24 of 240 frames.
>
> PR #6 (`ae169ea`) then answered the other half of issue #2: the frame rate is read from the
> file's own `/config` instead of being passed out of band, and the fixed 100 Hz tissue
> boundary gains a `--tissue-velocity` equivalent in mm/s (100 Hz ≈ 40 mm/s against a Nyquist
> velocity of 44.5 mm/s on this data).
>
> **What is left of this draft:** only the chunked complex128 Gram, and it is a much smaller
> point than drafted below. The 24 GB figure was computed on the retracted export's
> dimensions (700 frames × 2.13M voxels). On the corrected sample the full-copy cost is
> **~4.1 GB against a 2.0 GB complex64 original** — still avoidable, no longer alarming.
> The rest of this file is kept as the record of what we measured, not as something to send.

**Status: draft for Rome. Nothing has been pushed to `alephneuro/microbubbles`.**
Branch material lives in the scratchpad clone; the patch is reproduced at the bottom of
this file so it survives the scratchpad. Evidence: `docs/determinism-issue2.md`.

Target: `alephneuro/microbubbles`, on top of `1939006` (PR #4).

---

## Title

Adaptive SVD cutoff: score modes phase-invariantly, and keep the complex128 Gram out of RAM

## Body

#4 does what it says: running `track` twice on one beamformed acquisition (bug-report recipe,
different BLAS thread counts) goes from **93.8%** detection agreement before the merge to
**99.998%** after — 1 detection out of 43,040. Two things are left over from the same code
path.

Worth knowing alongside that: the same change moves the detection count from 38,282 to
43,039 on identical input (+12%). complex64 was over-suppressing the retained blood/bubble
subspace, so results either side of this commit are two different detection fields, not two
runs of one.

**1. The mode score still depends on a convention the eigensolver is free to change.**

`spectral_centroid_cutoff` scores each temporal mode with `|rfft(u.real)|**2`. A Hermitian
eigensolver defines each eigenvector only up to an arbitrary unit phase: `u` and
`e^{iθ}u` are the same mode, and which one comes back is up to the LAPACK build (and up to
cuSOLVER, for anyone running the GPU path). Taking `.real` turns that free phase into a
different spectrum, so the same mode can score differently — and the cutoff is a threshold
crossing, so a score that moves can select a different number of clutter components and
change every downstream detection.

Measured on synthetic tissue+blood data (128 frames, 222 Hz), re-drawing the eigenvector
phases 20 times and rescoring the *same* eigenvectors:

| score | max per-mode centroid shift under a phase re-convention |
|---|---|
| `\|rfft(u.real)\|²` (current) | **17.5 Hz** |
| `\|fft(u)\|²` over `\|f\|` (this PR) | **0.000 Hz** |

17.5 Hz of movement against a 100 Hz decision boundary is enough to move the cutoff on
data where a mode sits near the boundary. We hit the practical version of this in a fork:
the CPU (LAPACK) and GPU (cuSOLVER) paths chose different cutoffs on the same acquisition
and the track counts came out ~15× apart until the score was made phase-invariant.

The fix is to score the full complex spectrum and take the centroid over `|f|`. Besides
being invariant to the solver's phase, it is the frequency content of a complex
(signed-Doppler) mode — the current score folds the two Doppler directions together
before measuring them.

**2. `np.asarray(matrix, dtype=np.complex128)` doubles a matrix that was just made
streaming.**

#4 materialises a complex128 copy of the full (frames, voxels) matrix in both
`spectral_centroid_cutoff` and the `fast` branch. For one 700-frame acquisition at
25×225×378 that copy is ~24 GB, on top of the complex64 original — right after #3 stopped
holding all acquisitions in RAM. The decomposition only needs the (frames, frames) Gram in
double precision, and that can be accumulated from complex64 chunks: same double-precision
eigenvectors, a few hundred MB instead of 24 GB. The projection then runs in complex64,
matching the declared complex64 output.

**Tests** (`tests/test_svd_determinism.py`): phase-invariance of the mode score, bit-stable
cutoff across repeated calls, cutoff unchanged under a 1e-6 (float32-epsilon) input
perturbation, and reproducible `filter_svd_3d(method="adaptive")` output.

**One observation, no change requested.** On `sanitized_neutral_ultratrace.h5`, no mode's
centroid reaches the default 100 Hz tissue boundary on any acquisition we checked (acqs 0,
30, 59, 111, 222; max per acquisition 69.6–72.9 Hz), so `--svd-method adaptive` silently
takes the "10% of frames" fallback — it is a fixed cutoff of 70 on this data, not an adaptive
one. That may be intended for this recording, but it does mean the adaptive path is not
exercised by the public sample, and it is why the cutoff itself was not the source of the
run-to-run variation in #2 (what #4 fixed is the retained subspace at a fixed cutoff).

---

## Patch

Applied to `ultratrace_ulm/svd.py` on top of `1939006`:

- add `_temporal_gram(matrix, subtract_mean=True, chunk=300_000)` — complex128 Gram
  accumulated from complex64 chunks;
- add `mode_centroids(u, frame_rate_hz)` — power-weighted mean `|f|` from the full complex
  spectrum;
- `spectral_centroid_cutoff` uses both;
- the `fast` branch uses `_temporal_gram(..., subtract_mean=False)` and projects in
  complex64.

Plus `tests/test_svd_determinism.py` (4 tests, pass locally).
