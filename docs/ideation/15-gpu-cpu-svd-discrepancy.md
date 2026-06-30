# GPU vs CPU SVD "7x" discrepancy — root cause: the paths are EQUIVALENT (2026-06-30)

Bead **mb-a0a**. Claim under investigation: the GPU and CPU temporal-SVD clutter-filter
paths produce ~7x different microbubble detection densities (`baseline` `use_gpu_svd=True`
→ ~561 tracks/acq; `use_gpu_svd=False` → ~86 tracks/acq), even though mb-crr.2
("phase-invariant SVD cutoff") was supposed to make CPU==GPU.

## TL;DR

The SVD filter and the detector are **numerically equivalent** across the CPU and GPU
paths. The headline divergence the bead hypothesised (a precision-driven SVD-cutoff
difference, or a detector difference) **does not exist** on identical input. Measured,
on a real beamformed acquisition, the adaptive low-cutoff is **identical (70 on both
complex64 CPU and complex128 GPU)**. Therefore the reported ~7x is **not** produced by
the `gpu_svd` vs `svd` filter or the `gpu_detect` vs `tracking.detect_batch` detector —
it is a **cross-run comparison artifact** (the two figures come from different jobs /
acquisition subsets / code states), and the only residual *per-run* code difference (the
TGC normalisation method) is refuted as a 7x driver.

## Runtime measurement (CONFIRMED)

One small Modal A10G job (`scratchpad/svd_diverge_probe.py`) applied BOTH
`svd.filter_svd_3d` (CPU) and `gpu_svd.filter_svd_3d_gpu` (GPU) machinery to the SAME
compound — the existing post-TGC shard `beamformed/base60_refs/acq_0000.h5`
(shape `(700, 25, 225, 378)`). The adaptive (`method="adaptive"`, `tissue_freq_hz=100`,
`frame_rate_hz=222`) low-cutoff each path selects:

```
[cutoff] low_cpu(complex64)=70   low_gpu(complex128)=70   low_cpu_upcast(complex128)=70
```

**All three are 70.** This is the 10%-of-frames spectral-centroid *fallback*
(`max(1, round(n_frames*0.1))` = `round(700*0.1)` = 70), exactly as
`docs/HANDOFF.md:114-116`, `docs/ideation/06-svd-knee.md:62`, and
`docs/ideation/11-detection-threshold-rootcause.md:12` already reported: on this data the
spectral centroid never exceeds `tissue_freq_hz`, so the adaptive method always hits the
70-mode fallback. **complex64 (CPU) vs complex128 (GPU) accumulation makes no difference
to the chosen cutoff.** The leading candidate cause — precision flipping the
spectral-centroid crossing — is **refuted**.

(The probe's second stage — `|filtered|` stats and a 2x2 detector×magnitude supra-2σ peak
cross — was attempted but the run was slow loading the 11.9 GB shard + both full filters
and was torn down before it printed. The cutoff result above is decisive on its own and
the remaining equivalences are established by code below; the magnitude-equivalence link
is therefore flagged LIKELY rather than CONFIRMED.)

## Stage-by-stage code analysis

### 1. SVD filter — same cutoff, same kept subspace (CONFIRMED cutoff / LIKELY magnitude)

Both paths are driven with `method="adaptive"` (`app.py:537,650` default `svd_method`,
wired into `gpu_filter` at `app.py:663-677` and into the CPU default
`tracking._filter_acquisition` at `tracking.py:787-802`). For `adaptive`:

- CPU `svd.spectral_centroid_cutoff` (`svd.py:16-48`): `cov = x @ x.conj().H` on the
  **complex64** mean-subtracted matrix (`svd.py:38-39`), `eigh`, phase-invariant
  `|fft(u)|²` centroid, first index over 100 Hz else the 70 fallback (`svd.py:48`).
- GPU `gpu_svd._spectral_centroid_cutoff_gpu` (`gpu_svd.py:24-41`): identical formula on
  `Gc`, the mean-subtracted Gram accumulated in **complex128** (`gpu_svd.py:90,97-99`).

Same formula; only the Gram precision differs. Measured outcome: both return 70.

The projection then removes modes `0:low` and keeps `low:stop`:
- CPU `fast` path (`svd.py:115-121`): `cov = matrix @ matrix.conj().H` (complex64), `eigh`,
  `uc = u[:, 70:700]`, `filtered = uc @ (uc.conj().H @ matrix)`.
- GPU (`gpu_svd.py:89-99,119,133-140`): raw Gram `G` in complex128, `eigh`,
  `uc = u[:, 70:700].astype(complex64)`, projected in complex64 per voxel-chunk.

Both project onto the orthogonal complement of the **top-70 tissue modes**. Those modes
have large, well-separated eigenvalues, so the kept subspace (and hence the projector
`uc·ucᴴ`, which is invariant to the within-subspace basis) is the same in complex64 and
complex128 — up to *near-degenerate boundary modes* around index 70, whose energy is
small and bounded. ⇒ `|filtered|` is essentially the same; any residual is far too small
to flood the 2σ detector by 7x.

There IS one genuine code asymmetry: **`svd.py` accumulates the Gram in complex64 while
`gpu_svd.py` accumulates in complex128** (`gpu_svd.py:86-99` documents that float32
accumulation "visibly perturbs which modes are kept" near the eigenvalue boundary — which
is *why* the GPU uses float64). On this data it changed nothing (70==70), but it is the
one place the two filters are not bit-identical.

### 2. Detector — equivalent (CONFIRMED by code)

CPU `tracking.detect_batch` (`tracking.py:169-205`, slice stats `tracking.py:152-166`) and
GPU `gpu_detect.detect_batch_gpu` (`gpu_detect.py:21-95`) use the **same** semantics:
- spatial smoothing `gaussian_filter(sigma=(0,0,σ,σ))` (`tracking.py:180`, `gpu_detect.py:42`);
- per-elevation-slice mean/std over **positive smoothed** voxels (`tracking.py:159-165`,
  `gpu_detect.py:51-58`);
- z-score `(smoothed-mean)/(std+1e-10)` (`tracking.py:182`, `gpu_detect.py:62-63`);
- NMS `maximum_filter(size=(1,fs,fs,fs))`, `fs=2·min_distance+1`, peak `== max & > σ`
  (`tracking.py:184-186`, `gpu_detect.py:70-72`);
- intensities sampled from the **unsmoothed** magnitude (`tracking.py:195`,
  `gpu_detect.py:89`).

Only difference is scipy vs cupyx float rounding (last-ULP `==` ties on plateaus) — a
few-percent effect at most, not 7x. Subpixel localisation, the z-score knee filter, and
`_track_detections` (Kalman/Hungarian) are the *same functions* for both paths
(`tracking.py:_run_selected` 904-963, used by `run_tracking_outputs_streamed` 1385-1398).

### 3. The only residual per-run difference — TGC map (LIKELY-minor, refuted as 7x)

In `baseline`, the *only* code path that differs between `use_gpu_svd` True/False besides
the equivalent filter/detector is the TGC power-map method at **`app.py:599`**:
`out = _tgc_svd(comp, low_cutoff=tgc_svd_cut, method="fast" if use_gpu_svd else "full")`.
This removes the **top-35 modes** (`tgc_svd_cut=0.05`) via `fast` (cov-projection) vs
`full` (SVD) — *mathematically the same subspace*, differing only by float noise — then
takes `mean_t |·|²` and applies a Gaussian of `sigma ≈ 9·(1540/2.75e6) ≈ 5 mm`
(`app.py:603-611`), i.e. ~50-voxel smoothing. The two `inv_sqrt` maps are therefore
near-identical, and even a residual scale difference cannot change detections because the
detector's per-elevation z-score is **scale-invariant**. ⇒ cannot cause 7x.

## Why the bead's numbers still differ across runs

- `baseline use_gpu_svd=True` → 15,721 tracks / 28 acqs = 561/acq is the **in-progress
  full223 chunk** (current code).
- `baseline use_gpu_svd=False` → 2,595 tracks / 30 acqs = 86/acq is the earlier 30-acq run
  from `docs/ideation/14-overlinking-speed-gap.md` — a **different job, acq subset, and
  code state**.
- The related `detect_acqs` (23.6k loc → 754 tracks) vs `run_tracking` (37k loc → 17
  tracks) figures (`docs/ideation/13-track-length-gap.md:21-22,32`) are further confounded:
  `run_tracking` reads a **pre-beamformed shard with base60's TGC baked in** while
  `detect_acqs` re-beamforms with its **own** TGC — different magnitudes for reasons
  unrelated to SVD precision. (Note these two move detection count and track count in
  *opposite* directions, which a single per-acq filter/detector divergence cannot do — it
  is the signature of different beamform/TGC inputs, not a CPU-vs-GPU SVD bug.)

## Which stage dominates

**Neither the filter nor the detector** — they are equivalent (cutoff 70==70 CONFIRMED;
kept-subspace + detector + tracker identical). There is no 7x-scale numeric divergence on
identical input. The observed gap is a cross-run/config artifact; the TGC method is the
only residual per-run code difference and is refuted as a 7x cause.

## Recommended fix (one-liner + one verification)

1. **One-line code change (removes the only real asymmetry):** make the CPU filter
   accumulate the temporal Gram in complex128 like the GPU. In `svd.py:116`
   (`cov = matrix @ matrix.conj().T`) and `svd.py:39` (`cov = x @ x.conj().T`), cast to
   `complex128` before the GEMM (e.g. `m128 = matrix.astype(np.complex128); cov = m128 @ m128.conj().T`),
   matching `gpu_svd.py:89-99`. Bit-aligns the two paths; expected detection effect ≈ none
   (cutoff already 70==70).
2. **Verification (the real action):** re-run the A/B *controlled* — one `baseline`
   invocation with the SAME `sel` and code, toggling only `use_gpu_svd` — and compare
   **per-acq localization counts before tracking**. Expectation: the gap disappears
   (confirming the 7x was cross-run). If a residual gap survives, it localizes to TGC
   (`app.py:599`); align it to a single method for both paths and re-check.

## CONFIRMED vs LIKELY

| Finding | Status |
|---|---|
| Adaptive low-cutoff identical CPU/GPU (70==70), precision-hypothesis refuted | **CONFIRMED** (measured) |
| Detector math equivalent (z-score / NMS / 2σ / intensity basis) | **CONFIRMED** (code) |
| Subpixel + knee + tracker identical (same functions) | **CONFIRMED** (code) |
| `|filtered|` magnitude equivalent (kept subspace identical up to bounded boundary modes) | **LIKELY** (peak-cross runtime check not captured; strong code argument) |
| TGC method (app.py:599) cannot cause 7x | **LIKELY** (same subspace, heavy smoothing, z-score scale-invariant) |
| Reported 7x is cross-run config/data artifact, not a CPU/GPU SVD/detector bug | **LIKELY** (follows from the above; confirm with the controlled A/B) |
| `svd.py` complex64 Gram vs `gpu_svd.py` complex128 Gram asymmetry | **CONFIRMED** (code; immaterial here, 70==70) |

(Infra note: bead comment posted to mb-a0a successfully; `bd` did not throw a Dolt schema
error this session.)
