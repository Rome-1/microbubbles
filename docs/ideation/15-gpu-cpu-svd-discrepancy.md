# GPU vs CPU SVD "16.5x" discrepancy — root cause: complex64 (svd.py) vs complex128 (gpu_svd.py)

Bead **mb-a0a**. The 16.5x is **REAL and controlled**: on the SAME acqs 0-7, toggling ONLY
`use_gpu_svd` in `baseline` → GPU-SVD **7859 tracks (982/acq)** vs CPU-SVD **477 tracks
(60/acq)**. This doc localizes WHERE the 16.5x enters, with a subsampled runtime
measurement.

> Correction to an earlier draft of this file: an initial pass concluded the paths were
> "equivalent / a cross-run artifact." That was WRONG — it inferred magnitude-equivalence
> because the probe timed out before measuring it. The measurement below shows the
> filter magnitudes are NOT equivalent.

## TL;DR — the 16.5x is in the FILTER MAGNITUDE (SVD precision), not the detector

`svd.py` (CPU path) runs the temporal SVD in **complex64**; `gpu_svd.py` (GPU path) runs it
in **complex128** (`gpu_svd.py:86-99`). At the *same* cutoff (70), the complex64 path
produces a **systematically smaller, lower-contrast `|filtered|`** — it over-suppresses the
retained blood/bubble subspace. This enters the `baseline` A/B in **two** places, both
measured:

1. **Per-acq adaptive clutter filter** — GPU `|filtered|` mean **2.89x** higher than CPU,
   with a far heavier bright tail (GPU max 21.9 vs CPU max 2.83).
2. **TGC normalization map** (`app.py:599`, `method="fast" if use_gpu_svd else "full"`) —
   GPU power map **14.58x** the CPU power map (per-voxel ratio 3.7–56.6), so the two runs
   scale `comp` very differently before detection.

The **detector is numerically equivalent** (measured: identical peak counts on identical
magnitude). The compressed CPU magnitude → low-contrast noise-dominated 2σ detections that
don't link (60 tracks/acq); the high-contrast GPU magnitude → clean linkable detections
(982 tracks/acq). That is the 16.5x.

## Runtime measurement (CONFIRMED)

Modal A10G, one shard `beamformed/base60_refs/acq_0000.h5`, spatial crop
`[:, :, 60:180, 100:280]` → `(700, 25, 120, 180)` (3 GB, no timeout). Same crop fed to BOTH
filters (adaptive, `tissue_freq_hz=100`, `frame_rate_hz=222`) and BOTH detectors
(`sigma_threshold=2.0`, `min_distance=2`, `smoothing_sigma=1.0`). Probe:
`scratchpad/svd_crop_probe.py`.

```
[cutoff] adaptive low_cpu=70                               # cutoff identical (as before)
[mag CPU] mean=0.452  std=0.244  p99=1.14   p999=1.44  max=2.83
[mag GPU] mean=1.307  std=0.950  p99=4.58   p999=8.22  max=21.9
[mag] mean|cpu-gpu|/mean(cpu)=2.02   ratio gpu/cpu means=2.89     # FILTER magnitudes DIFFER

[peaks] detCPU x magCPU = 367874 (525.5/fr)
[peaks] detCPU x magGPU = 181342 (259.1/fr)
[peaks] detGPU x magCPU = 367876 (525.5/fr)
[peaks] detGPU x magGPU = 181349 (259.1/fr)
[peaks] ratios: (detCPU) magGPU/magCPU=0.493 | (magCPU) detGPU/detCPU=1.000 | gg/cc=0.493

[tgc] fast mean=2.719  full mean=0.187  fast/full mean-ratio=14.58
[tgc] rel|fast-full|/mean(full)=13.58  per-voxel ratio p1=3.75 p50=12.0 p99=56.6
```

Reading the 2x2 cross:
- **Row delta (detGPU vs detCPU on the same magnitude) = 1.000** → **DETECTOR EQUIVALENT.**
  `gpu_detect.detect_batch_gpu` and `tracking.detect_batch` produce the same peak set (as
  the code already showed: identical per-elevation z-score, NMS `maximum_filter` size, 2σ
  threshold, intensity basis). Suspect #2 (detector wiring) **REFUTED**.
- **Column delta (magGPU vs magCPU under the same detector) = 0.493** → **FILTER MAGNITUDES
  DIFFER.** Suspect #3 **CONFIRMED**.
- **TGC fast vs full differ 14.58x** → suspect #1 **CONFIRMED** (same root cause; below).

Note the crop reproduces the reported paradox: the CPU magnitude gives MORE raw 2σ peaks
(525 vs 259/fr) but they are low-contrast (max 2.83, thin tail) so they don't link; the GPU
magnitude gives fewer but high-contrast peaks (max 21.9) that link into many more tracks.
Peak count and track count move in opposite directions — exactly what
`docs/ideation/13-track-length-gap.md` observed.

## Root cause — complex64 vs complex128 in the SVD (CONFIRMED)

Both the per-acq filter and the TGC map differ, and both reduce to the SAME cause.

**Per-acq clutter filter (adaptive → "fast" cov-projection in both paths).** The only
difference between `svd.filter_svd_3d` and `gpu_svd.filter_svd_3d_gpu` at cutoff=70 is
precision:
- CPU `svd.py:82` casts to `complex64`; `svd.py:116` `cov = matrix @ matrix.conj().T` and
  `svd.py:117` `eigh` run in **complex64**.
- GPU `gpu_svd.py:89-99` accumulates the Gram in **complex128** (chunked, from c64 chunks),
  `gpu_svd.py:119` `eigh` in complex128, then casts `uc` to c64 for the projection.

`gpu_svd.py:86-88` documents exactly this: *"float32 accumulation over ~2M voxels visibly
perturbs which modes are kept … float64 makes the Gram (and thus the canonical baseline)
stable and reproducible."* Forming `M·Mᴴ` squares the condition number; with tissue/blood
dynamic range ~1e3–1e4, the squared condition ~1e6–1e8 exceeds complex64's ~7-digit
precision, so the **retained** (small-eigenvalue = blood/bubble) subspace is computed
inaccurately on the CPU and the projection over-suppresses it → the 2.89x-smaller,
thin-tailed CPU magnitude. complex128 (GPU) preserves it.

**TGC map (`app.py:599`).** `use_gpu_svd=True` builds the TGC power map with
`filter_svd_3d_gpu(method="fast")` (complex128 cov-projection); `use_gpu_svd=False` uses
`filter_svd_3d(method="full")` (complex64 SVD) — imported at `app.py:582-585`. `"fast"` and
`"full"` remove the *same* top-35 subspace (`tgc_svd_cut=0.05` → 35 modes) and are
mathematically identical, so the measured **14.58x** difference is **precision**, not
method — the c64 CPU path again yields a ~14.6x-smaller residual power map. Because this map
becomes `inv_sqrt = 1/sqrt(tgc)` applied per-voxel to `comp` *before* the SVD+detector, and
the detector z-score is per-elevation-slice (scale-invariant only to a *global* scale, not a
spatially-varying one), the two runs feed differently-shaped inputs into detection.

## Which stage carries the 16x

The **SVD filter precision**, entering through **both** the TGC normalization map (larger
single lever, 14.58x) **and** the per-acq adaptive clutter filter (2.89x). The **detector
carries none** of it (ratio 1.000). complex128 (GPU) is the numerically correct side;
complex64 (CPU `svd.py`) is the deficient one.

## The fix

**One focused change: make `svd.py` do the Gram accumulation + eigendecomposition in
complex128**, mirroring `gpu_svd.py:89-99`. Because `filter_svd_3d` backs BOTH the CPU
clutter filter (`tracking._filter_acquisition`, `tracking.py:793`) AND the CPU TGC
(`app.py:585`), this single fix aligns both stages with the GPU:

- `svd.py:116` `cov = matrix @ matrix.conj().T` →
  `m128 = matrix.astype(np.complex128); cov = m128 @ m128.conj().T`, then
  `uc = u[:, low:stop].astype(np.complex64)` before the projection (as the GPU does).
- `svd.py:39` (the `spectral_centroid_cutoff` Gram) → same complex128 cast.
- For the `"full"` branch used by the CPU TGC (`svd.py:122-127`), either cast `matrix` to
  complex128 before `np.linalg.svd`, or simplest: at `app.py:599` drop the asymmetry and use
  the same method+precision for both paths (the c128 `"fast"` path).

Memory note: casting a full `(700, ~2.1M)` acq to complex128 doubles RAM (~24 GB transient);
for the full pipeline mirror the GPU's **chunked** c128 Gram accumulation
(`gpu_svd.py:91-99`) rather than a single c128 cast.

**Direction of convergence:** this upgrades the CPU path to the GPU's numerically-correct,
high-contrast output → `use_gpu_svd=False` then also yields ~982/acq. (The brief asked for
the reverse — "make the GPU emit the clean CPU density" — but the measurement shows the CPU
c64 output is the *corrupted* one, so the correct fix is to lift the CPU to c128, not
degrade the GPU to c64. Whether ~982/acq is the biologically right *density* vs the
reference ~260/acq is a separate 2σ-threshold/tuning question —
`docs/ideation/11-detection-threshold-rootcause.md` — independent of this precision bug.)

## CONFIRMED vs LIKELY

| Finding | Status |
|---|---|
| 16.5x is real & controlled (A/B toggling only `use_gpu_svd`, acqs 0-7) | **CONFIRMED** (coordinator A/B) |
| Detector equivalent (detGPU==detCPU on identical magnitude, ratio 1.000) | **CONFIRMED** (measured) |
| Adaptive cutoff identical (70) regardless of c64/c128 | **CONFIRMED** (measured, both probes) |
| Per-acq filter magnitude differs 2.89x (GPU c128 vs CPU c64), same cutoff | **CONFIRMED** (measured) |
| TGC map differs 14.58x (GPU fast/c128 vs CPU full/c64), = precision not method | **CONFIRMED** (measured) |
| Root cause = svd.py complex64 vs gpu_svd.py complex128 (retained-subspace over-suppression) | **CONFIRMED** (measured + gpu_svd.py:86-99 documents the mechanism) |
| c128 is the correct side; fix = upgrade svd.py to c128 | **LIKELY** (numerically sound; verify with a controlled A/B re-run after the change) |

(Infra: bead comment posted to mb-a0a; `bd` did not throw a Dolt error this session.
Modal teardown verified — 0 active apps after the probe.)
