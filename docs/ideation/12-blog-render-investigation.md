# Blog-render investigation: what's available + why ours looks different (2026-06-29)

Triggered by Rome: our renders look nothing like the Aleph blog
(https://alephneuro.com/blog/ultrasound-brain). Investigated what data/methods are available.

## What's available (the answer)
- **Exactly one dataset.** The blog open-sources `github.com/alephneuro/braindump` ("the entire
  pipeline along with the dataset"). braindump = our pipeline (`ultratrace_ulm`); the repo is
  **README + code only** — no bundled data, no example images, no reference renders. The only data is
  the **98 GB `sanitized_neutral_ultratrace.h5`** on Cloudflare R2 (223 acqs) — the one we have.
  No second / volumetric / matrix-array dataset exists. (Our docs already said so: 07-codex-next-levers
  "this is not a matrix-array 3D ULM dataset", 00-context "one dataset".)
- **Two linked papers** (science, not data): `sciencedirect …/S0896627321001513`,
  `science.org/doi/10.1126/scitranslmed.adj3143`.

## Correction to the prior session message
The blog's render was made from THIS single-row dataset. So blog-quality is **NOT data-limited** — it
is a processing/rendering gap. (Earlier I wrongly concluded we'd need volumetric data. We don't.)

## Why ours looks nothing like the blog — two fixable causes
1. **We built the wrong render.** braindump ships two viewers: `track-viewer` (animated track-flow —
   what we built → the "spray") and **`volume`** ("rotatable 3D SVD volume viewer with track overlay").
   The blog is a 3D volumetric super-res image = the **volume viewer**, which we never built on the
   full data. Our track-spray and 2D coronal angiogram are both the wrong artifact.
   - Path: `volume3d` (GPU, `filter_svd_3d_gpu` over `*_refs` → `<base>_power.npy`) →
     `volume_export.export_svd_volume` → `web/volume_viewer` bundle (sparse SVD points + track overlay).
2. **We over-track 3–4×.** README production reference = **~260 tracks/acquisition** (adaptive SVD,
   `--temporal-sigma 0`, sigma 2.0, max_gap 3, min5, "genuinely flowing"). Ours: base223 **754/acq**,
   base223_best **1141/acq**. Too many tracks → dense haze instead of clean vessels. Calibrate (mb-289).

## Caveat this surfaces (mb-rvl is NOT closed)
The reference recipe uses **`--sigma-threshold 2.0` with adaptive SVD** and gets a clean ~260 tracks/acq.
So 2σ-adaptive is not itself a flood — the flood is specific to the `fast`/`knee` `detect_acqs_multi`
path (490k loc/acq) vs adaptive (23.6k). My earlier "2σ is the flood, base60≈5σ" came from a crude
local detector proxy and is suspect. Needs a 1–2-acq GPU repro through the real `detect_batch_gpu`.

## Canonical reference recipe (from braindump README)
```
ultratrace-ulm track --beamformed beamformed.h5 \
  --svd-method adaptive --frame-rate 222 --knee-filter --temporal-sigma 0 \
  --sigma-threshold 2.0 --svd-low-cutoff 0.1 --min-distance 2 --smoothing-sigma 1.0 \
  --tracking kalman --max-gap 3 --min-track-length 5 --max-cost 10
```

## Plan / beads
- **mb-8c5** (P1): build the 3D SVD volume viewer (blog-equivalent) on full 223 — START HERE.
- **mb-289** (P1): calibrate to ~260 tracks/acq (find the 3–4× over-track divergence).
- **mb-0c6** (P2): read the 2 linked papers for methods/specs.
- **mb-rvl** (updated): reconcile the fast/knee flood with a real GPU repro.
"Better" than the blog = our FRC + saturation validation quantifies quality; the blog does not.
