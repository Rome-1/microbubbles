# Render-reproduction session — handoff (2026-06-30 → 07-01)

Goal was to reproduce the Aleph blog's 3D ULM render (`renders/references/Screenshot 2026-06-29
at 11.51.16.png` = velocity-colored tracks, jet 0–38 mm/s, crisp branching vessels on black).
This session did NOT exactly match it, but resolved most of the puzzle and fixed a real bug.
Full detail: `docs/ideation/12-15`. This doc = the summary + the ONE experiment to run next.

## ⭐ THE NEXT EXPERIMENT (Rome's handoff question) — DO THIS FIRST
**"What happens if you just run EXACTLY what they shipped? No changes?"**
Run the PRISTINE upstream braindump pipeline end-to-end (NOT our fork — our fork has the
mb-a0a c128 fix + baseline edits) on the full 223 and render it, as the true baseline.
- Pristine = `pip install "git+https://github.com/alephneuro/braindump.git"` then
  `ultratrace-ulm run` (download→beamform→track→track-viewer) OR `track --beamformed …`.
  Do it on Modal (CPU containers scale past the 10-GPU cap; a GPU is only needed for beamform).
  We already proved pristine-upstream `track` on 1 shard = 28 tracks (≈ our fork's 17 pre-fix).
- **KEY HYPOTHESIS this tests:** the shipped `track` uses the CPU SVD path, which runs the Gram
  in **complex64** (what we called the mb-a0a "corruption"). That path is SPARSE (~28–60 tracks/
  acq) and high-contrast-selective. **The reference's CRISPNESS may come precisely from that
  sparseness** — meaning our c128 "fix" + 3.5σ produced a NUMERICALLY-more-complete but VISUALLY
  BUSIER render, i.e. we may have over-corrected for the render's purpose. So: render the
  pristine c64 output (all 223, the shipped track-viewer) and compare to the reference BEFORE
  assuming our modifications are improvements. If pristine ≈ reference → the recipe is literally
  "run it as shipped," and c128 is a separate correctness question, not a render fix.

## What we established (CONFIRMED)
1. **The render is velocity-colored TRACKS** (not a density/SVD volume). Our track-viewer + jet
   colormap is the right format. Individual ULM tracks are SHORT (5–6 frames) — normal; vessels
   emerge from accumulating many.
2. **We did NOT regress vs upstream** — pristine braindump `track` gives the same ~28 tracks/acq
   as our fork (measured on the same shard).
3. **Frame rate is NOT 222 Hz.** The H5 has NO PRF (only sampling 6.25 MHz, tx 2.75 MHz). Physics
   + a speed-scale calibration (our clean flow vs the reference's 0–38 mm/s) → **true rate ~160 Hz**
   (222 is ~1.4× too high; the 4-min blog anchor's 652 Hz is refuted by the speeds). FLAG TO DATA
   SOURCE — published speeds at 222 Hz are ~1.4× high. **Hz is a render-time knob** (speed =
   displacement × Hz; tracks are frame-rate-independent), adjustable after any run.
4. **Speed inflation was OVER-LINKING** (the tracking gate allows ~65 mm/s links; 22% of raw
   speeds were >100 mm/s = impossible). Velocity-filtering restores the 0–38 range.
5. **mb-a0a (FIXED, committed aee6332):** the CPU SVD Gram ran in complex64; squaring the
   condition number vs the ~1e3–1e4 tissue/blood dynamic range corrupts the RETAINED blood
   subspace. Controlled A/B (same acqs, toggle only use_gpu_svd): **GPU 7859 vs CPU 477 tracks =
   16.5×**. GPU (c128) is numerically correct; CPU (c64) over-suppresses. Fix: chunked c128 Gram
   in `svd.py` (`_gram_c128`). **BUT see the hypothesis above — sparser≠worse for the render.**
6. **The "reference recipe" we derived:** correct c128 + `sigma_threshold 3.5` → 268 tracks/acq
   (matches reference ~260) + physiological flow (median 5.8 mm/s). Full-223 render:
   `renders/reference_recipe/FINAL_223acq_jet.png` — real cortical vascular LAYERS + penetrating
   vessels + jet flow, but BUSIER than the reference's crisp individual vessels.

## Beads
- **mb-a0a** — c128 precision fix (applied, 33 tests pass; can close after merge to main).
- **mb-91j** — exposed `sigma_threshold` on `baseline` (done).
- **mb-45z** — render crispness gap (busier than reference): sub-pixel localization (centroid →
  Gaussian/PSF fit, `psf.py`), velocity-difference over-link cleanup, mask near-field z≈8mm skull
  band, try the 3D web viewer. **This is where the pristine-pipeline result should redirect: if
  pristine c64 is crisp, mb-45z becomes "match the shipped sparse recipe" not "clean up c128".**
- **mb-rvl** — detection flood: resolved (it's the 2σ threshold; 3.5σ is the answer).

## Infra state (all committed/pushed to origin/epic/mb-crr-foundation, Modal clean)
- `scripts/modal/app.py`: `baseline` reads local h5 + `cpu=16` + exposes `sigma_threshold`;
  `track_refs` (CPU clean-recipe on ref shards); local-read fix. `use_gpu_svd` toggles GPU vs
  CPU SVD (GPU = correct post-fix, but see hypothesis).
- Track tags on the `research` volume: `ref223_c*` (10 chunks, correct c128 + 3.5σ, 41k tracks,
  all 223), `gpusweep_s*` (2.5/3/3.5/4σ on acqs 0-7), `full223_c*` (partial CPU-c64 run, killed).
- Render/analysis scripts in the SCRATCHPAD (not committed): `vel_render.py` (jet velocity lines),
  `line_render.py`, `angiogram.py`, `shot.js` (playwright headless-webgl screenshot of the web
  viewer — uses cached chromium at `~/.cache/ms-playwright/chromium-1223`). Renders in
  `renders/reference_recipe/`, `renders/derisk_clean/`, `renders/angiogram/`.
- Budget raised to **$150** (from $100). This session spent heavily on GPU render runs; track it.
- Modal: **10-GPU concurrency cap**. CPU-SVD is slow on few-core containers (fixed with cpu=16).
  Launch `modal run --detach` via a background Bash `( … ) & wait` (foreground SIGTERMs at 2min).

## Also shipped earlier this session (independent of the render)
- **Saturation metric** (`bench.py:saturation_metrics`, mb-crr.19.6, committed) + the tracking-win
  headline `base223_best` (FRC + saturation validated). These are solid, done.
