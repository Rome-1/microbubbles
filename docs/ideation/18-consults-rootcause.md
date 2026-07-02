# Consults + root-cause: where the reference gap actually is (2026-07-02)

After Rome pushed back ("just run their code; why all these decisions?"), we (a) cloned
the pristine `braindump`, (b) ran a decisive stage-boundary experiment, and (c) got four
independent consults (Fable + Codex, each with minimal and full context). Strong consensus.
Consult transcripts: `docs/ideation/consults/` (codex) + inline below (fable).

## Decisive experiment: the fuzz enters at DETECTION, not tracking/render

Rendered the **raw detections** (`detections.positions_mm`, 1.43M points, PRE-tracking)
for the c128+3.5σ full-223 set as a density map (x,z), same projection as the viewer:
`renders/pristine_shipped/raw_detections_density_prewtracking.png`. The near-field band,
the ~5.5 mm-spaced depth bands, grid-quantization, and a **diffuse whole-FOV noise floor**
are ALL already present before any tracking, smoothing, masking, or rendering. So the gap
is the **clutter filter + z-score detection admitting a noise floor across the whole field**
instead of concentrating detections on vessels. Everything downstream (density, points-vs-
lines, 3dulm, masking) is cosmetics on a noisy detection field. (Codex-full and Fable-full
both proposed this exact experiment as the cheapest decisive one.)

## Consensus across all four consults

1. **The screenshot is NOT reproducible from public artifacts.** The reference's `s1/s2`
   switcher + 10 mm scale bar + `0–38 mm/s` legend are **not in the released viewer**
   (`web/track_viewer` is the bare "Brane" THREE.Points viewer). The figure was rendered by
   an internal/unreleased front-end. Pixel-match is impossible by construction.
2. **The shipped adaptive-SVD detection path is numerically non-deterministic.** Two bugs:
   (a) c64 Gram precision (mb-a0a); (b) **[Fable-full, new]** `spectral_centroid_cutoff`
   scores modes with a *phase-dependent* `|rfft(u.real)|²`, so the clutter cutoff `low`
   (integer # modes removed) changes with the eigensolver's phase convention → the ~15×
   CPU/GPU density swing. The authors THEMSELVES avoid this path in `track_viewer_export.py`
   ("the 'fast' variant is numerically unstable in float32 and not reproducible across
   backends") yet detection ships on it. Density is steered by a non-deterministic integer;
   pinning `--svd-n-components` bypasses it.
3. **Frame rate is NOT just a color knob** (corrects doc 16 claim #3). `frame_rate_hz`
   feeds `spectral_centroid_cutoff` (freqs vs `tissue_freq=100 Hz`), so it changes `low`,
   hence density and which subspace is "blood". Our "160 Hz" was circularly derived from the
   colorbar. Need the real PRF (not in the H5).
4. **Localization is NOT the main cause** (I over-weighted centroid). doc 14 measured 0.072 mm
   jitter; and `--subpixel gaussian_fit` is **advertised but unshipped** (falls through to
   centroid unless `parabolic`). The fuzz is false detections + over-linking, not sub-pixel.

## What I got wrong (own it)

- Flip-flopped between doc 13 (reference ≈ 144 frames/track → a **linkability** gap) and
  docs 16/17 (tracks inherently short, density accumulates) and picked the version that made
  our sparse c64 look like a match. The reference's isolated smooth 10–20 mm arcs in empty
  regions are single long trajectories → evidence for the linkability read I discounted.
- Masked ~49% of points as "artifact," but the reference KEEPS a near-field band → over-
  masking risks deleting real signal.
- Treated the whole thing as a render problem for too long; it's a detection/filter problem.

## Levers tested this session

- **3dulm arc-length smoothing** (`--smooth-method 3dulm`, the untested shipped knob both
  consults flagged): resamples each short track to a dense smooth arc (145k→933k pts).
  `renders/pristine_shipped/ref223_3dulm_smoothing_points.png`. Helps continuity marginally
  but does NOT close the gap — it smooths *within* 5–6-frame stubs; it can't fix a noisy
  detection field or manufacture long tracks.

## Running their code verbatim (Modal, pristine package from github)

`scripts/modal/pristine_app.py` installs `ultratrace-ulm-pipeline` from github and runs its
own CLI. Findings: **the shipped `beamform_mach` holds every acq's ~11.9 GB compound in RAM
before writing** (beamform_mach.py:119-158) → `--all-acqs` needs ~2.6 TB; even 8 acqs with
`--spatial-tgc` OOM'd a 256 GB box at the global-TGC step (TGC runs SVD on all held
compounds). Re-running 8 acqs without TGC succeeded. **Results (verbatim their code vs our fork):**
- **Beamform is BIT-IDENTICAL.** `compare_beamform` on acq 0: `max_abs_diff = 0.0`,
  `bit_identical = True` (pristine `beamform_mach` vs fork `beamform_all`, same 8 acqs, no
  TGC). Not just machine-eps — literally the same bits. The fork's `stream_accumulate`
  reorder is a true no-op numerically. So the beamform stage IS theirs, empirically.
- **Pristine verbatim `track` (c64, shipped recipe, 8 acqs):** 388 tracks = **48/acq**;
  track length median 5, mean 5.5, **max 11, zero ≥35**; from **335,286 raw detections
  (~42k/acq)** of which only 388 link → **~99.9% of shipped-c64+σ2 detections are unlinkable
  noise.** Renders: `renders/pristine_shipped/pristine_VERBATIM_8acq_{tracks,raw_detections}.png`.
  The raw-detection field is a grid-quantized diffuse noise cloud with faint structure.

**Conclusion:** running their EXACT code reproduces exactly what our fork produces — sparse
48/acq, 5–6-frame short tracks from a mostly-noise detection field. So the short-tracks +
noisy-field are inherent to the shipped pipeline on this data, NOT a fork artifact, and the
reference's continuous vessels are not what the released recipe yields. (Side note: without
TGC the near-field band drops from ~12% to ~1.4% of detections — the skull band is
TGC-amplified, and our earlier aggressive z<11 mm masking was partly fighting our own TGC.)

## The ask to the developers (what the release under-determines)

Reproducing the figure hinges on: the real PRF/frame-rate, the actual clutter cutoff `low`
per acq (non-deterministic in the released code), whether the public file is the figure's
exact input, and which viewer + any post-processing produced the screenshot. GitHub-issue
draft (Rome to approve before posting) lives in the session notes.
