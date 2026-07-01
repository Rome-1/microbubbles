# Pristine-shipped-pipeline render — RESULT (2026-07-01)

Answers Rome's DO-THIS-FIRST question (doc 16): **"What happens if you just run
EXACTLY what they shipped? No changes?"** Short answer: **the reference's crispness
comes from the shipped SPARSE path + rendering tracks as POINTS. Our c128 "fix" +
densification made the render busier/worse, not better.** The hypothesis in doc 16
is confirmed on the data we already had — no new compute needed for the conclusion.

Figure: `renders/pristine_shipped/compare_3way_ref_vs_pristine-c64_vs_ours-c128.png`

## What was compared (all rendered identically: points, Jet, coronal view down the
## elevation axis, vmax = 0.1713 mm/frame = 38 mm/s @222Hz, on near-black — i.e. the
## SHIPPED track-viewer's exact projection + colormap + speed scale)

| Render | Pipeline | Density | Character vs reference |
|---|---|---|---|
| **Reference** (Aleph blog) | shipped `ultratrace-ulm run` | ~thousands of tracks | crisp velocity-colored vessels on dark |
| **Pristine c64** (`full223_c*`, 79/223 acqs) | shipped recipe: adaptive **CPU c64** SVD, σ2.0, knee, τσ0, kalman, gap3, min5 | ~60 tracks/acq (7,639 total) | **MATCHES the reference's character** — discrete velocity-colored segments on black, incl. the prominent horizontal vascular band; just under-dense (⅓ the acqs) + a near-field artifact |
| **Our c128** (`base223_best`) | region-SVD + Kalman + stitch, **GPU c128**, over-linked | 254,590 tracks (1,141/acq) | dense saturated red/white **haze with heavy horizontal banding** — wrong character |

## The two things that actually produced the gap (both are OURS, not the data)

1. **We rendered LINES, not POINTS.** The shipped track-viewer renders each
   localization as a *point*, colored by smoothed speed; vessels emerge from the
   density of accumulated points. Our `FINAL_223acq_jet.png` drew *line segments*
   through each 5–6-frame track → confetti. Same tracks, rendered as points (see
   `render_points.py`), read completely differently.
2. **We over-densified.** The shipped CPU adaptive path is SPARSE (~28–60 tracks/
   acq). mb-a0a "fixed" the CPU SVD c64→c128, which is *numerically* correct but
   lifts density to ~982/acq (σ2.0); even our σ3.5 "reference recipe" (268/acq)
   and `base223_best` (1,141/acq) are far too dense → saturated haze. **Sparser is
   crisper for THIS render.** The c128 fix is a correctness win but a render loss.

## Key measurements (this session, all from existing volume data — $0 compute)

- Pristine `track` on 1 shard (`upstream_ref_test`): **28 tracks, ALL 5–6 frames**,
  zero ≥10. So the shipped viewer's default `--min-length 35` renders EMPTY on this
  data → **the reference was NOT min-length-filtered**; it accumulates short tracks.
- `base223_best`: 254,590 tracks, median len 6, max 79, only **242 (0.1%) ≥35 frames**.
  Our tracks are inherently short (normal for ULM) AND far too many.
- Pristine c64 (`full223_c0`): 477 tracks / 8 acqs = ~60/acq, adaptive σ2.0,
  `standalone_tracker=True` — the shipped recipe verbatim.
- Speed distributions (mm/frame): pristine c64 p50=0.073 p99=0.394 (physiological);
  ours p50=0.125 p99=1.10 (over-linked, inflated — matches doc 14).

## Status / next

- **Conclusion established** on free data. To make the side-by-side a knockout,
  render the pristine c64 recipe on the FULL 223 (the existing `full223_c*` covers
  only 79/223 — the run was killed early). ~3× the density fills in the branching.
  Launching that completion on Modal (GPU beamform + CPU c64 track; per-8-acq chunks
  demonstrably complete). Budget: ~$10–15, within the $150 gate.
- **mb-45z redirect (doc 16 anticipated this):** the crispness work is NOT "clean up
  c128." It is: (a) render points not lines [DONE — `render_points.py`]; (b) run the
  sparse shipped recipe, not the densified one; (c) mask the near-field z≈8mm skull
  band still visible at the bottom of the pristine render.
- **mb-a0a stays valid** as a *correctness* fix (c128 is the numerically right SVD);
  it is simply the wrong lever for *this render*, which wants the sparse shipped output.

## Repro
```
# points render of any shipped tracks.bin (the exact viewer projection + speed color):
python3 scripts/render_points.py <viewer>/data/tracks.bin out.png [--gain G --psf S]
# viewer bin from a tracks pickle: ultratrace_ulm.track_viewer_export.write_track_viewer(pkl, dir, min_length=5)
```
