# Reference artifacts received — issue #2 answered, calibration reframed

**Date:** 2026-07-06. **Source:** maintainer `ennucore` (CONTRIBUTOR) reply on
[alephneuro/microbubbles#2](https://github.com/alephneuro/microbubbles/issues/2#issuecomment-4898142040).
Ends the WAIT state. This doc records the reply, the artifacts, and what parsing them
locally proved. **Nothing was posted/pushed/published** — the maintainer reply came to us.

## What they gave us (all four asks answered)

1. **Non-determinism cause** — a **randomized SVD** on the GPU path (the *components*
   change run-to-run), NOT primarily the phase-sensitive `spectral_centroid_cutoff` we
   fingered in docs 11/18. They "pushed an update that removes the non-deterministic
   version." (That update is NOT in the public repo yet — newest upstream commit across
   all branches is `5fc46c4`, 2026-06-26, and they keep reproducibility code in a private
   repo per commit `17b5b3b`. We do not need it — see determinism below.)
2. **Viewer** — the blog front-end is unreleased; "the ultratrace we released corresponds
   to **s1**, not s2." Pixel-match remains off the table by construction; but they shipped
   the actual reference *tracks*.
3. **Reference tracks** (downloaded to `outputs/reference/`, gitignored):
   - s1 (released ultratrace): `tracks_v6.bin.gz` → 2,294 tracks / 124,736 pts
   - s2: `tracks_v5_v6.bin.gz` → 6,211 tracks / 690,254 pts
   - full pickle: `full_tracks_smoothed.pkl` (45 MB)
4. **PRF** — frame rate **222 Hz**, physical PRF **888 Hz** (4 plane waves/frame).
   ⚠️ Discrepancy: `00-context.md` recorded **5 transmit angles**; they say **4 plane
   waves/frame**. Reconcile against the raw IQ (raw shape had 5 tx-angle slots — maybe 4
   used for compounding + 1 other, or our count was wrong).

### Pickle safety
Opcode disassembly (no execution) showed the ONLY globals referenced are
`numpy._core.multiarray._reconstruct`, `numpy.ndarray`, `numpy.dtype` — pure numpy arrays,
no `os`/`subprocess`/`eval`. Loaded via a whitelist-restricted `Unpickler`. Safe. (Rome
also confirmed he trusts the devs and is fine unpickling directly.)

## The pickle = the full reference dataset

Top-level dict keys: `tracks`, `tracks_smoothed` (both len **50,456**), `detections`,
`density`, `grid_x/y/z`, `spacing`, `n_acquisitions`, `frames_per_acq`, `n_frames`, `params`.
- **n_acquisitions = 216** (not 223 — we assumed 223).
- **spacing:** dx=0.2004, dz=0.2008, **dy(elevation)=0.5547 mm** (2.8× coarser — the
  synthesized axis, quantified).
- **grid:** (25 elev, 154 z, 275 x); FOV ≈ 55 (x) × 13 (elev) × 31 (z) mm.
- **track dict:** `positions (N,3) f32`, `frames (N,) f32`, `length int`,
  `intensities (N,) f64`, `acq_index int`.

## The calibration reframe (this overturns mb-289's premise)

**The reference tracks are ALSO short.** Full reference set: median length **7 frames**
(mean 10, p90 18, max 191) — essentially identical to our median ~6. The reference pipeline
does NOT produce magically longer tracks. Reference length distribution:

| ≥ frames | count | % of 50,456 |
|---|---|---|
| ≥10 | 14,556 | 28.8% |
| ≥15 | 7,274 | 14.4% |
| ≥20 | 4,366 | 8.7% |
| ≥35 | 1,421 | 2.8% |
| ≥50 | 582 | 1.2% |

The "beautiful crisp render" is produced by **aggressive min-length filtering** — our own
`export_tracks_bin_v3` defaults to `min_length=35`; the shipped s1 (2,294) implies a
threshold ≈25.

### Head-to-head (v3 .bin — what actually renders)

| source | n_tracks rendered | median | mean | tracks ≥35 | max speed |
|---|---|---|---|---|---|
| **Reference** @min35 (ours-exported) | **1,421** | 46 | 52.2 | 1,421 | 0.419 mm/fr (~9 cm/s) |
| Reference s1 (shipped) | 2,294 | ~54 pts | — | — | 0.404 (~9 cm/s) |
| **Ours composite** | **254,590** | 6 | 7.0 | 242 | **3.347 (~74 cm/s)** |
| Ours baseline | 168,177 | 6 | 6.8 | 78 | 1.181 (~26 cm/s) |

### The crispness gap (mb-45z), decomposed — 4 distinct causes

1. **No render-side length filter on ours.** We render all 254k tracks (median 6); the
   reference filters to ≥25–35. Filtering our composite to ≥35 → **242 tracks**. Nearly-free,
   massive de-clutter. *(one line at export)*
2. **~5× over-detection.** 254k vs 50k total tracks — real upstream over-detection
   (mb-289's "3–4×" was directionally right; it's ~5× vs the full reference set).
3. **Genuinely worse continuity.** Even matched at ≥35: **242 (ours) vs 1,421 (ref)** —
   ~6× fewer real long tracks despite 5× more total. "Tracks too short" is real *relative
   to the reference*, not just across our own variants.
4. **Non-physiological fast links.** Our composite max speed **74 cm/s** vs reference
   **9 cm/s** — teleporting associations inflate our track length. The composite's length
   "win" (doc 10, validated by FRC) is partly clutter that FRC couldn't see but the
   reference exposes. A physiological velocity gate (~0.4–0.5 mm/frame) would prune these.

## Determinism (Step 3) — already solved in our fork, verified

- No unseeded randomness anywhere in the SVD paths (grep for `randn/rand/default_rng/
  random projection` across `gpu_svd*`, `svd*`, `detect*` is clean).
- Every SVD path uses deterministic `eigh`/`eigvalsh`/full `svd`. `gpu_svd.py:119` does
  `cp.linalg.eigh(G)` on a complex128 Gram (comment: made "stable and reproducible").
- The `"randomized"` method name is a **legacy alias** that falls through to the
  deterministic eigh path — it does NOT do random projection.
- `track_viewer_export.py:124` forces `method="full"` "for reproducibility."
→ Our pipeline is deterministic. The maintainer's randomized-SVD non-determinism was the
  **pristine upstream** behavior (what docs 18 measured via `pristine_app.py`); our fork
  already replaced it. **No code change needed.** Optional cosmetic: rename/remove the
  `"randomized"` alias to avoid confusion.

## What renders now
`outputs/reference/viewer_ref/` — reference tracks in our v3 viewer (min_length=35, 1,421
tracks). Compare launcher: `outputs/reference/compare/` (links reference + our composite +
baseline). Served locally on :8791 (tailnet only).

## Actionable next steps (the render is now a solved-in-principle target)
- **Cheap win:** re-export our composite/baseline with `min_length≈35` + a physiological
  velocity gate (≤~0.5 mm/frame) → should visually approach the reference immediately.
- **Real work:** cut the ~5× over-detection and lift genuine ≥35 continuity from 242→~1,400
  (this is exactly what the frontier brainstorm doc 19 targets: fewer/longer/cleaner via
  velocity-gated global association, TBD, min-cost-flow).
- **Do we need their private commit?** No — we reproduce deterministically and now have
  their reference tracks to calibrate against. Only ask upstream if we want their exact
  detection recipe; not blocking.
- **(Held for Rome)** a public thank-you reply on issue #2 + the 4-vs-5 planewave question.
