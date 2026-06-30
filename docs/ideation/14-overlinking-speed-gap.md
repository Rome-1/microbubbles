# The render quality gap is OVER-LINKING (+ a frame-rate ambiguity) — 2026-06-30

Investigated why our velocity-colored tracks don't look like the Aleph reference
(`renders/references/`). Rome's call to investigate the speed/quality gap before scaling to
the full 223 was correct — it surfaced the real issue.

## Findings (from 30-acq clean CPU-SVD tracks, 2595 tracks)
1. **Localization precision is fine** — perpendicular jitter 0.49 px (0.072mm); per-frame motion
   1.9 px is REAL, not jitter-dominated (jitter/step ratio 0.26). Ruled out.
2. **The tracker OVER-LINKS.** Speed distribution: median 39.5 mm/s, **p90 129, p95 155,
   and 22.7% of segments >100 mm/s** — physiologically impossible for microvasculature. The
   spatial gate `max_distance_mm = (0.29, 1.11, 0.29)` mm/frame → lateral **0.29mm × 222 Hz =
   64.7 mm/s max link speed**, so the tracker bridges distant detections into false-fast tracks.
   This is the scatter in every render — most "tracks" are over-linked junk.
3. **Velocity-plausibility filtering** (reject max-seg >60 mm/s or std >20) → speeds drop to
   **p50 21.8, p90 42.8 mm/s** (matching the reference's 0–38) but keeps only **14% (357/2595)**
   of tracks. The bad links are baked in at tracking time; a tighter GATE would avoid creating
   them (and keep real fast vessels), unlike post-hoc filtering.
4. **Slow flow IS present** (~20% under 15 mm/s) but swamped by the spurious fast tail.

## Frame-rate ambiguity (possible ROOT cause)
Our docs say "~4-minute scan, 223 acquisitions @ 222 Hz" (roadmap.md, flow-matching.md) and the
blog says a **4-minute acquisition**. But **223 × 700 frames ÷ 222 Hz = 703 s = 11.7 min**, not 4.
The two are inconsistent by ~3×. If the TRUE frame rate is ~650 Hz (consistent with 4 min), then:
- a 40 mm/s bubble moves only ~0.06mm/frame, so the gate (0.29mm) is **~5× too loose** → over-linking.
- our absolute speeds (computed at 222 Hz) are also mis-calibrated.
**This needs resolving** — it determines whether the gate is mis-tuned. Check the raw H5 timing
metadata (PRF / frame interval) or confirm with the data source.

## The fix (cheap, upstream of density/render)
A **tracking-gate tightening / velocity-difference constraint** + **resolved frame rate**, then a
CPU re-track (no GPU, no full-223 needed to test). This should clean the render by NOT creating the
spurious tracks. Scaling to 223 BEFORE this fix would have given 7× more over-linked scatter.

(Note: a dense band at the shallowest depth z≈8mm appears in all renders — near-field/skull clutter,
a separate artifact to mask.)
