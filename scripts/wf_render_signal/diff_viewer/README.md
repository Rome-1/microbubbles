# 3D+Time strategy-diff viewer

Self-contained, offline Three.js viewer that makes the **difference between any two ULM
vasculature reconstruction strategies** visually unmistakable — in 3D, with flow animation.
Design: `docs/diff-viewer-spec.md` (Opus-designed). Built + verified 2026-07-08 (mb-4k2/mb-y8i round).

## Run
```bash
cd scripts/wf_render_signal/diff_viewer
python -m http.server 8794         # then open http://localhost:8794/  (any modern browser)
```
Confirmed working in Chromium (software-WebGL headless snapshot + logic-verified in node).

## What it shows
- **Overlay + diff** (default, the core): both strategies co-registered; each streamline vertex
  classified at runtime by the *other* strategy's occupancy grid → **white = shared**,
  **strategy-A hue = A-only**, **strategy-B hue = B-only**. Default pair graph (validated win) vs
  gauss (baseline): cyan = graph-only (its higher recall), orange = gauss-only, white = agreed core.
- **Flow particles** advect along streamlines at per-vertex speed (fast arterial cores stream,
  slow vessels crawl) — colored by provenance in diff mode, by speed/direction/AV in science mode.
- View modes: Single A/B, Overlay+diff, Blink (A↔B, spacebar), Wipe (clip plane), Side-by-side.
- Science channels: speed (jet, fixed 0–38 mm/s), direction (DTI-RGB), artery/vein.
- Confidence-opacity (default ON — sparse/uncertain vessels render faint), depth/wipe clip,
  time controls (play/speed/density/pulsatility), reset camera, screenshot.

## Honesty (baked into the HUD banner)
A-only ≠ ground truth (may be smoothing hallucination); opacity encodes confidence; the elevation
axis is 2.77× coarser & partly synthesized; particles/pulsatility illustrate direction & *relative*
speed, not literal bubble paths or cardiac timing; fixed speed scale across strategies.

## Data (regenerate; `.bin` are gitignored)
```bash
python scripts/wf_render_signal/export_diff_viewer.py    # writes data/{manifest.json,<id>.lines.bin,<id>.grid.bin,av.grid.bin}
```
Strategies exported: graph (mb-ki9 win), gauss (baseline), graph_incomp, raw. Format: v4 lines.bin
(8 f32/vertex: pos, speed, dir, conf) + grid.bin (occupancy/dir/speed/confidence). `vendor/` is a
copy of the repo's vendored three.js r170 + lil-gui.
