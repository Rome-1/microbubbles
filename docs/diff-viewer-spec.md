# 3D+Time strategy-DIFF viewer — build spec (Opus-designed)

**Goal:** a self-contained, offline Three.js viewer that makes the DIFFERENCE between any two
ULM vasculature reconstruction STRATEGIES visually unmistakable — in 3D, with flow animation (time).

Folder: `scripts/wf_render_signal/diff_viewer/` — `index.html` (single self-contained file),
`vendor/` (three r170 + addons + lil-gui, already copied), `data/` (exported, ready).
Serve with `python -m http.server` in that folder. NO headless chrome in the fleet.

## Data (already exported by export_diff_viewer.py, verified)
- `data/manifest.json`: `{schema, grid{shape[138,13,77],spacing_mm,origin_mm}, frame_rate_hz 222.43,
  speed_scale_max_mm_s 38, default_pair ["graph","gauss"], av_grid, strategies[], honesty_notes[]}`.
  Each strategy: `{id,label,kind,uses_incompressibility,lines,grid,diff_hue,metrics{splithalf_dice,
  direction_cosine,n_streamlines,occ_vox}}`. Strategies: `graph` (validated win, Dice .715/cos .95,
  hue #20e0d0), `gauss` (baseline .61/.84, #ff8a3d), `graph_incomp` (#c78bff), `raw` (#8a8a8a).
- `data/<id>.lines.bin` v4, little-endian: header 64B = `magic u32=0x554C4D54, version u32=4,
  n_tracks u32, total_points u32, max_speed_mm_frame f32, bounds_min[3]f32, bounds_max[3]f32`, pad to 64.
  Then offset table `n_tracks × (point_offset u32, length u32)`. Then point data
  `total_points × 8 f32 = [x,y,z, speed_mm_frame, dirx,diry,dirz, conf01]` (dir = mm-space unit tangent).
- `data/<id>.grid.bin`: header 64B = `magic u32=0x554C4D47, version u32=1, nx,ny,nz u32, n_occupied u32`,
  pad. Then C-order blocks: `occupancy nx*ny*nz u8`, `dir nx*ny*nz*3 f32`, `speed nx*ny*nz f32`,
  `confidence nx*ny*nz f32`. C-order index = `((ix*ny)+iy)*nz + iz`.
- `data/av.grid.bin`: grid header + `nx*ny*nz u8` artery/vein labels (0 none / 1 artery / 2 vein).

## Primary diff approach: co-registered OVERLAY + runtime occupancy-PROVENANCE diff
Both strategies in ONE space. Classify each vertex at runtime via the OTHER strategy's occupancy grid
(`voxel = floor((p-ORG)/SP)`): A-vertex is `shared` if `occ_B[voxel]` else `A-only`; symmetric for B.
Color: **shared** = soft white `#e8e8e8` low-opacity (agreement skeleton); **A-only** = strategy A's
`diff_hue` (orange); **B-only** = B's `diff_hue` (cyan). Works for ANY pair, no per-pair precompute.
Sanity: shared-voxel fraction ≈ reported Dice.

## Secondary modes (reuse loaded geometry)
- **Blink**: hard A↔B toggle at settable rate + spacebar manual (catches sub-voxel shifts; fog off).
- **Wipe**: draggable clipping plane, A one side / B other.
- **Side-by-side**: two synced viewports (fallback/sanity).
- **Single A / Single B**.

## Science color regimes (single-strategy): speed jet (fixed 0–38 mm/s), direction DTI-RGB
(R=|dir_x| lateral, G=|dir_y| elevation, B=|dir_z| depth), artery/vein (#ff3838 / #3aa0ff / dim gray).
Diff-heatmap sub-modes on shared voxels: angular Δdir (acos(dir_A·dir_B)), Δspeed.

## Time = particle advection (physically honest)
Per streamline precompute cumulative arclength (mm). Particles `{trackId, s}` seeded ∝ track length
(density slider, ~8–15k). Per frame `s += speed(s)·dt·speedMult` → binary-search position; wrap mod L
with alpha fade at spawn/despawn. Comets ACCELERATE in fast arterial cores, crawl in slow vessels.
Color: jet(speed) in science regime, provenance hue in diff regime. In diff: a particle entering a
voxel absent from the other strategy goes full-opacity only-hue (the divergence lights up); shared → dim.
Pulsatility toggle (OFF default, labeled "illustrative — not measured").

## Scene / geometry
`THREE.LineSegments` (GL_LINES, one segment per streamline edge — never a strip across tracks), per-vertex
color + per-vertex alpha from conf01. One `THREE.Points` cloud for particles (additive, soft sprite, glow).
`camera.up=(0,0,1)`, near-coronal default; OrbitControls (damping); depth fog (near=diag*0.6,far=diag*2.2).

## Interactions (lil-gui, bottom-left): A/B dropdowns; view mode (Single A/B, Overlay+Diff, Blink, Wipe,
Side-by-side); diff channel (provenance / dir-Δ / speed-Δ); science channel (speed/direction/artery-vein);
time (play/pause, speed×, density, pulsatility); confidence-opacity toggle (default ON) + threshold clip;
depth clip plane (axis+pos); blink rate; line width; reset camera; screenshot (toDataURL). HUD (top): legend
that names A & B with metric badges (Dice/cos), colorbar (top-right), live 10 mm scale bar (bottom-right),
honesty banner.

## Honesty (mandatory)
Confidence must be visible (opacity∝conf01, default ON). Don't fake elevation resolution (y is 2.77× coarser,
partly synthesized — note it; if uses_incompressibility, flag elevation-derived structure). A-only ≠ "found a
real vessel B missed" — could be smoothing hallucination; caption neutrally + show Dice/cos badges. Particles &
pulsatility are illustrative of direction & RELATIVE speed, not literal bubbles or cardiac timing. Fixed 0–38
mm/s speed scale across strategies (no per-strategy autoscale). `raw`/`tracking` are different objects — banner it.

## Build + verify order
1. Loaders (manifest, v4 lines.bin, grid.bin) — read-back log n_tracks/bounds.
2. Static single-strategy LineSegments + jet + camera framing + scale bar.
3. A/B dropdowns + Single/Side-by-side.
4. Overlay + occupancy provenance (CORE) — shared fraction ≈ Dice.
5. Blink + Wipe.
6. Particle flow (single, then provenance-colored in overlay).
7. Channels (direction, artery/vein), confidence-opacity, depth clip.
8. Honesty banners + legend polish.
