# The real gap to the reference render: TRACK LENGTH (2026-06-29)

Rome pushed a reference target screenshot (`renders/references/`): velocity-colored tracks
(flow speed 0–38 mm/s), crisp and sparse, tracing individual branching vessels on black. That is
the repo's **track viewer**, just with clean LONG tracks. So the render is already solved — the gap
is track quality.

## Decisive measurement
Ran the EXACT braindump reference recipe (`track` Modal fn = `run_tracking_outputs`, adaptive SVD,
temporal-sigma 0, 2σ, max_gap 3, max_cost 10) on ONE beamformed acq (`base60_refs/acq_0000.h5`):

- **37,424 localizations** detected (53/frame) — **detection is fine** (not a count problem).
- But only **17 tracks**, all **5–7 frames long**.

The reference produces **~260 tracks/acq** (README). From 37k localizations that is **~144
localizations/track** — bubbles traced through whole vessels. **Our tracks are 5–12 frame stubs.**

| path | tracks/acq | typical length |
|---|---|---|
| braindump reference (README) | ~260 | ~144 frames (long vessel traces) |
| our GPU path (detect_acqs+track_acqs, base223) | 754 | median 6 |
| our CPU path (run_tracking, this test) | 17 | 5–7 |

## What it is / isn't
- NOT rendering: the track viewer + velocity color already matches the reference format. And no
  length filter rescues it — `frac_len≥35` is ~242 tracks TOTAL in base223_best.
- NOT detection count: 37k localizations/acq is plenty.
- NOT a TrackingOptions default regression: our defaults are byte-identical to upstream braindump
  (verified; the epic only ADDED `gate_on_prediction=False`, an identity).
- Both our paths use the SAME tracker (`_track_detections`), yet 17 vs 754 → the variable is
  **detection QUALITY / linkability**: our localizations are too scattered/noisy to link into long
  vessel traces. CPU detect (37k loc → 17 tracks) is even less linkable than GPU detect (23.6k → 754).

## Hypothesis
Our fork's localizations don't sit cleanly on vessel centerlines frame-to-frame (axial streaking,
elevation pile-up, residual clutter), so the gate/cost association breaks tracks after a few frames.
Upstream braindump reportedly gets ~144-frame tracks from the same recipe → either (a) upstream's
localization is cleaner, or (b) our fork regressed something in detection/localization (226 extra
lines in tracking.py + changes to gpu_detect/svd/beamform vs upstream).

## Decisive next step
Run **pristine upstream braindump end-to-end on Modal** (its own `run`/`track`, unmodified) on a few
acqs. If it yields ~260 long tracks → our fork regressed (diff to find it). If not → the ~260 figure
needs specific tuning. Either way it's the ground truth. This is the path to the reference render.

(Infra note: `bd`/Dolt threw a schema-migration error during this session — findings persisted here
in git instead of a bead.)
