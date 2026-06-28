# Bake-off results (this session)

Live log of the parallel bake-offs. **No ground truth** — all metrics are proxies
(bench.py). The split-half FRC arbiter (mb-crr.19.6) is needed to settle the
ambiguous cases below; until it lands, read "more/longer tracks" cautiously.

## #4 — elevation-anisotropic Kalman (retrack base60, 60 acqs, identical detections)

Isolates the TRACKING stage: retrack `base60`'s frozen detections with the new
`elev_gate_factor` / `elev_meas_factor`, `gate_on_prediction=False` (matches base60).
**Control `base60_iso` (gate=1,R=1) reproduced base60 exactly (49274 tracks) → harness
validated, params backward-compatible.**

| config | n_tracks | mean_curv_len_mm | straightness | occupied_frac | frag short/long | track_len_mean (pts) |
|--------|---------:|-----------------:|-------------:|--------------:|----------------:|---------------------:|
| base60 (iso)     | 49274 | 0.590 | 0.878 | 0.368 | 274 | 6.47 |
| g2 (gate×2)      | 55962 | 1.293 | 0.906 | 0.420 | 309 | 6.46 |
| g3 (gate×3)      | 58417 | 1.806 | 0.917 | 0.433 | 321 | 6.47 |
| **g3r10 (gate×3, R×10)** | **60638** | **1.902** | **0.926** | **0.442** | **283** | 6.52 |
| g4 (gate×4)      | 59657 | 2.057 | 0.921 | 0.439 | 338 | 6.48 |

**Read (honest):**
- **Coverage up** (+20 % occupied_fraction) and **straightness up** (+5 %) with **no
  fragmentation penalty** at g3r10 — encouraging, and the straightness gain argues
  against gross over-merge (over-merging unrelated bubbles lowers straightness, cf. the
  stitch idea #1 which dropped it −38 %).
- **BUT track lifetime is flat** (`track_len_mean` 6.47→6.52 points). The +222 %
  `mean_curv_len_mm` is therefore almost entirely **larger per-step distance**, not
  longer-lived tracks: the wider y-gate links detections across bigger elevation gaps
  (avg step 0.09→0.32 mm; mean speed 20→70 mm/s). Whether those elevation links are
  REAL (coarse y-sampling under-resolves genuine out-of-plane motion) or SPURIOUS
  (elevation localization jitter) cannot be told from these proxies.
- **R×10 adds beyond the gate on real data** (g3r10 > g4 on n_tracks/straightness/frag),
  unlike the synthetic unit test where the box gate masked R. So both knobs matter.
- **Sweet spot: g3r10** (gate×3, R×10).

### FRC verdict (split-half, acq-parity) — #4 is a REAL win

| config | xz_res_mm (finer=better) | fsc3d_res_mm | repro_score (↑=more reproducible) |
|--------|------:|------:|------:|
| base60 (iso)  | 0.957 | 0.777 | 0.614 |
| g2            | 0.911 | 0.666 | 0.637 |
| g3            | 0.927 | 0.435 | **0.645** |
| **g3r10**     | **0.899** | 0.436 | 0.644 |
| g4            | 0.930 | 0.435 | 0.644 |

The elevation lever **raises repro_score** (0.614→0.645, +5 %) — the extra coverage
**reproduces across independent acq-halves, so it is real, not artifact** — while making
the trustworthy lateral (**xz) resolution finer** (0.957→0.899 mm) and **3D resolution
+44 %** (0.777→0.435 mm). The flat track-lifetime caution is thus resolved: the
reconstruction is more reproducible AND finer, which is what matters for the vessel map.
(yz/xy res pinned at 0.224 across all configs — those planes weren't discriminating here,
a coarse-y/grid quirk; rely on xz + fsc3d + repro_score.) **VERDICT: ship the
elevation-anisotropic Kalman; g3r10 (gate×3, R×10) is the pick** — coverage +20 %,
straightness +5 %, xz resolution +6 % finer, 3D resolution +44 % finer, no frag penalty,
FRC-confirmed reproducible. This was the top cross-model prediction and it held up.

## SVD-cutoff sweep — IN FLIGHT (6 variants × 60 acqs, fused beamform)
Single-acq detection preview (acq 0): fix8 371k, fix17 486k, knee 489k (+8.7 % vs
floor 450k), fix30 470k, fix70/floor 450k, **kneeH (knee+MP noise cut) 542k (+20 %)**.
Track-level bench pending detection completion → track_acqs each → bench vs base60.
