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
- **Sweet spot: g3r10** (gate×3, R×10). Verdict PENDING split-half FRC: if the extra
  tracks reproduce across independent acq-halves they are real coverage; if not, the
  gate is manufacturing elevation-bridged tracks. **This is why FRC is now the critical path.**

## SVD-cutoff sweep — IN FLIGHT (6 variants × 60 acqs, fused beamform)
Single-acq detection preview (acq 0): fix8 371k, fix17 486k, knee 489k (+8.7 % vs
floor 450k), fix30 470k, fix70/floor 450k, **kneeH (knee+MP noise cut) 542k (+20 %)**.
Track-level bench pending detection completion → track_acqs each → bench vs base60.
