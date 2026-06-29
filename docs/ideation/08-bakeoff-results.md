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

### Combined tracking stack (retrack base60, FRC) — building on g3r10

| config | xz_res_mm | fsc3d_res_mm | repro_score | n_tracks |
|--------|------:|------:|------:|------:|
| base60 (iso)                       | 0.957 | 0.777 | 0.614 | 49274 |
| g3r10 (elev, gpred=off)            | 0.899 | 0.436 | 0.644 | 60638 |
| +gpred                             | 0.874 | 0.437 | 0.639 | 54312 |
| +gpred +intensity3                 | 0.903 | 0.730 | 0.636 | 52598 |
| **+gpred +intensity3 +max_gap6**   | **0.870** | **0.435** | **0.655** | 59112 |

**Clean attribution (isolation retracks):**

| config | xz_res_mm | fsc3d_res_mm | repro_score |
|--------|------:|------:|------:|
| base60 (iso)              | 0.955 | 0.777 | 0.615 |
| g3r10 (elev only)         | 0.890 | 0.436 | 0.645 |
| **g3r10 + max_gap6**      | **0.839** | **0.397** | **0.672** |
| g3r10 + gpred + gap6      | 0.852 | 0.431 | 0.660 |
| g3r10 + gpred + int3 + gap6 | 0.868 | 0.435 | 0.655 |

**BEST TRACKING = elevation-anisotropic Kalman (g3r10) + max_gap=6** → repro_score 0.672
(**+9.3 % vs base**), xz_res 0.839 (**+12 % finer**), fsc3d 0.397 (**+49 % finer**). max_gap=6
is a real secondary lever (#8, +4 % repro over g3r10 alone). **gate_on_prediction and
intensity cost (#9) both *hurt* on top** — coherent: once elevation is down-weighted,
extrapolating the unreliable elevation *velocity* (gpred) propagates noise, so a wide
elevation gate with NO prediction wins. The earlier `gp_i3_g6` "best" was a non-monotonic
artifact. → Composite uses **g3r10 + max_gap6, gpred off, intensity off**.

## SVD-cutoff sweep (6 variants × 41 acqs, fused beamform, baseline iso tracking)

Isolates the DETECTION/SVD change (identical baseline tracking for all 6; all 6 tags
trimmed to the identical 41-acq order-set). FRC + bench:

| variant (low cut) | xz_res_mm | repro_score | straightness | frag short/long | occupied | n_tracks |
|-------------------|------:|------:|------:|------:|------:|------:|
| fix70 (floor, =old default) | 0.791 | 0.774 | 0.872 | 289 | 0.834 | 694k |
| fix30  | 0.794 | 0.788 | 0.875 | 239 | 0.850 | 856k |
| fix17  | 0.656 | 0.798 | 0.869 | 105 | 0.860 | 1050k |
| **knee (≈17, data-driven)** | 0.719 | 0.807 | 0.872 | 200 | 0.856 | 988k |
| fix8   | 0.323 | **0.813** | **0.834** | 16.5 | 0.854 | 1116k |
| kneeH (knee+MP noise cut) | **0.323** | 0.806 | **0.894** | 3125 | 0.737 | 367k |

**Findings:**
1. **The floor-70 default is confirmed bad** — worst FRC (0.774) and coarsest xz (0.791).
   Removing FEWER tissue modes monotonically improves FRC + xz_res. **The knee (17) is a
   solid, principled win over the floor** (repro +4.3 %, xz +9 % finer, straightness held).
2. **METHODOLOGICAL CAVEAT (important):** FRC's acq-parity split is **confounded for the
   SVD-cutoff question** — static tissue clutter reproduces across even/odd halves too, so
   removing fewer tissue modes inflates repro_score with *clutter*, not just blood. The tell:
   **fix8 (remove only 8) has the best repro_score/xz BUT the lowest straightness (0.834)** —
   consistent with retained tissue adding reproducible-but-wandering tracks. So fix8's raw FRC
   "win" is partly clutter. (FRC was a clean arbiter for #4 because tracking only reorganizes
   fixed detections; it is NOT clean when the lever changes how much tissue is retained.)
   → **Pick the knee (data-driven ~17): high repro AND good straightness (0.872), the best
   balance.** Treat fix8 cautiously; validate any sub-knee cutoff against a clutter check
   (near-zero-velocity / low-straightness track fraction) before adopting.
3. **MP high-order noise cutoff (kneeH) is a real, separate lever:** 2.4× finer lateral
   resolution (xz 0.323) and the best straightness (0.894) — it sharpens by removing noise
   modes. Cost: lower coverage (0.737) + heavy fragmentation (it also removes weak signal).
   Best used *with* a small low cutoff and gentler tuning; promising for a resolution pass.

**Verdict:** SVD cutoff matters (floor-70 was 4–10× too aggressive); **use the data-driven
knee (low cut) for detection**, with the MP noise cutoff as an optional resolution lever.
Effect is real but more modest/confounded than the tracking lever — the **elevation-
anisotropic Kalman remains the headline win.**

## Synthesis + recommended composite

**Two validated levers this session (no-GT, FRC-arbitrated):**
1. **Elevation-anisotropic Kalman (#4) — the headline win.** `g3r10 + max_gap6`
   (elev gate ×3, R_yy ×10, max_gap 6, prediction OFF, intensity OFF): repro +9.3 %,
   lateral res +12 % finer, 3D res +49 % finer, no frag penalty. FRC-clean (tracking
   only reorganizes fixed detections). Predictive gating and intensity cost both *hurt*
   on top — once elevation is down-weighted, don't extrapolate its unreliable velocity.
2. **SVD low cutoff (mb-3k4) — confirmed real, more modest.** The old "adaptive" cut was a
   silent constant 70-mode removal; the data wants a SMALL cutoff. The data-driven knee
   (~17) beats the floor (repro +4.3 %, xz +9 % finer) with good straightness. MP high-order
   noise cutoff is a separate resolution lever (2.4× finer xz, coverage cost).

**Recommended composite for the 223 scale-up + render:**
detection = `svd_method=knee` (small data-driven low cut); tracking = `g3r10 + max_gap6`.
Then build the Three.js track-viewer render (mb-crr.19.7) on the composite track set.

**Caveats carried forward:** no ground truth; FRC repro_score is confounded by static tissue
for the SVD-cutoff axis (cross-check straightness/velocity); yz/xy FRC planes saturate at the
coarse-y limit (rely on xz + repro_score + straightness). The 223-acq best-tracking scale-up
hit a Modal detached-client issue (track_acqs on the 223-acq base tag) — retry with the
run_in_background launch pattern.
