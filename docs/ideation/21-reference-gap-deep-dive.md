# Reference-gap deep dive — why ours is worse, how to match the blog, how to beat both

**Date:** 2026-07-07. Multi-model investigation (Opus + Sonnet + 2× Fable + Codex), each
grounded in the reference's exact params (`docs/ideation/20`, pickle `params`) and the blog
figure (`renders/references/Screenshot 2026-06-29 at 11.51.16.png`). Answers Rome's three
questions: (1) match the blog render, (2) why is ours worse, (3) how to beat both.

---

## Q2 first — why our pipeline is so much worse (Opus, evidence-backed code diff)

**Headline reframe: our tracker is already ~reference-equivalent when run ISOTROPICALLY. The
composite's damage comes from the elevation-anisotropy knobs, and our "winning" composite was
actually a regression that gamed our own FRC proxy.**

The 5.05× total-track gap (254,590 vs 50,456) decomposes as:
- **3.01×** = raw frame count: we track **223 × ~700 = 156,100 frames** vs reference **216 ×
  240 = 51,840**. A beamform/framing difference, not a tracking param.
- **1.68×** = tracks-per-frame, and almost all of that is the composite's elevation anisotropy
  (+86,413 spurious tracks, +51%, over our own isotropic baseline).

Ranked diff with fixes:

| # | Difference | Our value (file:line) | Reference | Fix → effect |
|---|---|---|---|---|
| **1** | **Elevation velocity-gate ×3** — THE 74 cm/s teleports | `elev_gate_factor=3` (`tracking.py:296-298`; composite `app.py:1221`). Note our *default* gate `_tracking_gate` (`tracking.py:230-233`) already computes **exactly** the reference `(0.4008,1.1093,0.4016)` | `max_distance_mm (0.4008,1.1093,0.4016)` (no ×3) | set `elev_gate_factor=1.0` → max speed **74→26 cm/s**, tracks **254k→168k** |
| **2** | Elevation meas-noise ×10 | `elev_meas_factor=10` (`tracking.py:306-308`) | isotropic R | set `=1.0` (works with #1) |
| **3** | max_gap | `6` (`app.py:1217`; gate `tracking.py:437`) | `3` | set `=3` |
| **4** | **frames/acq (crispness driver)** | ~700 (beamform framing) | **240** | reconcile 4:1 planewave compounding — see below |
| **5** | Cross-acq tracking | single 156k-frame pass (`tracking.py:861-878`) | per-acq then combine (`boundary_safe_per_acq`) | track per-acq |
| **6** | Detection params | **IDENTICAL** to reference (sigma 2.0, min_distance 2, smoothing 1.0, centroid, window 5) | same | *not* the over-detection cause; keep detection on global **adaptive+knee** (region/knee floods 20× — `docs/11`) |
| 7 | min_track_length default | 15 in code (`tracking.py:47`, `cli.py:97`), 5 as-run | 5 | default to 5 |

**The FRC confound (important).** `docs/ideation/10` sold the composite as "real recovery"
(+223% curvilinear length, +6% FRC). Opus argues this was the **coarse-elevation analog of the
earlier static-tissue FRC confound**: the ×3 elevation gate makes tracks smear along the
2.77×-coarser, near-degenerate elevation axis; those elevation-smeared teleporting tracks
reproduce across split-half acquisitions (because the elevation axis is nearly degenerate) and
so *game FRC* while being non-physiological (74 cm/s). **Our "best" composite was a regression.**

**Single highest-impact change: `elev_gate_factor=1.0` + `elev_meas_factor=1.0`** — reverts the
composite to the reference's exact isotropic gate. After that, the residual gap (168k vs 50k,
and 78 vs 1,421 long tracks) is **upstream of tracking.py**: the frames/acq framing (#4) and
per-acq tracking (#5). The tracking stage itself is essentially reference-correct once the
elevation knobs are off.

**Why #4 is the crispness driver:** even our isotropic baseline yields only **78 tracks ≥35
frames (max len 72)** vs the reference's **1,421 ≥35 (max 191)** — an 18× long-track deficit at
matched per-frame count. Finer/noisier per-frame beamform (more frames, less compounding)
jitters detection positions, so the Kalman gate rejects the true continuation and tracks die at
median 6. The reference's 4-planewave-compounded 240-frame beamform gives cleaner per-frame
positions → long coherent vessel tracks → crisp render. **This ties directly to the 5-vs-4
transmit question (below): if we compound ~700 single/5-slot frames vs their 4-planewave 240,
that is plausibly the root of the crispness gap.**

---

## Q1 — matching the blog render (Sonnet, built + headless-Chromium verified)

**Blog figure = s1 tab · jet colormap · flow speed 0–38 mm/s (fixed, not auto-scaled) · pure
black bg · small points · 10 mm scalebar · near-coronal camera.**

**v6 binary format (reverse-engineered; unique byte-budget solution):**
- Header 64 B: bytes 0–43 identical to v3 (`<IIIIf` + 6 bbox floats); 44–63 partly resolved
  (scale const + flags).
- Track table: `n_tracks × 16 B` = `u32 point_offset, u32 length, u8 tag, 7×reserved`. The tag
  byte is uniform 0–255, uncorrelated to speed/length/duration → a per-track ID/seed, not physics.
- **Point data is COLUMNAR (struct-of-arrays), 9 B/point** — five back-to-back arrays of
  `total_points`: `int16 X`, `int16 Y`, `int16 Z` (each **anchor + cumulative int16 delta**
  scaled by `bbox_range/65535`), `uint16 packed_frame` (high byte ≈ frame-gap, clusters at 1,
  matches `max_gap=3`), `uint8 speed_byte` (jet color; encodes heavily-smoothed speed, not
  instantaneous velocity). Only the anchor+delta global-columnar scheme decodes all tracks
  inside the header bbox with physical ~0.4 mm/frame steps.
- **Caveat:** the shipped v6 and the given pickle are correlated but NOT byte-identical runs
  (no min_length reproduces s1's 2,294 exactly; v6's own min length = 35).

**What was built (bundles served on :8791, code committed):**
- `outputs/reference/viewer_blogmatch/` — reference s1-equivalent (2,295 tracks; jet 0–38 mm/s;
  black; legend + live 10 mm scalebar; near-coronal). **Close visual match to the blog.**
- `outputs/reference/viewer_ourmatch/` — our composite filtered to the reference's own bar
  (min_length≥25 & speed≤0.40 mm/frame) → **only 221 tracks**: "a sparse, all-blue,
  structureless debris field."
- `scripts/export_blogmatch.py`, `scripts/export_ourmatch.py`,
  `ultratrace_ulm/web/track_viewer/index_blogstyle.html` (shared viewer), minor
  `track_viewer_export.py` refactor (byte-identical regression-tested).

**Match recipe:** population `min_length≈28 & max_raw_speed≤0.48 mm/frame` → 2,295 tracks; color
jet fixed **0–38 mm/s** (= 38/222.43 = 0.171 mm/frame); black bg; point size 0.22; near-coronal
(look down elevation axis, Z-up).

**The stark quantified finding:** only **221 of our 254,590** tracks survive the reference's own
length+velocity bar, vs **~2,295 of their 50,456**. The render gap is **not** a rendering
problem — it is the tracking/detection quality gap from Q2. Remaining gaps to pixel-perfect are
the unreleased front-end (s1/s2 tabs, exact camera/fonts) — impossible by construction.

---

## Q3 — how to beat both systems (Fable A: pipeline · Fable B: render)

**Measured openings (Fable A, from the pickle):**
- The render **discards 83%** of the coverage the tracked data already touches (full tracks hit
  10.5% of grid voxels; the ≥35 render subset hits 1.8%).
- Tracking **discards 69%** of the reference's OWN high-confidence detections (acq 0: 9,761
  detections z=4.9–53.7 → 3,017 tracked points) → **~1.4M discarded real localizations**
  dataset-wide, because greedy Kalman+Hungarian couldn't link them into a ≥5 chain.
- **Killed two hypotheses:** cross-acq stitching is NOT the win (~59 plausible merges total);
  the elevation axis is NOT degenerate (±6.4 mm, std 3.14 — an accuracy/anisotropy problem, not
  emptiness).

**Highest-conviction "beat it" bet:** a **motion-confirmed, globally-associated dense
reconstruction, rendered as a velocity-colored super-resolution density map.** Recover the 69%
orphan detections via global min-cost-flow with the reference's physiological gate, and render
the full motion-verified field instead of length-filtering to 17% coverage. Zero new data, zero
training, numpy/CuPy on artifacts we hold. Success = one defensible number, **Coverage@equal-FRC**:
beat the reference's 1.8% grid-fill by a large multiple while (a) containing ≥95% of their 2,294
rendered vessels and (b) producing >1,421 ≥35-frame tracks at *their own* velocity gate.

Reusable GT-free yardsticks (the reference tracks are now the ruler): Ref-recall (≥95%),
split-half FRC (≤ reference), Coverage@equal-FRC (headline), ≥35-track count at the 0.40/1.11/0.40
gate (beat 1,421), held-out-acquisition recall.

Supporting levers (ranked): sub-frame spline interpolation (continuous streamlines);
orphan re-attachment onto a vessel-graph skeleton; sparse deconvolution for 40–130 bubbles/frame
overlap; anisotropic-covariance + flow-tangent elevation super-resolution; matched-filter +
coherence-factor localization over 2σ-centroid.

**The render that beats the blog (Fable B) — "the living angiogram":** the blog spends its whole
budget on one channel (scalar speed→jet). We own six + two free (velocity **direction**, already
computed then discarded at `track_viewer_export.py:55-64`; and a **cardiac time axis**). Compose
the near-reach wins: **drizzle density** (all localizations as sub-pixel splats → continuous
vessels, not dots) + **flow-direction color** (arteries vs antiparallel veins) + **eye-dome
lighting + fog** (reads as a solid 3D cast) + **animated flow comets** + **trust scaffolding**
(calibrated colorbar, mm cube, orientation gnomon, opacity = confidence). The viewer already has
the time engine (`aFrame`, `u_reveal`, `vGlow`).

**Moonshot (both agents converge):** a **self-gated 4D beating vasculature** — cardiac phase
estimated from the aggregate flow oscillation (no ECG; 5–7 Hz oversampled 30–45× at 222 Hz,
~1,200+ cycles pooled over 216 acqs), rendered as a 6 s loop of the pulse wave propagating
through an artery/vein-separated tree. Underneath it, a continuous **4D flow-field PINN**
(occupancy, velocity, confidence) regularized by divergence-free flow + tubularity + Womersley
profiles, validated by held-out-acquisition recall — detection and tracking dissolve into one
inverse problem. Ships pressure / wall-shear-stress: CFD-from-ULM.

---

## Immediate action list (ranked)
1. **Revert the composite elevation knobs** (`elev_gate_factor=1.0`, `elev_meas_factor=1.0`,
   `max_gap=3`) and re-render — kills the 74 cm/s teleports, ~254k→168k tracks. CPU/quick.
   Re-run the blog-match export on the fixed tracks. *(bead mb-d3d)*
2. **Reconcile the 700-vs-240 frames / 4-vs-5 planewave framing** (Q's below) — the crispness
   driver. Needs the raw `/config` (Modal `angle_probe`, running).
3. **Prototype the motion-confirmed dense render + global min-cost-flow** (the "beat it" bet) —
   scoped, GT-free-validated against the reference tracks.
4. **Build the "living angiogram" render** (direction color + drizzle + EDL) — mostly free /
   in-shader.

## Questions for the PR / issue-#2 reply (draft — NOT sent, awaiting Rome)
1. **Transmit slots (4 vs 5) — RESOLVED from our side, one question remains.** Modal
   `angle_probe` on acq 0 shows `tx_delays = [5, 134]`: **all 5 slots are real steered
   plane-wave transmits**, near-identical energy, forming a symmetric sweep (±steepest 1.81 µs,
   ±half 0.90 µs, **0° center = slot2, zero delays**). No junk/noise slot (aside from the 2
   `num_noise_loops`). So the released data is genuinely **5-angle**, and there is no
   "odd-one-out" to drop — which **conflicts** with the maintainer's "4 planewaves/frame"
   (888/222.43 = 4). **Question for the PR:** does production compound only 4 of the 5 released
   angles (if so, which — the 0° center or an extreme)? Or is the raw data 5-angle and the "4"
   a simplification? (Not blocking us — our beamform compounds all 5 legitimately; but it bears
   on the 700-vs-240 frame-scope and the crispness reconciliation.)
2. **The deterministic-tracking "update":** it isn't in the public repo (newest public commit
   `5fc46c4`, 2026-06-26, predates the fix). Is it in the private repo, and will it be pushed
   public? We already reproduce deterministically (our fork uses `eigh`), so this is confirmation
   not a blocker.
3. **(confirmed, no need to ask)** frame rate 222.43 Hz ✓, physical PRF 888 Hz ✓, 216 acqs ×
   240 frames ✓, per-acq-then-combine tracking ✓ — all read from the pickle params.

Codex cross-check: `docs/ideation/consults/codex-refdiff.md`.
