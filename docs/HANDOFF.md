# HANDOFF — microbubbles 3D ULM improvement project

Last updated mid-session (2026-06-28). Branch: **`epic/mb-crr-foundation`** on the
fork **`origin` = Rome-1/microbubbles** (NEVER push `upstream`=alephneuro). Epic
bead **`mb-crr`**; ideation/build sub-epic **`mb-crr.19`**. Everything below is
committed + pushed unless noted.

## TL;DR
We took Aleph's public 3D ULM pipeline, made it correct + deterministic + fast +
reliable on Modal, built an insight→diff→benchmark machine, ran a cross-model
(Opus+Codex) ideation to a consensus 10 improvements, built + cross-checked all 10,
and started the bake-off. Best validated improvement so far: **region-SVD + Kalman
predicted-gate composite (+50% track length vs baseline)**; **track stitching shows
+146% length but a straightness regression to tune**. A full 223-acq baseline exists.

## How to resume (environment)
- Repo: `/home/rome/gt/microbubbles/crew/cajal`. `source /home/rome/.venv/bin/activate`.
- Modal: `modal config set-environment rome` (profile research-exp). App:
  `scripts/modal/app.py`. Volume `research`, tenant subdir `/root/data/microbubbles/`.
  **All Modal spend needs Rome's OK; verify teardown (`modal app list`) after runs.**
- **The 98 GB raw HDF5 is downloaded to the volume** at
  `/root/data/microbubbles/sanitized_neutral_ultratrace.h5` (98.1 GiB, verified
  complete). Pipeline reads it locally (NOT lazy HTTP — R2 truncates large transfers).
- Codex CLI available + authed: `codex exec -s workspace-write -C <repo> --dangerously-bypass-approvals-and-sandbox -o <outfile> - < promptfile`. (Use the bypass flag; bwrap read-only sandbox is flaky here.)
- Local viz: `python3 scripts/local_view.py {volume|tracks|diff} ...`; metrics: `python3 scripts/bench.py <bin> [<bin2>] --labels a,b`.

## Pipeline + infra (built & validated)
`scripts/modal/app.py` functions: `inspect`, `probe`, `download` (resume-until-verified),
`detect_acqs` (Phase A: checkpointed/additive/instrumented per-acq beamform→filter→detect→localize;
opt-in `detector`/`svd_rank`/`nms_elev`/`elev_debias`/`low_conf`), `track_acqs` (Phase B:
cross-acq tracking from checkpoints), `baseline`/`beamform_all` (older fused path), `volume3d`
(3D SVD power volumes for diffs), `retrack`, `stitch`, `validate_svd`.
- **Data shape:** 223 acqs × 700 frames; beamformed (700,25,225,378) complex64 ≈11.9 GB/acq;
  IQ (702,5,1,134,256) → only 5 tx-angles, 1 physical receive row → 25 SYNTHESIZED elevation planes.
- **Architecture:** fused/streaming beamform→track (never store the 2.4 TB of beamformed volumes);
  per-acq detection CHECKPOINTS on the volume (`detections/<tag>/acq_XXXX.npz`) → resumable + additive.
- **Baselines on the volume:** `tracks/base60` (60-acq), `tracks/base223` (full 4-min, 168k tracks).
  Detection checkpoints `detections/base60` now hold all 223. Reference shards in `beamformed/<tag>_refs`.

## Key findings (the science + scars)
1. **Phase-invariant SVD cutoff (correctness fix):** Aleph's adaptive cutoff used
   `|rfft(eigvec.real)|²` which is phase-DEPENDENT → nondeterministic, CPU≠GPU by 15×.
   Fixed to `|fft(complex eigvec)|²`. The GPU pipeline is now the canonical, deterministic one.
2. **No ground truth:** all metrics are PROXIES. **Power-volume diffs are MISLEADING for
   clutter changes — track-level metrics (bench.py) are the arbiter.** (Motion correction
   looked great in the power volume but FRAGMENTED tracks +199% → excluded.)
3. **R2 truncates large HTTP transfers** silently (urllib EOF without error) — this caused a
   12-hour-timeout total loss. Fixed: download-to-volume (resume-until-verified) + per-acq checkpointing.
4. **GPU memory (A10G 24 GB):** the (700,2.13M) matrix is 11.9 GB; chunk, accumulate Gram in
   complex128 with complex64 chunks, `free_all_blocks()` between iterations, `del` loop-views,
   per-angle device free in beamform. Several OOMs came from violating these.
5. **Validated improvements (60-acq, vs base60):** Kalman predicted-gate ✓ (+43% length);
   region-adaptive SVD ✓ continuity (frag −47%, tracks≥35 +95%; coverage −27%); **composite
   region+Kalman ✓ best (+50% length, −39% frag)**; motion correction ✗.

## The agreed-10 (mb-crr.19) — status
Consensus set in `docs/ideation/03-agreed-10.md` (full trail: 00-context → 01/02 round lists →
03 agreed+confirmations → 04/05 build+integrate → 05-opus-crosscheck). Build split: Opus=tracking,
Codex=detection/SVD; both cross-checked each other (Codex found + Opus fixed a real stitch false-merge).
| # | idea | built | baked off |
|---|------|-------|-----------|
| 1 | track stitching (`track_stitch.py`) | ✅ tests | ✅ +146% len / −47% frag / straightness −38% (TUNE vel_cos_min) |
| 2 | CFAR/MAD detect (`detect_cfar.py`) | ✅ | ⏳ combo8 GPU-validating |
| 3 | adaptive per-block SVD rank (`svd_rank.py`) | ✅ | ⏳ combo8 |
| 4 | elevation-aware Kalman gate | ❌ TODO (Opus) | — |
| 5 | low-conf continuation-only | ✅ (Phase-A persists confidence) | ⏳ tracking-side consumption TODO |
| 6 | anisotropic/PSF NMS + midplane debias (`detect_cfar.py`) | ✅ | ⏳ combo8 |
| 7 | empirical PSF matched-filter (`psf.py`) | ✅ | ⏳ |
| 8 | larger max_gap + gating | ❌ TODO (param sweep) | — |
| 9 | intensity-aware association cost | ❌ TODO (Opus, in kalman) | — |
| 10 | coverage audit (`audit.py`) + split-half metric | half ✅ (audit), split-half TODO | — |

## Pending work (beaded — see `bd show mb-crr.19` children + below)
- Finish Opus tracking items #4, #8, #9 (edit `kalman_tracking_3d`).
- Consume low-conf (#5) in tracking (continuation-only).
- Run the GPU bake-offs (combo8 → 60-acq) for #2/#3/#6/#7; tune stitch `vel_cos_min`(~0.7) + `tol_elev`.
- Add no-GT metrics to bench.py: split-half reproducibility (#10), FRC, saturation (see `docs/literature/`).
- Scale the winning composite to full 223; build the real Three.js `track-viewer` 3D render
  (baseline vs improved) — the "beautiful" render Rome asked about (current MIPs are diagnostic only).
- Literature deep-dives running → `docs/literature/lit-detection-svd.md`, `lit-tracking-eval.md`
  (scientific basis for the knob values).

## Literature-grounded knob settings (next session — see `docs/literature/`)
Two cited reviews (`lit-detection-svd.md` = full Opus deep-dive; `lit-knobs-grounding.md` = Codex summary):
- **SVD cutoff is the big one:** ours floors to ~70 modes (10% of 700), but tissue is only a
  FEW-to-low-TENS of modes → **we over-remove → THIS is the coverage-loss cause.** Switch to a
  data-driven knee (singular-value gradient + spatial-vector correlation) + a separate high-order
  noise cutoff. Likely beats per-block rank for coverage. (Demené 2015, Baranger 2018, Lok/Song 2020.)
- **Stitch `vel_cos_min`: 0.8** (applied; was 0.3 → over-merge). 0.7 permissive edge.
- **Detection: two-tier** — spawn at 4–6σ, continue at 3.5–4.5σ (FN costs ~45% SSIM vs FP ~7% → bias sensitive).
- **Elevation: down-weight** — Kalman `R_yy` ×5–10 (test ×15–25 for our 1-row aperture), Mahalanobis gate, 2D-first.
- **max_gap sweep 4/6/8; SVD-rank delta sweep 0/4/8/12.**
- **First no-GT metric to add to bench.py: split-half FRC/FSC reproducibility** (the over-merge arbiter).

## Budget
~$50 of $100 spent (biggest: the 12-hr timeout ~$13, the full-223 baseline). GPU bake-offs ~$2.5 each.
