Implemented the DETECTION + SVD-RANK cluster as opt-in additions, without changing the baseline tracker or `kalman_tracking_3d`.

Added:

- [detect_cfar.py](/home/rome/gt/microbubbles/crew/cajal/ultratrace_ulm/detect_cfar.py): local CFAR/MAD detector, elevation debiasing, anisotropic NMS, GPU path, and high/low confidence detection output.
- [svd_rank.py](/home/rome/gt/microbubbles/crew/cajal/ultratrace_ulm/svd_rank.py): adaptive per-block rank maps, `delta` regularization, region SVD wrapper, GPU path.
- [psf.py](/home/rome/gt/microbubbles/crew/cajal/ultratrace_ulm/psf.py): empirical PSF mining and normalized matched-filter detector, CPU/GPU.
- [audit.py](/home/rome/gt/microbubbles/crew/cajal/ultratrace_ulm/audit.py): per-stage energy, density, accepted/rejected coverage audit.
- Local numpy tests in `tests/test_detect_cfar.py`, `tests/test_svd_rank.py`, `tests/test_psf.py`, and `tests/test_audit.py`.
- Build summary in [04-codex-build.md](/home/rome/gt/microbubbles/crew/cajal/docs/ideation/04-codex-build.md), including APIs, wiring examples, test results, and GPU validation asks.

Verification:

```bash
PYTHONPATH=. pytest -q
```

Result:

```text
43 passed, 1 skipped in 4.29s
```

No commit made. Note: `docs/ideation/03-codex-confirm.md` and `docs/ideation/03-opus-confirm.md` are also untracked in the worktree but were not created by this pass.

## Modal bake-off wiring and stitching cross-check

Updated [scripts/modal/app.py](/home/rome/gt/microbubbles/crew/cajal/scripts/modal/app.py) so `detect_acqs` keeps the existing default path (`detector="zscore"`, plain global/region GPU SVD, `detect_batch_gpu`) and adds opt-in bake-off switches:

- `detector="zscore"|"cfar"|"psf"` selects the GPU z-score detector, GPU CFAR/MAD detector, or empirical-PSF matched-filter detector.
- `svd_rank=True` or `svd_rank="elbow"/"energy"` enables adaptive per-block ranked region SVD; `rank_delta` controls the rank clamp width.
- `nms_elev=<planes>` enables anisotropic NMS for CFAR/PSF; `elev_debias=True` enables smooth elevation midplane debiasing.
- `low_conf=True` runs one sigma lower and persists a `confidence` array in each `detections/<tag>/acq_*.npz` (`1` = original high threshold, `0` = low continuation candidate). `track_acqs` loads the flag but still leaves Kalman consumption unchanged.

Example Modal runs:

```bash
modal run scripts/modal/app.py::detect_acqs --tag cfar_rank --detector cfar --svd-rank True --rank-delta 2 --nms-elev 5 --elev-debias True --low-conf True
modal run scripts/modal/app.py::track_acqs --tag cfar_rank --out-tag cfar_rank_tracks
modal run scripts/modal/app.py::stitch --tag baseline --out-tag baseline_stitched
```

Added `stitch` (CPU) to load `tracks/<tag>/tracks.pkl`, apply `track_stitch.stitch_pickle_data`, write `tracks/<out_tag>/tracks.pkl`, and export `tracks_min{5,20,50}.bin` for cheap stitched-vs-baseline benching.

Cross-check verdict on Opus's stitching risk: the velocity gate holds for a crossing-fragment adversary, but the loose elevation gate can still false-merge two distinct parallel bubbles if they are close in x/z, separated mostly in elevation, and have matching velocity/intensity. That stress case is now a strict xfail in [tests/test_track_stitch.py](/home/rome/gt/microbubbles/crew/cajal/tests/test_track_stitch.py); it documents the current failure mode rather than treating increased mean track length as automatically valid.
