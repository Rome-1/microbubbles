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