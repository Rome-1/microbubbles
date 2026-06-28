Implemented both tasks. No commit made.

Changed:
- Added adversarial stitch tests in [tests/test_track_stitch.py](/home/rome/gt/microbubbles/crew/cajal/tests/test_track_stitch.py:53).
  - Crossing fragments pass: velocity agreement prevents swapped links.
  - Nearby parallel elevation-separated bubbles are a real false-merge risk with current defaults, recorded as strict `xfail`.
- Wired `detect_acqs` bake-off params in [scripts/modal/app.py](/home/rome/gt/microbubbles/crew/cajal/scripts/modal/app.py:724):
  - `detector="zscore"|"cfar"|"psf"`
  - `svd_rank`, `rank_delta`
  - `nms_elev`, `elev_debias`
  - `low_conf`, persisted as `confidence` in detection `.npz`
- Added Modal CPU `stitch` function in [scripts/modal/app.py](/home/rome/gt/microbubbles/crew/cajal/scripts/modal/app.py:1100).
- Added confidence passthrough without touching Kalman in [tracking.py](/home/rome/gt/microbubbles/crew/cajal/ultratrace_ulm/tracking.py:681).
- Appended verdict and usage examples to [04-codex-build.md](/home/rome/gt/microbubbles/crew/cajal/docs/ideation/04-codex-build.md:23).

Verification:
- `python3 -c "import ast; ast.parse(open('scripts/modal/app.py').read())"` passed.
- `PYTHONPATH=. pytest -q` passed: `44 passed, 1 skipped, 1 xfailed`.
- Confidence-localization smoke passed under `uv run`.
- `git diff --check` passed.

Rafter:
- `rafter run` could not run because `RAFTER_API_KEY` is not configured.
- `rafter secrets .` passed: no secrets detected.
- Manual Rafter review: path writes/reads use `_guard`, detector is allow-listed, no shell execution added, `np.load` stays non-pickle, and pickle loading is limited to the project’s own Modal volume track artifact.