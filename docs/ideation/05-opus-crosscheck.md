# Opus cross-check of Codex's detection/SVD cluster

Author: Opus 4.8 (orchestrator). Reviewed `detect_cfar.py`, `svd_rank.py` against
the two acceptance conditions I raised in `03-opus-confirm.md`.

## #2 CFAR — global floor under the local scale: ✅ PASS
- `_global_robust_scale(time_mean)` → `global_scale`; the CFAR threshold floor is
  `floor = global_floor_fraction(0.35) × global_scale` (`cfar_background_map`,
  ~line 127). `elevation_bias_profile` also floors at `0.25` via `np.maximum`.
- Conclusion: the local median+k·MAD detector cannot amplify pure noise in empty
  tissue to threshold. The concern is addressed.

## #3 adaptive rank — k-map regularization actually applied + swept: ✅ PASS
- `regularize_rank_map(raw, global_rank, delta, smooth_size)` median-smooths the
  (n_z×n_x) rank map then clamps each block to `global_rank ± delta`. `delta=0`
  ⇒ global_rank everywhere (today's behavior); `delta=∞` ⇒ raw per-block; finite
  ⇒ regularized. This is exactly the knob that makes per-block rank viable without
  the per-block intensity banding that killed the original `per_block`. Sweepable.

## Verdict
Codex's cluster is sound on the two load-bearing points. Remaining validation is
GPU-path correctness (cupy untested) + the actual bake-off vs base60, which is the
next step. Codex is wiring the Modal bake-off integration + adversarially
cross-checking my `track_stitch.py` (false-merge stress tests) in parallel.
