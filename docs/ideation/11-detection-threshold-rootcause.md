# mb-rvl root cause: the detection flood is the 2σ threshold, NOT the SVD cutoff

Resolved by a local, no-GPU repro on one downloaded `base60_refs` beamformed shard
(`acq_0000.h5`, the same post-TGC beamform that fed base60). Cost: ~$0 (bandwidth only).

## The puzzle
The `swp_*` detection sweep + the knee-223 re-detect emitted ~450–550k localizations/acq
(~700/frame), ~20× base60's sane 23.6k/acq (~34/frame), and density was INSENSITIVE to the SVD
mode count (fix8≈fix70). Re-detecting the knee "with more modes removed" therefore could not help.

## The repro (CPU, `ultratrace_ulm.svd` on a central patch of the real beamform)
**Cutoff each method picks (modes removed):** `adaptive=70  fast70=70  knee=26`.
- `adaptive ≡ fast70` exactly (both hit the 70 fallback → identical projection → identical output).
- knee removes 26 — and gives essentially the same detection density as 70.

**Detection density at 2σ (same patch):** `adaptive=331  fast70=331  knee=322` det/frame.
→ The SVD method/cutoff has ~no effect on density. All flood equally at 2σ.

**Threshold sweep (adaptive-filtered, scaled patch→acq):**

| σ threshold | localizations/acq |
|---|---|
| 2.0 (sweep + knee-223 used this) | ~1.2M  **(flood)** |
| 3.0 | ~394k |
| 3.5 | ~206k |
| 4.5 | ~44k |
| **5.0** | **~19.6k ≈ base60's 23.6k** |

Detection count is EXPONENTIALLY sensitive to the σ threshold and collapses to base60's density
at ~5σ. base60 was made at ~4.5–5σ; the sweep/knee-223 used 2σ.

## Root cause
`sigma_threshold` is **hardcoded at 2.0** in both detection paths (`detect_acqs` opts ~`app.py:809`,
`detect_acqs_multi` opts ~`app.py:1067`) and is **not exposed as a CLI param**. 2σ is far below the
literature-recommended spawn threshold (4–6σ; see `lit-next-levers.md`). At 2σ the z-score detector
fires on noise everywhere → ~1M+ localizations/acq → clutter saturation. The SVD cutoff (adaptive/
fast/knee) is a red herring for the flood.

## Fix
1. Expose `sigma_threshold` as a param on `detect_acqs` / `detect_acqs_multi`.
2. Default it to ~4–5σ (literature spawn 4–6σ), not 2.0. Optionally two-tier (spawn high ~5σ,
   continue lower ~3.5σ) via the existing `low_conf` path.
3. ONLY THEN is the knee/coverage lever testable: re-detect knee vs adaptive at the SAME sane
   threshold (~5σ), track with g3r10+gap6, FRC+saturation. At 2σ every variant floods, so all prior
   inter-variant detection rankings (incl. the session-2 "knee wins FRC") were confounded.

## Consequence for the session-2 results
The session-2 "knee beats the floor (+4.3% repro)" detection bake-off compared 2σ-flooded variants —
it is not a trustworthy ranking. The TRACKING win (`base223_best`, g3r10+gap6) is unaffected: it uses
base60's sane ~5σ adaptive detections and is dual-arbiter validated.
