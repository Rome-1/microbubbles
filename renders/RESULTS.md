# Insight results (60-acq evaluation, track-level metrics vs base60 baseline)

All diffed apples-to-apples (same 60 acquisitions) via the checkpointed pipeline
+ `scripts/bench.py`. Track-level metrics (not just power-volume MIPs, which were
shown to be misleading). "Better" for ULM = longer, less-fragmented tracks.

| Insight | Verdict | Key track-level deltas vs baseline |
| --- | --- | --- |
| Predicted-state Kalman gate (mb-crr.10) | ✅ positive | mean curv-length +43%, less fragmentation |
| Region-adaptive SVD (mb-crr.12) | ✅ positive (continuity) | fragmentation −47%, tracks≥20 +79%, tracks≥35 +95%; trade-off: coverage −26% |
| Motion correction (mb-crr.11) | ❌ negative | fragmentation **+199%**, lost all ≥50-frame tracks — de-clutters the power volume but fragments TRACKS. Excluded from composite. |
| **Composite = region-SVD + Kalman** | ✅ **best** | **mean track length +50% (0.59→0.89mm), tracks≥35 +79%, fragmentation −39%, ≥20-frame tracks ~2x**; trade-off: coverage/density lower |

Key methodological finding: **power-volume diffs are misleading for clutter changes;
track-level metrics are the real arbiter** (motion looked good visually, hurt tracks).

Renders: 08 = baseline track density; 09 = composite track density (denser, longer tracks).
