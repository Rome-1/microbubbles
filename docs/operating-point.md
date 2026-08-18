# Operating point — SUPERSEDED

> This document recommended a 130/130/130 mm/s gate with `min_track_length` 8, giving 1,531
> tracks ≥35. Both halves have since been corrected:
>
> * **elevation should not be tightened.** Leaving it at the Aleph 2-voxel default gives 1,786
>   tracks ≥35 instead of 1,531 — the voxel scaling encodes per-axis localization uncertainty,
>   which on the synthesized-elevation axis is the dominant term.
> * **the floor is 10, not 8**, once re-derived at the wider in-plane gate.
>
> The canonical explanation, with figures and the full change ledger, is
> **`docs/from-baseline-to-improved.md`**. This file is kept only so the earlier numbers are
> traceable rather than silently rewritten.
