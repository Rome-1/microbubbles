Written to [docs/literature/lit-knobs-grounding.md](/home/rome/gt/microbubbles/crew/cajal/docs/literature/lit-knobs-grounding.md).

Key recommendations captured there:
- SVD rank delta sweep: `0, 4, 8, 12`, with smoothed/clamped per-block k-map.
- Detection: local MAD/CFAR normalization, starts at `k=4.5-6.0`, continuation at `k=3.5-4.5`.
- NMS/localization: PSF-tied and anisotropic, especially in elevation.
- Elevation: down-weight tracking variance `5-10x`, and test `15-25x` for this 1-row aperture.
- Tracking gaps: sweep `4, 6, 8` frames with velocity/predicted-position gating.
- Stitching: use velocity cosine `>=0.8` by default, `0.7` as permissive edge; avoid `0.3`.
- First no-GT metric for `bench.py`: split-half FRC/FSC-style reproducibility.

I did not use live web access. I also left the pre-existing untracked `docs/literature/lit-detection-svd.md` untouched.