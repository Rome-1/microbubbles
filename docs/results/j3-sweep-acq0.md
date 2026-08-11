# J3 operating-point sweep

acq 0 of `corrected_bf_1.h5`, volume (240, 25, 154, 275), 150s GPU

**Measured PSF** (400 isolated patches): FWHM elev 1.66 mm, z 1.00 mm, x 1.00 mm; axial carrier 3.096 rad/voxel (pi = 3.142, the 2k carrier lambda/4 sampling predicts).

**Injection**: 6 realizations x 300 bubbles, 29 bubble-frames/frame added; compounding gain applied: True.

## Resolved SVD cutoffs

| arm | low (modes removed) | high_remove (MP) | modes kept |
|---|---|---|---|
| rank24 | 24 | 0 | 216 |
| rank24+mp | 24 | 122 | 94 |
| knee | 21 | 0 | 219 |
| knee+mp | 21 | 122 | 97 |
| knee_spatial | 2 | 0 | 238 |
| knee_spatial+mp | 2 | 122 | 116 |

## Operating points at matched density = 34 detections/frame

| rank | filter | normalization | low | recovery | vs status quo | +-SE | depth non-unif. |
|---|---|---|---|---|---|---|---|
| 1 | rank24 | spatial_tgc | 24 | 0.396 | +0.030 | 0.003 | 0.633 |
| 2 | knee | spatial_tgc | 21 | 0.390 | +0.024 | 0.003 | 0.770 |
| 3 | knee | per_elev_zband | 21 | 0.385 | +0.019 | 0.003 | 0.872 |
| 4 | rank24 | per_elev_zband | 24 | 0.385 | +0.018 | 0.003 | 0.860 |
| 5 | knee | per_elev | 21 | 0.369 | +0.003 | 0.003 | 0.729 |
| 6 | rank24 | per_elev | 24 | 0.366 | +0.000 | 0.003 | 0.903 |
| 7 | rank24+mp | spatial_tgc | 24 | 0.272 | -0.094 | 0.003 | 0.824 |
| 8 | knee+mp | spatial_tgc | 21 | 0.269 | -0.097 | 0.003 | 0.909 |
| 9 | rank24+mp | per_elev_zband | 24 | 0.267 | -0.099 | 0.003 | 0.921 |
| 10 | knee+mp | per_elev_zband | 21 | 0.266 | -0.101 | 0.003 | 0.882 |
| 11 | knee+mp | per_elev | 21 | 0.250 | -0.116 | 0.003 | 1.220 |
| 12 | rank24+mp | per_elev | 24 | 0.245 | -0.121 | 0.003 | 1.186 |
| 13 | knee_spatial | per_elev | 2 | 0.003 | -0.363 | 0.000 | nan |
| 14 | knee_spatial+mp | per_elev | 2 | 0.003 | -0.363 | 0.000 | nan |
| 15 | knee_spatial+mp | per_elev_zband | 2 | 0.003 | -0.364 | 0.000 | nan |
| 16 | knee_spatial | per_elev_zband | 2 | 0.003 | -0.364 | 0.000 | nan |
| 17 | knee_spatial | spatial_tgc | 2 | 0.002 | -0.364 | 0.000 | nan |
| 18 | knee_spatial+mp | spatial_tgc | 2 | 0.002 | -0.364 | 0.000 | nan |

## Stratified recovery -- `rank24|per_elev` (STATUS QUO)

speed (mm/s): 5=0.244(n=6552)  20=0.435(n=6096)  45=0.460(n=6216)  70=0.411(n=5952)  88.97=0.336(n=6125)  110=0.353(n=5488)  130=0.330(n=5615)
depth band: 0=0.306(n=2182)  1=0.283(n=5683)  2=0.391(n=6151)  3=0.375(n=6790)  4=0.349(n=6509)  5=0.403(n=6629)  6=0.447(n=6114)  7=0.250(n=1986)
SNR (dB): 6=0.017(n=6853)  9=0.064(n=7594)  12=0.304(n=6895)  15=0.526(n=7073)  18=0.592(n=6546)  21=0.721(n=7083)
direction: axial=0.293(n=11064)  elev=0.293(n=9851)  lateral=0.515(n=10080)  oblique=0.368(n=11049)

depth non-uniformity (spread/mean) by SNR: 6dB=nan  9dB=1.823  12dB=0.983  15dB=0.767  18dB=0.665  21dB=0.279

recovery by speed x direction (the compounding null is AXIAL only):

| direction | 5 | 20 | 45 | 70 | 88.97 | 110 | 130 |
|---|---|---|---|---|---|---|---|
| axial | 0.591 | 0.709 | 0.511 | 0.107 | 0.002 | 0.020 | 0.044 |
| elev | 0.051 | 0.097 | 0.203 | 0.383 | 0.392 | 0.596 | 0.487 |
| lateral | 0.077 | 0.376 | 0.535 | 0.666 | 0.630 | 0.624 | 0.653 |
| oblique | 0.209 | 0.561 | 0.575 | 0.489 | 0.362 | 0.256 | 0.148 |
| *predicted 4-angle gain (dB)* | -0.0 | -0.7 | -3.8 | -11.4 | -92.1 | -13.8 | -11.3 |

recovery by depth band at fixed SNR:

| SNR (dB) | band 0 | band 1 | band 2 | band 3 | band 4 | band 5 | band 6 | band 7 |
|---|---|---|---|---|---|---|---|---|
| 6 | 0.000 | 0.002 | 0.013 | 0.032 | 0.019 | 0.027 | 0.018 | 0.003 |
| 9 | 0.002 | 0.016 | 0.030 | 0.070 | 0.107 | 0.082 | 0.098 | 0.054 |
| 12 | 0.042 | 0.135 | 0.237 | 0.369 | 0.301 | 0.440 | 0.384 | 0.212 |
| 15 | 0.284 | 0.403 | 0.535 | 0.495 | 0.596 | 0.621 | 0.636 | 0.268 |
| 18 | 0.604 | 0.510 | 0.595 | 0.583 | 0.605 | 0.640 | 0.681 | 0.305 |
| 21 | 0.733 | 0.792 | 0.777 | 0.715 | 0.592 | 0.669 | 0.782 | 0.668 |

## Stratified recovery -- `rank24|spatial_tgc` (WINNER)

speed (mm/s): 5=0.267(n=6552)  20=0.462(n=6096)  45=0.515(n=6216)  70=0.454(n=5952)  88.97=0.356(n=6125)  110=0.356(n=5488)  130=0.363(n=5615)
depth band: 0=0.388(n=2182)  1=0.377(n=5683)  2=0.438(n=6151)  3=0.391(n=6790)  4=0.347(n=6509)  5=0.402(n=6629)  6=0.465(n=6114)  7=0.270(n=1986)
SNR (dB): 6=0.021(n=6853)  9=0.095(n=7594)  12=0.365(n=6895)  15=0.556(n=7073)  18=0.625(n=6546)  21=0.739(n=7083)
direction: axial=0.326(n=11064)  elev=0.325(n=9851)  lateral=0.534(n=10080)  oblique=0.402(n=11049)

depth non-uniformity (spread/mean) by SNR: 6dB=nan  9dB=1.289  12dB=0.391  15dB=0.653  18dB=0.503  21dB=0.332

recovery by speed x direction (the compounding null is AXIAL only):

| direction | 5 | 20 | 45 | 70 | 88.97 | 110 | 130 |
|---|---|---|---|---|---|---|---|
| axial | 0.613 | 0.757 | 0.611 | 0.123 | 0.002 | 0.020 | 0.088 |
| elev | 0.065 | 0.108 | 0.245 | 0.447 | 0.430 | 0.608 | 0.538 |
| lateral | 0.097 | 0.410 | 0.572 | 0.682 | 0.626 | 0.610 | 0.691 |
| oblique | 0.245 | 0.576 | 0.613 | 0.570 | 0.408 | 0.272 | 0.152 |
| *predicted 4-angle gain (dB)* | -0.0 | -0.7 | -3.8 | -11.4 | -92.1 | -13.8 | -11.3 |

recovery by depth band at fixed SNR:

| SNR (dB) | band 0 | band 1 | band 2 | band 3 | band 4 | band 5 | band 6 | band 7 |
|---|---|---|---|---|---|---|---|---|
| 6 | 0.000 | 0.007 | 0.037 | 0.034 | 0.017 | 0.025 | 0.016 | 0.018 |
| 9 | 0.029 | 0.103 | 0.078 | 0.088 | 0.124 | 0.077 | 0.142 | 0.057 |
| 12 | 0.138 | 0.391 | 0.349 | 0.415 | 0.284 | 0.433 | 0.411 | 0.199 |
| 15 | 0.510 | 0.497 | 0.576 | 0.501 | 0.593 | 0.614 | 0.644 | 0.299 |
| 18 | 0.676 | 0.610 | 0.630 | 0.602 | 0.601 | 0.655 | 0.697 | 0.392 |
| 21 | 0.836 | 0.836 | 0.803 | 0.722 | 0.591 | 0.676 | 0.791 | 0.654 |

## False-alarm cross-check on null data

| filter/norm | null_block1 (/frame) | null_block20 (/frame) | null_phase (/frame) | null_quiet_crop (/frame) |
|---|---|---|---|---|
| knee+mp|per_elev | 34.03 | 34.01 | 79.36 | 6.58 |
| knee+mp|per_elev_zband | 33.95 | 34.00 | 80.62 | 5.05 |
| knee+mp|spatial_tgc | 34.02 | 33.99 | 68.40 | 5.87 |
| knee_spatial+mp|per_elev | 34.00 | 34.00 | 69.86 | 8.78 |
| knee_spatial+mp|per_elev_zband | 34.00 | 33.99 | 58.01 | 6.91 |
| knee_spatial+mp|spatial_tgc | 34.00 | 34.00 | 57.10 | 8.46 |
| knee_spatial|per_elev | 34.00 | 34.00 | 70.02 | 8.82 |
| knee_spatial|per_elev_zband | 34.00 | 34.00 | 57.83 | 6.90 |
| knee_spatial|spatial_tgc | 34.00 | 34.00 | 57.12 | 8.44 |
| knee|per_elev | 33.98 | 34.00 | 82.30 | 6.40 |
| knee|per_elev_zband | 33.98 | 34.02 | 84.12 | 5.30 |
| knee|spatial_tgc | 34.00 | 34.03 | 72.87 | 5.85 |
| rank24+mp|per_elev | 34.09 | 34.00 | 79.73 | 5.61 |
| rank24+mp|per_elev_zband | 34.02 | 34.01 | 84.77 | 3.95 |
| rank24+mp|spatial_tgc | 33.97 | 33.95 | 71.36 | 5.30 |
| rank24|per_elev | 33.98 | 33.98 | 83.14 | 5.65 |
| rank24|per_elev_zband | 34.01 | 34.02 | 86.40 | 4.30 |
| rank24|spatial_tgc | 33.98 | 34.00 | 76.75 | 5.35 |

What each null preserves and breaks:

- **block_shuffle**: Contiguous blocks of frames permuted. PRESERVES: every frame's spatial content exactly (so speckle, PSF, depth-dependent gain and the amplitude distribution are untouched) and, within a block, the short-timescale temporal coherence the SVD needs to concentrate tissue in a few leading modes. BREAKS: trajectories longer than the block, and the long-timescale tissue modes across block boundaries. The block size is the dial: block=1 (full shuffle) destroys the clutter structure entirely and leaves tissue residue the SVD cannot remove, so it is a HARD null that OVER-states false alarms; large blocks are gentler but leave real short bubble tracks intact, understating them. Sweep it rather than trusting one value.

- **phase_surrogate**: Per-voxel temporal Fourier surrogate: each voxel's time series keeps its power spectrum exactly and gets an INDEPENDENT random phase per bin. PRESERVES: the per-voxel temporal spectrum (so tissue still looks slow and noise still looks fast -- the axis the clutter filter cuts on) and the marginal amplitude scale. BREAKS: inter-voxel coherence, and the mechanism is exact rather than hand-wavy -- the randomized array is still supported on the same frequency bins, so the clutter's rank goes from 'number of spatial modes' (a few) to 'bandwidth in bins' (tens). A fixed-rank cut therefore removes much less of it, leaving tissue residue and OVER-stating false alarms. Two corollaries worth stating: a monochromatic clutter mode would survive with rank 1 intact (so this null is only hard on realistically broadband clutter), and a single GLOBAL per-bin phase would leave the Gram spectrum exactly unchanged while failing to remove bubbles -- which is why the draw is per-voxel. Complementary to block_shuffle: hard on the SPATIAL low-rank assumption where block_shuffle is hard on the TEMPORAL one.

- **quiet_crop**: A spatial crop of the real volume chosen where the baseline detector finds the fewest suprathreshold voxels. PRESERVES: everything -- it is real data, with real clutter, real noise and the real PSF. BREAKS: nothing, which is the problem: it is not guaranteed bubble-free, so any detection counted here may be a real bubble. It therefore gives a CONSERVATIVE (upper-bound) false-alarm rate and is the closest thing to an assumption-free null we have. Treat disagreement between quiet_crop and the synthetic nulls as the honest error bar on the false-alarm axis.

- **clean**: Not a null: the un-injected real volume. Used for MATCHED DETECTION DENSITY, the primary operating-point match. It makes no assumption at all, which is why it, and not a synthetic null, is the primary axis; the nulls are the cross-check.

