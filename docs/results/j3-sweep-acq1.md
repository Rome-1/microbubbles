# J3 operating-point sweep

acq 0 of `j3_bf_acq1.h5`, volume (240, 25, 154, 275), 139s GPU

**Measured PSF** (400 isolated patches): FWHM elev 1.66 mm, z 1.00 mm, x 1.00 mm; axial carrier 2.891 rad/voxel (pi = 3.142, the 2k carrier lambda/4 sampling predicts).

**Injection**: 6 realizations x 300 bubbles, 29 bubble-frames/frame added; compounding gain applied: True.

## Resolved SVD cutoffs

| arm | low (modes removed) | high_remove (MP) | modes kept |
|---|---|---|---|
| rank24 | 24 | 0 | 216 |
| rank24+mp | 24 | 121 | 95 |
| knee | 24 | 0 | 216 |
| knee+mp | 24 | 121 | 95 |
| knee_spatial | 2 | 0 | 238 |
| knee_spatial+mp | 2 | 121 | 117 |

## Operating points at matched density = 34 detections/frame

| rank | filter | normalization | low | recovery | vs status quo | +-SE | depth non-unif. |
|---|---|---|---|---|---|---|---|
| 1 | rank24 | spatial_tgc | 24 | 0.564 | +0.013 | 0.003 | 0.452 |
| 2 | knee | spatial_tgc | 24 | 0.564 | +0.013 | 0.003 | 0.452 |
| 3 | rank24 | per_elev | 24 | 0.551 | +0.000 | 0.003 | 0.621 |
| 4 | knee | per_elev | 24 | 0.551 | +0.000 | 0.003 | 0.621 |
| 5 | rank24 | per_elev_zband | 24 | 0.537 | -0.014 | 0.003 | 0.494 |
| 6 | knee | per_elev_zband | 24 | 0.537 | -0.014 | 0.003 | 0.494 |
| 7 | rank24+mp | spatial_tgc | 24 | 0.425 | -0.126 | 0.003 | 0.763 |
| 8 | knee+mp | spatial_tgc | 24 | 0.425 | -0.126 | 0.003 | 0.763 |
| 9 | rank24+mp | per_elev | 24 | 0.410 | -0.140 | 0.003 | 0.828 |
| 10 | knee+mp | per_elev | 24 | 0.410 | -0.140 | 0.003 | 0.828 |
| 11 | rank24+mp | per_elev_zband | 24 | 0.407 | -0.143 | 0.003 | 0.556 |
| 12 | knee+mp | per_elev_zband | 24 | 0.407 | -0.143 | 0.003 | 0.556 |
| 13 | knee_spatial+mp | per_elev_zband | 2 | 0.002 | -0.549 | 0.000 | nan |
| 14 | knee_spatial+mp | per_elev | 2 | 0.002 | -0.549 | 0.000 | nan |
| 15 | knee_spatial+mp | spatial_tgc | 2 | 0.002 | -0.549 | 0.000 | nan |
| 16 | knee_spatial | per_elev | 2 | 0.002 | -0.549 | 0.000 | nan |
| 17 | knee_spatial | per_elev_zband | 2 | 0.002 | -0.549 | 0.000 | nan |
| 18 | knee_spatial | spatial_tgc | 2 | 0.002 | -0.549 | 0.000 | nan |

## Stratified recovery -- `rank24|per_elev` (STATUS QUO)

speed (mm/s): 5=0.343(n=6552)  20=0.668(n=6096)  45=0.685(n=6216)  70=0.636(n=5952)  88.97=0.486(n=6125)  110=0.515(n=5488)  130=0.530(n=5615)
depth band: 0=0.434(n=2182)  1=0.484(n=5683)  2=0.534(n=6151)  3=0.556(n=6790)  4=0.574(n=6509)  5=0.593(n=6629)  6=0.636(n=6114)  7=0.417(n=1986)
SNR (dB): 6=0.175(n=6853)  9=0.392(n=7594)  12=0.528(n=6895)  15=0.664(n=7073)  18=0.754(n=6546)  21=0.805(n=7083)
direction: axial=0.463(n=11064)  elev=0.429(n=9851)  lateral=0.756(n=10080)  oblique=0.558(n=11049)

depth non-uniformity (spread/mean) by SNR: 6dB=1.240  9dB=1.386  12dB=0.280  15dB=0.493  18dB=0.156  21dB=0.168

recovery by speed x direction (the compounding null is AXIAL only):

| direction | 5 | 20 | 45 | 70 | 88.97 | 110 | 130 |
|---|---|---|---|---|---|---|---|
| axial | 0.682 | 0.938 | 0.822 | 0.316 | 0.006 | 0.147 | 0.276 |
| elev | 0.137 | 0.220 | 0.262 | 0.562 | 0.587 | 0.736 | 0.683 |
| lateral | 0.193 | 0.739 | 0.818 | 0.883 | 0.873 | 0.823 | 0.910 |
| oblique | 0.315 | 0.794 | 0.815 | 0.783 | 0.537 | 0.442 | 0.266 |
| *predicted 4-angle gain (dB)* | -0.0 | -0.7 | -3.8 | -11.4 | -92.1 | -13.8 | -11.3 |

recovery by depth band at fixed SNR:

| SNR (dB) | band 0 | band 1 | band 2 | band 3 | band 4 | band 5 | band 6 | band 7 |
|---|---|---|---|---|---|---|---|---|
| 6 | 0.007 | 0.067 | 0.120 | 0.177 | 0.284 | 0.242 | 0.237 | 0.096 |
| 9 | 0.055 | 0.400 | 0.323 | 0.356 | 0.519 | 0.418 | 0.541 | 0.197 |
| 12 | 0.326 | 0.522 | 0.462 | 0.603 | 0.491 | 0.614 | 0.561 | 0.264 |
| 15 | 0.590 | 0.546 | 0.602 | 0.685 | 0.703 | 0.753 | 0.759 | 0.445 |
| 18 | 0.700 | 0.688 | 0.704 | 0.737 | 0.800 | 0.805 | 0.798 | 0.761 |
| 21 | 0.849 | 0.859 | 0.837 | 0.778 | 0.749 | 0.762 | 0.848 | 0.724 |

## Stratified recovery -- `rank24|spatial_tgc` (WINNER)

speed (mm/s): 5=0.364(n=6552)  20=0.673(n=6096)  45=0.699(n=6216)  70=0.645(n=5952)  88.97=0.504(n=6125)  110=0.520(n=5488)  130=0.552(n=5615)
depth band: 0=0.522(n=2182)  1=0.562(n=5683)  2=0.552(n=6151)  3=0.551(n=6790)  4=0.551(n=6509)  5=0.586(n=6629)  6=0.638(n=6114)  7=0.439(n=1986)
SNR (dB): 6=0.189(n=6853)  9=0.412(n=7594)  12=0.545(n=6895)  15=0.686(n=7073)  18=0.759(n=6546)  21=0.806(n=7083)
direction: axial=0.467(n=11064)  elev=0.451(n=9851)  lateral=0.767(n=10080)  oblique=0.576(n=11049)

depth non-uniformity (spread/mean) by SNR: 6dB=0.909  9dB=0.924  12dB=0.285  15dB=0.285  18dB=0.095  21dB=0.216

recovery by speed x direction (the compounding null is AXIAL only):

| direction | 5 | 20 | 45 | 70 | 88.97 | 110 | 130 |
|---|---|---|---|---|---|---|---|
| axial | 0.695 | 0.935 | 0.826 | 0.326 | 0.006 | 0.135 | 0.286 |
| elev | 0.146 | 0.228 | 0.300 | 0.602 | 0.598 | 0.733 | 0.738 |
| lateral | 0.210 | 0.756 | 0.828 | 0.876 | 0.907 | 0.840 | 0.904 |
| oblique | 0.356 | 0.793 | 0.817 | 0.780 | 0.562 | 0.459 | 0.301 |
| *predicted 4-angle gain (dB)* | -0.0 | -0.7 | -3.8 | -11.4 | -92.1 | -13.8 | -11.3 |

recovery by depth band at fixed SNR:

| SNR (dB) | band 0 | band 1 | band 2 | band 3 | band 4 | band 5 | band 6 | band 7 |
|---|---|---|---|---|---|---|---|---|
| 6 | 0.034 | 0.202 | 0.138 | 0.142 | 0.225 | 0.228 | 0.267 | 0.098 |
| 9 | 0.204 | 0.520 | 0.347 | 0.376 | 0.487 | 0.388 | 0.533 | 0.182 |
| 12 | 0.435 | 0.596 | 0.469 | 0.596 | 0.472 | 0.627 | 0.569 | 0.342 |
| 15 | 0.742 | 0.598 | 0.620 | 0.681 | 0.718 | 0.754 | 0.754 | 0.561 |
| 18 | 0.753 | 0.728 | 0.733 | 0.734 | 0.771 | 0.800 | 0.792 | 0.746 |
| 21 | 0.889 | 0.878 | 0.848 | 0.771 | 0.736 | 0.753 | 0.845 | 0.715 |

## False-alarm cross-check on null data

| filter/norm | null_block1 (/frame) | null_block20 (/frame) | null_phase (/frame) | null_quiet_crop (/frame) |
|---|---|---|---|---|
| knee+mp|per_elev | 33.88 | 33.98 | 102.19 | 7.30 |
| knee+mp|per_elev_zband | 33.95 | 33.98 | 92.28 | 5.24 |
| knee+mp|spatial_tgc | 33.87 | 33.92 | 83.16 | 5.73 |
| knee_spatial+mp|per_elev | 33.99 | 34.00 | 66.69 | 8.79 |
| knee_spatial+mp|per_elev_zband | 34.00 | 34.00 | 69.72 | 5.84 |
| knee_spatial+mp|spatial_tgc | 33.99 | 34.00 | 64.58 | 8.93 |
| knee_spatial|per_elev | 34.00 | 34.00 | 66.70 | 8.76 |
| knee_spatial|per_elev_zband | 34.00 | 34.00 | 69.83 | 5.84 |
| knee_spatial|spatial_tgc | 34.00 | 34.00 | 64.59 | 8.91 |
| knee|per_elev | 34.01 | 33.99 | 108.42 | 7.42 |
| knee|per_elev_zband | 34.01 | 34.02 | 94.44 | 5.51 |
| knee|spatial_tgc | 34.00 | 34.00 | 87.40 | 6.15 |
| rank24+mp|per_elev | 33.88 | 33.98 | 102.19 | 6.96 |
| rank24+mp|per_elev_zband | 33.95 | 33.98 | 92.28 | 4.50 |
| rank24+mp|spatial_tgc | 33.87 | 33.92 | 83.16 | 5.44 |
| rank24|per_elev | 34.01 | 33.99 | 108.42 | 7.15 |
| rank24|per_elev_zband | 34.01 | 34.02 | 94.44 | 4.92 |
| rank24|spatial_tgc | 34.00 | 34.00 | 87.40 | 5.82 |

What each null preserves and breaks:

- **block_shuffle**: Contiguous blocks of frames permuted. **MEASURED VERDICT: not a usable null for a per-frame detector -- do not report it as one.** Permuting frames leaves every frame's contents, including every real bubble, exactly intact; it removes trajectories, not echoes. A detector that works one frame at a time therefore sees the same material, and in the J3 sweep both block=1 and block=20 returned the matched target density (34.00/frame) for all 18 operating points -- i.e. zero information. Its only real effect is on the SVD, which loses the temporal coherence it uses to identify tissue; that shifts WHICH modes are removed but not how much signal is present. Keep it as the null for TRACKER-level claims, where destroying trajectories is exactly the right intervention, and use phase_surrogate or quiet_crop for detector-level false alarms.

- **phase_surrogate**: Per-voxel temporal Fourier surrogate: each voxel's time series keeps its power spectrum exactly and gets an INDEPENDENT random phase per bin. PRESERVES: the per-voxel temporal spectrum (so tissue still looks slow and noise still looks fast -- the axis the clutter filter cuts on) and the marginal amplitude scale. BREAKS: inter-voxel coherence, and the mechanism is exact rather than hand-wavy -- the randomized array is still supported on the same frequency bins, so the clutter's rank goes from 'number of spatial modes' (a few) to 'bandwidth in bins' (tens). A fixed-rank cut therefore removes much less of it, leaving tissue residue and OVER-stating false alarms. Two corollaries worth stating: a monochromatic clutter mode would survive with rank 1 intact (so this null is only hard on realistically broadband clutter), and a single GLOBAL per-bin phase would leave the Gram spectrum exactly unchanged while failing to remove bubbles -- which is why the draw is per-voxel. Complementary to block_shuffle: hard on the SPATIAL low-rank assumption where block_shuffle is hard on the TEMPORAL one.

- **quiet_crop**: A spatial crop of the real volume chosen where the baseline detector finds the fewest suprathreshold voxels. PRESERVES: everything -- it is real data, with real clutter, real noise and the real PSF. BREAKS: nothing, which is the problem: it is not guaranteed bubble-free, so any detection counted here may be a real bubble. It therefore gives a CONSERVATIVE (upper-bound) false-alarm rate and is the closest thing to an assumption-free null we have. Treat disagreement between quiet_crop and the synthetic nulls as the honest error bar on the false-alarm axis.

- **clean**: Not a null: the un-injected real volume. Used for MATCHED DETECTION DENSITY, the primary operating-point match. It makes no assumption at all, which is why it, and not a synthetic null, is the primary axis; the nulls are the cross-check.

