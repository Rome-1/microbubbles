## 1. Why Ours Is Worse

1. **Not uncapped, but not equivalent: elevation gate is widened 3x.**  
   The Kalman tracker has a hard box gate: `diffs <= max_dist_mm` and `gaps <= max_gap` in [tracking.py](/home/rome/gt/microbubbles/crew/cajal/ultratrace_ulm/tracking.py:436). Default gate derives from spacing as `2*dx, 2*dy, 2*dz`, matching reference-ish `(0.4008, 1.1093, 0.4016)` [tracking.py](/home/rome/gt/microbubbles/crew/cajal/ultratrace_ulm/tracking.py:223). But current `g3r10` multiplies elevation via `elev_gate_factor` [tracking.py](/home/rome/gt/microbubbles/crew/cajal/ultratrace_ulm/tracking.py:295). With factor 3, y cap becomes `~3.33 mm`, giving max one-frame displacement `sqrt(.40^2+3.33^2+.40^2)=~3.38 mm/frame`, exactly explaining observed `3.35 mm/frame` / `~74 cm/s`. Reference cap is fixed `(0.4008, 1.1093, 0.4016)` per grounding line 53.

2. **`max_gap=6` worsens false continuity, but is secondary.**  
   Reference uses `max_gap 3`; current composite uses `6` per grounding lines 68-70. In the Kalman path the gate is **not multiplied by gap** [tracking.py](/home/rome/gt/microbubbles/crew/cajal/ultratrace_ulm/tracking.py:437), so this does not permit farther spatial jumps, but it keeps stale tracks alive twice as long, increasing accidental relinks and short false tracks.

3. **We likely process ~3x too many frames/opportunities.**  
   Reference scope is `216 acqs * 240 frames = 51,840` frames, grounding lines 44-46. Our code uses `compound.shape[0]` as frames-per-acq with no clamp to 240 [tracking.py](/home/rome/gt/microbubbles/crew/cajal/ultratrace_ulm/tracking.py:925). Grounding flags our assumption around `~700` frames/acq and `223` acqs, lines 87-94. That alone explains much of “5x” over-production.

4. **Region-SVD differs from reference global adaptive SVD.**  
   Reference is global adaptive SVD + knee filter. Our region path partitions z/x blocks and filters local temporal subspaces [svd_region.py](/home/rome/gt/microbubbles/crew/cajal/ultratrace_ulm/svd_region.py:158), wired in Modal when `filter_variant == "region"` [scripts/modal/app.py](/home/rome/gt/microbubbles/crew/cajal/scripts/modal/app.py:755). This changes residual clutter statistics before identical-looking detection.

5. **Detector args mostly match; over-detection is upstream + spawn policy.**  
   Detector uses sigma `2.0`, NMS `min_distance`, smoothing, centroid/window from opts [tracking.py](/home/rome/gt/microbubbles/crew/cajal/ultratrace_ulm/tracking.py:820). Every unmatched detection spawns a new track [tracking.py](/home/rome/gt/microbubbles/crew/cajal/ultratrace_ulm/tracking.py:590), so extra frames/residual clutter convert directly into many short tracks.

## 2. Top 3 Fixes

1. **Restore reference physical gate.**  
   Set `elev_gate_factor=1.0`, `elev_meas_factor=1.0`, and explicitly pass:
   ```text
   max_dist=(0.4008, 1.1093, 0.4016)
   ```
   This should kill the `3.35 mm/frame` elevation teleports.

2. **Use reference gap and acquisition boundaries.**  
   Set:
   ```text
   max_gap=3
   output_per_acq=True
   ```
   or otherwise track each acquisition independently, then combine. Reference does this; our non-per-acq path tracks across concatenated frames [tracking.py](/home/rome/gt/microbubbles/crew/cajal/ultratrace_ulm/tracking.py:963).

3. **Run the reference SVD/detection scope first.**  
   Disable region-SVD for the parity run:
   ```text
   svd_method="adaptive"
   knee_filter=True
   svd_low_cutoff=0.1
   tissue_freq_hz=100
   sigma_threshold=2.0
   min_distance=2
   smoothing_sigma=1.0
   subpixel="centroid"
   window_size=5
   n_acqs=216
   frames_per_acq=240
   ```
   The code already supports these params; the missing part is enforcing the same frame/acq scope.

## 3. Beat It

Use the clean reference-like run above, then apply strict post-hoc stitching to reduce fragmentation without re-detecting. Existing stitcher links fragment ends to starts using gap-extrapolated position, velocity agreement, and intensity continuity [track_stitch.py](/home/rome/gt/microbubbles/crew/cajal/ultratrace_ulm/track_stitch.py:60). Sweep conservative params on the fixed dataset:

```text
max_gap=8..12
tol_lateral_mm=0.4..0.6
tol_elev_mm=1.0
vel_cos_min=0.8
intensity_log_tol=1.0..1.5
```

Validate by speed cap, longer-track yield, and odd/even split-half stability. This targets the reference’s main weakness: 50,456 tracks but median length only 7 and only ~2,300 rendered.
