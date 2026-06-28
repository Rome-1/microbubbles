## Critique Of Opus Ideas

- **AGREE: Short tracks are primarily an association failure.** Greedy frame-to-frame Hungarian plus `max_gap=3` is exactly the kind of setup that fragments tracks under missed detections, crossing vessels, and variable bubble intensity. Build tracklet stitching before spending heavily on detection.

- **AGREE: Add post-hoc tracklet stitching.** This is cheap, testable on 60 acquisitions, and directly targets median/mean track length, long-track fraction, and usable velocity estimates.

- **AGREE: Replace purely greedy association with motion-aware linking.** Constant-velocity or Kalman-gated costs should reduce identity breaks and reconnect plausible gaps without requiring ground truth.

- **AGREE: Tune `max_gap`, gating radii, and link costs systematically.** This is a high-value bake-off because it changes only association policy and can be scored with existing track metrics.

- **AGREE: Elevation from 25 synthetic planes derived from one physical row is untrustworthy.** Treating this as reliable 3D localization risks baking in a bright-midplane prior. Use elevation cautiously unless the metric proves it improves track consistency.

- **AGREE: Bright-midplane bias is a real risk.** If the elevation estimator collapses detections toward the center plane, apparent 3D tracks may look smoother while being physically wrong.

- **AGREE: Validate elevation using self-consistency, not visual plausibility.** Test whether z improves continuity, velocity smoothness, bidirectional stability, and vessel-map sharpness; otherwise prefer 2D/lateral association with conservative z use.

- **AGREE: Detection thresholding/intensity normalization still matters.** Association may be the largest failure, but missed detections create gaps that association must bridge. Detection tuning remains useful if judged by downstream track metrics.

- **DISAGREE: Deprioritize detection too far.** If the detector drops dim bubbles or splits bright bubbles, stitching can only hide the problem. Detection and association should be baked off independently.

- **AGREE: Use dataset-level ablations instead of ground-truth-dependent claims.** With no labels, compare track length, gap rate, velocity smoothness, vessel-map concentration, bidirectional consistency, and acquisition-to-acquisition stability.

## Unified Ranked Top-10 To Build

1. **CONSENSUS — Post-hoc tracklet stitching.**  
   Build a second-pass linker over terminated tracklets using temporal gap, position, velocity, and intensity continuity.  
   **Moves:** median track length, long-track fraction, fragmentation rate.

2. **CONSENSUS — Motion-aware association cost.**  
   Replace or augment greedy frame-to-frame costs with constant-velocity/Kalman prediction and Mahalanobis-style gating.  
   **Moves:** track length, velocity smoothness, ID-break proxy.

3. **CONSENSUS — Association parameter sweep.**  
   Bake off `max_gap`, spatial gate, velocity gate, cost weights, and minimum-track-length thresholds over 60 acquisitions.  
   **Moves:** Pareto frontier of track count vs track length vs smoothness.

4. **CONSENSUS — Conservative 2D-first / z-optional linker.**  
   Run association primarily in reliable lateral coordinates, then compare z-aware linking only where elevation passes self-consistency tests.  
   **Moves:** track stability, false 3D fragmentation, vessel-map sharpness.

5. **CONSENSUS — Elevation bias audit.**  
   Quantify z-plane occupancy, midplane attraction, z-jitter, z-velocity plausibility, and whether z improves or harms linking metrics.  
   **Moves:** elevation trust score, z-aware association decision.

6. **CONSENSUS — Detection-threshold downstream sweep.**  
   Sweep detection thresholds, blob size filters, and local contrast normalization, scored only by downstream track metrics.  
   **Moves:** detection continuity, gap rate, usable track count.

7. **CONSENSUS — Gap-aware interpolation with flags.**  
   Allow short missing spans to be bridged for tracking while marking interpolated detections separately from real detections.  
   **Moves:** usable track length, velocity continuity, gap statistics.

8. **CONTESTED — Bidirectional tracking consistency check.**  
   Track forward and backward, then score agreement of links and stitched tracklets as a no-ground-truth reliability proxy.  
   **Moves:** association confidence, ID-break proxy, parameter selection.

9. **CONTESTED — Vessel-map sharpness bake-off metric.**  
   Render tracks into vascular density maps and score concentration/sharpness under matched detection counts.  
   **Moves:** practical ULM image quality, over-linking detection.

10. **CONTESTED — Intensity-aware link scoring.**  
   Add bubble intensity/shape continuity as a weak association feature, downweighted relative to motion.  
   **Moves:** crossing robustness, false-link rate proxy, track continuity.