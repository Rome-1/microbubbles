1. **Overlap-and-feather region SVD**
   - **Stage:** SVD / clutter filtering.
   - **Implementation:** Run region SVD on overlapping 3D tiles, discard or downweight tile margins, then blend filtered volumes with smooth spatial weights. Add explicit tile-boundary diagnostics.
   - **Metric:** `coverage`, `track length`, `fragmentation`. Region-SVD coverage loss is not acceptable if it creates spatial holes; it is likely recoverable if boundary/rank artifacts are the cause.
   - **Validate without GT:** Compare detection density and track survival near tile interiors vs tile borders; coverage should stop showing grid-like dropouts while track velocity distributions stay stable.
   - **A10G risk:** Low-Medium. More compute, but memory is manageable if tiles stream sequentially.
   - **Effort:** M.

2. **Adaptive SVD rank per region instead of fixed cutoffs**
   - **Stage:** SVD.
   - **Implementation:** Estimate low-rank clutter cutoff per tile using singular-value elbow, temporal autocorrelation, Marchenko-Pastur-like noise floor, or retained-energy bounds. Clamp ranks to sane min/max.
   - **Metric:** `coverage`, `track continuity`. Over-aggressive rank removal likely deletes weak bubbles and creates intermittent detections.
   - **Validate without GT:** Sweep rank policy and plot coverage, median track length, detections/frame, and velocity smoothness. Pick the knee before velocity noise rises.
   - **A10G risk:** Low. Mostly metadata from existing SVD spectra.
   - **Effort:** S-M.

3. **Track-aware detection threshold sweep**
   - **Stage:** Detection.
   - **Implementation:** Lower detection threshold modestly, but evaluate by downstream tracks, not raw detections. Keep candidate confidence scores and let tracking reject isolated false positives.
   - **Metric:** `track length`, `fragmentation`, `coverage`. The short-track problem is probably detection-dominated if bubbles disappear for single frames before tracking can link them.
   - **Validate without GT:** Plot median/90th percentile track length, fraction of one-frame tracks, accepted-link rate, and velocity outlier rate across thresholds.
   - **A10G risk:** Low-Medium. More detections increase linker cost.
   - **Effort:** S.

4. **Local CFAR / MAD-based adaptive detection**
   - **Stage:** Detection.
   - **Implementation:** Replace global intensity threshold with local noise-normalized threshold by depth/region, e.g. local median + `k*MAD`, optionally with per-depth normalization after beamforming/SVD.
   - **Metric:** `coverage`, `track length`. Recovers weak/deep/elevation-edge bubbles without flooding bright regions.
   - **Validate without GT:** Detection density should become less depth-biased; tracks gained should have plausible velocities and multi-frame persistence.
   - **A10G risk:** Low. Local reductions/convolutions are cheap.
   - **Effort:** M.

5. **Gap-closing tracker with bounded missed frames**
   - **Stage:** Tracking.
   - **Implementation:** Allow links across 1-3 missing frames using predicted position from last velocity/Kalman state; penalize gaps but do not terminate immediately.
   - **Metric:** `track length`, `fragmentation`, `continuity`. This directly targets intermittent detection dropout.
   - **Validate without GT:** Track length should increase while acceleration and velocity distributions remain physiologic; inspect gap frequency and gap length histograms.
   - **A10G risk:** Low. Mostly CPU/GPU assignment complexity; candidate pruning matters.
   - **Effort:** M.

6. **Motion-model gating instead of distance-only linking**
   - **Stage:** Tracking.
   - **Implementation:** Use constant-velocity Kalman prediction with anisotropic covariance reflecting axial/lateral/elevation localization uncertainty. Gate by Mahalanobis distance, not raw Euclidean distance.
   - **Metric:** `track length`, `ID continuity`, `fragmentation`. Helps preserve tracks through dense regions without over-linking.
   - **Validate without GT:** Lower abrupt direction reversals and acceleration spikes; longer tracks at same false-link proxy rate.
   - **A10G risk:** Low.
   - **Effort:** M.

7. **Two-pass association: conservative first, recovery second**
   - **Stage:** Tracking.
   - **Implementation:** First link high-confidence detections with tight gates. Then run a second pass to attach low-confidence detections or bridge short fragments using relaxed gates and velocity consistency.
   - **Metric:** `track length`, `fragmentation`, `coverage`.
   - **Validate without GT:** Report how many fragments are merged, velocity discontinuity at merges, and whether merged tracks remain spatially vessel-consistent.
   - **A10G risk:** Low-Medium depending detection count.
   - **Effort:** M.

8. **PSF-aware 3D localization fit**
   - **Stage:** Detection/localization.
   - **Implementation:** Fit anisotropic 3D Gaussian or matched PSF around each local maximum instead of using voxel peak/centroid only. Use confidence from residual/SNR.
   - **Metric:** `track continuity`, `velocity smoothness`, `elevation stability`. Better localization reduces tracking gate failures.
   - **Validate without GT:** Track jitter perpendicular to flow should drop; repeated localizations should show narrower residual distribution.
   - **A10G risk:** Medium if many candidates; batch small ROIs on GPU.
   - **Effort:** M-L.

9. **Treat elevation as low-confidence unless independently validated**
   - **Stage:** Beamforming / localization / tracking.
   - **Implementation:** Inflate elevation uncertainty in tracker gates and avoid hard elevation-based rejection. Report 3D tracks plus 2D-projected metrics separately.
   - **Metric:** `continuity`, `false fragmentation`. Elevation synthesized from one physical row is not fully trustworthy for quantitative 3D ULM; it can help disambiguate, but should not dominate linking.
   - **Validate without GT:** Compare projected 2D track stability vs full 3D stability; check whether elevation velocities are noisy, biased, or unphysiologic.
   - **A10G risk:** Low.
   - **Effort:** S.

10. **Depth/elevation PSF calibration from isolated bubbles**
   - **Stage:** Beamforming / detection.
   - **Implementation:** Automatically mine high-SNR isolated bubble events and estimate empirical PSF widths by depth and elevation. Feed widths into detection NMS, localization fit, and tracker covariance.
   - **Metric:** `track length`, `continuity`, `localization stability`.
   - **Validate without GT:** PSF residuals should tighten; NMS duplicates should fall without reducing multi-frame tracks.
   - **A10G risk:** Low.
   - **Effort:** M.

11. **Connected low-confidence candidate retention**
   - **Stage:** Detection/tracking interface.
   - **Implementation:** Keep two detection sets: high-confidence detections for track starts, low-confidence detections only eligible to continue existing tracks.
   - **Metric:** `track length`, `fragmentation`, while protecting false-positive starts.
   - **Validate without GT:** One-frame tracks should not explode; existing-track continuation rate should improve.
   - **A10G risk:** Low-Medium due to more candidates.
   - **Effort:** S-M.

12. **Explicit duplicate/merge handling in dense bubble frames**
   - **Stage:** Detection/tracking.
   - **Implementation:** Add NMS radius tied to empirical PSF and reject implausibly close duplicate detections unless separable by fit residual. In tracking, prevent one detection from spawning competing fragments.
   - **Metric:** `fragmentation`, `track purity`, `length`.
   - **Validate without GT:** Reduce near-identical parallel short tracks; track density should remain stable after NMS tuning.
   - **A10G risk:** Low.
   - **Effort:** S-M.

13. **Region-SVD coverage recovery audit before further 3D claims**
   - **Stage:** Evaluation.
   - **Implementation:** Add per-stage maps: beamformed energy, post-SVD energy, detections, accepted tracks, rejected detections, tile boundaries. Quantify loss from each step.
   - **Metric:** Indirectly moves `coverage` and `track length` by identifying whether loss comes from SVD, detection, or tracking.
   - **Validate without GT:** If detections vanish before tracking, root cause is SVD/detection; if detections exist but tracks stay short, root cause is linker/gating.
   - **A10G risk:** Low, but logging volumes can use disk.
   - **Effort:** S.

14. **Temporal batching with overlap**
   - **Stage:** SVD / detection / tracking.
   - **Implementation:** Process temporal chunks with overlapping frame margins, then drop margin outputs or reconcile duplicate tracks across chunk seams.
   - **Metric:** `track length`, `fragmentation`. Prevents artificial track breaks at processing chunk boundaries.
   - **Validate without GT:** Track termination rate should not spike near chunk edges.
   - **A10G risk:** Medium. Temporal overlap increases memory/compute but is controllable.
   - **Effort:** M.

15. **Use vessel-consistency priors only after initial unbiased tracking**
   - **Stage:** Post-tracking / refinement.
   - **Implementation:** Build a coarse vessel probability map from long reliable tracks, then use it only to rescue short fragments aligned with established paths, not to suppress novel coverage.
   - **Metric:** `coverage`, `track length`, `fragmentation`.
   - **Validate without GT:** Rescued fragments should align with stable flow directions and improve spatial continuity without erasing low-density branches.
   - **A10G risk:** Low.
   - **Effort:** L.