**(1) CRITIQUE of our diagnosis**

Your diagnosis is directionally plausible, but it is overfit and too confident.

The strongest confirmed facts are: the shipped viewer is a point renderer, not a line renderer; `THREE.Points` is used directly, with Jet default and `maxSpeed: 0.1713` in [index.html](/home/rome/braindump_pristine/ultratrace_ulm/web/track_viewer/index.html:277) and [index.html](/home/rome/braindump_pristine/ultratrace_ulm/web/track_viewer/index.html:367). The default `track-viewer --min-length 35` is also real in [cli.py](/home/rome/braindump_pristine/ultratrace_ulm/cli.py:379), and export hard-fails if no tracks pass it in [track_viewer_export.py](/home/rome/braindump_pristine/ultratrace_ulm/track_viewer_export.py:159). So min-length is not cosmetic; it is a reproducibility blocker.

The weakest claim is “centroid localization is the remaining root cause.” The code does default to centroid, and the implementation is just a center-of-mass over a local window in [tracking.py](/home/rome/braindump_pristine/ultratrace_ulm/tracking.py:97). But your own doc 14 says measured perpendicular jitter was only 0.49 px / 0.072 mm and initially “ruled out” localization precision. Also, pristine CLI advertises `--subpixel gaussian_fit` in [cli.py](/home/rome/braindump_pristine/ultratrace_ulm/cli.py:89), but the pristine localizer only special-cases `"parabolic"`; every other method falls through to centroid in [tracking.py](/home/rome/braindump_pristine/ultratrace_ulm/tracking.py:86). So “use Gaussian fit” may be an unshipped feature, not merely an unused flag.

The “faithfully reproduce their code” claim is mostly true for defaults, but incomplete. The fork’s beamform streaming option is documented as numerically identical in [beamform_core.py](/home/rome/gt/microbubbles/crew/cajal/ultratrace_ulm/beamform_core.py:280), while shipped `beamform_mach` still stores every compound in `compounds` before TGC/write in [beamform_mach.py](/home/rome/braindump_pristine/ultratrace_ulm/beamform_mach.py:119). Tracking knobs are default-neutral, but SVD is not: pristine CPU adaptive uses a complex64 covariance in [svd.py](/home/rome/braindump_pristine/ultratrace_ulm/svd.py:71), whereas the fork’s patched path accumulates complex128 Gram matrices in [svd.py](/home/rome/gt/microbubbles/crew/cajal/ultratrace_ulm/svd.py:16) and the GPU path does the same in [gpu_svd.py](/home/rome/gt/microbubbles/crew/cajal/ultratrace_ulm/gpu_svd.py:85). That is a real algorithmic fork, not just hardware trivia.

The “tracks must be much longer” inference is probably wrong. It came from dividing README density by detections. But doc 17 later reports pristine tracks are all 5-6 frames and that only 0.1% of `base223_best` are >=35 frames. The reference image also visually looks like accumulated short point samples, not continuous spline tracks. Ask about min-length and filtering, not “why are our tracks short?”

The degraded-public-data hypothesis is possible but currently the least evidenced. Your own artifacts show major pipeline/render choices can swing the image from empty/sparse to haze. Before implying degraded data, first localize whether the mismatch is already present in raw detections, in tracking/smoothing, or only in viewer/export state.

**(2) CHEAPEST DECISIVE EXPERIMENT to localize the gap**

Do a stage-boundary render from an existing full-223 or 79-acq pickle, no re-beamforming and no new SVD:

Render the exact same masked population three ways with the same camera/Jet/maxSpeed/point size: raw `detections.positions_mm` before tracking, unsmoothed track positions, and `tracks_smoothed` positions. Also include a random downsample matched to the reference density.

Interpretation is clean:

If raw detections are already fuzzy/banded, the gap is upstream of tracking: SVD/clutter filtering, elevation-edge normalization, localizer, or input data.

If raw detections are crisp but unsmoothed/smoothed tracks are fuzzy, the gap is tracking, over-linking, smoothing, or length/speed filtering.

If random downsampling fixes the haze, density/thresholding is the main lever.

If none match after mask + density control, then the public pipeline lacks a production post-filter/localizer/mask, or the screenshot used a different derived artifact/data version.

This is cheaper and more decisive than completing pristine c64 on all 223: the 79-acq c64 render already fails topologically, and full c64 mostly tests density, not where the error enters.

**(3) TIGHTENED GITHUB ISSUE — rewrite our 5-question draft into the minimal exact set, ready to post**

Title: Exact reproduction details for the ultrasound-brain track-viewer render

Body:

We are trying to reproduce the published ultrasound-brain hero render from the public `sanitized_neutral_ultratrace.h5` and current `ultratrace-ulm` pipeline.

We can match the viewer style: point-flow track viewer, Jet, near-black background, `maxSpeed = 0.1713 mm/frame` / 38 mm/s at 222 Hz. But the public recipe does not reproduce the published vessel map for us. In particular, `track-viewer --min-length 35` would render essentially empty for our outputs, while lower min-length renders short accumulated point tracks. We also see strong near-field/elevation-edge artifacts and residual depth bands unless we mask them.

Could you clarify the exact production path for the screenshot?

1. Was the screenshot generated from the public `sanitized_neutral_ultratrace.h5` currently linked by the repo? If yes, can you provide the file checksum/version? If not, what input was used?

2. Can you share the exact generated `viewer/data/tracks.bin` or `tracks_smoothed.pkl` used for the screenshot? If not, please provide the exact command/config, including acquisition subset/all-acqs, spatial TGC, SVD method/precision, sigma threshold, knee filter, temporal sigma, subpixel method, min-track-length, max-gap/max-cost, export/viewer min-length, and any masks or post-filters.

3. Was any non-shipped processing used between public tracking output and the screenshot: near-field/skull masking, elevation-edge masking, speed/length gates, point-size/intensity settings, or an unshipped Gaussian/PSF localizer? The CLI accepts `--subpixel gaussian_fit`, but the public localizer appears to fall through to centroid unless `parabolic` is selected.

4. What frame rate/PRF was used for the 0-38 mm/s color scale and tracking-gate tuning? The H5 appears to include sampling/tx frequencies but not acquisition frame rate, and 222 Hz conflicts with the stated scan duration.
