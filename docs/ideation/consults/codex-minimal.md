**(1) DIAGNOSIS**

I would not assume the blog figure is reproducible from the public artifacts as released.

Ranked likely reasons:

1. **The blog renderer/asset is not the released track viewer.**  
   The screenshot has `s1/s2`, zoom/fullscreen controls, a 10 mm scale bar, and a fixed `0–38 mm/s` colorbar. The released `track_viewer/index.html` has none of that; it is a “Brane” Three.js point viewer with a tiny GUI. It also defaults to `useIntensity: true`, which colors by sampled B-mode intensity, not velocity.  
   Inspect: blog network assets / exact `tracks.bin` or JSON loaded by the blog; compare against released `track_viewer_export.py` binary format and viewer JS.

2. **The public data may not be the data used for the figure.**  
   The released file is sanitized neutral IQ with a single physical receive row and synthesized elevation. That can plausibly produce discrete elevation-plane bands or repeated/fuzzy structures, especially if the reference used internal multi-row data, a different beamformed H5, or postprocessed tracks.  
   Inspect: H5 `/config` attrs, `iq_frames.shape`, `row_index`, `tx_delays_elev`, `grid_y`, per-elevation detection histograms, and whether detections concentrate on the 25 synthetic planes.

3. **Exact acquisition/scene selection is missing.**  
   The blog has `s1/s2`; the repo has no scene definitions. README quick start processes one acquisition unless `--all-acqs` is added, while the target sounds like a curated aggregate. Different subsets of 223 acquisitions could radically change continuity and apparent vessel density.  
   Inspect: `selected_acq_ids` in the track pickle, track/detection counts per acquisition, and whether any subset matches the screenshot footprint.

4. **Tracking may be incorrectly linked across acquisition boundaries.**  
   The default `run` path tracks the selected acquisitions as one continuous frame sequence, not per-acquisition isolated tracks. If acquisitions are independent bursts, cross-boundary links can create bogus long tracks or fuzzy aggregates.  
   Inspect: tracks spanning frames near acquisition boundaries; compare with `--output-per-acq` followed by combine.

5. **Hidden production parameters likely matter.**  
   The repo exposes many sensitive knobs: adaptive SVD cutoff, knee filtering, sigma threshold, min distance, max gap, max cost, smoothing method, min display length, spatial TGC, elevation planes, grid coarseness. The README says adaptive SVD “matches the production reference,” but does not provide the exact figure recipe.  
   Inspect: distributions of detections/acq, tracks/acq, length, speed, z-score knee, and speed cap. The README target of ~260 tracks/acquisition is a useful sanity check.

6. **Velocity/color scaling is ambiguous.**  
   The released `track_viewer_export.py` stores speed in `mm/frame`; the viewer hard-codes `maxSpeed: 0.1713`, which is approximately `38 / 222`, but the released UI does not label that as `38 mm/s`. The general exporter elsewhere stores `mm/s` in a different binary version.  
   Inspect: which exporter produced the blog asset; verify whether color uses speed or intensity.

**Single cheapest decisive experiment:**  
Using an existing faithful public-pipeline track pickle, make a read-only diagnostic projection: collapse/ignore `y`, render only `x,z` tracks colored by speed in `mm/s` capped at 38, with the same acquisition subset. If the vessels become thin and blog-like, the gap is elevation reconstruction / 3D rendering / sanitized single-row data. If they remain fuzzy, the gap is upstream filtering, localization, tracking, or a different dataset.

**(2) DRAFT GITHUB ISSUE**

Title: Exact reproduction recipe for ultrasound-brain ULM hero figure

Hi Aleph Neuro team,

I am trying to exactly reproduce the ULM brain hero figure from the ultrasound-brain blog post using the public `braindump` repo and `sanitized_neutral_ultratrace.h5`. A faithful run of the released pipeline produces a fuzzy, horizontally banded point cloud rather than the thin continuous velocity-colored vessels in the blog figure.

Could you clarify the minimal missing reproduction details?

1. Was the blog hero figure generated from the public `sanitized_neutral_ultratrace.h5` and the released `braindump` pipeline? If not, what dataset or precomputed artifact was used?

2. What exact acquisition IDs correspond to the blog `s1` and `s2` scenes? Were all 223 acquisitions used, a subset, or a curated/registered composite?

3. What exact command/config produced the tracks shown in the figure, including beamforming params, SVD params, detection params, tracking params, smoothing method, min track length, and whether tracking was run per-acquisition or across acquisition boundaries?

4. Does the public sanitized single-row dataset with synthesized elevation contain enough information to reproduce the 3D vessel geometry in the blog figure, or did the figure use internal multi-row / differently beamformed / additionally corrected data?

5. Can you publish the exact viewer artifact used by the blog figure: `tracks.pkl` / `tracks_smoothed.pkl` or `tracks.bin`, plus the HTML/JS/settings for the `s1/s2` switcher, 10 mm scale bar, camera, point size, and `0–38 mm/s` colorbar?

Without those details, the released repo appears to reproduce a related demo pipeline, but not the exact blog figure.
