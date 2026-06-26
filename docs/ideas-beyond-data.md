# Ideas beyond our current data — GENERATE FREELY, DO NOT ACT

Rome: "creating a separate doc with ideas beyond that I encourage, but acting (generating is
encouraged) on those insights is out of scope for now."

This is an idea garden. Everything here needs data, hardware, or acquisition control we **do not
have** with the single sanitized 4-minute scan. Nothing here is beaded for action. When one of
these matures into something doable within our data, *then* it gets a bead.

## Acquisition / hardware (need raw device control — we only have sanitized neutral IQ)

- **Access raw ADC / vendor frames** (the release excludes them). Would enable true RF-domain
  processing, harmonic/sub-harmonic bubble-specific imaging, and coded excitation.
- **Multi-pulse amplitude/phase modulation (CPS/AM/PI)** for contrast-specific detection — needs
  control of the transmit sequence; our data is already demodulated and compounded.
- **Higher frame rate / more transmit angles** to raise reflection-matrix rank for UMI (`SKULL`).
- **Aberration-correcting transmit** (closed-loop adaptive transmit) rather than receive-only
  correction — needs to drive the probe live.

## Algorithms that need ground truth or more subjects

- **Supervised localization / denoising nets** trained on simulated or reference-labeled data.
- **Cross-subject vascular atlas** + registration to a standard brain space.
- **Self-supervised representation learning** over many scans for the telepathy decoder.

## Telepathy / functional decoding (downstream of everything)

- Map the flow-matching **residual/surprise field** (`flow-matching.md`) to task labels or stimuli —
  requires synchronized behavioral/stimulus data we don't have here.
- Functional ULM (fULM): correlate flow modulation with events; needs event timing.
- Closed-loop neurofeedback from the surprise map.

## Physics / reconstruction stretch

- **Full-waveform inversion** for a true volumetric SoS map (very expensive; needs raw data).
- **Wave-equation / matrix migration** beamforming instead of delay-and-sum.
- **Differentiable beamformer** so localization/aberration are learned end-to-end from a loss —
  conceptually the ultimate "joint inverse problem," far beyond current scope.

## House rule

If an idea here becomes feasible *with the data we have*, move it to a bead under `mb-crr` (or its
successor epic) and delete it from this list. Otherwise it lives here as inspiration, not a task.
