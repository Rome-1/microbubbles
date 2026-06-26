# Skull-aware beamforming — context (bead `SKULL`, mb-crr.13)

Rome's framing: the device is a small **wearable**, so there is **no CT** to derive a
subject-specific speed-of-sound (SoS) map from. The two tractable, CT-free families are
**ultrasound-derived SoS maps** and **local phase-aberration estimation**. Rome asked
specifically what *"reflection matrix → ultrasound matrix imaging (UMI) correction"* and
*"guide-star correction using tissue targets"* would buy us. This doc answers that and ties
it to the data we actually have.

## Why the skull hurts ULM specifically

The baseline beamformer (`beamform_core.beamform_iq`) assumes a **homogeneous medium, one
speed of sound, geometric delays**. A skull adds spatially-varying attenuation, **phase
aberration**, reverberation, and multiple scattering. Because ULM localization precision is
set by the **point-spread function (PSF)**, an aberrated PSF does two bad things at once:

1. **Biases bubble positions** — the focused spot shifts and distorts, so sub-voxel
   localization is systematically wrong (and wrong *differently* across the field of view).
2. **Destroys detections** — a smeared, dimmer PSF drops below the z-score threshold, so the
   low-SNR bubbles (often the deep / interesting ones) are simply never detected.

So skull correction is not a cosmetic gain: it changes *which* bubbles enter the pipeline at
all, and where they're placed. That's why it's flagged as the largest fidelity lever — and why
it's also the highest risk and sequenced after the cheaper correctness/PSF/motion work.

The single most important property: **the aberration law is spatially local.** There is no one
"skull delay correction." Each region of the brain is seen through a different patch of skull,
so the correction must vary across the field of view (the *isoplanatic patch* is small).

---

## 1. Reflection matrix → ultrasound matrix imaging (UMI)

**The object.** Record, instead of a single beamformed image, the full **reflection matrix
`R`**: the response measured at every receive element (output basis) for every transmit event
(input basis). With plane-wave transmits, the input basis is *transmit angle*; the output basis
is *receive element* (or a focused/virtual basis you synthesize afterward). `R` contains far
more than any one image — it's every input→output Green's-function sample the probe can see.

**The trick (focused reflection matrix + distortion matrix).** Numerically focus `R` at both
input and output to get the **focused reflection matrix** `R_focused(r_in, r_out)` — the signal
that went in focused at `r_in` and came out focused at `r_out`. In a homogeneous medium this is
diagonal (energy comes back where you sent it). The skull spreads it off-diagonal. You then form
the **distortion matrix** `D` by removing the *expected geometric* wavefront from `R_focused`,
leaving only the part due to aberration. Within an **isoplanatic patch** (a region where the
aberration is ~constant), an **SVD of `D`** concentrates the aberration into its first singular
vector: that singular vector *is* the estimated aberration transmittance (the phase screen) for
that patch. Apply its conjugate as a correction, re-focus, and the PSF tightens and brightens.
Iterate per patch, tile the patches, and you've restored resolution and contrast — **estimated
purely from the recorded backscatter, no CT, no contrast agent, no guide star.**

**What it buys us.** This is the technique behind the 3D ultrasound-matrix-imaging head-phantom
results (Nature Comms 2023, the paper Rome cited). It is the most principled CT-free transcranial
aberration corrector because it (a) estimates a *spatially-local* law, exactly what the skull
needs, and (b) needs nothing but the data the probe already records.

**Does our data support it?** *Partially, and this is the key feasibility point.* Our neutral
IQ is `(frames, transmit_angles, probe_rows, probe_cols, time_samples)` — i.e. **angle-resolved
input × element-resolved output**, which is precisely the raw material of a reflection matrix.
We don't have arbitrary focused transmits, but plane-wave (angle) input + per-element receive is
a standard, sufficient `R` parameterization. The honest caveats: (i) the number of transmit
angles bounds the input-side rank — few angles → coarse aberration estimate; (ii) the matrix is
*per-acquisition* and tissue/bubbles move, so `R` must be built within a short temporal window;
(iii) 3D isoplanatic-patch UMI is compute-heavy (per-patch SVDs over many voxels). **First step
for the bead:** read the actual `tx_angles` / `rows` / `cols` shapes from `/config` to size the
achievable input/output rank before committing — this decides whether full UMI is in reach or
whether we fall back to coherence-based aberration estimation (below).

## 2. Guide-star correction — and why "tissue targets" matter for us

**Adaptive-optics analogy.** A *guide star* is a known point source you measure the wavefront
distortion against: a point should produce a spherical wave at the aperture; any deviation is
the aberration, and you conjugate it to correct. In ultrasound, a **bright isolated scatterer**
plays that role.

**Why not bubbles for us.** The natural ULM guide star is a strong, isolated microbubble. Rome's
point is right: in this wearable / low-concentration transcranial regime we **won't reliably get
bright isolated bubbles** through the skull — the very aberration we're trying to fix dims and
smears them, and concentration is low. So bubble guide-stars are largely unavailable here.

**Tissue targets = the way in.** Two CT-free, contrast-free options use *tissue* itself:

- **Specular/anatomical reflectors as guide stars.** Strong, persistent tissue interfaces
  (sulci boundaries, vessel walls, ventricle edges) are bright and *stationary*, unlike bubbles.
  Treat such a reflector as a virtual point source: the received wavefront from it, compared to
  the geometric expectation, gives the local aberration law for its neighborhood. Persistence is
  the advantage — you can average over many frames to beat SNR, which you can't do with a
  transient bubble.

- **Spatial-coherence / "virtual guide star" from speckle (CLASS-like).** You don't even need a
  discrete bright target. The **spatial coherence of the backscattered tissue speckle** carries
  the aberration: the van Cittert–Zernike relationship says the across-aperture coherence of
  echoes from a diffuse (random) scatterer field is degraded in a way that encodes the phase
  screen. Coherence-factor / coherence-based adaptive focusing and CLASS-type methods synthesize
  a **virtual guide star** from the speckle and iterate the correction from it. This is
  essentially the *physically-motivated cousin* of the UMI distortion-matrix SVD: both extract a
  local aberration law from ordinary tissue backscatter; UMI does it in the matrix/SVD formalism,
  coherence methods do it in the spatial-coherence formalism. **This is the most robust fallback
  when the reflection-matrix rank is too low for full UMI.**

**What "tissue targets" buys us specifically:** a *stationary, repeatable, always-present*
reference to estimate the local phase screen — exactly when bubbles are too dim/sparse to serve
as guide stars. The catch: tissue reflectors are extended, not ideal points, so the estimate is
noisier than an ideal guide star, and it still must be done **per isoplanatic patch**.

---

## How these relate (one mental model)

All four CT-free options estimate **a spatially-local aberration / SoS correction from data we
already have**, differing in *what reference* they lean on:

| Method | Reference used | Needs bubbles? | Locality | Our-data feasibility |
|---|---|---|---|---|
| US-derived SoS map | bulk delay/coherence optimization | no | global→regional | High (cheapest first step) |
| Local phase-aberration est. | per-patch wavefront fit | no | per-patch | High |
| Reflection-matrix / UMI | full input×output matrix + distortion-matrix SVD | no | per-patch (isoplanatic) | Medium — bounded by #tx-angles; check `/config` first |
| Guide-star (tissue target) | bright stationary tissue reflector or speckle coherence | **no** (that's the point) | per-patch | Medium-High — robust fallback when UMI rank too low |

## Concrete plan for the bead (ordered, cheapest-first)

1. **Read `/config` + IQ shapes** to size achievable input (tx-angle) and output (element) rank.
   This decides UMI feasibility before any compute is spent.
2. **Global/regional US SoS estimate** — sweep `speed_of_sound_m_s` (and a coarse regional map),
   pick the value(s) maximizing a focusing-quality metric (coherence factor / sharpness). Cheap,
   immediate, and already an improvement over the single hardcoded SoS. **Diff vs baseline here.**
3. **Coherence-based local aberration (virtual guide star from speckle)** per isoplanatic patch —
   the robust tissue-target path. Re-beamform with per-patch correction; diff vs baseline.
4. **Reflection-matrix / UMI distortion-matrix SVD** — only if step 1 says the angle/element rank
   supports it. Highest fidelity, highest cost; validate on a single acquisition first.

Each step is its own 3D volume + diff. We expect the SoS sweep and coherence method to land
first; UMI is the stretch goal conditioned on the acquisition's angular diversity.

> Sources Rome cited for grounding: 3D ultrasound matrix imaging (Nature Communications, 2023,
> `s41467-023-42338-8`). Specific numerical claims here (rank bounds, expected gains) are our
> working estimates and must be checked against the real `/config` and a single-acq trial before
> we rely on them.
