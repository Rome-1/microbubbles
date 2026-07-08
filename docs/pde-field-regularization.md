# Graph-Laplacian along-vessel field regularization (mb-ki9)

**Result in one line.** Replacing the isotropic-Gaussian field regularizer in our
validated ULM tractography with an **anisotropic graph-Laplacian low-pass that
diffuses *along* vessels** raises split-half reproducibility from **Dice 0.61 /
direction-cosine 0.84 to Dice 0.71 / 0.95** (+17% / +13%) — a measured, seed-robust
improvement. A matched-diffusivity *isotropic* control collapses (Dice 0.07, cos ≈ 0),
proving the gain comes from anisotropy, not from more smoothing.

Author: microbubbles/crew (Gas Town). Date: 2026-07-08. Data: Aleph Neuro sanitized
ultratrace, acq-0 slice (216 acquisitions, odd/even split-half). Builds on
`docs/regularized-field-tractography.md` §8.1. Method: `scripts/wf_render_signal/tractography_pde.py`.

---

## 1. Motivation

The validated tractography result reconstructs vessels by integrating streamlines
through a *confidence-weighted-smoothed* microbubble-velocity field. The smoothing is
the essential lever (raw field → 17 mm streamlines; smoothed → ~50 mm), but the
baseline uses an **isotropic Gaussian**:

```
V_smooth = gaussian(V·C, σ) / gaussian(C, σ)          # σ = (2.5,1.4,2.5) voxels
```

An isotropic kernel smears flow **across** vessel walls as much as **along** them —
it bleeds neighbouring vessels into each other and blurs the field's fine transverse
structure to buy along-vessel coherence. The physically-right regularizer smooths
*along* the flow and *not* across it (`docs/regularized-field-tractography.md` §8.1).

---

## 2. Method

### 2.1 Anisotropic graph-Laplacian (along-vessel) smoothing

Keep the confidence-weighting that makes the baseline work (smooth `V·C` and `C`,
then divide), but replace the isotropic Gaussian with **anisotropic heat diffusion**
whose conductivity is steered by the local flow direction.

On the regular grid, the 6-neighbour along-axis edge diffusivity for axis *a* reduces
to a clean form:

```
D_a(voxel) = |d_a|^γ + ε_iso          d = mm-space unit flow direction, γ = 1.5
```

so a voxel whose flow runs along *x* (`d ≈ (1,0,0)`) diffuses strongly along *x* and
weakly across into elevation/*z*; `ε_iso = 0.2` is a small isotropic floor that keeps
coverage gaps fillable where the direction is unknown. Directions `d` are bootstrapped
from one mild Gaussian pass. The field is then evolved by explicit anisotropic heat
diffusion (graph Laplacian = divergence of `D·∇`, Neumann boundaries) for 200 steps:

```
(V·C), C  ←  (V·C), C  +  dt · Lₐₙᵢₛₒ(·)           dt = 0.12
V_reg = smooth(V·C) / smooth(C)
```

With **γ = 0** this is exactly isotropic diffusion (a Gaussian) — the control below.

### 2.2 Incompressibility reconstruction of the weak elevation velocity

Blood flow is ≈ incompressible: `∂ₓvₓ + ∂_y v_y + ∂_z v_z = 0`, where *y* = elevation
(the 2.77×-coarser, weakly-synthesized axis). Recover `v_elev` from the well-resolved
in-plane components by integrating `∂_y v_y = −(∂ₓvₓ + ∂_z v_z)` along elevation,
DC-anchored to the measured elevation velocity, then confidence-blended back in.

---

## 3. Validation (GT-free, identical metric to the baseline)

Build + regularize + integrate on odd-parity acquisitions and, independently, on
even-parity acquisitions; compare the two streamline maps by occupancy Dice and
shared-voxel direction cosine — the *same* `tractogram` and metric as the Gaussian
baseline, so this is head-to-head. Mean ± sd over 3 streamline-seed draws:

| field regularizer | split-half Dice | direction cosine | max vessel (p99.5) |
|---|---|---|---|
| baseline Gaussian (2.5,1.4,2.5) | 0.609 ± 0.004 | 0.844 ± 0.004 | 95 mm |
| **iso control** (γ=0, matched diffusivity) | **0.074 ± 0.005** | **−0.00 ± 0.04** | 6 mm |
| **graph along-vessel** (γ=1.5, 200 it) | **0.715 ± 0.001** | **0.952 ± 0.002** | 81 mm |
| graph + incompressibility (κ=0.5) | 0.713 ± 0.002 | 0.956 ± 0.002 | 80 mm |

Artifact: `outputs/reference/wf/r2/signal/velocity-field/splithalf_pde.json`
(regenerate: `python scripts/wf_render_signal/tractography_pde.py`).

**Reading the table.**
- **The win is real and robust.** Anisotropic along-vessel smoothing beats the
  validated Gaussian baseline by +0.11 Dice (+17%) and +0.11 cosine (+13%), with
  seed sd ±0.001 — not an artifact of which streamlines are seeded.
- **Anisotropy is the lever, not more smoothing.** At *matched total diffusivity*,
  the *isotropic* control (γ=0) collapses: it over-diffuses the coarse elevation axis,
  washes the field toward uniform, and streamline directions become random (cos ≈ 0).
  Same diffusion energy, opposite outcome — the improvement is the *directional*
  steering. (This mirrors, at matched budget, the doc's warning that isotropic
  over-smoothing inflates length without reproducibility.)
- **Not length-inflation.** Dice/cosine rise while the p99.5 streamline length
  *falls* (95 → 81 mm) — the signature of genuine denoising, not of over-smoothing
  stretching spurious streamlines. Occupancy grows (44k → 55k voxels) while directions
  stay coherent (cos 0.95), i.e. more vessel is recovered *coherently*.

**Honest finding — incompressibility is marginal here.** The divergence-free
elevation reconstruction nudges the cosine up (0.952 → 0.956) but the Dice down
(0.715 → 0.713): net ≈ neutral. On this data the along-vessel graph smoothing already
captures the elevation coherence, and the coarse, synthesized elevation axis carries
too little reliable in-plane divergence signal to reconstruct a *helpful* `v_elev`.
It is kept as an option (`mode="graph+incomp"`), off by default. This is a real
constraint of the data, not a failure of the physics — it would likely matter more on
genuinely 3D-resolved elevation.

---

## 4. Why this is the right next step

- **Physically principled.** Vessels are the flow's own coherence direction; diffusing
  along them (not across) is the correct low-pass, and the numbers agree.
- **Drop-in.** `regularize(V, C, mode="graph")` replaces `smooth_field` with the same
  interface; the rest of the tractography/render pipeline is unchanged.
- **Compounds the validated result.** The already-defensible split-half number
  (Dice 0.60 → **0.71**; cos 0.85 → **0.95**) gets materially stronger on the exact
  test that separates real structure from smoothing artifact.

## 5. Caveats

- Validated on the acq-0 slice (same scope as the baseline); dataset-wide (216 acqs)
  is bead mb-9ay.
- γ, iters chosen at a *moderate* operating point (higher iters/lower γ push Dice to
  ~0.73 but risk over-smoothing); we deliberately did not metric-maximize.
- Incompressibility is neutral on this data (above); revisit on 3D-resolved elevation.

Provenance: `docs/ideation/19–24`, `docs/regularized-field-tractography.md`. Bead: mb-ki9 (epic mb-k25).
