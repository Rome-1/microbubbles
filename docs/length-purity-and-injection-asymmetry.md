# Two settled questions: where temporal persistence starts being informative (A), and whether the injector can adjudicate the axial deficit (B)

Beads mb-isb (A) and mb-9u0 (B).

- **A** — `python3 scripts/wf_render_signal/length_sweep.py` (local, CPU, ~9 min at 4 workers).
  60 acquisitions, 603,254 cached detections from `outputs/corrected_full`, retracked through the
  production path with nothing changed but the emission floor and the gate.
- **B** — `modal run scripts/modal/inject_track_app.py::inject_track --n-realizations 12`
  (A100-80GB, 201 s, ~$0.6), rendered by `scripts/wf_render_signal/inject_track_report.py`.
  Reused `corrected_bf_1.h5` on the volume; no re-beamforming.

---

## VERDICT A — `min_track_length` is an output dial, and 8 is where persistence starts being informative

`min_track_length` is an **emission predicate**, not an association parameter. It appears in
`tracking.py` only as `if length >= min_track_length: emit` (`:343` in `_retire`, `:722`, `:729`),
never in the Hungarian cost, the Kalman update, the gate, or the spawn logic. Lowering it recovers
**zero** associations. The ~49% of detections it withholds were already chained; they are simply
not printed.

Two consequences, both of which the earlier framing got wrong:

1. Its yield must never be added to, or compared against, the 42.2% the pipeline failed to
   associate. Those are different populations — one never linked, the other linked and withheld.
2. Because the chain set is independent of the floor, every floor is an exact **subset** of one
   retrack at `min_track_length=2`. `verify_subset()` proves this by fingerprint against a real
   `min_len=15` retrack and aborts if it fails; it passes on every gate. So the cells below are
   exactly comparable, not merely similar.

So the question is purity, not recovery. The null is the same detections with frame order
permuted **within each acquisition** — identical per-frame spatial distribution, density, and
vascular clustering, with only the temporal correspondence destroyed. Any chain the tracker still
forms is a chance chain.

`purity(L) = 1 - N_null(>=L) / N_real(>=L)`

**This is the calibration the project has been missing.** Temporal persistence is the only
bubble/noise discriminator available without ground truth, and nobody had measured where it starts
working.

### The purity curve (frozen gate, 60 acquisitions, null averaged over 3 permutations)

| min_len L | tracks emitted | null tracks | **purity** | linked dets | med len |
|---|---|---|---|---|---|
| 2 | 94,779 | 60,229 | 36.5% | — | — |
| 3 | 45,353 | 15,349 | 66.2% | — | — |
| 4 | 25,157 | 5,214 | 79.3% | — | — |
| 5 | 15,810 | 2,107 | 86.7% | 152,022 (25.2%) | 7 |
| 6 | 11,053 | 967 | 91.3% | — | — |
| 7 | 8,213 | 474 | 94.2% | — | — |
| **8** | **6,439** | **247** | **96.2%** | **98,779 (16.4%)** | **11** |
| 9 | 5,190 | 138 | 97.3% | — | — |
| 10 | 4,356 | 77 | 98.2% | 81,281 (13.5%) | 14 |
| 12 | 3,146 | 30 | 99.0% | 68,649 (11.4%) | 17 |
| 15 (shipped) | 2,118 | 9 | 99.6% | 55,454 (9.2%) | 21 |
| 20 | 1,267 | 2 | 99.9% | 41,294 (6.8%) | 28 |
| 25 | 801 | 0 | 100.0% | — | — |

At the wider gate (130/130) the knee sits in the same place and every purity value is ~0.5–1 point
lower — a wider gate buys longer chains for the null too:

| min_len L | tracks emitted | null tracks | purity |
|---|---|---|---|
| 5 | 16,666 | 2,524 | 84.9% |
| 8 | 6,837 | 328 | 95.2% |
| 10 | 4,591 | 112 | 97.6% |
| 15 | 2,225 | 12 | 99.5% |
| 20 | 1,351 | 1 | 100.0% |

Purity above is a **track-level** count. The **detection-level** contamination is smaller, because
null chains are short: at L=8 the null links 0.39% of all detections against the real 16.37%, i.e.
2.4% of published linked detections are chance. At L=5 that rises to 8.3%, and at the shipped
L=15 it is 0.3%.

### Two-point chains are half noise, and the curve is steep below 8

A floor of 2 is 36.5% pure — a permutation of the data reproduces **63.5%** of all 2-point chains.
Below L=5 the null is a first-order term. Purity then climbs steeply and flattens: the cost of
going from 8 to 15 is a factor of 3 in yield (6,439 → 2,118 tracks) to buy 3.4 points of purity
(96.2% → 99.6%).

### The added tracks are signal, by two independent discriminators

**Planted-crossing link precision does not degrade at all.** Same identity-restricted harness
`gate_sweep.py` uses, production tracker, gate frozen, only the floor varying:

| min_len | precision (no crossings) | precision (50 crossings) | recall (50 crossings) |
|---|---|---|---|
| 5 | 0.949 | 0.912 | 0.437 |
| 8 | 0.951 | 0.914 | 0.331 |
| 10 | 0.954 | 0.918 | 0.271 |
| 12 | 0.962 | 0.924 | 0.220 |
| 15 (shipped) | 0.949 | 0.933 | **0.131** |
| 20 | 0.937 | 0.933 | 0.078 |

Precision is flat within noise across the whole range (0.91–0.96) while **recall more than
triples** from 15 to 5. The shipped floor is buying essentially no link precision and paying for
it with three-quarters of the recall.

**Spatial coherence — the added tracks lie on the vasculature and run along it.** Scaffold built
from a random half of the long (>=15) tracks; the other half is the held-out positive reference.
`dist` = median distance to the scaffold; `|cos|` = alignment with the scaffold's local mean
direction (sign-free, via the principal eigenvector of the local direction outer product).

Frozen gate:

| length band | n real | real dist | real \|cos\| | n null | null dist | null \|cos\| |
|---|---|---|---|---|---|---|
| 5–8 | 9,371 | 0.440 | 0.792 | 1,873 | 0.115 | 0.762 |
| 8–10 | 2,083 | 0.241 | 0.852 | 165 | 0.119 | 0.772 |
| 10–12 | 1,210 | 0.201 | 0.878 | 42 | 0.174 | 0.811 |
| 12–15 | 1,028 | 0.178 | 0.898 | 22 | 0.076 | 0.852 |
| **>=15 (held out)** | **1,040** | **0.143** | **0.894** | 12 | 0.075 | 0.830 |

Read this one carefully, because the obvious reading is wrong. **Distance is inverted and useless
as a discriminator**: the null's chains sit *closer* to the scaffold than real short tracks do,
because chance chains form preferentially in the densest — that is, the most vascular — regions.
Alignment is the half that discriminates: real short tracks beat the null in every band and rise
monotonically toward the held-out long-track reference (0.792 → 0.898 vs the reference's 0.894),
while the null's chains do not converge on it. But the margin at 5–8 is only 0.792 vs 0.762, and
the null's own alignment is high in absolute terms — chance chains inherit the vessels' elongation
because that is where the clutter is. Coherence is therefore the **weakest** of the three
discriminators and is offered as corroboration, not proof. The decoy survival ratio carries the
argument. (At 130/130 the gap widens — 0.763 vs 0.678 at 5–8 — because the wider gate lets null
chains wander further off-axis.)

### Recommendation: lower `min_track_length` from 15 to **8**, and what it costs

- **Yield**: 2,118 → 6,439 tracks (3.0x). Linked detections 9.2% → 16.4% of the pool.
- **Purity cost**: 99.6% → 96.2%. About **247 of the 6,439** emitted tracks are chance chains,
  against 9 today. In exchange, 4,321 tracks that the pipeline had already built are published.
  At the detection level the contamination is milder — chance chains are short — at 2.4% of
  published linked detections, against 0.3% today.
- **Link precision cost**: none measurable (0.933 → 0.914 under a 50-crossing load, inside seed
  noise; 0.949 → 0.951 with no crossings).
- **Tracks >=35 are unchanged** (372, necessarily — the floor cannot touch them). Any downstream
  metric restricted to long tracks is unaffected by this change.
- **Median length drops 21 → 11**, which is close to the reference release's 9.99 and is a
  consequence of publishing the short tracks, not of shorter tracks being made.

Do not go to 5 (86.7% pure: 2,107 chance chains, a 13% contamination that would show up in any
density or flow-direction map) and do not go below. **8 is the operating point**: it is the point
where purity clears 95% and the curve flattens, so it takes essentially all the available yield
before the noise term turns first-order.

### Caveat this prices on the gate result too

The null also lifts when the gate widens, so part of the gate's gain is chance association rather
than recovered bubbles. Linked-detection fraction, real vs null, at both gates:

| floor L | real default | real 130/130 | gain | null default | null 130/130 | null gain | **% of gain the null reproduces** |
|---|---|---|---|---|---|---|---|
| 5 | 25.200% | 26.741% | +1.540 | 2.084% | 2.524% | +0.441 | **28.6%** |
| **8** | 16.374% | 17.486% | +1.112 | 0.385% | 0.515% | +0.131 | **11.8%** |
| 10 | 13.474% | 14.346% | +0.872 | 0.149% | 0.218% | +0.068 | 7.8% |
| 12 | 11.380% | 12.188% | +0.808 | 0.069% | 0.100% | +0.031 | 3.8% |
| **15 (shipped)** | 9.192% | 9.802% | +0.609 | 0.026% | 0.033% | +0.007 | **1.2%** |
| 20 | 6.845% | 7.380% | +0.535 | 0.006% | 0.002% | −0.003 | ~0% |

**At the shipped floor the caveat is negligible — 1.2% of the "+6.6% links" is null-reproducible,
so that number stands.** But the contamination grows sharply as the floor drops: at the
recommended floor of 8 it is 11.8%, and at 5 it is 28.6%. The two changes interact, and the gate's
gain should be quoted with the floor it was measured at. At L=8 the gate's honest gain is roughly
+0.98 points of linked detections, not +1.11.

---

## VERDICT B — neither arm reproduces the real asymmetry; the injector cannot adjudicate it

**Outcome (c).** And per the stopping rule agreed in advance: no post-compound injector can settle
this, so this line stops here rather than iterating.

Design: matched-speed bubbles injected on both axes, 5–130 mm/s, into the beamformed **complex**
volume pre-SVD; same measured PSF, same noise field, same `rank24|per_elev` filter, same
matched-density threshold **fixed on the clean volume for both arms**, same production tracker,
same gate. 12 realizations x 300 bubbles per arm, identical seeds — literally the same bubbles.
The tracker gate derived from this volume's spacing comes out at 89.15 / 246.75 / 89.32 mm/s,
matching the real run's wall exactly, so the two tables are on the same axis.

Both `compounding_gain_on=True` and `False` were run, because `inject.py` applies the 4-angle
compounding gain to each bubble by hand and that term *is* the axial suppression under
investigation. A single arm reproducing the effect would have shown only that a model containing
the effect by construction reproduces it.

### The headline comparison — lateral:axial per-step ratio, by band of the old gate

| band (x old gate) | **real data** | injection, gain ON | injection, gain OFF |
|---|---|---|---|
| 0.00–0.25 | **0.98** | 0.04 | 0.03 |
| 0.25–0.50 | **0.95** | 0.40 | 0.29 |
| 0.50–0.75 | **1.20** | 1.66 | 0.37 |
| 0.75–1.00 | **1.53** | **169** | **0.80** |

(Injection at `min_track_length=5`; at the production floor of 15 the two fastest bands contain
16 and 0 steps respectively and cannot be read at all — see the caveat below. Real column is
`docs/tracking-gate-censoring.md`, "Approach to the wall", frozen gate, 60 acquisitions.)

Raw counts behind the top band: gain ON 169 lateral / **1** axial; gain OFF 169 lateral / **212**
axial. The lateral count is identical because the gain only touches axial motion.

### Three things this settles

**1. The modelled notch is ~110x too violent to explain the real asymmetry.** Propagated through
detection and the production tracker, `compounding_gain_on=True` predicts a lateral:axial ratio of
**169** in the 0.75–1.00 band. The real data shows **1.53**. The gain term wipes out 211 of 212
axial linked steps (99.5%) in that band; the real data is missing about a third of its axial
steps. Whatever produces the real 1.51, a compounding null of this depth is not it — it would have
left near-total axial extinction, and the real distribution does not show that.

**2. Turn the notch off and the deficit does not appear.** Gain OFF gives **0.80** in the top
band — on the *wrong side of 1*, mildly axial-*favoring*. So within this instrument the high-speed
axial deficit is entirely manufactured by the term we imposed. That is outcome (b) in local terms
and, as anticipated, it is not evidence: it says only that the model contains what we put in it.
The useful part is the magnitude mismatch in (1), which is arm-independent in its implication.

**3. The instrument disagrees with the real data by ~25x at the low end, in BOTH arms.** Real
0.98 vs injection 0.03–0.04 at 0.00–0.25. Both arms agree because the compounding gain is ~1.0
below 20 mm/s, so this is not the notch — it is the slow-lateral clutter-filter blind spot,
reproduced here at detection level (recovery 0.026 lateral vs 0.420 axial at 5 mm/s, consistent
with J3's 0.077 vs 0.591 at a higher SNR ladder). **The real ratio table is isotropic at low
speed (0.98) and therefore contradicts it.** Either the real low-speed steps are not coming from
slow bubbles, or the injected slow-lateral motion model is wrong. Given the instrument is
internally consistent and the real data is not reproducible from it, the injected lateral motion
model is the suspect.

### Putting J1 and J3 in the same units — linking amplifies, it does not reconcile

This was the point of the exercise, and the answer is clean. At 5 mm/s the **detection** ratio is
0.06; the **linked-step** ratio is 0.00 (0 lateral steps vs 187 axial). Linking is multiplicative
in per-frame recovery — a track needs a run of consecutive detections — so a per-frame asymmetry
of 16:1 becomes an unbounded one at the track level. **Conditioning on linking exponentiates J3's
asymmetry rather than reversing it.** The J1/J3 disagreement is therefore not a units artifact:
J3's picture propagated through the tracker predicts ~0.01–0.04 at low speed where the real data
says 0.98.

### Caveats, stated plainly

- **The fast bands are starved at production settings.** At `min_track_length=15` the 0.75–1.00
  band holds 16 (gain ON) and 17/16 (gain OFF) linked steps. The comparison above uses
  `min_len=5`, where it holds 169 vs 1 and 169 vs 212. This is a real limitation: a bubble at
  80 mm/s with a 24-frame lifetime rarely stays in the volume long enough to yield a 15-point
  track, so the instrument is intrinsically weak exactly where the anomaly lives.
- **Construction differs between the two tables.** The real column bands a step by its own
  per-axis displacement and so mixes oblique movers into both columns; the injection uses
  pure-axis movers. This matters for interpreting a 1.2x difference. It does not rescue a 25x or
  110x one.
- **The leading unmodelled candidate is intra-frame smear.** At ~89 mm/s axial a bubble crosses
  ~2 voxels *within* the 4-transmit window, averaging the λ/4 carrier and gutting the peak whether
  or not the sum is coherent. `inject.py:59-64` explicitly models fast bubbles as marginally
  *easier*, the opposite sign. Testing it needs per-transmit injection before compounding — a
  different instrument and a separate decision.
- The compounding notch is left neither established nor refuted. This experiment bounds its
  *depth*: if it exists at the modelled depth, the real data would look far more axially extinct
  than it does.

---

## Reproduce

```
python3 scripts/wf_render_signal/length_sweep.py --workers 4
modal run scripts/modal/inject_track_app.py::inject_track --n-realizations 12
modal volume get research microbubbles/inject_track.json outputs/inject_track/inject_track.json
python3 scripts/wf_render_signal/inject_track_report.py outputs/inject_track/inject_track.json
```

Both scripts abort rather than report if their reproduction gates fail: `length_sweep.py` checks
that the retrack of cached detections reproduces the shipped 216-acquisition pkl track-for-track
and that the floor is a pure post-filter; `inject_track_app.py` fixes the detection threshold on
the clean volume so neither arm can pick its own operating point.
