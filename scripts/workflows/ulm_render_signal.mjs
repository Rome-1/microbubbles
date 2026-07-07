export const meta = {
  name: 'ulm-render-signal',
  description: 'Multi-model iterate on the best ULM render + most-insightful brain signal, scored vs the Aleph reference tracks',
  phases: [
    { title: 'Design' },
    { title: 'Synthesize' },
    { title: 'Implement-r1' },
    { title: 'Evaluate-r1' },
    { title: 'Implement-r2' },
    { title: 'Evaluate-r2' },
  ],
}

const GROUND = `
CONTEXT — Ultratrace 3D ULM (ultrasound localization microscopy) of human brain vasculature.
Repo root = /home/rome/gt/microbubbles/crew/cajal. Read these for full grounding BEFORE acting:
docs/ideation/20-reference-artifacts-received.md, docs/ideation/21-reference-gap-deep-dive.md,
docs/ideation/19-frontier-brainstorm-fable.md.

We have the Aleph REFERENCE (their actual output, our GT-free yardstick):
- outputs/reference/full_tracks_smoothed.pkl : dict with tracks_smoothed (50,456 tracks; each a
  dict positions(N,3)f32, frames(N,)f32, length int, intensities(N,)f64, acq_index int),
  detections (dict per acq, z-scores 4.9..53.7), grid_x/y/z (25 elev,154 z,275 x), spacing
  (dx0.2004,dy0.5547,dz0.2008 mm), n_acquisitions 216, frames_per_acq 240, params.
  SAFE-LOAD (do NOT plain-pickle-load blindly; use this whitelist unpickler):
    import pickle,numpy as np
    SAFE={("numpy._core.multiarray","_reconstruct"),("numpy.core.multiarray","_reconstruct"),
          ("numpy._core.multiarray","scalar"),("numpy.core.multiarray","scalar"),
          ("numpy","ndarray"),("numpy","dtype")}
    class SU(pickle.Unpickler):
      def find_class(self,m,n):
        for mm in (m,m.replace("numpy._core","numpy.core"),m.replace("numpy.core","numpy._core")):
          if (mm,n) in SAFE: return super().find_class(mm,n)
        raise pickle.UnpicklingError(m+"."+n)
    obj=SU(open("outputs/reference/full_tracks_smoothed.pkl","rb")).load()
- outputs/reference/tracks_v6.bin (s1 render, 2,294 tracks), outputs/reference/viewer_ref/,
  outputs/reference/viewer_blogmatch/ (jet 0-38 mm/s, black bg — matches the blog figure),
  renders/references/Screenshot 2026-06-29 at 11.51.16.png (the BLOG target).
- OUR pipeline output: renders/track_viewer_composite/data/tracks.bin (v3: 64B header, table
  n_tracks*(u32 offset,u32 length), then 6 float32/point x,y,z,frame,speed,intensity).

KEY FACTS (established): reference tracks are median-7-frames SHORT; the crisp render = aggressive
min_length filter (~28) to ~2,300 long tracks. Our composite over-produces (254k tracks) with
74 cm/s TELEPORT links from elev_gate_factor=3 (fix: 1.0). Reference max speed ~0.40 mm/frame
(~9 cm/s) physiological. The reference render DISCARDS 83% of touched coverage and tracking
discards 69% of its own detections (~1.4M localizations) — that is the opening.

GT-FREE YARDSTICKS (score every candidate on these, with ACTUAL measured numbers):
- Coverage@equal-FRC: grid-voxel fill fraction at matched resolution (reference ~1.8%). HEADLINE.
- Ref-recall: fraction of the reference render's occupied voxels our map also covers (target >=95%).
- >=35-frame track count at the physiological gate (0.40,1.11,0.40 mm) (reference 1,421). Beat it.
- max per-frame speed must stay physiological (<= ~0.5 mm/frame; our composite's 3.35 is clutter).
- held-out-acquisition recall (fit on 215 acqs, predict acq-216 localizations) where applicable.

TWO DELIVERABLES:
(A) BEST RENDER — beat the blog: denser + crisper + honest. Direction is free (velocity vector is
    computed then discarded at track_viewer_export.py:55-64). Ideas: drizzle super-res density
    (all localizations as sub-pixel splats -> continuous vessels not dots), flow-DIRECTION color,
    depth cueing/eye-dome-lighting, animated flow, trust scaffolding (calibrated legend/scalebar).
(B) MOST-INSIGHTFUL SIGNAL for downstream tasks — velocity field, pulsatility (self-gated cardiac,
    222 Hz oversamples the ~5-7 Hz cardiac band; 240-frame acqs span several cycles), artery/vein
    separation, flow-field reconstruction. MUST also come with a dense rendering of the signal.

RULES: work in the main working dir (reference data is gitignored, present only here — do NOT use
worktrees). Write all artifacts under outputs/reference/wf/<round>/<candidate>/ (create dirs).
Do NOT git commit/push (the parent handles commits). Do NOT run Modal/GPU (CPU numpy only). GROUND
every claim by ACTUALLY running python on the reference pickle and reporting measured numbers.

CODEX: the OpenAI codex CLI (gpt-5.5, a strong algorithmic reasoner) is available. When a stage
says "use codex", run: timeout 900 codex exec --dangerously-bypass-approvals-and-sandbox
--skip-git-repo-check '<detailed prompt>' 2>/dev/null ; read + sanity-check its output, then
implement/synthesize. If codex errors or returns empty, do the work yourself and note the fallback.
`

const PLAN = { type:'object', additionalProperties:true,
  properties:{ report:{type:'string'}, top_ideas:{type:'array',items:{type:'string'}},
               highest_conviction:{type:'string'} }, required:['report','highest_conviction'] }
const SPEC = { type:'object', additionalProperties:true,
  properties:{ report:{type:'string'}, render_plan:{type:'array',items:{type:'string'}},
               signal_plan:{type:'array',items:{type:'string'}}, metrics:{type:'array',items:{type:'string'}} },
  required:['report','render_plan','signal_plan'] }
const IMPL = { type:'object', additionalProperties:true,
  properties:{ report:{type:'string'}, artifacts:{type:'array',items:{type:'string'}},
               measured:{type:'string'}, next:{type:'array',items:{type:'string'}} },
  required:['report','measured'] }
const EVAL = { type:'object', additionalProperties:true,
  properties:{ report:{type:'string'}, verdicts:{type:'string'},
               ranked:{type:'array',items:{type:'string'}}, next_round:{type:'array',items:{type:'string'}} },
  required:['report','verdicts','next_round'] }

// ---------------- Design (4 parallel, diverse models incl. codex) ----------------
phase('Design')
const design = await parallel([
  () => agent(GROUND + `\nYOUR TASK (opus, rigor): design the MOST-INSIGHTFUL SIGNAL pipeline (B).
Focus on the motion-confirmed dense reconstruction that recovers the 69% discarded detections, a
per-voxel velocity field, self-gated pulsatility, and artery/vein separation. Give concrete,
runnable numpy algorithms on the reference pickle detections/tracks, and how each is VALIDATED
GT-free. Be exact about the math. Return the plan.`,
    {label:'design:signal', model:'opus', phase:'Design', schema:PLAN}),
  () => agent(GROUND + `\nYOUR TASK (fable, vision): design the BEST RENDER (A) that decisively beats
the blog — denser AND crisper AND honest. Push the "living angiogram": drizzle density, flow-
direction color, eye-dome-lighting depth, animated flow, trust scaffolding. Also propose how to
render the dense SIGNAL (B) beautifully. Be concrete about how each maps to our Three.js viewer +
the per-point channels we have. Return the plan.`,
    {label:'design:render', model:'fable', phase:'Design', schema:PLAN}),
  () => agent(GROUND + `\nYOUR TASK (fable, wild): propose 2-3 NON-OBVIOUS downstream-task signals a
neuroscientist/clinician would prize (e.g. wall-shear-stress, resistance maps, vascular-graph
topology, perfusion territories) that this data can plausibly yield, each with a dense render and a
GT-free sanity check. Swing big but stay grounded in the data. Return the plan.`,
    {label:'design:downstream', model:'fable', phase:'Design', schema:PLAN}),
  () => agent(GROUND + `\nYOUR TASK (codex workhorse): USE CODEX (gpt-5.5) to design the CORE
ALGORITHM for the motion-confirmed dense reconstruction + global velocity-gated min-cost-flow
association that recovers discarded detections without reintroducing teleport clutter, plus the
exact implementation of the GT-free yardstick metrics (Coverage@equal-FRC, ref-recall, >=35-track
count at the physiological gate). Run codex per the CODEX instructions above with a detailed prompt;
sanity-check its algorithm against the pickle; return the concrete algorithm + metric code sketch.`,
    {label:'design:algo-codex', phase:'Design', schema:PLAN}),
]).then(r => r.filter(Boolean))

// ---------------- Synthesize (opus -> single implementation spec) ----------------
phase('Synthesize')
const spec = await agent(GROUND + `\nYOU ARE THE SYNTHESIZER (opus). Merge these ${design.length}
design plans into ONE concrete, prioritized implementation spec with two tracks (render + signal),
exact algorithms, output paths under outputs/reference/wf/, and the measured yardsticks each impl
must report. Plans:\n` + design.map((d,i)=>`--- PLAN ${i+1} ---\n`+JSON.stringify(d)).join('\n'),
  {label:'synthesize', model:'opus', phase:'Synthesize', schema:SPEC})

// ---------------- Implement -> Evaluate, 2 iterate rounds ----------------
const rounds = []
let feedback = 'Round 1: implement the synthesized spec faithfully.'
for (let r = 1; r <= 2; r++) {
  phase(`Implement-r${r}`)
  const impls = await parallel([
    () => agent(GROUND + `\nIMPLEMENT THE RENDER (sonnet), round ${r}. Spec:\n${JSON.stringify(spec)}
\nFEEDBACK to address this round:\n${feedback}\nBuild a self-contained viewer bundle under
outputs/reference/wf/r${r}/render/ that beats the blog (apply the elev_gate_factor=1.0 fix to our
composite tracks first, then drizzle density + flow-direction color + depth). VERIFY with a real
headless-chromium screenshot (google-chrome --headless=new --enable-unsafe-swiftshader) saved
alongside. Report measured yardsticks (Coverage@equal-FRC, ref-recall, >=35-track count, max speed)
and artifact paths.`,
      {label:`impl-r${r}:render`, model:'sonnet', phase:`Implement-r${r}`, schema:IMPL}),
    () => agent(GROUND + `\nIMPLEMENT THE SIGNAL (codex workhorse), round ${r}. Spec:\n${JSON.stringify(spec)}
\nFEEDBACK to address this round:\n${feedback}\nUSE CODEX (gpt-5.5) for the core algorithm, then
implement in python (CPU numpy) on the reference pickle: motion-confirmed dense reconstruction +
velocity field + (if reachable) pulsatility / artery-vein. Write the dense signal + a viewer/render
under outputs/reference/wf/r${r}/signal/. Report measured yardsticks with ACTUAL numbers and how
each signal is GT-free-validated. If codex fails, fall back to your own implementation and say so.`,
      {label:`impl-r${r}:signal`, phase:`Implement-r${r}`, schema:IMPL}),
  ]).then(x => x.filter(Boolean))

  phase(`Evaluate-r${r}`)
  const ev = await agent(GROUND + `\nEVALUATE (opus), round ${r}. Two implementations:\n` +
    impls.map((m,i)=>`--- IMPL ${i+1} ---\n`+JSON.stringify(m)).join('\n') +
    `\nAdversarially verify the MEASURED numbers by re-running spot checks on the artifacts (do the
files exist? are the metrics real? is the render honest or is the "signal" clutter/teleports?).
Rank render + signal vs the reference yardsticks. Then give a SHORT concrete next_round punch-list
of the highest-leverage improvements for round ${r+1}.`,
    {label:`eval-r${r}`, model:'opus', phase:`Evaluate-r${r}`, schema:EVAL})

  rounds.push({ round:r, impls, ev })
  const nr = (ev && Array.isArray(ev.next_round)) ? ev.next_round : []
  feedback = nr.join('\n') || (ev && ev.report) || feedback
  log(`Round ${r} done. Next: ${String(feedback).slice(0,200)}`)
}

return { spec, rounds }
