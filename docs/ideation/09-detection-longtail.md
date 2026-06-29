# 09 — `detect_acqs_multi` GPU long-tail: root cause + fix

**Context.** A 223-acq `swp_knee` SVD re-detection was run by launching **8 parallel
`modal run --detach`** invocations of `detect_acqs_multi`
(`scripts/modal/app.py:1005`), one A10G GPU each, splitting 223 acqs into 8 equal-**COUNT**
chunks (`--acq-start {0,28,56,…,196} --num-acqs 28`). The function is additive/checkpointed
(skips an acq whose `acq_XXXX.npz` already exists). 41 of 223 were already done. It took
**~2h54m wall** (01:51→04:45) instead of the expected ~70 min, and "the later ones take ages."

**Verdict (one line).** The blow-up is a **load-balance / scheduling tail**, not a per-acq
slowdown. Equal-**count** static partitioning collided with (a) pre-existing checkpoints
concentrated in the low orders and (b) ~5× per-container host-speed heterogeneity, so the
run collapsed from 8 GPUs to a **single GPU running solo for the final ~38 min**.

---

## Evidence

### A. Per-chunk real work was wildly uneven (CONFIRMED — saved container stdout)

The 41 pre-existing checkpoints were concentrated in the **low** orders, exactly the ranges
assigned to the first chunks. Equal-count slicing therefore handed the early-range GPUs almost
nothing and the high-range GPUs a full load. From the per-container logs
(`scratchpad/modal_logs/pathA_chunk_*.log`):

| chunk `--acq-start` | order range | real acqs done | median beamform | finished |
|---|---|---|---|---|
| 0   | [0,28)    | **6**  | 250.5 s | 02:26 |
| 28  | [28,56)   | **13** | 315.3 s | 03:10 |
| 56  | [56,84)   | 24 | 49.1 s  | ~early (fast host) |
| 84  | [84,112)  | 28 | 185.6 s | 04:07 |
| 112 | [112,140) | 28 | 177.7 s | — |
| 140 | [140,168) | 28 (1 host-OOM restart) | 115.3 s | — |
| 168 | [168,196) | 28 | 75.6 s  | — |
| 196 | [196,223) | **27** | 278.0 s | **04:45 (LAST)** |

Real acqs = 6+13+24+28+28+28+28+27 = **182**, +41 pre-existing = 223. ✔
The first two GPUs had **6 and 13** acqs; the last had **27** — a >4× spread per worker.

### B. The run collapsed 8 → 1 GPU and finished as a solo tail (CONFIRMED — monitors)

Aggregate completion rate (`/tmp/curve.txt`, from checkpoint-count polling): steady
**~2.0–2.4 acq/min** for the first ~45 min, decaying to **~0.17 acq/min = exactly one A10G**
for the final ~40 min. The "ephemeral app count" monitor pins it down
(`pathA_monitor2.log` / `pathA_monitor3.log`):

```
03:55  swp_knee=209  ephemeral=3
03:58  swp_knee=212  ephemeral=2
04:07  swp_knee=216  ephemeral=1   ← only chunk196 left; SOLO from here
04:45  swp_knee=223  ephemeral=1   PATH_A_COMPLETE
```

The final **38 min (04:07→04:45) was a single GPU** (chunk196) grinding orders ~216–222 alone.
The last ~50 min ran at ≤2 GPUs. That collapse — not any per-acq regression — is the wall-time.

### C. Per-acq cost is roughly constant *within* a container; it varies *between* hosts (CONFIRMED)

Within any one chunk log the per-acq beamform time is steady (no upward drift across the 27–28
iterations → **no intra-container memory-pool creep**). But **median beamform differs 5× across
containers**: 49 s (chunk56) vs 315 s (chunk28), 278 s (chunk196). The slow containers stayed
slow even when running solo (chunk196's last orders 220–222 = 182/286/308 s, with no peers) —
so this is **host/CPU heterogeneity** (the `mach.kernel` "array is not contiguous, rearranging
will add latency" CPU rearrange dominates), **not** GPU contention. Static assignment is
maximally exposed to this: the unlucky slow host (chunk196, 278 s/acq) also drew a *full* 27-acq
load → it is the long pole on both axes at once.

### D. Ruled out as the primary cause

- **GPU concurrency cap / queueing waves — REFUTED.** The decorator at
  `scripts/modal/app.py:1005` has **no** `max_containers`/`concurrency_limit`
  (grep: none in `app.py`). All 8 detached apps dispatched together (`pathA_dispatch.log`,
  8× `DISPATCH_EXIT 0`) and the aggregate rate is ~2.2 acq/min from the first poll — consistent
  with **all 8 A10Gs granted ~immediately**, no wave. `modal app list` now shows the 8 detection
  apps gone (only the composite job `ap-4Pe240AJeRRaIGol1wvbhW` remains) — nothing stranded billing.
- **GPU memory not freed between acqs — REFUTED as a tail cause.** The hot paths *do* free the
  cupy pool: `beamform_core.py:387-392` frees per-angle device blocks inside `stream_accumulate`;
  `gpu_svd.py:78,144` and `gpu_detect.py:36-81` call `free_all_blocks()`. Per-acq beamform is flat
  within a container (Evidence C), confirming no pool growth. **Not the lever.**
- **`vol.commit()` per acq (`app.py:1141`) — minor.** One commit per ~245 s of compute is
  negligible; not a measurable contributor.

### E. Secondary: one host-RAM OOM (CONFIRMED, minor)

chunk140 took a `SIGKILL exit 137` ("runs out of memory") at order 142 and **restarted**
(`pathA_chunk_140.log:41`). This is **host RAM** (the function requests `memory=98304`; the
beamform accumulator peaks ~2× the volume per `beamform_core.py:382`), not GPU. Additive
checkpointing absorbed it (re-run skipped 140–141), costing only one cold-start. The per-acq
loop never `del`s `comp`/`iq`/`gx,gy,gz` or calls `gc.collect()`, so host arrays can transiently
stack. Worth a cheap fix but not the headline.

---

## Quantifying the tail

Total GPU-seconds actually spent (median beamform + ~70 s detect + ~7 s read per acq, summed
over all 182 real acqs) ≈ **44.6k GPU-s ≈ 744 GPU-min**; mean **≈ 245 s/acq** (not the assumed
180 s — beamform on contended/slow hosts runs 250–315 s).

| scenario | wall |
|---|---|
| Operator's mental model (180 s/acq, 8 GPU, perfect balance) | ~68 min |
| **Achievable** balanced wall (real 744 GPU-min ÷ 8, +startup) | **~95 min** |
| **Actual** (static 8-chunk partition) | **174 min** |

The static partition cost **~80 min** (174 vs 95) — a **~1.8× inflation** purely from the
8→1-GPU collapse, on top of the ~25 min that the higher-than-assumed per-acq cost already added
over the 70-min expectation. The solo 38-min tail is the bulk of the waste.

---

## Recommended fix (the single biggest lever)

**Replace the hand-rolled fixed 8-chunk `modal run --detach` launch with a fine-grained Modal
`.starmap()` over the *missing* global orders.** Modal pulls the next input as each container
frees up → automatic work-stealing, no solo tail, and host-speed heterogeneity is *shared* across
inputs instead of concentrated in one worker. This is the one change that turns 174 min → **~95–100 min**.

### Design

1. **Planner (local entrypoint / cheap CPU fn).** Compute `canonical = all_ids[::acq_step]`
   (order→aid map; for this dataset order==aid), then list `…/detections/<tag>/` on the volume and
   emit **only the missing orders**: `missing = [(order, aid) for order,aid in enumerate(canonical)
   if not exists(f"acq_{order:04d}.npz")]`. This is the "plan over only the missing orders" step —
   no container is ever spun up for an already-done acq (kills the wasted chunk-`[0,28)` GPU).

2. **Worker = the current loop body for ONE acq (or a small K-batch).** Factor lines
   `app.py:1109-1141` into `detect_one(order, aid, variants, …)`: beamform once, run all variants
   (keep the fused sweep — beamform is shared), write `acq_{order:04d}.npz`, `vol.commit()`, return
   timing. TGC `inv_sqrt` is already cached on the volume (`inv_sqrt.npy`, reused from `base60`,
   `app.py:1075-1077`) — each worker just `np.load`s it; no recompute. **Checkpoint layout is
   unchanged** (`acq_XXXX.npz` by global order), so it stays drop-in compatible with `track_acqs`.

3. **Launch with auto load-balance + an explicit cap.**

```python
@app.function(image=gpu_image, gpu="A10G", timeout=12*3600, memory=98304,
              volumes={"/root/data": vol}, max_containers=8)   # explicit, predictable fan-out
def detect_one(order: int, aid: int, variants: str, ...): ...   # = current per-acq body

@app.local_entrypoint()
def detect_sweep(variants: str, acq_step: int = 1, tag: str = "swp_knee"):
    missing = plan_missing(variants, acq_step, tag)             # (order, aid) tuples, volume-listed
    args = [(o, a, variants, ...) for o, a in missing]
    list(detect_one.starmap(args))                              # Modal work-steals across ≤8 GPUs
```

`starmap` keeps ≤`max_containers` A10Gs busy and dispatches the next acq the instant any GPU
frees — the 8→1 collapse cannot happen. With ~245 s/acq and ~182 inputs over 8 GPUs the slow
host does ~6–8 acqs while fast hosts do ~30; the tail is one acq, not 27.

### Supporting tweaks
- **Keep per-acq `vol.commit()`** (checkpoint safety; it's not the bottleneck). If batching K
  acqs/input, still commit per acq.
- **Host-RAM insurance (Evidence E):** at the end of each acq `del comp, iq, txd, txde, gx, gy, gz, d`
  + `gc.collect()` (GPU pool already freed). Cheap; avoids the one OOM restart.
- **Make concurrency explicit** via `max_containers=` rather than relying on N separate
  `modal run` calls — predictable cost and no chance of an accidental wave.

### Expected speedup
**~174 min → ~95–100 min (~1.7–1.8×)**, the solo 38-min single-GPU tail eliminated, plus
robustness to per-host speed variance and to skewed pre-existing checkpoints.

---

*CONFIRMED = read from code or saved container/monitor logs. LIKELY = inferred (live
`modal app logs` for the 8 detection apps have aged out and return nothing; the saved
`scratchpad/modal_logs/pathA_*` files ARE those containers' stdout, captured at run time).*
