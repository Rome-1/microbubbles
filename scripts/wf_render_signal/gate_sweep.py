"""J1 (bead mb-5pd) — is the step-speed distribution censored by our own tracking gate?

MEASUREMENT THAT MOTIVATES THIS. Per-step velocities over the 216-acquisition corrected run wall
off at exactly the tracker's default box gate: 2 voxels/frame at (dx,dy,dz) = (0.2004, 0.5547,
0.2008) mm and 222.43 Hz is (89.15, 246.76, 89.33) mm/s for (lateral x, elevation y, axial z),
and the observed maxima are 89.2 / 246.7 / 89.3. So the gate is not a loose safety rail that the
data never touches — it IS the edge of the distribution. Two consequences: every in-plane mover
at or above ~89 mm/s is structurally barred from being linked, and the gate is 2.77x LOOSER on
elevation, the WORST-localized axis (dy is 2.77x coarser than dx), which is backwards.

STRUCTURAL CONFOUND (do not skip). The gate wall and the 4-angle coherent-compounding null are
THE SAME SPEED by algebra, not by coincidence:
    gate  = 2 voxels/frame = 2*(lambda/4)*FR = lambda*FR/2
    null  = lambda*PRF/8 with PRF = 4*FR     = lambda*FR/2   ->  both 88.97 mm/s
They separate only by AXIS: the compounding null suppresses AXIAL motion only (it is a Doppler
phase effect along the beam), while the box gate censors all three axes. Therefore the axis
composition of whatever a wider gate recovers is the scientific readout, not the total:
  * lateral-dominant recovery  => gate censoring dominated; the mass was there and we cut it.
  * an axial-specific DEFICIT  => the compounding null killed those detections upstream, before
                                  the tracker ever saw them; widening the gate cannot recover
                                  what detection never produced.

WHAT THIS SCRIPT DOES. Retracks CACHED detections (outputs/corrected_full/detections_XXXX.npz)
through the production tracking path (ultratrace_ulm.tracking._track_detections, i.e. the exact
code the 216-acq run used) changing NOTHING but the gate, via the existing `max_dist_mms` option
(mm/s in, per-frame mm out, through `_tracking_gate`). No re-detection, no GPU, no Modal.

GT-FREE SCORING (there is no ground truth, and "matches the reference" is not validation):
  (a) does supra-89 mm/s step mass appear AND live in PERSISTENT tracks (a censored distribution
      un-censoring, vs. threshold-shuffling which only reshuffles short fragments);
  (b) AXIS COMPOSITION of the recovered mass (lateral / axial / elevation);
  (c) planted-crossing link precision as the gate widens (simulation harness borrowed from
      track_sim_validate.py: identity-restricted, clutter held fixed), run against THIS tracker;
  (d) unlinked-detection fraction vs gate, and whether the newly linked detections are the ones
      that were previously orphaned (link-level attribution, not just a count).

Fleet-safe: nice(15), 1 BLAS thread per worker, worker count hard-capped, load-gated.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ultratrace_ulm.tracking import TrackingOptions, _track_detections, _tracking_gate

DET_DIR = Path("outputs/corrected_full")
OUTDIR = Path("outputs/gate_sweep")
BATCHES = (0, 12, 24, 36, 48)
FPS = 222.43
# grid spacing recorded in every corrected_full tracks pkl (params/spacing); the default gate is
# 2 voxels/frame on each axis, i.e. 2*SPACING mm/frame -> 2*SPACING*FPS mm/s.
SPACING = {"dx": 0.20040900000000006, "dy": 0.5546669999999998, "dz": 0.2007840000000023}
AXES = ("lateral", "elevation", "axial")  # position columns are (x, y, z)

MAX_WORKERS = 4
LOAD_PAUSE = 30.0

# the frozen operating point of the 216-acq run (recreate_full_app.py -> `ultratrace-ulm track`
# with --sigma-threshold 2.0 --knee-filter --temporal-sigma 0 and CLI defaults). ONLY the gate
# fields are ever changed below.
BASE_OPTS = TrackingOptions(
    beamformed_path=Path("/dev/null"),
    tracks_path=Path("/dev/null"),
    frame_rate_hz=FPS,
    max_gap=3,
    min_track_length=15,
    max_cost=10.0,
    reversal_penalty=10.0,
    tracking="kalman",
)

IN_PLANE = (89.0, 110.0, 130.0, 180.0)
ELEVATION = (247.0, 130.0, 90.0)


def load1() -> float:
    with open("/proc/loadavg") as fh:
        return float(fh.read().split()[0])


def wait_for_load(tag=""):
    while True:
        load = load1()
        if load <= LOAD_PAUSE:
            return load
        print(f"  [load-guard] {load:.1f} > {LOAD_PAUSE} — pausing {tag}", flush=True)
        time.sleep(20)


def _init_worker():
    os.nice(15)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"


# ----------------------------------------------------------------------------- data


def load_batch(b0: int):
    """Rebuild the per-frame detection lists exactly as the production run consumed them.

    `frame_indices` in the npz are the run's GLOBAL frame index (acq_order*frames_per_acq +
    frame_in_acq), so grouping by it reproduces detections_by_frame byte-for-byte, including the
    cross-acquisition frame ordering the tracker actually saw."""
    d = np.load(DET_DIR / f"detections_{b0:04d}.npz")
    pos = np.asarray(d["positions_mm"], np.float32)
    inten = np.asarray(d["intensities"], np.float32)
    frames = np.asarray(d["frame_indices"], np.int32)
    nframes = int(frames.max()) + 1
    order = np.argsort(frames, kind="stable")
    pos, inten, frames = pos[order], inten[order], frames[order]
    bounds = np.searchsorted(frames, np.arange(nframes + 1))
    dets = [pos[bounds[f]:bounds[f + 1]] for f in range(nframes)]
    ints = [inten[bounds[f]:bounds[f + 1]] for f in range(nframes)]
    # detection row id per frame, so tracks can be mapped back to detections (d, below)
    rows = [np.arange(bounds[f], bounds[f + 1]) for f in range(nframes)]
    return dets, ints, rows, pos, frames


def _row_lookup(dets, rows):
    """(frame, position-bytes) -> detection row. Track positions are copies of the detections,
    so the float32 bit pattern matches exactly; no tolerance needed."""
    look = {}
    for f, (P, R) in enumerate(zip(dets, rows)):
        for p, r in zip(P, R):
            look[(f, p.tobytes())] = int(r)
    return look


# ----------------------------------------------------------------------------- metrics


def track_steps(tracks):
    """Per-step displacement of every output track, as mm/frame PER AXIS plus the frame gap.

    Returns (D, gaps, tid, tlen): D is (n_steps, 3) signed mm/frame (already divided by the gap,
    so it is directly comparable to the per-frame gate), tid indexes the parent track and tlen is
    that track's length (used to ask whether supra-gate steps live in persistent tracks)."""
    D, gaps, tid, tlen = [], [], [], []
    for i, t in enumerate(tracks):
        p = np.asarray(t["positions"], float)
        f = np.asarray(t["frames"], float)
        if len(p) < 2:
            continue
        g = np.diff(f)
        D.append(np.diff(p, axis=0) / g[:, None])
        gaps.append(g)
        tid.append(np.full(len(g), i))
        tlen.append(np.full(len(g), len(p)))
    if not D:
        return (np.zeros((0, 3)), np.zeros(0), np.zeros(0, int), np.zeros(0, int))
    return (np.vstack(D), np.concatenate(gaps), np.concatenate(tid).astype(int),
            np.concatenate(tlen).astype(int))


def wall_profile(tracks, base_gate_mm, bands=(0.25, 0.5, 0.75, 1.0, 1.5, 2.0)):
    """Per-axis step counts in bands of the OLD gate — the control the axis-composition claim
    needs. A lateral:axial ratio measured only ABOVE the wall is uninterpretable on its own: if
    in-plane motion is already lateral-dominant BELOW the wall, a lateral-dominant recovery is
    just the same anisotropy continued, not evidence about the compounding null. The null's
    signature is a ratio that JUMPS as you approach and cross the wall, because it removes axial
    detections specifically at ~89 mm/s."""
    D, _, _, _ = track_steps(tracks)
    if not len(D):
        return {}
    frac = np.abs(D) / base_gate_mm[None, :]
    out = {}
    lo = 0.0
    for hi in bands:
        sel = (frac >= lo) & (frac < hi)
        out[f"{lo:.2f}-{hi:.2f}"] = {name: int(sel[:, k].sum()) for k, name in enumerate(AXES)}
        lo = hi
    return out


def summarize(tracks, dets, rows, look, n_det, base_gate_mm, baseline_linked=None):
    lens = np.array([int(t["length"]) for t in tracks]) if tracks else np.zeros(0, int)
    D, gaps, tid, tlen = track_steps(tracks)
    speed = np.linalg.norm(D, axis=1) * FPS
    per_axis = np.abs(D) * FPS

    # A step is "supra-baseline" on an axis when its per-frame displacement exceeds what the OLD
    # gate allowed on that axis — i.e. it could not have existed in the 216-acq run.
    over = np.abs(D) > base_gate_mm[None, :]
    any_over = over.any(axis=1)

    linked = set()
    for t in tracks:
        for p, f in zip(np.asarray(t["positions"], np.float32), np.asarray(t["frames"])):
            r = look.get((int(f), p.tobytes()))
            if r is not None:
                linked.add(r)

    out = {
        "n_tracks": len(tracks),
        "n_linked": len(linked),
        "linked_frac": len(linked) / max(1, n_det),
        "unlinked_frac": 1.0 - len(linked) / max(1, n_det),
        "len_med": float(np.median(lens)) if len(lens) else 0.0,
        "len_mean": float(lens.mean()) if len(lens) else 0.0,
        "n_ge35": int((lens >= 35).sum()),
        "n_steps": int(len(speed)),
        "speed_med": float(np.median(speed)) if len(speed) else 0.0,
        "speed_p99": float(np.percentile(speed, 99)) if len(speed) else 0.0,
        "speed_max": float(speed.max()) if len(speed) else 0.0,
        "n_supra": int(any_over.sum()),
        "supra_frac": float(any_over.mean()) if len(speed) else 0.0,
        "linked_rows": linked,
    }
    for k, name in enumerate(AXES):
        out[f"max_{name}"] = float(per_axis[:, k].max()) if len(speed) else 0.0
        out[f"supra_{name}"] = int(over[:, k].sum())
    if any_over.any():
        out["supra_len_med"] = float(np.median(tlen[any_over]))
        out["supra_len_mean"] = float(tlen[any_over].mean())
        out["supra_in_ge35"] = float((tlen[any_over] >= 35).mean())
        out["supra_tracks"] = int(len(np.unique(tid[any_over])))
    else:
        out.update(supra_len_med=0.0, supra_len_mean=0.0, supra_in_ge35=0.0, supra_tracks=0)
    if baseline_linked is not None:
        out["newly_linked"] = len(linked - baseline_linked)
        out["lost_links"] = len(baseline_linked - linked)
        out["kept_frac"] = len(linked & baseline_linked) / max(1, len(baseline_linked))
    return out


# ----------------------------------------------------------------------------- baseline check


def _fingerprint(tracks):
    return sorted((int(t["frames"][0]), float(t["positions"][0][0]), int(t["length"]))
                  for t in tracks)


def verify_baseline(b0=0):
    """Nothing downstream is trustworthy unless the retrack of cached detections reproduces the
    216-acq run BIT-FOR-BIT, and unless the mm/s gate path reproduces the spacing-derived one at
    the equivalent numbers. Both are checked here; the sweep aborts if either fails."""
    import pickle
    dets, ints, rows, pos, frames = load_batch(b0)
    ours = _track_detections(dets, ints, BASE_OPTS, SPACING)
    with open(DET_DIR / f"tracks_{b0:04d}.pkl", "rb") as fh:
        ref = pickle.load(fh)["tracks"]
    same_run = _fingerprint(ours) == _fingerprint(ref)
    equiv = replace(BASE_OPTS, max_dist_mms=tuple(2 * SPACING[k] * FPS for k in ("dx", "dy", "dz")))
    mms = _track_detections(dets, ints, equiv, SPACING)
    same_path = _fingerprint(ours) == _fingerprint(mms)
    print(f"  baseline check on batch {b0:04d}: retrack == production pkl ({len(ours)} vs "
          f"{len(ref)} tracks): {'PASS' if same_run else 'FAIL'}")
    print(f"  baseline check: max_dist_mms path at {tuple(round(v,2) for v in equiv.max_dist_mms)} "
          f"mm/s == spacing-derived gate: {'PASS' if same_path else 'FAIL'}")
    if not (same_run and same_path):
        raise SystemExit("baseline did not reproduce — refusing to run the sweep")


# ----------------------------------------------------------------------------- sweep


def gate_opts(inplane_mms, elev_mms):
    """The ONLY thing that varies in this experiment. mm/s in; _tracking_gate divides by the frame
    rate to get the per-frame box half-width, exactly as --max-dist-mms does on the CLI."""
    if inplane_mms is None:
        return BASE_OPTS  # spacing-derived default = the frozen 216-acq operating point
    return replace(BASE_OPTS, max_dist_mms=(float(inplane_mms), float(elev_mms), float(inplane_mms)))


def _job(args):
    b0, inplane, elev = args
    dets, ints, rows, pos, frames = load_batch(b0)
    opts = gate_opts(inplane, elev)
    t0 = time.time()
    tracks = _track_detections(dets, ints, opts, SPACING)
    return (b0, inplane, elev, tracks, len(pos), time.time() - t0)


def run_sweep(batches, workers, out_json):
    import json
    import multiprocessing as mp

    base_gate_mm = np.array(_tracking_gate(BASE_OPTS, SPACING))
    print(f"default (frozen) gate: {base_gate_mm} mm/frame = {base_gate_mm*FPS} mm/s "
          f"for {AXES}")

    cells = [(None, None)] + [(ip, ev) for ip in IN_PLANE for ev in ELEVATION]
    jobs = [(b0, ip, ev) for (ip, ev) in cells for b0 in batches]
    print(f"{len(cells)} gate cells x {len(batches)} batches = {len(jobs)} tracking runs")

    lookups, ndets = {}, {}
    for b0 in batches:
        dets, ints, rows, pos, frames = load_batch(b0)
        lookups[b0] = (_row_lookup(dets, rows), dets, rows, len(pos))
        ndets[b0] = len(pos)
        print(f"  batch {b0:04d}: {len(pos)} detections over {len(dets)} frames")

    raw = {}
    workers = max(1, min(int(workers), MAX_WORKERS))
    wait_for_load("sweep")
    with mp.Pool(workers, initializer=_init_worker) as pool:
        for i in range(0, len(jobs), workers):
            wait_for_load(f"chunk {i//workers+1}")
            for b0, ip, ev, tracks, npos, secs in pool.map(_job, jobs[i:i + workers]):
                raw[(ip, ev, b0)] = tracks
                print(f"  gate={('default' if ip is None else f'{ip:.0f}/{ev:.0f}')} "
                      f"batch {b0:04d}: {len(tracks)} tracks in {secs:.0f}s", flush=True)

    # baseline first (its linked set is the reference for the attribution columns)
    results = {}
    base_linked = {}
    walls = {}
    for cell in cells:
        per_batch = []
        wp = {}
        for b0 in batches:
            for band, counts in wall_profile(raw[(cell[0], cell[1], b0)], base_gate_mm).items():
                for ax, n in counts.items():
                    wp.setdefault(band, {}).setdefault(ax, 0)
                    wp[band][ax] += n
        walls[cell] = wp
        for b0 in batches:
            look, dets, rows, ndet = lookups[b0]
            prev = base_linked.get(b0)
            s = summarize(raw[(cell[0], cell[1], b0)], dets, rows, look, ndet, base_gate_mm, prev)
            if cell == (None, None):
                base_linked[b0] = s["linked_rows"]
            per_batch.append(s)
        results[cell] = _pool_batches(per_batch, ndets, batches)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump({"cells": {f"{k[0]}/{k[1]}": v for k, v in results.items()},
                   "wall_profile": {f"{k[0]}/{k[1]}": v for k, v in walls.items()}},
                  fh, indent=1)
    return results, base_gate_mm, walls


def _pool_batches(per_batch, ndets, batches):
    """Pool the batches: counts add, fractions are recomputed from the pooled counts, and the
    speed/length summaries are detection-count weighted."""
    tot_det = sum(ndets[b] for b in batches)
    out = {}
    add = ("n_tracks", "n_linked", "n_ge35", "n_steps", "n_supra", "supra_tracks",
           "supra_lateral", "supra_elevation", "supra_axial", "newly_linked", "lost_links")
    for k in add:
        if k in per_batch[0]:
            out[k] = int(sum(s[k] for s in per_batch))
    mx = [k for k in per_batch[0] if k.startswith("max_") or k == "speed_max"]
    for k in mx:
        out[k] = float(max(s[k] for s in per_batch))
    w = np.array([s["n_steps"] for s in per_batch], float)
    wl = np.array([s["n_tracks"] for s in per_batch], float)
    ws = np.array([s["n_supra"] for s in per_batch], float)
    for k, weight in (("speed_med", w), ("speed_p99", w), ("len_med", wl), ("len_mean", wl),
                      ("supra_len_med", ws), ("supra_len_mean", ws), ("supra_in_ge35", ws),
                      ("kept_frac", wl)):
        if k in per_batch[0]:
            v = np.array([s[k] for s in per_batch], float)
            out[k] = float((v * weight).sum() / max(1e-9, weight.sum()))
    out["linked_frac"] = out["n_linked"] / tot_det
    out["unlinked_frac"] = 1.0 - out["linked_frac"]
    out["supra_frac"] = out["n_supra"] / max(1, out["n_steps"])
    out["n_det"] = tot_det
    return out


# ----------------------------------------------------------------------------- (d) pool split


def _pool_job(args):
    b0, ip, ev, min_len = args
    dets, ints, rows, pos, frames = load_batch(b0)
    opts = replace(gate_opts(ip, ev), min_track_length=int(min_len))
    tracks = _track_detections(dets, ints, opts, SPACING)
    look = _row_lookup(dets, rows)
    linked = set()
    for t in tracks:
        for p, f in zip(np.asarray(t["positions"], np.float32), np.asarray(t["frames"])):
            r = look.get((int(f), p.tobytes()))
            if r is not None:
                linked.add(r)
    return b0, ip, ev, min_len, len(linked), len(pos), len(tracks)


def pool_attribution(batches, workers, gates=((None, None), (130.0, 247.0), (180.0, 247.0),
                                              (180.0, 130.0))):
    """Split the unlinked pool into its two causes, which the sweep table conflates.

    Retracking at min_track_length=2 keeps every chain the ASSOCIATION step formed, so:
      never-chained  = detections that no gate/cost ever linked to a neighbour  -> gate-limited
      chained-then-pruned = linked into a chain shorter than the production min length of 15
                            -> length-filter-limited, NOT gate-limited
    Only the first term is available to a gate change, so this bounds how much of the unlinked
    pool a wider gate could ever recover."""
    import multiprocessing as mp
    jobs = [(b0, ip, ev, ml) for (ip, ev) in gates for ml in (2, 15) for b0 in batches]
    workers = max(1, min(int(workers), MAX_WORKERS))
    out = {}
    with mp.Pool(workers, initializer=_init_worker) as pool:
        for i in range(0, len(jobs), workers):
            wait_for_load("pool-attrib")
            for b0, ip, ev, ml, nl, nd, nt in pool.map(_pool_job, jobs[i:i + workers]):
                k = (ip, ev, ml)
                a, b = out.get(k, (0, 0))
                out[k] = (a + nl, b + nd)
    rows = []
    for (ip, ev) in gates:
        chained, ndet = out[(ip, ev, 2)]
        kept, _ = out[(ip, ev, 15)]
        rows.append(dict(inplane=ip, elev=ev, n_det=ndet,
                         chained_frac=chained / ndet, kept_frac=kept / ndet,
                         never_chained_frac=1 - chained / ndet,
                         pruned_frac=(chained - kept) / ndet))
    return rows


# ----------------------------------------------------------------------------- (c) sim


def sim_precision(gate_cells, n_seeds=3, nframes=400, verbose=True):
    """Planted-crossing link precision as the gate widens, on THIS tracker.

    Reuses track_sim_validate.simulate/gt_links (identity-restricted edge scoring, clutter held
    fixed so crossings ADD detections rather than displacing noise) but drives the production
    Kalman/Hungarian tracker instead of the RTS tracker that module scores, and calibrates the
    regime to the corrected_full data rather than the reference pkl."""
    from track_sim_validate import gt_links, simulate

    d = np.load(DET_DIR / "detections_0000.npz")
    P = np.asarray(d["positions_mm"], float)
    F = np.asarray(d["frame_indices"], int)
    import pickle
    with open(DET_DIR / "tracks_0000.pkl", "rb") as fh:
        base = pickle.load(fh)["tracks"]
    lens = np.array([t["length"] for t in base])
    sp = np.concatenate([np.linalg.norm(np.diff(np.asarray(t["positions"], float), axis=0), axis=1)
                         * FPS for t in base if t["length"] > 1])
    nf_all = int(F.max()) + 1
    st = dict(lo=P.min(0), hi=P.max(0), nframes=int(nframes),
              dets_per_frame=len(P) / nf_all,
              linked_per_frame=lens.sum() / nf_all,
              speed_med=float(np.median(sp)), speed_lo=float(np.percentile(sp, 10)),
              speed_hi=float(np.percentile(sp, 90)),
              len_mean=float(lens.mean()), len_min=int(lens.min()), len_max=int(lens.max()))
    if verbose:
        print(f"\n  sim regime from corrected_full acq-batch 0: {st['dets_per_frame']:.1f} det/frame, "
              f"{st['linked_per_frame']:.1f} linked/frame, speed med {st['speed_med']:.1f} mm/s, "
              f"len mean {st['len_mean']:.1f}")

    rows = []
    for (ip, ev) in gate_cells:
        for n_cross in (0, 50):
            pr, rc = [], []
            for seed in range(n_seeds):
                Ps, Fs, ID = simulate(st, seed=seed, n_cross=n_cross)
                nfr = int(Fs.max()) + 1
                dets = [Ps[Fs == f].astype(np.float32) for f in range(nfr)]
                ints = [np.ones(len(x), np.float32) for x in dets]
                idx = [np.where(Fs == f)[0] for f in range(nfr)]
                tracks = _track_detections(dets, ints, gate_opts(ip, ev), SPACING)
                look = {(f, p.tobytes()): int(r)
                        for f, (D_, R_) in enumerate(zip(dets, idx)) for p, r in zip(D_, R_)}
                pred = set()
                for t in tracks:
                    ridx = [look.get((int(f), np.float32(p).tobytes()))
                            for p, f in zip(np.asarray(t["positions"], np.float32),
                                            np.asarray(t["frames"]))]
                    ridx = [r for r in ridx if r is not None]
                    pred.update(zip(ridx[:-1], ridx[1:]))
                G = gt_links(Fs, ID)
                tp = len(pred & G)
                pr.append(tp / max(1, len(pred)))
                rc.append(tp / max(1, len(G)))
            rows.append(dict(inplane=ip, elev=ev, n_cross=n_cross,
                             prec=float(np.mean(pr)), rec=float(np.mean(rc))))
            if verbose:
                tag = "default" if ip is None else f"{ip:.0f}/{ev:.0f}"
                print(f"    gate {tag:>9}  crossings {n_cross:>3}  "
                      f"precision {np.mean(pr):.3f}  recall {np.mean(rc):.3f}", flush=True)
    return rows


# ----------------------------------------------------------------------------- report


def _cell_name(cell):
    return "default (2 vox)" if cell[0] is None else f"{cell[0]:.0f} / {cell[1]:.0f}"


def report(results, base_gate_mm, walls=None, sim_rows=None, pool_rows=None):
    lines = []
    A = lines.append
    A("### Gate sweep — pooled over the retracked batches\n")
    A("| gate in-plane/elev (mm/s) | tracks | linked | linked% | med len | >=35 | med step (mm/s) "
      "| p99 | max lat | max ax | max elev | supra-gate steps | in tracks | med len of those "
      "| in tracks >=35 |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for cell, s in results.items():
        A(f"| {_cell_name(cell)} | {s['n_tracks']} | {s['n_linked']} | {s['linked_frac']*100:.1f}% "
          f"| {s['len_med']:.0f} | {s['n_ge35']} | {s['speed_med']:.1f} | {s['speed_p99']:.0f} "
          f"| {s['max_lateral']:.0f} | {s['max_axial']:.0f} | {s['max_elevation']:.0f} "
          f"| {s['n_supra']} ({s['supra_frac']*100:.1f}%) | {s['supra_tracks']} "
          f"| {s['supra_len_med']:.0f} | {s['supra_in_ge35']*100:.0f}% |")
    A("\n### Axis composition of the recovered (supra-old-gate) step mass\n")
    A("| gate | supra steps | lateral | axial | elevation | lateral:axial |")
    A("|---|---|---|---|---|---|")
    for cell, s in results.items():
        la, ax, ev = s["supra_lateral"], s["supra_axial"], s["supra_elevation"]
        A(f"| {_cell_name(cell)} | {s['n_supra']} | {la} | {ax} | {ev} | "
          f"{(la/ax if ax else float('nan')):.2f} |")
    if walls:
        A("\n### Approach to the wall — lateral:axial step counts in bands of the OLD gate\n")
        A("Control for the axis-composition read: if in-plane motion is already lateral-dominant "
          "below the wall, a lateral-dominant recovery above it is not evidence of anything. "
          "A ratio that CLIMBS as the wall is approached and crossed is the axial-suppression "
          "signature.\n")
        for cell in [(None, None), (130.0, 130.0), (180.0, 130.0)]:
            if cell not in walls:
                continue
            A(f"\n**gate {_cell_name(cell)}**\n")
            A("| band (x old gate) | lateral | axial | lateral:axial |")
            A("|---|---|---|---|")
            for band, c in walls[cell].items():
                if c["lateral"] + c["axial"] == 0:
                    continue
                r = c["lateral"] / c["axial"] if c["axial"] else float("nan")
                A(f"| {band} | {c['lateral']} | {c['axial']} | {r:.2f} |")
    A("\n### Link attribution vs the frozen baseline\n")
    A("| gate | newly linked dets | lost | baseline links kept | unlinked pool |")
    A("|---|---|---|---|---|")
    for cell, s in results.items():
        if "newly_linked" not in s:
            continue
        A(f"| {_cell_name(cell)} | {s['newly_linked']} | {s['lost_links']} | "
          f"{s['kept_frac']*100:.1f}% | {s['unlinked_frac']*100:.1f}% |")
    if pool_rows:
        A("\n### What the unlinked pool is made of (min_len=2 vs the production min_len=15)\n")
        A("| gate | detections | never chained (gate-limited) | chained but pruned (<15) | kept |")
        A("|---|---|---|---|---|")
        for r in pool_rows:
            tag = "default" if r["inplane"] is None else f"{r['inplane']:.0f} / {r['elev']:.0f}"
            A(f"| {tag} | {r['n_det']} | {r['never_chained_frac']*100:.1f}% | "
              f"{r['pruned_frac']*100:.1f}% | {r['kept_frac']*100:.1f}% |")
    if sim_rows:
        A("\n### Planted-crossing link precision (simulation, production tracker)\n")
        A("| gate | crossings | precision | recall |")
        A("|---|---|---|---|")
        for r in sim_rows:
            tag = "default" if r["inplane"] is None else f"{r['inplane']:.0f} / {r['elev']:.0f}"
            A(f"| {tag} | {r['n_cross']} | {r['prec']:.3f} | {r['rec']:.3f} |")
    return "\n".join(lines)


def figure(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = [c for c in results if c[0] is not None]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for evi, ev in enumerate(ELEVATION):
        sel = [c for c in cells if c[1] == ev]
        x = [c[0] for c in sel]
        ax[0].plot(x, [results[c]["linked_frac"] * 100 for c in sel], "o-", label=f"elev {ev:.0f}")
        ax[1].plot(x, [results[c]["n_supra"] for c in sel], "o-", label=f"elev {ev:.0f}")
        ax[2].plot(x, [results[c]["n_ge35"] for c in sel], "o-", label=f"elev {ev:.0f}")
    base = results[(None, None)]
    for a, v in zip(ax, (base["linked_frac"] * 100, base["n_supra"], base["n_ge35"])):
        a.axhline(v, color="k", ls="--", lw=0.8, label="frozen gate")
        a.set_xlabel("in-plane gate (mm/s)")
    ax[0].set_ylabel("detections linked (%)")
    ax[1].set_ylabel("steps beyond the old gate")
    ax[2].set_ylabel("tracks >= 35 points")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batches", type=int, nargs="*", default=list(BATCHES))
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    ap.add_argument("--sim-seeds", type=int, default=3)
    ap.add_argument("--no-sim", action="store_true")
    ap.add_argument("--no-fig", action="store_true")
    a = ap.parse_args()
    _init_worker()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print(f"gate sweep | load {load1():.1f} | batches {a.batches}")
    verify_baseline(a.batches[0])

    results, base_gate_mm, walls = run_sweep(a.batches, a.workers, OUTDIR / "gate_sweep.json")
    pool_rows = pool_attribution(a.batches, a.workers)
    sim_rows = None
    if not a.no_sim:
        sim_rows = sim_precision([(None, None), (110.0, 130.0), (130.0, 130.0), (180.0, 130.0)],
                                 n_seeds=a.sim_seeds)
    md = report(results, base_gate_mm, walls, sim_rows, pool_rows)
    (OUTDIR / "gate_sweep.md").write_text(md + "\n")
    print("\n" + md)
    if not a.no_fig:
        figure(results, OUTDIR / "gate_sweep.png")


if __name__ == "__main__":
    main()
