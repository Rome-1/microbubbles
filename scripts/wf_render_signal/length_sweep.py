"""EXP-A (bead mb-isb) — is `min_track_length=15` throwing away signal or noise?

WHY THIS IS THE LEVER. The detection-pool decomposition (docs/tracking-gate-censoring.md, 60
acquisitions) found that only 9.2% of detections survive to a production track, that 42.2% never
join even a 2-point chain, and that the remaining ~49% ARE chained by the association step and are
then discarded by `min_track_length=15`. The length filter, not the gate, is the dominant term in
the unlinked pool. The reference release shipped `min_track_length=5` and its tracks have a median
length of 9.99 — i.e. more than half of the reference's tracks are shorter than our filter's floor
and could not exist in our output at all.

THE STRUCTURAL FACT THAT MAKES THIS CHEAP AND EXACT. `min_track_length` is never consulted by the
association step. In `kalman_tracking_3d` it appears exactly once, at retirement
(`if _lengths[i] >= min_track_length`), and in the non-kalman path likewise only in `_retire`.
So the chain SET is independent of it, and every min_track_length cell is an exact SUBSET of a
single retrack at min_track_length=1. We therefore retrack once per (gate, batch) and filter,
which makes the cells exactly comparable rather than merely similar. `verify_subset()` proves the
claim by direct fingerprint comparison against a real min_len=15 retrack and aborts if it fails.

THE QUESTION THAT ACTUALLY MATTERS is not how many tracks a lower floor adds — lowering a
threshold always adds tracks — but whether the ADDED ones are signal. Three GT-free
discriminators, none of which requires ground truth:

  (1) FRAME-SHUFFLE DECOY. Within each acquisition, permute which frame each detection cloud is
      presented as. This preserves the per-frame spatial detection distribution EXACTLY — same
      clutter density, same vascular clustering, same count — and destroys only the temporal
      correspondence. Any chain the tracker still forms is a chance chain. n_decoy(len>=L) /
      n_real(len>=L) is then a direct, per-length false-discovery estimate. The length at which
      that ratio stops being small IS the answer to "where do added tracks stop being
      distinguishable from noise".

      Time reversal is NOT used as the decoy, despite being the more commonly cited control, and
      the reason is worth stating: a real track played backwards is still a physically valid
      track with identical step statistics. Reversal preserves exactly the structure the decoy is
      supposed to destroy, so it would understate the noise floor toward zero. Frame-shuffle is
      the correct null here. (Reversal is still computed, as a positive control: it should look
      like REAL data, not like the decoy, and if it does not, the shuffle is doing something
      other than what is claimed.)

  (2) PLANTED-CROSSING LINK PRECISION vs min_track_length, on the production tracker, reusing the
      identity-restricted simulation from track_sim_validate.py exactly as gate_sweep.py does.

  (3) SPATIAL COHERENCE against the existing vasculature. Long tracks (>=15) define the vascular
      scaffold. For each added short track we ask two things: how far its points sit from that
      scaffold, and whether it runs ALONG the local flow or across it (|cos| between its own
      direction and the mean direction of nearby scaffold steps). A short track that is a real
      bubble on a real vessel lies on the scaffold and moves along it; a chance chain assembled
      out of clutter does neither. Both statistics are compared against the decoy's short tracks,
      which is the comparison that controls for the fact that clutter is itself concentrated in
      bright vascular regions.

Fleet-safe: nice(15), 1 BLAS thread/worker, <=4 workers, load-gated. CPU only, no Modal.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gate_sweep import (  # reuse the frozen operating point and the verified loaders
    AXES,
    BASE_OPTS,
    BATCHES,
    DET_DIR,
    FPS,
    MAX_WORKERS,
    SPACING,
    _fingerprint,
    _init_worker,
    _row_lookup,
    load1,
    load_batch,
    wait_for_load,
)
from ultratrace_ulm.tracking import _track_detections, _tracking_gate

OUTDIR = Path("outputs/length_sweep")

MIN_LENS = (5, 8, 10, 12, 15, 20)
GATES = ((None, None), (130.0, 130.0))  # frozen default, and the recommended operating point
N_DECOY_SEEDS = 3
PROD_MIN_LEN = 15  # what we ship today; "added" tracks are those with length < this


# ----------------------------------------------------------------------------- decoys


def shuffle_frames(dets, ints, n_acq=12, seed=0):
    """Frame-shuffle decoy: permute frame slots WITHIN each acquisition.

    Each frame keeps its detection cloud intact and simply appears at a different time. Spatial
    statistics (density, clustering onto vessels, per-frame count) are preserved bit-for-bit; only
    the frame-to-frame correspondence is destroyed. Permuting within an acquisition rather than
    across the whole batch matters: acquisitions differ in bubble load and depth, so a global
    shuffle would also destroy the between-acquisition structure and inflate the apparent
    difficulty of chaining for reasons that have nothing to do with time."""
    rng = np.random.default_rng(seed)
    nfr = len(dets)
    per = nfr // n_acq
    order = np.arange(nfr)
    for a in range(n_acq):
        lo, hi = a * per, min((a + 1) * per, nfr)
        blk = order[lo:hi].copy()
        rng.shuffle(blk)
        order[lo:hi] = blk
    return [dets[i] for i in order], [ints[i] for i in order], order


def reverse_frames(dets, ints):
    """Positive control, not a null — see the module docstring."""
    return dets[::-1], ints[::-1]


# ----------------------------------------------------------------------------- tracking jobs


def gate_opts(inplane, elev, min_len):
    o = BASE_OPTS if inplane is None else replace(
        BASE_OPTS, max_dist_mms=(float(inplane), float(elev), float(inplane)))
    return replace(o, min_track_length=int(min_len))


def _job(args):
    """One retrack at min_track_length=2; every min_len cell of interest is a filter of the result.

    2 rather than 1 because a "chain" of one detection is not a chain — it is an unlinked
    detection, it is never reported at any floor we consider, and keeping them would push a
    quarter-million singleton dicts per batch across the pool boundary for nothing."""
    b0, ip, ev, kind, seed = args
    dets, ints, rows, pos, frames = load_batch(b0)
    if kind == "shuffle":
        dets, ints, order = shuffle_frames(dets, ints, seed=seed)
        rows = [rows[i] for i in order]   # detection ids must follow their frame
    elif kind == "reverse":
        dets, ints = reverse_frames(dets, ints)
        rows = rows[::-1]
    t0 = time.time()
    tracks = _track_detections(dets, ints, gate_opts(ip, ev, 2), SPACING)
    # Linked-detection counts must be computed HERE, against the lookup built from the frame
    # order the tracker actually saw. Computing them in the parent against the unpermuted
    # lookup silently returns zero for every null arm, because a detection's frame index is
    # exactly what the permutation changed.
    look = _row_lookup(dets, rows)
    linked = {}
    for ml in MIN_LENS:
        acc = set()
        for t in tracks:
            if int(t["length"]) < ml:
                continue
            for p, f in zip(np.asarray(t["positions"], np.float32), np.asarray(t["frames"])):
                r = look.get((int(f), p.tobytes()))
                if r is not None:
                    acc.add(r)
        linked[ml] = len(acc)
    # keep only what downstream needs, so the pickle crossing the pool boundary stays small
    slim = [{"length": int(t["length"]),
             "positions": np.asarray(t["positions"], np.float32),
             "frames": np.asarray(t["frames"], np.float32)} for t in tracks]
    return b0, ip, ev, kind, seed, slim, len(pos), time.time() - t0, linked


# ----------------------------------------------------------------------------- correctness gate


def verify_subset(b0, ip, ev):
    """Prove the subset claim instead of trusting the code read.

    A real retrack at min_track_length=15 must equal the min_track_length=1 retrack filtered to
    length>=15, track-for-track. If that fails, the filter is not a pure post-filter and the whole
    cheap-sweep design is invalid, so this aborts."""
    dets, ints, rows, pos, frames = load_batch(b0)
    full = _track_detections(dets, ints, gate_opts(ip, ev, 2), SPACING)
    direct = _track_detections(dets, ints, gate_opts(ip, ev, PROD_MIN_LEN), SPACING)
    filtered = [t for t in full if int(t["length"]) >= PROD_MIN_LEN]
    ok = _fingerprint(filtered) == _fingerprint(direct)
    tag = "default" if ip is None else f"{ip:.0f}/{ev:.0f}"
    print(f"  subset check batch {b0:04d} gate {tag}: min_len=1 filtered to >=15 "
          f"({len(filtered)}) == direct min_len=15 retrack ({len(direct)}): "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def verify_baseline_pkl(b0=0):
    """The same reproduction gate gate_sweep.py runs: the retrack of cached detections at the
    frozen operating point must reproduce the shipped 216-acq tracks pkl track-for-track."""
    dets, ints, rows, pos, frames = load_batch(b0)
    ours = _track_detections(dets, ints, BASE_OPTS, SPACING)
    with open(DET_DIR / f"tracks_{b0:04d}.pkl", "rb") as fh:
        ref = pickle.load(fh)["tracks"]
    ok = _fingerprint(ours) == _fingerprint(ref)
    print(f"  baseline check batch {b0:04d}: retrack == production pkl "
          f"({len(ours)} vs {len(ref)}): {'PASS' if ok else 'FAIL'}")
    return ok


# ----------------------------------------------------------------------------- metrics


def linked_count(tracks, look, min_len):
    linked = set()
    for t in tracks:
        if t["length"] < min_len:
            continue
        for p, f in zip(np.asarray(t["positions"], np.float32), np.asarray(t["frames"])):
            r = look.get((int(f), p.tobytes()))
            if r is not None:
                linked.add(r)
    return linked


def cell_summary(tracks_by_batch, looks, ndets, min_len):
    n_tracks = n_ge35 = 0
    lens_all = []
    linked_tot = 0
    for b0, tracks in tracks_by_batch.items():
        keep = [t for t in tracks if t["length"] >= min_len]
        n_tracks += len(keep)
        n_ge35 += sum(1 for t in keep if t["length"] >= 35)
        lens_all.append(np.array([t["length"] for t in keep], float))
        linked_tot += len(linked_count(tracks, looks[b0], min_len))
    lens = np.concatenate(lens_all) if lens_all else np.zeros(0)
    tot_det = sum(ndets.values())
    return dict(n_tracks=n_tracks, n_ge35=n_ge35, n_linked=linked_tot,
                linked_frac=linked_tot / max(1, tot_det),
                len_med=float(np.median(lens)) if len(lens) else 0.0,
                len_mean=float(lens.mean()) if len(lens) else 0.0,
                n_det=tot_det)


def survival(tracks_by_batch, lmax=45):
    """N(length >= L) for L = 1..lmax, pooled over batches."""
    lens = np.concatenate([np.array([t["length"] for t in tr], int)
                           for tr in tracks_by_batch.values()]) if tracks_by_batch else np.zeros(0, int)
    return {L: int((lens >= L).sum()) for L in range(2, lmax + 1)}


# ----------------------------------------------------------------------------- (3) coherence


def _scaffold(tracks, min_len=PROD_MIN_LEN, half=None, seed=0):
    """Points and local step directions of the long tracks — the 'existing vascular structure'.

    `half` splits the long tracks in two. The scaffold is built from one half so that the OTHER
    half can be scored against it as a held-out positive reference. Without that split, tracks of
    length >= min_len score a distance of exactly 0 to the scaffold because they ARE the scaffold,
    and the reference row of the table is meaningless."""
    sel = [t for t in tracks if t["length"] >= min_len]
    if half is not None:
        rng = np.random.default_rng(seed)
        mask = rng.random(len(sel)) < 0.5
        sel = [t for t, m in zip(sel, mask) if (m if half == 0 else not m)]
    P, V = [], []
    for t in sel:
        p = np.asarray(t["positions"], float)
        d = np.diff(p, axis=0)
        n = np.linalg.norm(d, axis=1, keepdims=True)
        good = n[:, 0] > 1e-9
        P.append(p[:-1][good])
        V.append(d[good] / n[good])
    if not P:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.vstack(P), np.vstack(V)


def _split_half(tracks, min_len, half, seed=0):
    """The complement of the half `_scaffold(half=0)` used, with the same seed and rule."""
    sel = [t for t in tracks if t["length"] >= min_len]
    rng = np.random.default_rng(seed)
    mask = rng.random(len(sel)) < 0.5
    return [t for t, m in zip(sel, mask) if (m if half == 0 else not m)]


def coherence(tracks, lo, hi, scaffold_P, scaffold_V, max_pts=20000, seed=0):
    """How well do tracks with lo <= length < hi sit on, and run along, the scaffold?

    Returns the median distance from a short track's points to the nearest scaffold point, and the
    median |cos| between the short track's own mean direction and the scaffold's mean direction in
    its neighbourhood. Distance alone is weak — clutter is concentrated in bright vascular regions
    too, so a chance chain will also sit near vessels. Direction agreement is the discriminating
    half: chance chains assembled from clutter have no reason to point along the vessel."""
    from scipy.spatial import cKDTree

    if not len(scaffold_P):
        return dict(n=0, dist_med=float("nan"), cos_med=float("nan"))
    tree = cKDTree(scaffold_P)
    rng = np.random.default_rng(seed)
    sel = [t for t in tracks if lo <= t["length"] < hi]
    if not sel:
        return dict(n=0, dist_med=float("nan"), cos_med=float("nan"))
    if len(sel) > max_pts:
        sel = [sel[i] for i in rng.choice(len(sel), max_pts, replace=False)]
    dists, coss = [], []
    for t in sel:
        p = np.asarray(t["positions"], float)
        d, idx = tree.query(p, k=1)
        dists.append(np.median(d))
        v = p[-1] - p[0]
        nv = np.linalg.norm(v)
        if nv < 1e-9:
            continue
        v /= nv
        # mean scaffold direction in the neighbourhood of this track, sign-free (vessels carry
        # flow both ways and a track's sense is not the question — its ALIGNMENT is)
        nbr = tree.query_ball_point(p.mean(0), r=0.6)
        if not nbr:
            continue
        S = scaffold_V[nbr]
        M = S.T @ S  # sign-invariant mean direction = principal eigenvector of the outer product
        w, U = np.linalg.eigh(M)
        coss.append(abs(float(v @ U[:, -1])))
    return dict(n=len(sel),
                dist_med=float(np.median(dists)) if dists else float("nan"),
                cos_med=float(np.median(coss)) if coss else float("nan"))


# ----------------------------------------------------------------------------- (2) sim


def sim_precision_vs_minlen(min_lens, gate=(None, None), n_seeds=3, nframes=400, verbose=True):
    """Planted-crossing link precision as the LENGTH FLOOR drops, on the production tracker.

    Identical harness to gate_sweep.sim_precision (identity-restricted edge scoring, clutter held
    fixed so crossings add detections rather than displace noise); the gate is frozen and
    min_track_length is what varies. Precision here is a LINK-level quantity, so it answers a
    question the decoy cannot: of the edges that a lower floor lets through, what fraction connect
    the same simulated bubble?"""
    from track_sim_validate import gt_links, simulate

    d = np.load(DET_DIR / "detections_0000.npz")
    P = np.asarray(d["positions_mm"], float)
    F = np.asarray(d["frame_indices"], int)
    with open(DET_DIR / "tracks_0000.pkl", "rb") as fh:
        base = pickle.load(fh)["tracks"]
    lens = np.array([t["length"] for t in base])
    sp = np.concatenate([np.linalg.norm(np.diff(np.asarray(t["positions"], float), axis=0), axis=1)
                         * FPS for t in base if t["length"] > 1])
    nf_all = int(F.max()) + 1
    st = dict(lo=P.min(0), hi=P.max(0), nframes=int(nframes),
              dets_per_frame=len(P) / nf_all, linked_per_frame=lens.sum() / nf_all,
              speed_med=float(np.median(sp)), speed_lo=float(np.percentile(sp, 10)),
              speed_hi=float(np.percentile(sp, 90)),
              len_mean=float(lens.mean()), len_min=int(lens.min()), len_max=int(lens.max()))

    rows = []
    for n_cross in (0, 50):
        for seed in range(n_seeds):
            Ps, Fs, ID = simulate(st, seed=seed, n_cross=n_cross)
            nfr = int(Fs.max()) + 1
            dets = [Ps[Fs == f].astype(np.float32) for f in range(nfr)]
            ints = [np.ones(len(x), np.float32) for x in dets]
            idx = [np.where(Fs == f)[0] for f in range(nfr)]
            tracks = _track_detections(dets, ints, gate_opts(gate[0], gate[1], 1), SPACING)
            look = {(f, p.tobytes()): int(r)
                    for f, (D_, R_) in enumerate(zip(dets, idx)) for p, r in zip(D_, R_)}
            G = gt_links(Fs, ID)
            for ml in min_lens:
                pred = set()
                for t in tracks:
                    if len(t["positions"]) < ml:
                        continue
                    ridx = [look.get((int(f), np.float32(p).tobytes()))
                            for p, f in zip(np.asarray(t["positions"], np.float32),
                                            np.asarray(t["frames"]))]
                    ridx = [r for r in ridx if r is not None]
                    pred.update(zip(ridx[:-1], ridx[1:]))
                tp = len(pred & G)
                rows.append(dict(min_len=ml, n_cross=n_cross, seed=seed,
                                 prec=tp / max(1, len(pred)), rec=tp / max(1, len(G))))
    agg = {}
    for r in rows:
        k = (r["min_len"], r["n_cross"])
        agg.setdefault(k, []).append((r["prec"], r["rec"]))
    out = [dict(min_len=k[0], n_cross=k[1],
                prec=float(np.mean([v[0] for v in vs])), rec=float(np.mean([v[1] for v in vs])))
           for k, vs in sorted(agg.items())]
    if verbose:
        for r in out:
            print(f"    min_len {r['min_len']:>3}  crossings {r['n_cross']:>3}  "
                  f"precision {r['prec']:.3f}  recall {r['rec']:.3f}", flush=True)
    return out


# ----------------------------------------------------------------------------- driver


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batches", type=int, nargs="*", default=list(BATCHES))
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    ap.add_argument("--decoy-seeds", type=int, default=N_DECOY_SEEDS)
    ap.add_argument("--sim-seeds", type=int, default=3)
    ap.add_argument("--no-sim", action="store_true")
    a = ap.parse_args()
    import multiprocessing as mp

    _init_worker()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print(f"length sweep | load {load1():.1f} | batches {a.batches}")

    if not verify_baseline_pkl(a.batches[0]):
        raise SystemExit("baseline did not reproduce — refusing to run")
    for ip, ev in GATES:
        if not verify_subset(a.batches[0], ip, ev):
            raise SystemExit("min_track_length is not a pure post-filter — design invalid")

    looks, ndets = {}, {}
    for b0 in a.batches:
        dets, ints, rows, pos, frames = load_batch(b0)
        looks[b0] = _row_lookup(dets, rows)
        ndets[b0] = len(pos)
        print(f"  batch {b0:04d}: {len(pos)} detections over {len(dets)} frames")

    jobs = []
    for ip, ev in GATES:
        for b0 in a.batches:
            jobs.append((b0, ip, ev, "real", 0))
            jobs.append((b0, ip, ev, "reverse", 0))
            for s in range(a.decoy_seeds):
                jobs.append((b0, ip, ev, "shuffle", s))
    print(f"{len(jobs)} retracks (min_len=1; every cell is a filter of these)")

    store, linked_store = {}, {}
    workers = max(1, min(int(a.workers), MAX_WORKERS))
    with mp.Pool(workers, initializer=_init_worker) as pool:
        for i in range(0, len(jobs), workers):
            wait_for_load(f"chunk {i//workers+1}")
            for b0, ip, ev, kind, seed, tracks, npos, secs, lk in pool.map(_job,
                                                                            jobs[i:i + workers]):
                store[(ip, ev, kind, seed, b0)] = tracks
                linked_store[(ip, ev, kind, seed, b0)] = lk
                tag = "default" if ip is None else f"{ip:.0f}/{ev:.0f}"
                print(f"  gate {tag} {kind}{seed if kind=='shuffle' else ''} batch {b0:04d}: "
                      f"{len(tracks)} chains in {secs:.0f}s", flush=True)

    out = {"cells": {}, "survival": {}, "coherence": {}}
    for ip, ev in GATES:
        key = "default" if ip is None else f"{ip:.0f}/{ev:.0f}"
        real = {b0: store[(ip, ev, "real", 0, b0)] for b0 in a.batches}
        rev = {b0: store[(ip, ev, "reverse", 0, b0)] for b0 in a.batches}
        for ml in MIN_LENS:
            out["cells"][f"{key}|{ml}"] = cell_summary(real, looks, ndets, ml)
        # the SAME summary on the null, so every yield number can be priced against the
        # yield the null produces for free
        dec_cells = {}
        tot_det = sum(ndets.values())
        for ml in MIN_LENS:
            per = []
            for sd in range(a.decoy_seeds):
                tr = {b0: store[(ip, ev, "shuffle", sd, b0)] for b0 in a.batches}
                nl = sum(linked_store[(ip, ev, "shuffle", sd, b0)][ml] for b0 in a.batches)
                c = cell_summary(tr, looks, ndets, ml)
                c["n_linked"], c["linked_frac"] = nl, nl / max(1, tot_det)
                per.append(c)
            dec_cells[ml] = {k: float(np.mean([q[k] for q in per])) for k in per[0]}
        out.setdefault("cells_null", {}).update(
            {f"{key}|{ml}": v for ml, v in dec_cells.items()})
        sr = survival(real)
        srev = survival(rev)
        sds = [survival({b0: store[(ip, ev, "shuffle", s, b0)] for b0 in a.batches})
               for s in range(a.decoy_seeds)]
        sd = {L: float(np.mean([x[L] for x in sds])) for L in sr}
        out["survival"][key] = {"real": sr, "shuffle": sd, "reverse": srev}

        # THE HEADLINE: purity(L) = 1 - N_null(>=L)/N_real(>=L), against the yield it buys.
        # Temporal persistence is the only bubble/noise discriminator available without ground
        # truth, and this is the curve that says where it starts being informative.
        pur = []
        for L in range(2, 41):
            nr, nn = sr[L], sd[L]
            c = out["cells"].get(f"{key}|{L}")
            cn = out.get("cells_null", {}).get(f"{key}|{L}")
            pur.append(dict(L=L, n_real=nr, n_null=nn,
                            purity=(1.0 - nn / nr) if nr else float("nan"),
                            linked_frac=(c["linked_frac"] if c else None),
                            linked_frac_null=(cn["linked_frac"] if cn else None),
                            n_ge35=(c["n_ge35"] if c else None)))
        out.setdefault("purity", {})[key] = pur

        # coherence: scaffold from the REAL long tracks in each case, so the decoy's short chains
        # are scored against the same vasculature the real short chains are scored against.
        allreal = [t for b0 in a.batches for t in real[b0]]
        alldec = [t for b0 in a.batches for t in store[(ip, ev, "shuffle", 0, b0)]]
        # scaffold from HALF the long tracks; the other half is the held-out positive reference
        sP, sV = _scaffold(allreal, half=0)
        heldout = [t for t in _split_half(allreal, PROD_MIN_LEN, 1)]
        coh = {}
        for lo, hi in ((5, 8), (8, 10), (10, 12), (12, 15)):
            coh[f"{lo}-{hi}"] = {
                "real": coherence(allreal, lo, hi, sP, sV),
                "decoy": coherence(alldec, lo, hi, sP, sV),
            }
        coh[">=15 (held out)"] = {
            "real": coherence(heldout, PROD_MIN_LEN, 10**6, sP, sV),
            "decoy": coherence(alldec, PROD_MIN_LEN, 10**6, sP, sV),
        }
        out["coherence"][key] = coh

    if not a.no_sim:
        print("\n  planted-crossing precision vs min_track_length (frozen gate):")
        out["sim"] = sim_precision_vs_minlen(MIN_LENS, gate=(None, None), n_seeds=a.sim_seeds)

    with open(OUTDIR / "length_sweep.json", "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    md = report(out)
    (OUTDIR / "length_sweep.md").write_text(md + "\n")
    print("\n" + md)


def report(out):
    lines = []
    A = lines.append

    A("### HEADLINE — yield vs purity of the temporal-persistence discriminator\n")
    A("`min_track_length` is an EMISSION predicate, not an association parameter: it appears in "
      "`tracking.py` only as `if length >= min_track_length: emit`, never in the Hungarian cost, "
      "the Kalman update, the gate, or the spawn logic. Lowering it therefore recovers exactly "
      "ZERO associations — every chain it admits was already formed and simply not printed. It is "
      "an output dial, and its yield must NOT be added to, or compared against, the 42.2% of "
      "detections the pipeline failed to associate. Those are different populations: one was "
      "never linked, the other was linked and withheld.\n")
    A("So the question is purity, not recovery. `purity(L) = 1 - N_null(>=L)/N_real(>=L)`, where "
      "the null is the same detections with frame order permuted within each acquisition — "
      "identical spatial statistics, no temporal correspondence. This is the calibration of "
      "temporal persistence as a bubble/noise discriminator, which is the only such discriminator "
      "available here without ground truth.\n")
    for key, rows in out.get("purity", {}).items():
        A(f"\n**gate {key}**\n")
        A("| min_len L | tracks emitted | null tracks | purity | linked% | null linked% | >=35 |")
        A("|---|---|---|---|---|---|---|")
        for r in rows:
            if r["L"] not in (2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 35):
                continue
            lf = f"{r['linked_frac']*100:.1f}%" if r["linked_frac"] is not None else "-"
            lfn = f"{r['linked_frac_null']*100:.1f}%" if r["linked_frac_null"] is not None else "-"
            g = r["n_ge35"] if r["n_ge35"] is not None else "-"
            A(f"| {r['L']} | {r['n_real']} | {r['n_null']:.0f} | {r['purity']*100:.1f}% | "
              f"{lf} | {lfn} | {g} |")

    A("\n### min_track_length x gate — pooled over the retracked batches\n")
    A("| gate | min_len | tracks | >=35 | linked dets | linked% | med len | mean len |")
    A("|---|---|---|---|---|---|---|---|")
    for key, ml in [(k.split("|")[0], int(k.split("|")[1])) for k in out["cells"]]:
        s = out["cells"][f"{key}|{ml}"]
        A(f"| {key} | {ml} | {s['n_tracks']} | {s['n_ge35']} | {s['n_linked']} | "
          f"{s['linked_frac']*100:.1f}% | {s['len_med']:.0f} | {s['len_mean']:.1f} |")

    A("\n### Frame-shuffle decoy — chance-chain false-discovery vs length floor\n")
    A("`decoy/real` at length L is the fraction of tracks surviving a floor of L that a "
      "time-destroyed copy of the SAME detections produces anyway. It is a direct false-discovery "
      "estimate for the added tracks. `reverse/real` is the positive control and should stay "
      "near 1.0 — time-reversed real data is still real data.\n")
    for key, s in out["survival"].items():
        A(f"\n**gate {key}**\n")
        A("| min_len | real tracks >=L | decoy tracks >=L | decoy/real | reverse/real |")
        A("|---|---|---|---|---|")
        for L in (2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 35):
            r, d, v = s["real"].get(str(L), s["real"].get(L, 0)), None, None
            d = s["shuffle"].get(str(L), s["shuffle"].get(L, 0))
            v = s["reverse"].get(str(L), s["reverse"].get(L, 0))
            A(f"| {L} | {r} | {d:.0f} | {(d/r if r else float('nan')):.3f} | "
              f"{(v/r if r else float('nan')):.3f} |")

    A("\n### Spatial coherence of tracks by length band\n")
    A("`dist` = median distance (mm) from the track's points to the nearest point of the long-track "
      "(>=15) vascular scaffold. `|cos|` = median alignment between the track's own direction and "
      "the scaffold's local mean direction. Real short tracks that are signal should match the "
      "long tracks on both; chance chains should be indistinguishable from the decoy's.\n")
    for key, c in out["coherence"].items():
        A(f"\n**gate {key}**\n")
        A("| length band | n real | real dist | real \\|cos\\| | n decoy | decoy dist | decoy \\|cos\\| |")
        A("|---|---|---|---|---|---|---|")
        for band, v in c.items():
            r, d = v["real"], v["decoy"]
            A(f"| {band} | {r['n']} | {r['dist_med']:.3f} | {r['cos_med']:.3f} | "
              f"{d['n']} | {d['dist_med']:.3f} | {d['cos_med']:.3f} |")

    if "sim" in out:
        A("\n### Planted-crossing link precision vs min_track_length (frozen gate)\n")
        A("| min_len | crossings | precision | recall |")
        A("|---|---|---|---|")
        for r in out["sim"]:
            A(f"| {r['min_len']} | {r['n_cross']} | {r['prec']:.3f} | {r['rec']:.3f} |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
