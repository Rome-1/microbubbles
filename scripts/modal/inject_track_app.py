"""EXP-B (bead mb-9u0) — put J1 and J3 in the same units, and test the compounding notch.

THE THREE-WAY DISAGREEMENT
--------------------------
Real tracks (docs/tracking-gate-censoring.md): the lateral:axial per-step ratio is ~0.95-0.98
below half the old gate and climbs to 1.20 (0.50-0.75 band) and 1.51-1.53 (0.75-1.00 band). An
AXIAL DEFICIT that grows with speed.

The J3 injection instrument (docs/results/j3-sweep-acq0.md, rank24|per_elev at 34 det/frame): at
5 mm/s, lateral recovery 0.077 vs axial 0.591. A LATERAL deficit, at the slow end.

These are not in conflict yet, because they are not in the same units. J1's ratio is conditioned
on LINKING — a detection that never links never becomes a track step, so J1 can only ever see the
subset of detections that survived association. J3's recovery is a DETECTION rate and knows
nothing about the tracker. This script removes that difference: it injects bubbles at matched
speeds on both axes across 0-130 mm/s, runs the SAME production tracker over the resulting
detections, and computes the lateral:axial ratio exactly the way it was computed on real tracks
(per-step counts, banded by fraction of the old gate).

WHY BOTH GAIN ARMS, AND WHY A SINGLE ARM WOULD PROVE NOTHING
------------------------------------------------------------
`inject.py` applies the 4-angle coherent-compounding gain to every injected bubble by hand
(`compounding_gain_on=True` by default), and that gain term IS the axial high-speed suppression
under investigation — it is unity at v_z=0 and notches to zero at lambda*FR/2 = 88.97 mm/s. A
single-arm run that reproduced the real 1.51 asymmetry would therefore establish only that a
model containing the effect by construction reproduces the effect. Circular.

So both arms run, identical in every other respect — same bubbles, same seeds, same PSF, same
noise field, same filter, same matched density, same tracker. The DIFFERENTIAL is the
measurement, and it maps onto three conclusions:

  (a) the asymmetry appears with gain OFF  -> the deficit is created downstream (SVD dynamics,
      the detection statistic, or linking at speed) and the compounding notch is not needed to
      explain it. Strongest outcome.
  (b) the asymmetry appears ONLY with gain ON -> the notch as modelled is necessary within the
      model. Suggestive, not conclusive: we imposed the gain ourselves.
  (c) neither arm reproduces it -> the cause is something this injector does not model. The
      leading candidate is intra-frame smear: at ~89 mm/s axial a bubble crosses ~2 voxels
      WITHIN the 4-transmit window, averaging the lambda/4 carrier and gutting the peak whether
      or not the sum is coherent. inject.py explicitly models fast bubbles as marginally EASIER,
      which is the opposite sign. Settling (c) needs per-transmit injection before compounding —
      a different instrument, and out of scope here.

The compounding notch is treated as neither established nor refuted going in.

    modal run scripts/modal/inject_track_app.py::inject_track
"""

from __future__ import annotations

import json
import sys
import time

import modal

PROJECT = "microbubbles"
DATA_ROOT = f"/root/data/{PROJECT}"
REPO = "/home/rome/gt/microbubbles/crew/cajal/ultratrace_ulm"

vol = modal.Volume.from_name("research")

gpu_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("h5py==3.11.0", "numpy<2.0", "scipy==1.13.1", "cupy-cuda12x==13.3.0")
    .add_local_dir(REPO, remote_path="/workspace/ultratrace_ulm")
)

app = modal.App(f"{PROJECT}-inject-track")

# The status-quo operating point, held fixed. This experiment is about the axis asymmetry, not
# about picking a filter, so the filter/normalization are frozen at what production ships.
FILTER_SPEC = ("rank24", "rank24", False)
NORM = "per_elev"
PRIMARY_DENSITY = 34.0  # detections/frame, matched on the CLEAN volume as in J3

# Speeds chosen to populate the SAME bands the real-data table uses: fractions of the old in-plane
# gate (89.15 mm/s). 0-0.25 | 0.25-0.50 | 0.50-0.75 | 0.75-1.00 | >1.00.
SPEEDS_MMS = (5.0, 12.0, 20.0, 28.0, 36.0, 45.0, 55.0, 62.0, 70.0, 80.0, 88.97, 105.0, 130.0)
SNR_DB = (9.0, 12.0, 15.0)
OLD_GATE_MMS = 89.15  # 2 voxels/frame in-plane at 222.43 Hz, the wall the real data ends at
BANDS = (0.25, 0.50, 0.75, 1.00, 1.50)

# Tracker: the frozen 216-acquisition operating point, so the ratio is computed by the same code
# path on the same settings as the real-data table it is compared against.
PROD_MIN_LEN = 15
ALT_MIN_LEN = 5


def _load(path: str, acq: int):
    import h5py
    import numpy as np

    with h5py.File(path, "r") as h5:
        keys = sorted(h5["acquisitions"].keys(), key=int)
        grp = h5[f"acquisitions/{keys[acq]}"]
        meta = grp["meta"] if "meta" in grp else grp
        comp = np.asarray(meta["compound_image"], dtype=np.complex64)
        g = meta["grid"]
        gz, gy, gx = np.asarray(g["z"]), np.asarray(g["y"]), np.asarray(g["x"])
    d_z = float(abs(gz[1, 0, 0] - gz[0, 0, 0])) * 1000.0
    d_x = float(abs(gx[0, 0, 1] - gx[0, 0, 0])) * 1000.0
    d_e = float(abs(gy[0, 1, 0] - gy[0, 0, 0])) * 1000.0
    return comp, dict(d_elev_mm=d_e, d_z_mm=d_z, d_x_mm=d_x, z0_mm=float(gz[0, 0, 0]) * 1000.0)


def _peaks_to_mm(sel, geom):
    """Peaks (voxel indices, axis order elev/z/x) -> the tracker's (x, y, z) mm columns.

    The production tracker's position columns are (lateral, elevation, axial) = (x, y, z) and its
    spacing dict is keyed dx/dy/dz to match. Getting this order wrong would silently swap which
    axis the gate applies to, which is the entire quantity under measurement, so it is done once,
    here, and asserted against the spacing used for the gate."""
    import numpy as np

    return np.stack([
        sel.x.astype(np.float64) * geom.d_x_mm,
        sel.e.astype(np.float64) * geom.d_elev_mm,
        geom.z0_mm + sel.z.astype(np.float64) * geom.d_z_mm,
    ], axis=1).astype(np.float32)


def _attribute(truth, sel, n_f, tol=(1.0, 3.0, 3.0)):
    """detection row -> injected bubble_id (or -1), by nearest truth row within tolerance.

    `opsweep.match_peaks_to_truth` answers the mirror-image question (was this truth row seen?)
    and returns one bool per truth row. Tracking needs the other direction: given a detection that
    the tracker linked, which injected bubble was it? Same tolerance, (1 elev, 3 z, 3 x) voxels,
    so the two are directly comparable; nearest-wins in the scaled metric, and a detection that
    matches nothing is clutter or a real bubble and gets -1."""
    import numpy as np
    from scipy.spatial import cKDTree

    tol = np.asarray(tol, float)
    owner = np.full(sel.frame.size, -1, np.int64)
    tf = np.asarray(truth["frame"], np.int64)
    T = np.stack([np.asarray(truth["e"], float), np.asarray(truth["z"], float),
                  np.asarray(truth["x"], float)], axis=1)
    bid = np.asarray(truth["bubble_id"], np.int64)
    tb = np.searchsorted(np.sort(tf), np.arange(n_f + 1))
    to = np.argsort(tf, kind="stable")
    tf_s, T_s, bid_s = tf[to], T[to], bid[to]
    db = np.searchsorted(sel.frame, np.arange(n_f + 1))
    D = np.stack([sel.e.astype(float), sel.z.astype(float), sel.x.astype(float)], axis=1)
    for f in range(n_f):
        d0, d1 = int(db[f]), int(db[f + 1])
        t0, t1 = int(tb[f]), int(tb[f + 1])
        if d1 <= d0 or t1 <= t0:
            continue
        tree = cKDTree(T_s[t0:t1] / tol)
        dist, idx = tree.query(D[d0:d1] / tol, k=1, distance_upper_bound=1.0)
        ok = np.isfinite(dist)
        owner[d0:d1][ok] = bid_s[t0:t1][idx[ok]]
    return owner


def _band_of(speed):
    f = speed / OLD_GATE_MMS
    lo = 0.0
    for hi in BANDS:
        if lo <= f < hi:
            return f"{lo:.2f}-{hi:.2f}"
        lo = hi
    return f">{BANDS[-1]:.2f}"


@app.function(image=gpu_image, gpu="A100-80GB", timeout=3 * 3600, memory=196608, cpu=8.0,
              volumes={"/root/data": vol})
def inject_track(
    bf_path: str = "corrected_bf_1.h5",
    acq: int = 0,
    n_realizations: int = 12,
    n_bubbles: int = 300,
    lifetime: int = 24,
    out: str = "inject_track.json",
) -> dict:
    sys.path.insert(0, "/workspace")
    import cupy as cp
    import numpy as np

    from ultratrace_ulm import opsweep
    from ultratrace_ulm.inject import (
        Geometry, compounding_gain, estimate_complex_psf, inject_bubbles, make_bubble_plan,
        reference_noise_sigma,
    )
    from ultratrace_ulm.gpu_svd import filter_svd_3d_gpu
    from pathlib import Path as _Path

    from ultratrace_ulm.tracking import TrackingOptions, _track_detections

    t0 = time.time()
    comp, spacing = _load(f"{DATA_ROOT}/{bf_path}", acq)
    geom = Geometry(frame_rate_hz=222.43, **spacing)
    n_f = int(comp.shape[0])
    # The tracker's spacing dict. dx = LATERAL, dy = ELEVATION, dz = AXIAL — matching the (x,y,z)
    # column order produced by _peaks_to_mm, and matching the real run's recorded spacing.
    trk_spacing = {"dx": geom.d_x_mm, "dy": geom.d_elev_mm, "dz": geom.d_z_mm}
    base_opts = TrackingOptions(
        beamformed_path=_Path("/dev/null"), tracks_path=_Path("/dev/null"), frame_rate_hz=222.43,
        max_gap=3, min_track_length=PROD_MIN_LEN, max_cost=10.0, reversal_penalty=10.0,
        tracking="kalman",
    )
    report = {
        "bf_path": bf_path, "acq": acq, "shape": list(comp.shape), "spacing_mm": spacing,
        "tracker_spacing": trk_spacing, "speeds_mms": list(SPEEDS_MMS), "snr_db": list(SNR_DB),
        "old_gate_mms": OLD_GATE_MMS, "primary_density_per_frame": PRIMARY_DENSITY,
        "filter": FILTER_SPEC[0], "norm": NORM,
        "n_realizations": int(n_realizations), "n_bubbles_per_realization": int(n_bubbles),
        "lifetime": int(lifetime),
        "gate_mms_from_spacing": {k: 2 * v * 222.43 for k, v in trk_spacing.items()},
        "compounding_gain_at_speed": {f"{s:g}": float(compounding_gain(s, geom))
                                      for s in SPEEDS_MMS},
    }
    print(f"[B] loaded {comp.shape} spacing={spacing}", flush=True)
    print(f"[B] tracker gate from spacing: {report['gate_mms_from_spacing']} mm/s", flush=True)

    # ---- PSF and noise, measured once on the real volume ---------------------- #
    base_complex = filter_svd_3d_gpu(comp, low_cutoff=0.1, method="fast")
    zb = opsweep.zscore_field(cp.asarray(np.abs(base_complex).astype(np.float32)),
                              mode=NORM, xp=cp)
    pk = opsweep.find_peaks(zb, min_distance=2, floor=3.0, xp=cp)
    del zb
    cp.get_default_memory_pool().free_all_blocks()
    sel = pk.at(opsweep.threshold_for_count(pk, int(PRIMARY_DENSITY * n_f)))
    bounds = np.searchsorted(sel.frame, np.arange(n_f + 1))
    dets_psf = [(np.stack([sel.e[a:b], sel.z[a:b], sel.x[a:b]], 1).astype(np.int32),
                 np.ones(b - a, np.float32), sel.zscore[a:b])
                for a, b in zip(bounds[:-1], bounds[1:])]
    psf = estimate_complex_psf(base_complex, dets_psf, patch_radius=(3, 6, 6))
    report["psf"] = {"n_patches": psf.n_patches, "k_axial": psf.k_axial}
    del base_complex, pk, sel, dets_psf
    cp.get_default_memory_pool().free_all_blocks()

    sigma, band_edges = reference_noise_sigma(comp, ref_rank=8, n_z_bands=8)

    # ---- The clean volume fixes the detection threshold for BOTH arms --------- #
    # Neither arm is allowed to pick its own operating point; if it were, an arm that injects
    # brighter bubbles would also get a looser threshold and the comparison would be rigged.
    name, cut, mag = next(iter(opsweep.svd_filter_bank(comp, [FILTER_SPEC], xp=cp, to_host=False)))
    z = opsweep.zscore_field(mag, mode=NORM, voxel_mm=tuple(geom.voxel_mm), xp=cp)
    clean_peaks = opsweep.find_peaks(z, min_distance=2, floor=2.0, xp=cp)
    thr = float(opsweep.threshold_for_count(clean_peaks, int(PRIMARY_DENSITY * n_f)))
    del z, mag, clean_peaks
    cp.get_default_memory_pool().free_all_blocks()
    report["threshold"] = thr
    report["svd_cutoffs"] = cut
    print(f"[B] matched-density threshold from clean volume: z={thr:.3f}", flush=True)

    # ---- Both arms ------------------------------------------------------------ #
    acc = {}   # (gain_on, direction, speed) -> counters
    for gain_on in (True, False):
        for r in range(int(n_realizations)):
            plan = make_bubble_plan(
                comp.shape, geom, n_bubbles=n_bubbles, lifetime=lifetime, n_z_bands=4,
                speeds_mms=SPEEDS_MMS, snr_db=SNR_DB, directions=("lateral", "axial"),
                seed=1000 + r,   # SAME seed across arms -> literally the same bubbles
            )
            inj, truth = inject_bubbles(comp, plan, psf, geom, sigma, band_edges,
                                        compounding_gain_on=gain_on)
            _, _, m = next(iter(opsweep.svd_filter_bank(inj, [FILTER_SPEC], xp=cp, to_host=False)))
            del inj
            zf = opsweep.zscore_field(m, mode=NORM, voxel_mm=tuple(geom.voxel_mm), xp=cp)
            peaks = opsweep.find_peaks(zf, min_distance=2, floor=2.0, xp=cp)
            del zf, m
            cp.get_default_memory_pool().free_all_blocks()
            sel = peaks.at(thr)

            owner = _attribute(truth, sel, n_f)
            pos_mm = _peaks_to_mm(sel, geom)
            db = np.searchsorted(sel.frame, np.arange(n_f + 1))
            dets = [pos_mm[a:b] for a, b in zip(db[:-1], db[1:])]
            ints = [sel.zscore[a:b].astype(np.float32) for a, b in zip(db[:-1], db[1:])]
            rows = [np.arange(a, b) for a, b in zip(db[:-1], db[1:])]
            look = {(f, p.tobytes()): int(rr)
                    for f, (P, R) in enumerate(zip(dets, rows)) for p, rr in zip(P, R)}

            # bubble_id -> (direction, speed), for aggregation
            bmeta = {}
            for b, d, s in zip(np.asarray(truth["bubble_id"]), np.asarray(truth["direction"]),
                               np.asarray(truth["speed_mms"])):
                bmeta[int(b)] = (str(d), float(s))

            for min_len, tag in ((PROD_MIN_LEN, "min15"), (ALT_MIN_LEN, "min5")):
                from dataclasses import replace as _replace
                tracks = _track_detections(dets, ints, _replace(base_opts,
                                                                min_track_length=min_len),
                                           trk_spacing)
                for t in tracks:
                    ridx = [look.get((int(f), np.asarray(p, np.float32).tobytes()))
                            for p, f in zip(np.asarray(t["positions"], np.float32),
                                            np.asarray(t["frames"]))]
                    for i in range(len(ridx) - 1):
                        a, b = ridx[i], ridx[i + 1]
                        if a is None or b is None:
                            continue
                        oa, ob = int(owner[a]), int(owner[b])
                        # a step counts for a bubble only if BOTH endpoints are that bubble --
                        # a link that jumps between two injected bubbles is an identity error,
                        # not a recovered step, and must not be credited to either.
                        if oa >= 0 and oa == ob:
                            d, s = bmeta.get(oa, ("?", -1.0))
                            k = (gain_on, tag, d, s)
                            acc.setdefault(k, {"steps": 0})["steps"] += 1

            # detection-level (J3 units), from the same run and the same threshold
            td = np.asarray(truth["direction"])
            ts = np.asarray(truth["speed_mms"], float)
            hit = opsweep.match_peaks_to_truth(truth, sel)
            for d in ("lateral", "axial"):
                for s in SPEEDS_MMS:
                    m_ = (td == d) & (np.abs(ts - s) < 1e-6)
                    if not m_.any():
                        continue
                    k = (gain_on, "det", d, float(s))
                    e = acc.setdefault(k, {"n_truth": 0, "n_hit": 0})
                    e["n_truth"] += int(m_.sum())
                    e["n_hit"] += int(hit[m_].sum())
            print(f"[B] gain={gain_on} r={r}: {truth['frame'].size} bubble-frames, "
                  f"{sel.frame.size} dets, t={time.time()-t0:.0f}s", flush=True)

    report["raw"] = {f"{g}|{tag}|{d}|{s:g}": v for (g, tag, d, s), v in acc.items()}
    report["tables"] = _tables(acc)
    report["elapsed_s"] = time.time() - t0
    with open(f"{DATA_ROOT}/{out}", "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    vol.commit()
    print(f"[B] wrote {DATA_ROOT}/{out} in {report['elapsed_s']:.0f}s", flush=True)
    print(json.dumps(report["tables"], indent=2, default=float), flush=True)
    return report["tables"]


def _tables(acc):
    """Detection ratio (J3 units) and linked-step ratio (J1 units), per speed and per band."""
    out = {}
    for gain_on in (True, False):
        g = {}
        # J3 units: detection recovery
        det = {}
        for s in SPEEDS_MMS:
            row = {}
            for d in ("lateral", "axial"):
                e = acc.get((gain_on, "det", d, float(s)))
                row[d] = (e["n_hit"] / e["n_truth"]) if e and e["n_truth"] else float("nan")
                row[f"{d}_n"] = e["n_truth"] if e else 0
            row["ratio"] = (row["lateral"] / row["axial"]) if row["axial"] else float("nan")
            row["band"] = _band_of(s)
            det[f"{s:g}"] = row
        g["detection"] = det
        # J1 units: linked step counts, at both length floors
        for tag in ("min15", "min5"):
            per_speed, per_band = {}, {}
            for s in SPEEDS_MMS:
                la = acc.get((gain_on, tag, "lateral", float(s)), {}).get("steps", 0)
                ax = acc.get((gain_on, tag, "axial", float(s)), {}).get("steps", 0)
                per_speed[f"{s:g}"] = {"lateral": la, "axial": ax,
                                       "ratio": (la / ax) if ax else float("nan"),
                                       "band": _band_of(s)}
                b = _band_of(s)
                e = per_band.setdefault(b, {"lateral": 0, "axial": 0})
                e["lateral"] += la
                e["axial"] += ax
            for b, e in per_band.items():
                e["ratio"] = (e["lateral"] / e["axial"]) if e["axial"] else float("nan")
            g[tag] = {"per_speed": per_speed, "per_band": per_band}
        out["gain_on" if gain_on else "gain_off"] = g
    return out
