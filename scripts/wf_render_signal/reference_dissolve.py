"""Dissolve the reference's OWN linked tracks into per-acquisition point clouds.

CONTEXT (bead-worthy correction to a standing assumption): `outputs/reference/full_tracks_smoothed.pkl`
was treated as "acq-0 detections only" because its `detections` key IS acq-0-only (9,761 raw
detections). But its `tracks` key holds 50,456 tracks spanning ALL 216 acquisitions (acq_index
0-215, 504,090 points total) -- the reference's own linked output, with the reference's own
link labels. We do not have the reference's raw detections for acq 1-215 (those were never
released), but we DO have its LINKED subset for every acquisition. This script turns that into
a per-acq point cloud + link-label array so a held-out validation set can be built from it.

  positions_mm (M,3) | frame_in_acq (M,) 0..239 | intensities (M,) | track_id (M,) reference's own link id

*** IMPORTANT CAVEAT -- read before trusting any downstream precision/recall number ***
This cloud is ONLY the reference's LINKED subset: on acq-0 that is 3,017 of 9,761 raw
detections (30.9%), matching the doc's "~31% linked" figure. A tracker fed this cloud is
solving an EASIER association problem than real operation (no unlinkable noise detections
to reject, no need to discover the ~69% the reference itself declined to link). Any
precision/recall computed against this cloud is OPTIMISTIC relative to real operation on raw
per-acq detections we do not have. See scripts/wf_render_signal/reference_holdout_eval.py.

Frame mapping: verified empirically (not assumed) below -- absolute `frames` in the pickle
run 0..51839 (216 acqs x 240 frames_per_acq), contiguous per acq: frame_in_acq = absolute_frame
- acq_index * frames_per_acq, and this is asserted to land in [0, frames_per_acq) for every
single point before anything is cached.

CPU numpy only. Fleet-safe: nice -15, single-threaded, no plotting.
"""
from __future__ import annotations
import os, sys, argparse
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

REF = "outputs/reference/full_tracks_smoothed.pkl"
OUTDIR = "outputs/reference/dissolved"


def _load_pickle():
    from render_flow_diversity import SU
    return SU(open(REF, "rb")).load()


def dissolve(o=None):
    """Returns {acq: dict(positions_mm, frame_in_acq, intensities, track_id, track_length)},
    plus frames_per_acq. track_id is a per-acq-local integer (0..ntracks_in_acq-1); points
    from the same reference track are CONTIGUOUS in the arrays (they are concatenated
    track-by-track), which downstream code (reference_holdout_eval.py) relies on to recover
    true link edges as adjacent-index pairs sharing track_id."""
    if o is None:
        o = _load_pickle()
    fpa = int(o["frames_per_acq"])
    tracks = o["tracks"]
    by_acq: dict[int, list] = {}
    for t in tracks:
        by_acq.setdefault(int(t["acq_index"]), []).append(t)

    out = {}
    for acq, trs in sorted(by_acq.items()):
        pos_chunks, fia_chunks, inten_chunks, tid_chunks, lens = [], [], [], [], []
        for local_tid, t in enumerate(trs):
            p = np.asarray(t["positions"], np.float64)
            fr = np.asarray(t["frames"], np.int64)
            it = np.asarray(t["intensities"], np.float64)
            fia = fr - acq * fpa
            assert fia.min() >= 0 and fia.max() < fpa, (
                f"acq {acq}: frame_in_acq out of [0,{fpa}) range "
                f"(got {fia.min()}..{fia.max()}) -- frame-mapping assumption is WRONG")
            pos_chunks.append(p); fia_chunks.append(fia); inten_chunks.append(it)
            tid_chunks.append(np.full(len(p), local_tid, np.int32))
            lens.append(len(p))
        out[acq] = dict(
            positions_mm=np.vstack(pos_chunks).astype(np.float32),
            frame_in_acq=np.concatenate(fia_chunks).astype(np.int32),
            intensities=np.concatenate(inten_chunks).astype(np.float32),
            track_id=np.concatenate(tid_chunks),
            n_tracks=len(trs),
            track_lengths=np.array(lens, np.int32),
        )
    return out, fpa


def save(dissolved):
    os.makedirs(OUTDIR, exist_ok=True)
    for acq, d in sorted(dissolved.items()):
        path = f"{OUTDIR}/acq_{acq:04d}.npz"
        np.savez_compressed(path, positions_mm=d["positions_mm"], frame_in_acq=d["frame_in_acq"],
                            intensities=d["intensities"], track_id=d["track_id"])
    print(f"  wrote {len(dissolved)} acq files to {OUTDIR}/")


def load_dissolved(acq: int) -> dict:
    """Load one acq's cached npz (written by save()). Raises FileNotFoundError if not yet built."""
    path = f"{OUTDIR}/acq_{acq:04d}.npz"
    z = np.load(path)
    return {k: z[k] for k in z.files}


def summary_table(dissolved, fpa):
    print(f"\n{'acq':>4} {'points':>7} {'tracks':>7} {'mean_len':>9} {'med_len':>8} "
          f"{'frame_occ%':>11}")
    print("-" * 52)
    rows = []
    for acq, d in sorted(dissolved.items()):
        L = d["track_lengths"]
        occ = len(np.unique(d["frame_in_acq"])) / fpa * 100
        rows.append((acq, len(d["positions_mm"]), d["n_tracks"], L.mean(), np.median(L), occ))
    for acq, npts, ntr, ml, medl, occ in rows:
        print(f"{acq:>4} {npts:>7} {ntr:>7} {ml:>9.2f} {medl:>8.1f} {occ:>10.1f}%")
    print("-" * 52)
    npts_all = np.array([r[1] for r in rows]); ntr_all = np.array([r[2] for r in rows])
    print(f"TOTAL: {npts_all.sum()} points, {ntr_all.sum()} tracks over {len(rows)} acqs "
          f"(mean {npts_all.mean():.0f} pts/acq, {ntr_all.mean():.0f} tracks/acq)")


def verify_acq0_subset(dissolved, o):
    """The dissolved acq-0 linked cloud MUST be a subset of the 9,761 released raw detections
    (position match within tolerance). If not, the dissolve logic is wrong -- STOP."""
    from scipy.spatial import cKDTree
    d0 = dissolved[0]
    raw = np.asarray(o["detections"]["positions_mm"], np.float64)
    tree = cKDTree(raw)
    dist, _ = tree.query(d0["positions_mm"].astype(np.float64), k=1)
    tol = 1e-4
    match_rate = float((dist < tol).mean())
    print(f"\nACQ-0 SUBSET VERIFICATION: {len(d0['positions_mm'])} dissolved linked points vs "
          f"{len(raw)} released raw detections")
    print(f"  nearest-neighbor distance: max={dist.max():.6g} mm, mean={dist.mean():.6g} mm "
          f"(tol={tol} mm)")
    print(f"  match rate: {match_rate*100:.2f}%  "
          f"({'PASS -- dissolve verified as a true subset' if match_rate > 0.999 else 'FAIL -- dissolve logic is WRONG, STOP'})")
    print(f"  linked fraction acq-0: {len(d0['positions_mm'])}/{len(raw)} = "
          f"{len(d0['positions_mm'])/len(raw)*100:.1f}% (doc says ~31%)")
    return match_rate


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()
    print("Loading reference pickle...")
    o = _load_pickle()
    print(f"  n_acquisitions={o['n_acquisitions']} frames_per_acq={o['frames_per_acq']} "
          f"n_frames={o['n_frames']}  |  {len(o['tracks'])} tracks total")

    dissolved, fpa = dissolve(o)
    assert set(dissolved) == set(range(int(o["n_acquisitions"]))), \
        "not all 216 acquisitions present in dissolved tracks"

    summary_table(dissolved, fpa)
    match_rate = verify_acq0_subset(dissolved, o)

    if not a.no_save:
        save(dissolved)

    print("\nCAVEAT: this is the reference's LINKED subset only (~31% of raw detections on "
          "acq-0). Association on this cloud is an EASIER problem than real operation on raw "
          "per-acq detections (which we do not have for acq 1-215). Treat downstream "
          "precision/recall as optimistic. See module docstring.")
    return dissolved, match_rate


if __name__ == "__main__":
    main()
