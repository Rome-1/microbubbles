"""TEMPORAL tracking on acq-0 raw detections — prototype the reference's ACTUAL method
(link each bubble into a track over frames) on OUR data, to test whether temporal
denoising recovers the reference's diverse short-track look (orientation coherence
cl ~0.53) that our spatial velocity-field averaging destroys (cl ~0.99).

Root cause (docs/flow-diversity-gap.md §2, §6): build_field averages velocity to ONE
vector per voxel -> crossings averaged away -> cl 0.99. Spatial multi-vector fix reached
cl 0.83 but stalled because spatial denoising MERGES near-crossings. The reference reaches
cl 0.53 by TEMPORAL denoising: a track follows ONE bubble, so it preserves diversity. This
script tests whether temporal linking is even viable at our concentration.

Method: frame-to-frame nearest-neighbour linking with velocity (Kalman-ish) gating.
  - dt = 1/222.43 s; max bubble speed 38 mm/s -> max displacement/frame = 0.171 mm.
  - Predict each active track forward by its EMA velocity; gate candidates within
    GATE_FACTOR * 0.171 mm (scaled by frame gap for coasting).
  - Global per-frame assignment via Hungarian (scipy.linear_sum_assignment) on the
    gated cost matrix; unmatched detections seed new tracks; tracks coast up to
    MAX_GAP missed frames before closing.
  - Filter to tracks of >= MIN_LEN detections (mirrors the reference discarding ~69%).

Measure: coherence cl on the resulting tracks (flow_diversity.coherence_map +
segdirs_from_tracks) vs reference tracks (cl~0.53) and our field streamlines (cl~0.99);
track count, length distribution, % detections linked, within-frame NN margin.

Render: renders/tracking_vs_reference.png (my acq-0 tracks vs reference acq-0 tracks,
direction-hue via render_flow_diversity.panel).

Fleet-safe: numpy/scipy/matplotlib(Agg) only, single process, run niced.
"""
from __future__ import annotations
import sys, os, pickle, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flow_diversity import coherence_map, segdirs_from_tracks
from render_flow_diversity import SU, panel
from tractography_field import GS, SP, ORG

REF = "outputs/reference/full_tracks_smoothed.pkl"
FPS = 222.43                       # Hz (from reference acquisition)
DT = 1.0 / FPS                     # s / frame
VMAX = 38.0                        # mm/s  (reference bubble speed range 0..38)
STEP_MM = VMAX * DT                # 0.171 mm : max displacement per frame
GATE_FACTOR = 2.0                  # link gate = GATE_FACTOR * STEP_MM per frame gap
GATE = GATE_FACTOR * STEP_MM       # 0.342 mm
MAX_GAP = 2                        # frames a track may coast unmatched before closing
EMA = 0.6                          # velocity smoothing (inst vs history)
MIN_LEN = 4                        # min detections/track kept (>=3 segments)


# ----------------------------------------------------------------------------- load
def load_detections():
    o = SU(open(REF, "rb")).load()
    det = o["detections"]
    pos = np.asarray(det["positions_mm"], float)          # (N,3) mm  (x, elev, z)
    fr = np.asarray(det["frame_indices"], int)
    acq = np.asarray(det["acq_indices"], int)
    m = acq == 0
    return pos[m], fr[m]


def frame_groups(pos, fr):
    order = np.argsort(fr, kind="stable")
    pos, fr = pos[order], fr[order]
    frames = np.unique(fr)
    return {int(f): pos[fr == f] for f in frames}, frames


# ------------------------------------------------------------------- trackability
def nn_margin(groups):
    """Within-frame nearest-neighbour distance distribution: the linking ambiguity.
    NN >> gate => frame-to-frame links are locally unambiguous."""
    nn = []
    cand = []
    for f, P in groups.items():
        if len(P) < 2:
            continue
        t = cKDTree(P)
        d, _ = t.query(P, k=2)
        nn.append(d[:, 1])
        # how many detections fall within one gate step of each detection (competition)
        cand.append(np.array([len(t.query_ball_point(p, GATE)) - 1 for p in P]))
    nn = np.concatenate(nn); cand = np.concatenate(cand)
    return nn, cand


# ------------------------------------------------------------------------ tracker
class Track:
    __slots__ = ("pos", "frames", "vel", "miss")
    def __init__(self, p, f):
        self.pos = [p]; self.frames = [f]; self.vel = np.zeros(3); self.miss = 0
    def predict(self, f):
        return self.pos[-1] + self.vel * (f - self.frames[-1])
    def extend(self, p, f):
        gap = f - self.frames[-1]
        inst = (p - self.pos[-1]) / gap
        self.vel = EMA * inst + (1 - EMA) * self.vel if len(self.pos) > 1 else inst
        self.pos.append(p); self.frames.append(f); self.miss = 0


def track(groups, frames):
    active, done = [], []
    for f in frames:
        P = groups[int(f)]
        if not active:
            active = [Track(p, int(f)) for p in P]
            continue
        preds = np.array([tr.predict(f) for tr in active])       # (A,3)
        gaps = np.array([f - tr.frames[-1] for tr in active])    # (A,)
        # cost = distance; disallow beyond gate*gap
        D = np.linalg.norm(preds[:, None, :] - P[None, :, :], axis=2)   # (A,B)
        gate = GATE * gaps[:, None]
        BIG = 1e6
        cost = np.where(D <= gate, D, BIG)
        used_b = np.zeros(len(P), bool)
        matched_rows = set()
        if len(active) and len(P):
            ri, ci = linear_sum_assignment(cost)
            for i, j in zip(ri, ci):
                if cost[i, j] < BIG:
                    active[i].extend(P[j], int(f)); used_b[j] = True; matched_rows.add(i)
        # unmatched active -> miss/coast; unmatched detections -> new tracks
        still = []
        for i, tr in enumerate(active):
            if i in matched_rows:
                still.append(tr)
            else:
                tr.miss += 1
                (still if tr.miss <= MAX_GAP else done).append(tr)
        active = still
        for j in range(len(P)):
            if not used_b[j]:
                active.append(Track(P[j], int(f)))
    done.extend(active)
    return done


# ------------------------------------------------------------------------ measure
def tracks_to_dicts(tracks, min_len, parity_by_index=True):
    out = []
    for i, tr in enumerate(tracks):
        if len(tr.pos) >= min_len:
            out.append({"positions": np.array(tr.pos),
                        "acq_index": i if parity_by_index else 0})
    return out


def cl_stats(tracks_dicts, want_splithalf=False):
    mids, dirs, par = segdirs_from_tracks(tracks_dicts)
    res = {}
    for tag, ip in (("3D", False), ("xz", True)):
        cl, cnt = coherence_map(mids, dirs, inplane=ip)
        occ = cnt >= 4
        res[tag] = {"mean_cl": float(cl[occ].mean()),
                    "crossing_frac": float((cl[occ] < 0.4).mean()),
                    "occ_vox": int(occ.sum())}
        if want_splithalf:
            clo, cno = coherence_map(mids[par == 0], dirs[par == 0], inplane=ip)
            cle, cne = coherence_map(mids[par == 1], dirs[par == 1], inplane=ip)
            both = (cno >= 4) & (cne >= 4)
            if both.sum() > 20:
                res[tag]["splithalf_r"] = float(np.corrcoef(clo[both], cle[both])[0, 1])
    return res


# ------------------------------------------------------------------------- render
def render(my_polys, ref_polys, path):
    fig, ax = plt.subplots(1, 2, figsize=(15, 6.2), facecolor="black")
    panel(ax[0], my_polys,
          f"OUR temporal tracks, acq-0  ({len(my_polys)}, len>={MIN_LEN})\n"
          f"nearest-neighbour + velocity gating · hue = x-z flow direction")
    panel(ax[1], ref_polys,
          f"Aleph reference tracks, acq-0  ({len(ref_polys)}, len>={MIN_LEN})\n"
          f"the reference's actual temporal method · hue = x-z flow direction")
    fig.suptitle("Temporal tracking on OUR acq-0 detections vs the reference's acq-0 tracks",
                 color="white", fontsize=12.5)
    os.makedirs("renders", exist_ok=True)
    fig.savefig(path, dpi=140, facecolor="black", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- main
def main():
    print(f"gate = {GATE_FACTOR} x {STEP_MM:.4f} mm = {GATE:.4f} mm/frame ; MAX_GAP={MAX_GAP} ; MIN_LEN={MIN_LEN}")
    pos, fr = load_detections()
    groups, frames = frame_groups(pos, fr)
    N = len(pos)
    print(f"\n[data] acq-0 detections={N}  frames={len(frames)} ({frames.min()}..{frames.max()})  "
          f"dets/frame mean={N/len(frames):.1f}")

    # trackability: within-frame NN margin vs the 0.17 mm/frame link budget
    nn, cand = nn_margin(groups)
    print(f"[trackability] within-frame NN dist mm: median={np.median(nn):.3f} "
          f"p10={np.percentile(nn,10):.3f} p25={np.percentile(nn,25):.3f}  "
          f"(link budget/frame={STEP_MM:.3f}mm, gate={GATE:.3f}mm)")
    print(f"[trackability] competitors within one gate of a detection: "
          f"mean={cand.mean():.2f} frac_with_>=1={np.mean(cand>=1):.3f}")

    # run the tracker
    tracks = track(groups, frames)
    lens = np.array([len(t.pos) for t in tracks])
    kept = lens >= MIN_LEN
    linked_pts = int(lens[kept].sum())
    print(f"\n[tracks] total={len(tracks)}  kept(len>={MIN_LEN})={int(kept.sum())}  "
          f"singletons(len=1)={int((lens==1).sum())}")
    print(f"[tracks] length (detections/track): kept mean={lens[kept].mean():.1f} "
          f"median={int(np.median(lens[kept]))} max={int(lens.max())}  "
          f"p90={int(np.percentile(lens[kept],90))}")
    print(f"[tracks] detections linked into kept tracks: {linked_pts}/{N} = {linked_pts/N:.1%}  "
          f"(=> discarded {1-linked_pts/N:.1%}; reference discards ~69%)")

    my_dicts = tracks_to_dicts(tracks, MIN_LEN, parity_by_index=True)
    my = cl_stats(my_dicts, want_splithalf=True)
    print(f"\n[cl OUR temporal tracks acq-0] "
          f"3D mean_cl={my['3D']['mean_cl']:.3f} crossing={my['3D']['crossing_frac']:.3f} "
          f"occ={my['3D']['occ_vox']}  |  "
          f"xz mean_cl={my['xz']['mean_cl']:.3f} crossing={my['xz']['crossing_frac']:.3f}"
          + (f"  split-half(track) r_3D={my['3D'].get('splithalf_r', float('nan')):.3f}"
             if 'splithalf_r' in my['3D'] else ""))

    # reference tracks: acq-0 only (fair, same-data) AND all-216 (reproduce doc 0.53)
    o = SU(open(REF, "rb")).load()
    rtr = o["tracks_smoothed"]
    ref0 = [t for t in rtr if int(t["acq_index"]) == 0]
    ref0_dicts = [{"positions": np.asarray(t["positions"], float), "acq_index": 0}
                  for t in ref0 if len(t["positions"]) >= MIN_LEN]
    rmA, rdA, rpA = segdirs_from_tracks(rtr)
    refall = {}
    for tag, ip in (("3D", False), ("xz", True)):
        cl, cnt = coherence_map(rmA, rdA, inplane=ip); occ = cnt >= 4
        refall[tag] = {"mean_cl": float(cl[occ].mean()),
                       "crossing_frac": float((cl[occ] < 0.4).mean())}
    ref0m = cl_stats(ref0_dicts)
    print(f"[cl REFERENCE acq-0]           "
          f"3D mean_cl={ref0m['3D']['mean_cl']:.3f} crossing={ref0m['3D']['crossing_frac']:.3f} "
          f"occ={ref0m['3D']['occ_vox']}  |  xz mean_cl={ref0m['xz']['mean_cl']:.3f}")
    print(f"[cl REFERENCE all-216acq]      "
          f"3D mean_cl={refall['3D']['mean_cl']:.3f} crossing={refall['3D']['crossing_frac']:.3f}  "
          f"|  xz mean_cl={refall['xz']['mean_cl']:.3f}   (doc target 0.53)")
    print(f"[cl OUR field streamlines]     3D mean_cl~0.99 crossing~0.00  "
          f"(established pipeline, docs/flow-diversity-gap.md §5-6)")
    print(f"[cl spatial multi-vector fix]  3D mean_cl~0.83 crossing~0.05  (doc §6, the bar to beat)")

    # render (cap polyline counts for weight)
    rng = np.random.default_rng(0)
    my_polys = [d["positions"] for d in my_dicts]
    ref_polys = [d["positions"] for d in ref0_dicts]
    def cap(polys, n):
        return polys if len(polys) <= n else [polys[i] for i in rng.choice(len(polys), n, replace=False)]
    render(cap(my_polys, 6000), cap(ref_polys, 6000), "renders/tracking_vs_reference.png")
    print("\nwrote renders/tracking_vs_reference.png")

    # machine-readable summary
    summary = {
        "n_detections": N, "n_frames": int(len(frames)), "dets_per_frame": round(N/len(frames), 1),
        "nn_median_mm": round(float(np.median(nn)), 4), "nn_p10_mm": round(float(np.percentile(nn, 10)), 4),
        "link_budget_mm": round(STEP_MM, 4), "gate_mm": round(GATE, 4),
        "competitors_mean": round(float(cand.mean()), 3),
        "n_tracks_total": len(tracks), "n_tracks_kept": int(kept.sum()),
        "track_len_mean": round(float(lens[kept].mean()), 2), "track_len_max": int(lens.max()),
        "pct_detections_linked": round(linked_pts / N, 4),
        "cl_ours_3D": round(my["3D"]["mean_cl"], 3), "cl_ours_xz": round(my["xz"]["mean_cl"], 3),
        "crossing_ours_3D": round(my["3D"]["crossing_frac"], 3),
        "cl_ref_acq0_3D": round(ref0m["3D"]["mean_cl"], 3),
        "cl_ref_all_3D": round(refall["3D"]["mean_cl"], 3),
        "cl_field_3D": 0.99, "cl_multivector_3D": 0.83,
    }
    print("\nSUMMARY_JSON " + json.dumps(summary))


if __name__ == "__main__":
    main()
