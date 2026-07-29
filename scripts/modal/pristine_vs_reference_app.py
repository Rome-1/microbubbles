"""Does upstream's OWN current pipeline reproduce upstream's OWN released detections? (mb-ivt)

The registration diagnostic ruled out geometry: our grid covers the reference's range on every
axis, 97.8% of our detections fall inside the reference's bounding box, and no global
translation improves agreement beyond chance. Yet agreement IS chance -- 1.2% same-frame, and
the 70% "frame-agnostic" figure is exactly what 264k random points in a ~25,000 mm^3 volume
would score at a 0.3 mm radius (1 - exp(-nv/V) = 0.70).

That leaves two possibilities, and they have very different consequences:

  (1) our detection path is finding different objects than theirs, or
  (2) the released acq-0 detections do not correspond to what the shipped pipeline produces
      on the public data at all -- in which case the whole "reproduce the reference" program
      has been calibrating against an artifact it cannot reach.

This distinguishes them without touching our code. Today's determinism runs already produced
detections from the PRISTINE upstream package (post-#4 HEAD) run on the PRISTINE upstream
beamform of the same public sample: `determinism_postfix_a.pkl` (43,039 detections). Score
those against the reference's released 9,761 the same way. If upstream's own code also lands
at chance against upstream's own artifact, the answer is (2) and it is not our bug.
"""

import modal

PROJECT = "microbubbles"
DATA_ROOT = f"/root/data/{PROJECT}"
SCRATCH = ("/home/rome/tmp/claude-1000/-home-rome-gt-microbubbles-crew-cajal/"
           "d06d6857-7ce7-4afc-9f8d-3f4a28ac48e3/scratchpad/refdet")

vol = modal.Volume.from_name("research")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy<2.0", "scipy==1.13.1")
    .add_local_dir(SCRATCH, remote_path="/workspace/refdet")
)

app = modal.App(f"{PROJECT}-pristine-vs-reference")


@app.function(image=image, timeout=3600, memory=65536, cpu=8.0,
              volumes={"/root/data": vol})
def compare(pkl: str = "determinism_postfix_a.pkl", window: int = 240,
            match_mm: float = 0.3) -> dict:
    import json
    import pickle

    import numpy as np
    from scipy.spatial import cKDTree

    ref = np.load("/workspace/refdet/acq0_reference_detections.npz")
    ref_pos = ref["positions_mm"].astype(np.float64)
    ref_frame = ref["frame_indices"].astype(np.int64)
    ref_frame -= ref_frame.min()

    vol.reload()
    with open(f"{DATA_ROOT}/{pkl}", "rb") as fh:
        d = pickle.load(fh)
    det = d["detections"]
    P = np.asarray(det["positions_mm"], dtype=np.float64)
    F = np.asarray(det["frame_indices"], dtype=np.int64)
    A = np.asarray(det["acq_indices"], dtype=np.int64)
    m = A == A.min()
    P, F = P[m], F[m] - F[m].min()
    n_frames = int(F.max()) + 1
    print(f"[pristine] {len(P)} detections over {n_frames} frames "
          f"({len(P)/max(1,n_frames):.1f}/frame)", flush=True)
    print(f"[ref]      {len(ref_pos)} detections over {int(ref_frame.max())+1} frames "
          f"({len(ref_pos)/240:.1f}/frame)", flush=True)

    vol_mm3 = float(np.prod(ref_pos.max(0) - ref_pos.min(0)))
    sphere = 4.0 / 3.0 * np.pi * match_mm ** 3

    def chance(n):
        return float(1.0 - np.exp(-n * sphere / vol_mm3))

    tree = cKDTree(P)
    d_any, _ = tree.query(ref_pos)
    out = {
        "pkl": pkl, "match_mm": match_mm, "bbox_mm3": round(vol_mm3, 1),
        "n_pristine": int(len(P)), "n_reference": int(len(ref_pos)),
        "pristine_frames": n_frames,
        "frame_agnostic_within": round(float((d_any <= match_mm).mean()), 3),
        "frame_agnostic_chance": round(chance(len(P)), 3),
        "median_nn_mm": round(float(np.median(d_any)), 3),
    }

    # best 240-frame window, scored per-frame, offset searched like the reference's
    # `selected_sequence` start is unknown
    best = {"offset": 0, "recall": -1.0}
    for off in range(0, max(1, n_frames - window + 1), 5):
        hit = 0
        tot = 0
        for fr in range(0, window, 4):
            rm = ref_frame == fr
            om = F == off + fr
            if not rm.any() or not om.any():
                continue
            dd, _ = cKDTree(P[om]).query(ref_pos[rm])
            hit += int((dd <= match_mm).sum())
            tot += int(rm.sum())
        r = hit / max(1, tot)
        if r > best["recall"]:
            best = {"offset": off, "recall": round(float(r), 3)}
    out["best_window"] = best
    per_frame_n = len(P) / max(1, n_frames)
    out["per_frame_chance"] = round(chance(per_frame_n), 3)

    with open(f"{DATA_ROOT}/pristine_vs_reference.json", "w") as fh:
        json.dump(out, fh, indent=2)
    vol.commit()
    print(json.dumps(out, indent=2), flush=True)
    return out
