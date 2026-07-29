"""Which acquisition (if any) do the reference's released detections correspond to? (mb-ivt)

The pristine upstream package at HEAD, run on the pristine upstream beamform of the public
sample, matches the reference's released acq-0 detections at 24.3% frame-agnostic against a
22.7% chance rate, and 0.2% per-frame at the best of 140 candidate window offsets. Before
concluding the released artifact is unreachable from the public inputs, rule out the boring
explanation: that their "acq 0" is not the raw file's acquisition 0.

Test: score the pristine detections against ALL 216 dissolved reference acquisition clouds
(`outputs/reference/dissolved/`, the reference's own linked points per acquisition) and report
the best matches. If some acquisition scores far above chance, this is an indexing offset. If
none does, the artifact does not correspond to any acquisition of the public data as processed
by the public code.

Positive control included: the reference's raw acq-0 detections vs the dissolved acq-0 cloud
must score ~1.0 (the dissolved points are a verified exact subset). If the control fails, the
comparison machinery is wrong, not the data.
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
    .add_local_dir("outputs/reference/dissolved", remote_path="/workspace/dissolved")
)

app = modal.App(f"{PROJECT}-acq-identity")


@app.function(image=image, timeout=3600, memory=65536, cpu=8.0,
              volumes={"/root/data": vol})
def identify(pkl: str = "determinism_postfix_a.pkl", match_mm: float = 0.3) -> dict:
    import glob
    import json
    import os
    import pickle

    import numpy as np
    from scipy.spatial import cKDTree

    ref = np.load("/workspace/refdet/acq0_reference_detections.npz")
    ref_pos = ref["positions_mm"].astype(np.float64)

    vol.reload()
    with open(f"{DATA_ROOT}/{pkl}", "rb") as fh:
        d = pickle.load(fh)
    det = d["detections"]
    A = np.asarray(det["acq_indices"], dtype=np.int64)
    P = np.asarray(det["positions_mm"], dtype=np.float64)[A == A.min()]
    tree = cKDTree(P)
    print(f"[pristine] {len(P)} detections", flush=True)

    sphere = 4.0 / 3.0 * np.pi * match_mm ** 3

    def score(target):
        """Fraction of `target` points with a pristine detection within match_mm,
        against the chance rate for this point count in this bounding box."""
        bbox = float(np.prod(target.max(0) - target.min(0)))
        dd, _ = tree.query(target)
        return (float((dd <= match_mm).mean()),
                float(1.0 - np.exp(-len(P) * sphere / max(bbox, 1e-9))))

    # positive control: the machinery must recover a known-exact subset
    ctrl = np.load("/workspace/dissolved/acq_0000.npz")
    ctrl_pos = ctrl["positions_mm"].astype(np.float64)
    ctrl_tree = cKDTree(ref_pos)
    dctl, _ = ctrl_tree.query(ctrl_pos)
    control = round(float((dctl <= 1e-6).mean()), 4)
    print(f"[control] dissolved acq-0 exactly inside released acq-0: {control}", flush=True)

    rows = []
    for path in sorted(glob.glob("/workspace/dissolved/acq_*.npz")):
        acq = int(os.path.basename(path).split("_")[1].split(".")[0])
        pos = np.load(path)["positions_mm"].astype(np.float64)
        hit, ch = score(pos)
        rows.append({"acq": acq, "n": int(len(pos)), "hit": round(hit, 3),
                     "chance": round(ch, 3), "excess": round(hit - ch, 3)})
    rows.sort(key=lambda r: -r["excess"])
    for r in rows[:8]:
        print(f"[match] acq {r['acq']:>3} n={r['n']:>5} hit={r['hit']:.3f} "
              f"chance={r['chance']:.3f} excess={r['excess']:+.3f}", flush=True)

    out = {"pkl": pkl, "match_mm": match_mm, "n_pristine": int(len(P)),
           "positive_control_exact_subset": control,
           "best": rows[:8],
           "excess_mean": round(float(np.mean([r["excess"] for r in rows])), 4),
           "excess_max": rows[0]["excess"]}
    with open(f"{DATA_ROOT}/acq_identity.json", "w") as fh:
        json.dump(out, fh, indent=2)
    vol.commit()
    print(json.dumps({k: out[k] for k in
                      ("positive_control_exact_subset", "excess_mean", "excess_max")},
                     indent=2), flush=True)
    return out
