"""Why does our detection set share ~1% of the reference's acq-0 detections? (mb-ivt)

`detector_ab_app.py` scored all four detector variants at ~1% recall / ~0% precision against
the reference's 9,761 acq-0 detections. That is chance: with ~1,130 detections per frame in a
~25,000 mm^3 volume, a 0.3 mm match radius covers ~0.5% of it by accident alone. A result at
chance is a registration failure, not a detector-quality measurement, so nothing about pooled
vs robust normalization can be concluded until this is explained.

Three candidate explanations, tested here in order of how cheap they are to rule out:

  GEOMETRY   the reference grid is (25, 154, 275) spanning x +-27.46, y +-6.66, z 10.0-40.72 mm;
             our beamformed shard is (25, 225, 378) -- a larger FOV at the same 0.2 mm pitch.
             If the two grids do not share an origin, every position is offset by a constant.
             Tested by (a) comparing per-axis percentiles, (b) restricting our detections to the
             reference's bounding box, and (c) searching for a single global translation that
             maximizes agreement.

  TIME       the window offset may be wrong, or our frame order may not correspond to theirs.
             Tested by scoring frame-agnostically: match each reference detection to ANY of our
             detections anywhere in the acquisition. If time is the only problem, this is high.

  DETECTION  our detections may simply not contain the reference's bubbles. This is the residual
             explanation, and only believable once the two above are excluded.

CPU-only, one acquisition, minutes.
"""

import modal

PROJECT = "microbubbles"
DATA_ROOT = f"/root/data/{PROJECT}"
SCRATCH = ("/home/rome/tmp/claude-1000/-home-rome-gt-microbubbles-crew-cajal/"
           "d06d6857-7ce7-4afc-9f8d-3f4a28ac48e3/scratchpad/refdet")

vol = modal.Volume.from_name("research")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("h5py==3.11.0", "numpy<2.0", "scipy==1.13.1")
    .add_local_dir("ultratrace_ulm", remote_path="/workspace/ultratrace_ulm")
    .add_local_dir(SCRATCH, remote_path="/workspace/refdet")
)

app = modal.App(f"{PROJECT}-detector-diag")


@app.function(image=image, timeout=3 * 3600, memory=131072, cpu=16.0,
              volumes={"/root/data": vol})
def diagnose(acq_h5: str = "beamformed/base60_refs/acq_0000.h5",
             frame_rate: float = 222.0, sigma_threshold: float = 2.0,
             min_distance: int = 2, smoothing_sigma: float = 1.0,
             window: int = 240, offset: int = 458) -> dict:
    import json
    import sys

    sys.path.insert(0, "/workspace")
    import numpy as np
    from scipy.ndimage import gaussian_filter, maximum_filter
    from scipy.spatial import cKDTree

    from ultratrace_ulm.h5_io import open_h5, acq_keys, load_compound, grid_arrays
    from ultratrace_ulm.svd import filter_svd_3d
    from ultratrace_ulm.tracking import indices_to_mm, subpixel_localize_3d

    ref = np.load("/workspace/refdet/acq0_reference_detections.npz")
    ref_pos = ref["positions_mm"].astype(np.float64)
    ref_idx = ref["indices"].astype(np.float64)
    ref_frame = ref["frame_indices"].astype(np.int64)
    ref_frame -= ref_frame.min()

    vol.reload()
    with open_h5(f"{DATA_ROOT}/{acq_h5}") as h5:
        aid = acq_keys(h5)[0]
        gx, gy, gz = grid_arrays(h5, aid)
        comp = load_compound(h5, aid)[offset:offset + window]

    out: dict = {"acq": str(aid), "offset": offset, "window": window}
    out["our_grid_shape"] = [int(v) for v in gx.shape]
    out["our_grid_range_mm"] = {ax: [round(float(g.min()), 2), round(float(g.max()), 2)]
                                for ax, g in (("x", gx), ("y", gy), ("z", gz))}
    out["ref_pos_range_mm"] = {ax: [round(float(ref_pos[:, i].min()), 2),
                                    round(float(ref_pos[:, i].max()), 2)]
                               for i, ax in enumerate("xyz")}
    print(json.dumps({k: out[k] for k in ("our_grid_shape", "our_grid_range_mm",
                                          "ref_pos_range_mm")}, indent=2), flush=True)

    mag = np.abs(filter_svd_3d(comp, method="adaptive",
                               frame_rate_hz=frame_rate)).astype(np.float32)
    del comp
    sm = gaussian_filter(mag, sigma=(0, 0, smoothing_sigma, smoothing_sigma))
    n_elev = sm.shape[1]
    mean = np.zeros(n_elev, np.float32)
    std = np.ones(n_elev, np.float32)
    for e in range(n_elev):
        v = sm[:, e]
        vals = v[v > 0]
        if vals.size:
            mean[e] = vals.mean()
            s = vals.std()
            std[e] = s if s > 1e-10 else 1.0
    z = (sm - mean[None, :, None, None]) / (std[None, :, None, None] + 1e-10)
    del sm
    fs = 2 * int(min_distance) + 1
    pk = (z == maximum_filter(z, size=(1, fs, fs, fs))) & (z > float(sigma_threshold))
    coords = np.array(np.where(pk))
    zs = z[pk].astype(np.float32)
    del z, pk

    pos, frame, zsc, idxs = [], [], [], []
    for f in np.unique(coords[0]):
        m = coords[0] == f
        pix = coords[1:, m].T.astype(np.int32)
        sub = subpixel_localize_3d(mag[f], pix, method="centroid", window_size=5)
        pos.append(indices_to_mm(sub, gx, gy, gz))
        idxs.append(sub)
        frame.append(np.full(int(m.sum()), int(f), np.int64))
        zsc.append(zs[m])
    P = np.concatenate(pos).astype(np.float64)
    IDX = np.concatenate(idxs).astype(np.float64)
    F = np.concatenate(frame)
    Z = np.concatenate(zsc)
    del mag
    print(f"[ours] {len(P)} detections over {window} frames", flush=True)

    pct = lambda a, i: [round(float(np.percentile(a[:, i], q)), 2) for q in (1, 50, 99)]
    out["our_pos_pct_1_50_99"] = {ax: pct(P, i) for i, ax in enumerate("xyz")}
    out["ref_pos_pct_1_50_99"] = {ax: pct(ref_pos, i) for i, ax in enumerate("xyz")}
    out["our_idx_pct_1_50_99"] = {ax: pct(IDX, i) for i, ax in enumerate(("elev", "z", "x"))}
    out["ref_idx_pct_1_50_99"] = {ax: pct(ref_idx, i) for i, ax in enumerate(("elev", "z", "x"))}

    # --- TIME: frame-agnostic match (any of ours, anywhere in the window) --------
    tree_all = cKDTree(P)
    d_any, _ = tree_all.query(ref_pos)
    out["frame_agnostic"] = {
        "median_nn_mm": round(float(np.median(d_any)), 3),
        "within_0.3mm": round(float((d_any <= 0.3).mean()), 3),
        "within_1.0mm": round(float((d_any <= 1.0).mean()), 3),
    }

    # --- GEOMETRY: does a single global translation line the clouds up? ----------
    best = {"shift_mm": [0.0, 0.0, 0.0], "within_0.3mm": float((d_any <= 0.3).mean())}
    for dx in np.arange(-6, 6.01, 1.0):
        for dy in np.arange(-6, 6.01, 1.0):
            for dz in np.arange(-6, 6.01, 1.0):
                d, _ = tree_all.query(ref_pos + np.array([dx, dy, dz]))
                hit = float((d <= 0.3).mean())
                if hit > best["within_0.3mm"]:
                    best = {"shift_mm": [float(dx), float(dy), float(dz)],
                            "within_0.3mm": hit}
    out["best_global_shift"] = {k: (round(v, 3) if isinstance(v, float) else v)
                                for k, v in best.items()}

    # --- GEOMETRY: restrict ours to the reference's bounding box -----------------
    lo = ref_pos.min(0) - 0.5
    hi = ref_pos.max(0) + 0.5
    inbox = np.all((P >= lo) & (P <= hi), axis=1)
    out["ours_inside_ref_bbox"] = round(float(inbox.mean()), 3)

    # --- DETECTION: same-frame match, and the brightest-only subset --------------
    same_hit = np.zeros(len(ref_pos), bool)
    for fr in range(window):
        rm, om = ref_frame == fr, F == fr
        if not rm.any() or not om.any():
            continue
        d, _ = cKDTree(P[om]).query(ref_pos[rm])
        same_hit[np.where(rm)[0]] = d <= 0.3
    out["same_frame_recall"] = round(float(same_hit.mean()), 3)

    keep = np.argsort(Z)[::-1][:len(ref_pos)]
    d_top, _ = cKDTree(P[keep]).query(ref_pos)
    out["top_by_z_frame_agnostic_within_0.3mm"] = round(float((d_top <= 0.3).mean()), 3)

    with open(f"{DATA_ROOT}/detector_diag.json", "w") as fh:
        json.dump(out, fh, indent=2)
    vol.commit()
    print(json.dumps(out, indent=2), flush=True)
    return out
