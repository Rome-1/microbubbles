"""Detector A/B on acq-0 against the reference's OWN released detections (mb-ivt follow-on).

Why this exists. Every accuracy number in `docs/bubble-tracking-acq0.md` was computed on
detections the reference produced; our detector has never been scored against anything. The
crossed experiment in bead `mb-bfl` says that is where the whole gap lives: our tracker on the
reference's acq-0 detections beats the reference tracker (8 vs 6 tracks >= 35 frames), while the
same tracker on OUR detections for the same acquisition yields zero tracks >= 35. Tracker fixed,
input swapped, result collapses. So measure the input.

The reference's acq-0 detections are a usable proxy ground truth: 9,761 detections over
240 frames, 100% frame occupancy, 40.7/frame (min 15, max 130), and a minimum z-score of 4.90
despite `sigma_threshold: 2.0` -- i.e. their knee filter, not the 2 sigma, is the operative
threshold. Ours flood: 7% of frames hold 68% of detections, 61% mean occupancy (mb-5gv).

Two hypotheses are tested, crossed:

  NORMALIZATION  pooled (shipped: per-elevation mean/std over ALL frames of the acquisition)
                 vs per-frame robust (per-frame, per-elevation median/MAD).
                 A pooled statistic cannot survive a contrast bolus: it sits at the
                 acquisition's average brightness, so dim frames fall entirely below threshold
                 and bright frames clear it everywhere. That single mechanism predicts all
                 three symptoms -- low occupancy, flood frames, and in-frame crowding.

  FRAME WINDOW   our ~700 beamformed frames vs the reference's 240-frame `selected_sequence`.
                 The reference runs the SAME pooled code without flooding, which it can only do
                 because it normalizes over a near-stationary 240-frame window. The window
                 offset is not assumed -- it is found by maximizing spatial agreement with the
                 reference detections over all candidate offsets.

Scoring against the reference detections, per variant: detection-level precision/recall at a
0.3 mm match radius, count ratio, per-frame occupancy and burstiness, plus a DENSITY-MATCHED
score (keep our top-N per frame by z, N = the reference's own count for that frame) which
separates ranking quality from threshold choice.

CPU-only, one acquisition, minutes. Verify teardown after the run.
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

app = modal.App(f"{PROJECT}-detector-ab")

MATCH_MM = 0.3


@app.function(image=image, timeout=4 * 3600, memory=131072, cpu=16.0,
              volumes={"/root/data": vol})
def detector_ab(acq_h5: str = "beamformed/base60_refs/acq_0000.h5",
                frame_rate: float = 222.0, sigma_threshold: float = 2.0,
                min_distance: int = 2, smoothing_sigma: float = 1.0,
                window: int = 240) -> dict:
    import json
    import sys
    import time

    sys.path.insert(0, "/workspace")
    import numpy as np
    from scipy.ndimage import gaussian_filter, maximum_filter
    from scipy.spatial import cKDTree

    from ultratrace_ulm.h5_io import open_h5, acq_keys, load_compound, grid_arrays
    from ultratrace_ulm.svd import filter_svd_3d
    from ultratrace_ulm.tracking import indices_to_mm, subpixel_localize_3d

    ref = np.load("/workspace/refdet/acq0_reference_detections.npz")
    ref_pos = ref["positions_mm"].astype(np.float64)
    ref_frame = ref["frame_indices"].astype(np.int64)
    ref_frame -= ref_frame.min()
    ref_counts = np.bincount(ref_frame, minlength=window)
    print(f"[ref] {len(ref_pos)} detections over {ref_frame.max()+1} frames, "
          f"{ref_counts.mean():.1f}/frame", flush=True)

    vol.reload()
    with open_h5(f"{DATA_ROOT}/{acq_h5}") as h5:
        aid = acq_keys(h5)[0]
        gx, gy, gz = grid_arrays(h5, aid)
        comp = load_compound(h5, aid)
    n_all = int(comp.shape[0])
    print(f"[in] acq={aid} {comp.shape}", flush=True)

    # ---------- detector variants ------------------------------------------------
    def zscore_pooled(mag):
        """Shipped statistic: per-elevation mean/std pooled over every frame."""
        sm = gaussian_filter(mag, sigma=(0, 0, smoothing_sigma, smoothing_sigma)) \
            if smoothing_sigma > 0 else mag
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
        return (sm - mean[None, :, None, None]) / (std[None, :, None, None] + 1e-10)

    def zscore_robust(mag):
        """Per-FRAME, per-elevation median/MAD -- immune to a per-frame gain excursion."""
        sm = gaussian_filter(mag, sigma=(0, 0, smoothing_sigma, smoothing_sigma)) \
            if smoothing_sigma > 0 else mag
        z = np.empty_like(sm)
        for f in range(sm.shape[0]):
            for e in range(sm.shape[1]):
                v = sm[f, e]
                pos = v[v > 0]
                if pos.size < 16:
                    z[f, e] = 0.0
                    continue
                med = np.median(pos)
                mad = np.median(np.abs(pos - med))
                scale = 1.4826 * mad if mad > 1e-10 else (pos.std() or 1.0)
                z[f, e] = (v - med) / scale
        return z

    def peaks(z):
        fs = 2 * int(min_distance) + 1
        mx = maximum_filter(z, size=(1, fs, fs, fs))
        pk = (z == mx) & (z > float(sigma_threshold))
        c = np.array(np.where(pk))
        return c, z[pk].astype(np.float32)

    def localize(mag, coords, zs):
        """Sub-voxel localize + mm, matching the shipped detection path."""
        out_pos, out_frame, out_z = [], [], []
        for f in np.unique(coords[0]):
            m = coords[0] == f
            pix = coords[1:, m].T.astype(np.int32)
            sub = subpixel_localize_3d(mag[f], pix, method="centroid", window_size=5)
            out_pos.append(indices_to_mm(sub, gx, gy, gz))
            out_frame.append(np.full(m.sum(), f, np.int64))
            out_z.append(zs[m])
        if not out_pos:
            return (np.zeros((0, 3), np.float64), np.zeros(0, np.int64), np.zeros(0, np.float32))
        return (np.concatenate(out_pos).astype(np.float64),
                np.concatenate(out_frame), np.concatenate(out_z))

    # ---------- scoring ----------------------------------------------------------
    def score(pos, frame, zs, offset, tag):
        """Detection-level precision/recall vs the reference, at `offset`, plus a
        density-matched variant that keeps our top-N per frame by z."""
        sel = (frame >= offset) & (frame < offset + window)
        p, f, z = pos[sel], frame[sel] - offset, zs[sel]
        counts = np.bincount(f, minlength=window)
        rec_hit = np.zeros(len(ref_pos), bool)
        ours_hit = np.zeros(len(p), bool)
        top_hit_ref = np.zeros(len(ref_pos), bool)
        top_n = 0
        for fr in range(window):
            rm, om = ref_frame == fr, f == fr
            if not rm.any() or not om.any():
                continue
            tree = cKDTree(p[om])
            d, _ = tree.query(ref_pos[rm])
            rec_hit[np.where(rm)[0]] = d <= MATCH_MM
            d2, _ = cKDTree(ref_pos[rm]).query(p[om])
            ours_hit[np.where(om)[0]] = d2 <= MATCH_MM
            # density-matched: our top-N by z for this frame, N = reference's count
            n = int(ref_counts[fr])
            idx = np.where(om)[0]
            keep = idx[np.argsort(z[idx])[::-1][:n]]
            top_n += len(keep)
            dt, _ = cKDTree(p[keep]).query(ref_pos[rm])
            top_hit_ref[np.where(rm)[0]] = dt <= MATCH_MM
        rep = {
            "variant": tag, "offset": int(offset), "n_ours": int(len(p)),
            "dets_per_frame": round(float(counts.mean()), 1),
            "max_per_frame": int(counts.max()) if len(counts) else 0,
            "occupancy": round(float((counts > 0).mean()), 3),
            "count_ratio_vs_ref": round(float(len(p) / max(1, len(ref_pos))), 2),
            "recall": round(float(rec_hit.mean()), 3),
            "precision": round(float(ours_hit.mean()), 3),
            "density_matched_recall": round(float(top_hit_ref.mean()), 3),
            "density_matched_n": int(top_n),
        }
        print(f"[score] {rep}", flush=True)
        return rep

    def best_offset(pos, frame):
        """Find the 240-frame window that best matches the reference -- no assumption
        about where `selected_sequence` starts."""
        best, best_hits = 0, -1
        for off in range(0, max(1, n_all - window + 1), 10):
            hits = 0
            for fr in range(0, window, 8):          # subsample frames for speed
                rm = ref_frame == fr
                om = frame == off + fr
                if not rm.any() or not om.any():
                    continue
                d, _ = cKDTree(pos[om]).query(ref_pos[rm])
                hits += int((d <= MATCH_MM).sum())
            if hits > best_hits:
                best, best_hits = off, hits
        for off in range(max(0, best - 9), min(n_all - window, best + 10)):
            hits = 0
            for fr in range(0, window, 8):
                rm = ref_frame == fr
                om = frame == off + fr
                if not rm.any() or not om.any():
                    continue
                d, _ = cKDTree(pos[om]).query(ref_pos[rm])
                hits += int((d <= MATCH_MM).sum())
            if hits > best_hits:
                best, best_hits = off, hits
        print(f"[align] best offset {best} ({best_hits} subsampled matches)", flush=True)
        return best, best_hits

    results = []

    # ---- A: filter the FULL 700-frame acquisition (what we do today) -------------
    t0 = time.time()
    mag_full = np.abs(filter_svd_3d(comp, method="adaptive", frame_rate_hz=frame_rate)) \
        .astype(np.float32)
    print(f"[svd] full {mag_full.shape} in {time.time()-t0:.0f}s", flush=True)

    for tag, fn in (("pooled_700", zscore_pooled), ("robust_700", zscore_robust)):
        t = time.time()
        z = fn(mag_full)
        coords, zs = peaks(z)
        del z
        pos, frame, zsc = localize(mag_full, coords, zs)
        counts_all = np.bincount(frame, minlength=n_all)
        print(f"[{tag}] {len(pos)} dets, {counts_all.mean():.0f}/frame, "
              f"max {counts_all.max()}, occupancy {(counts_all>0).mean():.2f} "
              f"({time.time()-t:.0f}s)", flush=True)
        off, hits = best_offset(pos, frame)
        rep = score(pos, frame, zsc, off, tag)
        rep["full_acq_dets"] = int(len(pos))
        rep["full_acq_occupancy"] = round(float((counts_all > 0).mean()), 3)
        rep["full_acq_max_per_frame"] = int(counts_all.max())
        rep["align_hits"] = int(hits)
        results.append(rep)

    aligned = max(results, key=lambda r: r["align_hits"])["offset"]
    del mag_full

    # ---- B: filter ONLY the aligned 240-frame window (what the reference does) ---
    t0 = time.time()
    mag_win = np.abs(filter_svd_3d(comp[aligned:aligned + window], method="adaptive",
                                   frame_rate_hz=frame_rate)).astype(np.float32)
    print(f"[svd] window@{aligned} {mag_win.shape} in {time.time()-t0:.0f}s", flush=True)

    for tag, fn in (("pooled_240", zscore_pooled), ("robust_240", zscore_robust)):
        z = fn(mag_win)
        coords, zs = peaks(z)
        del z
        pos, frame, zsc = localize(mag_win, coords, zs)
        rep = score(pos, frame, zsc, 0, tag)
        rep["window_start"] = int(aligned)
        results.append(rep)

    out = {"acq": str(aid), "match_mm": MATCH_MM, "window": window,
           "n_reference_detections": int(len(ref_pos)),
           "reference_dets_per_frame": round(float(ref_counts.mean()), 1),
           "aligned_offset": int(aligned), "variants": results}
    with open(f"{DATA_ROOT}/detector_ab.json", "w") as fh:
        json.dump(out, fh, indent=2)
    vol.commit()
    print(json.dumps(out, indent=2), flush=True)
    return out
