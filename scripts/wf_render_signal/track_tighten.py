"""Close the remaining acq-0 gaps of the production tracker against the reference:
track count (249 vs 292), turning (11.3 vs 8.1 deg), coherence cl (0.65 vs 0.73).

Two disentanglements, both grounded in the data (not just "match the reference"):

  E1 TRACK COUNT — is the deficit the z>=p25 prefilter removing linkable detections, or the
     KF under-linking? Run reference-config with the prefilter OFF vs ON, and check whether the
     extra tracks are REAL (bootstrap-stable + physiological) or noise.

  E2 TURNING/cl — sweep the RTS process noise and compare RTS vs the reference's own Gaussian
     sigma=2 smoothing. Gate every candidate by the FILTERED one-step error (link-quality anchor
     that smoothing cannot fake): a turning drop that leaves filtered one-step flat is masking,
     not a real gain.

Fleet-safe: nice -15, single-thread, load-gated by the caller. No writes except stdout.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from render_flow_diversity import SU
from track_rts import track_kf_rts, tracks_rts, tracks_filt
from track_bubbles import (phys_plausibility, direction_agreement, onestep_consistency,
                           length_stats, load_acq0)
from flow_diversity import coherence_map, segdirs_from_tracks

REF = "outputs/reference/full_tracks_smoothed.pkl"
RCFG = dict(sigma_a=0.05, gate_chi2=9.0, max_step_scale=1.0, max_gap=3, min_len=5)


def acq0_dets(zpct=None):
    o = SU(open(REF, "rb")).load()
    d = o["detections"]
    P = np.asarray(d["positions_mm"], float); F = np.asarray(d["frame_indices"], int)
    Z = np.asarray(d["zscores"], float)
    if zpct is not None:
        keep = Z >= np.percentile(Z, zpct); P, F, Z = P[keep], F[keep], Z[keep]
    return P, F, Z


def gauss_smooth(tracks, sigma=2.0):
    from scipy.ndimage import gaussian_filter1d
    out = []
    for p, fr in tracks:
        if len(p) >= 3:
            p = np.column_stack([gaussian_filter1d(p[:, d], sigma, mode="nearest") for d in range(3)])
        out.append((p, fr))
    return out


def cl_of(tracks):
    m, d, _ = segdirs_from_tracks([{"positions": p, "acq_index": 0} for p, _ in tracks])
    cl, cn = coherence_map(m, d)
    return float(cl[cn >= 4].mean())


def bootstrap(P, F, drop=0.15, seed=0):
    from scipy.spatial import cKDTree
    full = tracks_rts(track_kf_rts(P, F, **RCFG))
    rng = np.random.default_rng(seed); keep = rng.random(len(P)) > drop
    boot = tracks_rts(track_kf_rts(P[keep], F[keep], **RCFG))
    if not full or not boot:
        return float("nan")
    tree = cKDTree(np.vstack([p for p, _ in boot]))
    return float(np.mean([np.mean(tree.query(p)[0] < 0.3) for p, _ in full]))


def summarize(tracks, P):
    st = length_stats(tracks); sp, ang = phys_plausibility(tracks)
    return (st["n"], st["mean"], st["mx"], st["pts"] / len(P) * 100,
            float(np.median(sp)), float(np.median(ang)), cl_of(tracks))


def main():
    P0, F0, I0, ref = load_acq0()
    rsp, rang = phys_plausibility([(p, np.arange(len(p))) for p in ref])
    print(f"REFERENCE: 292 tracks, mean 10.3, max 68, speed {np.median(rsp):.1f}, "
          f"turn {np.median(rang):.1f}, cl 0.73\n")

    # ---- E1: track count — prefilter OFF vs ON ----
    print("E1  TRACK COUNT: z-prefilter OFF vs ON (reference-config, RTS-smoothed)")
    print(f"    {'prefilter':<14} {'dets':>5} {'trk':>4} {'mean':>5} {'max':>4} {'link%':>6} "
          f"{'spMed':>6} {'turn':>5} {'cl':>5} {'boot':>5}")
    for tag, zpct in (("OFF (all)", None), ("ON (z>=p25)", 25)):
        P, F, Z = acq0_dets(zpct)
        tr = tracks_rts(track_kf_rts(P, F, **RCFG))
        n, mean, mx, lk, sp, tn, cl = summarize(tr, P)
        bs = bootstrap(P, F)
        print(f"    {tag:<14} {len(P):>5} {n:>4} {mean:>5.1f} {mx:>4} {lk:>6.1f} "
              f"{sp:>6.1f} {tn:>5.1f} {cl:>5.2f} {bs*100:>4.0f}%")

    # ---- E2: turning/cl — RTS sigma_a sweep + Gaussian sigma=2, anti-masking anchor ----
    print("\nE2  TURNING/cl: smoother comparison (prefilter OFF; filtered 1-step = anti-masking anchor)")
    print(f"    {'smoother':<20} {'turn':>5} {'cl':>5} {'filt-1step':>10}  (ref turn 8.1, cl 0.73)")
    P, F, Z = acq0_dets(None)
    for sa in (0.02, 0.03, 0.05, 0.08):
        recs = track_kf_rts(P, F, **{**RCFG, "sigma_a": sa})
        rts = tracks_rts(recs); filt = tracks_filt(recs)
        f1 = float(np.median(onestep_consistency(filt)))
        _, ang = phys_plausibility(rts)
        print(f"    RTS sigma_a={sa:<10.2f} {np.median(ang):>5.1f} {cl_of(rts):>5.2f} {f1:>10.3f}")
    # reference's own scheme: Gaussian sigma=2 on the filtered states
    recs = track_kf_rts(P, F, **RCFG); filt = tracks_filt(recs)
    for sig in (1.0, 2.0, 3.0):
        g = gauss_smooth(filt, sig); _, ang = phys_plausibility(g)
        f1 = float(np.median(onestep_consistency(filt)))
        print(f"    Gaussian sigma={sig:<8.1f} {np.median(ang):>5.1f} {cl_of(g):>5.2f} {f1:>10.3f}")


if __name__ == "__main__":
    main()
