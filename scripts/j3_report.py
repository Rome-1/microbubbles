#!/usr/bin/env python3
"""Render the J3 sweep JSON into the results tables (bead mb-fmj).

    modal volume get research microbubbles/j3_sweep.json -
    python3 scripts/j3_report.py j3_sweep.json

Prints, in order: the resolved SVD cutoffs per arm, the 18-row operating-point
table at the primary detection density, recovery stratified by speed / depth /
SNR for the status quo and the winner, and the false-alarm cross-check on each
null. Every recovery number is measured at MATCHED detection density on the
un-injected volume -- see ultratrace_ulm/opsweep.py for why nothing here is ever
compared at a matched threshold.
"""

from __future__ import annotations

import json
import sys


def _fmt(x, n=3):
    if x is None:
        return "-"
    try:
        f = float(x)
    except (TypeError, ValueError):
        return str(x)
    return "nan" if f != f else f"{f:.{n}f}"


def _gain_db(v_mms: float, lam: float = 0.8, fr: float = 222.43, n: int = 4) -> str:
    """Analytic N-angle compounding gain for AXIAL motion, in dB (see inject.py)."""
    import math

    dphi = 4 * math.pi * v_mms / (lam * n * fr)
    den = n * math.sin(dphi / 2)
    if abs(den) < 1e-12:
        return "0.0"
    g = abs(math.sin(n * dphi / 2) / den)
    return "-inf" if g < 1e-6 else f"{20 * math.log10(g):.1f}"


def main(path: str, density: str | None = None) -> None:
    rep = json.load(open(path))
    res = rep["results"]
    density = density or f"{rep['primary_density_per_frame']:g}"

    print(f"# J3 operating-point sweep\n")
    print(f"acq {rep['acq']} of `{rep['bf_path']}`, volume {tuple(rep['shape'])}, "
          f"{rep['elapsed_s']:.0f}s GPU\n")
    p = rep["psf"]
    print(f"**Measured PSF** ({p['n_patches']} isolated patches): FWHM "
          f"elev {p['fwhm_mm']['elev']:.2f} mm, z {p['fwhm_mm']['z']:.2f} mm, "
          f"x {p['fwhm_mm']['x']:.2f} mm; axial carrier {p['k_axial_rad_per_voxel']:.3f} rad/voxel "
          f"(pi = 3.142, the 2k carrier lambda/4 sampling predicts).\n")
    inj = rep["injection"]
    print(f"**Injection**: {inj['n_realizations']} realizations x "
          f"{inj['n_bubbles_per_realization']} bubbles, "
          f"{list(inj['bubble_frames_per_frame'].values())[0]:.0f} bubble-frames/frame added; "
          f"compounding gain applied: {inj['compounding_gain_applied']}.\n")

    print("## Resolved SVD cutoffs\n")
    print("| arm | low (modes removed) | high_remove (MP) | modes kept |")
    print("|---|---|---|---|")
    seen = set()
    for key, slot in res.items():
        f = slot["filter"]
        if f in seen:
            continue
        seen.add(f)
        c = slot["cutoffs"]
        print(f"| {f} | {c['low']} | {c['high_remove']} | {c['kept']} |")

    print(f"\n## Operating points at matched density = {density} detections/frame\n")
    print("| rank | filter | normalization | low | recovery | vs status quo | +-SE | depth non-unif. |")
    print("|---|---|---|---|---|---|---|---|")
    for i, row in enumerate(rep["ranking"]["rows"], 1):
        f, n = row["config"].split("|")
        print(f"| {i} | {f} | {n} | {row['low']} | {_fmt(row['recovery'])} | "
              f"{row['vs_status_quo']:+.3f} | {_fmt(row.get('vs_status_quo_se'))} | "
              f"{_fmt(row.get('depth_nonuniformity'))} |")

    best = rep["ranking"]["rows"][0]["config"]
    for key in ("rank24|per_elev", best):
        r = res[key]["recovery"].get(density)
        if not r:
            continue
        print(f"\n## Stratified recovery -- `{key}`"
              f"{' (STATUS QUO)' if key == 'rank24|per_elev' else ' (WINNER)'}\n")
        for axis, label in (("by_speed_mms", "speed (mm/s)"), ("by_z_band", "depth band"),
                            ("by_snr_db", "SNR (dB)"), ("by_direction", "direction")):
            if axis not in r:
                continue
            print(f"{label}: " + "  ".join(
                f"{k}={_fmt(v['recovery'], 3)}(n={v['n']})" for k, v in sorted(
                    r[axis].items(),
                    key=lambda kv: (float(kv[0]) if kv[0].replace(".", "").isdigit() else 0))))
        print(f"\ndepth non-uniformity (spread/mean) by SNR: " + "  ".join(
            f"{k}dB={_fmt(v)}" for k, v in sorted(
                r.get("depth_nonuniformity_by_snr", {}).items(), key=lambda kv: float(kv[0]))))

        cross = r.get("by_speed_x_direction")
        if cross:
            speeds = sorted({float(k) for v in cross.values() for k in v})
            print("\nrecovery by speed x direction (the compounding null is AXIAL only):\n")
            print("| direction | " + " | ".join(f"{s:g}" for s in speeds) + " |")
            print("|---" * (len(speeds) + 1) + "|")
            for d in sorted(cross):
                print(f"| {d} | " + " | ".join(
                    _fmt(cross[d].get(f"{s:g}", {}).get("recovery")) for s in speeds) + " |")
            print("| *predicted 4-angle gain (dB)* | " + " | ".join(
                _gain_db(s) for s in speeds) + " |")

        prof = r.get("depth_profile_by_snr", {})
        if prof:
            bands = sorted({int(b) for v in prof.values() for b in v})
            print("\nrecovery by depth band at fixed SNR:\n")
            print("| SNR (dB) | " + " | ".join(f"band {b}" for b in bands) + " |")
            print("|---" * (len(bands) + 1) + "|")
            for snr in sorted(prof, key=float):
                print(f"| {snr} | " + " | ".join(
                    _fmt(prof[snr].get(str(b), {}).get("recovery")) for b in bands) + " |")

    print("\n## False-alarm cross-check on null data\n")
    nulls = sorted({n for s in res.values() for n in s.get("false_alarms", {}).get(density, {})})
    print("| filter/norm | " + " | ".join(f"{n} (/frame)" for n in nulls) + " |")
    print("|---" * (len(nulls) + 1) + "|")
    for key, slot in sorted(res.items()):
        fa = slot.get("false_alarms", {}).get(density, {})
        if not fa:
            continue
        print(f"| {key} | " + " | ".join(_fmt(fa.get(n, {}).get("per_frame"), 2) for n in nulls) + " |")

    print("\nWhat each null preserves and breaks:\n")
    for k, v in rep["null_descriptions"].items():
        print(f"- **{k}**: {v}\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
