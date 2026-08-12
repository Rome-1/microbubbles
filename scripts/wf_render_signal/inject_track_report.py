"""Render EXP-B (bead mb-9u0) — the injection lateral:axial tables, beside the real-data one.

Reads the JSON written by scripts/modal/inject_track_app.py (fetch with
`modal volume get research microbubbles/inject_track.json -`) and emits the comparison that
settles which of (a)/(b)/(c) holds. The real-data column is transcribed from
docs/tracking-gate-censoring.md, "Approach to the wall", frozen gate, which is the table this
experiment exists to be compared against.

    python3 scripts/wf_render_signal/inject_track_report.py outputs/inject_track.json
"""
from __future__ import annotations

import json
import sys

# docs/tracking-gate-censoring.md, "Approach to the wall", gate = default (2 vox), 60 acquisitions.
REAL = {
    "0.00-0.25": {"lateral": 30143, "axial": 30861},
    "0.25-0.50": {"lateral": 16285, "axial": 17149},
    "0.50-0.75": {"lateral": 4557, "axial": 3792},
    "0.75-1.00": {"lateral": 2351, "axial": 1534},
}


def ratio(a, b):
    return a / b if b else float("nan")


def main(path):
    rep = json.load(open(path))
    T = rep["tables"]
    L = []
    A = L.append

    A("## EXP-B — lateral:axial, injection vs real data, in the same units\n")
    A(f"Source: `{rep['bf_path']}` acq {rep['acq']}, shape {rep['shape']}, "
      f"{rep.get('n_realizations', '?')} realizations per arm, filter `{rep['filter']}|{rep['norm']}` "
      f"at {rep['primary_density_per_frame']}/frame (threshold z={rep['threshold']:.3f} fixed on "
      f"the CLEAN volume for both arms).\n")
    A(f"Tracker gate derived from this volume's spacing: "
      f"{ {k: round(v, 2) for k, v in rep['gate_mms_from_spacing'].items()} } mm/s — "
      "the same 89.15 / 246.75 / 89.32 the real run walled off at.\n")

    A("\n### The headline comparison — per-step lateral:axial by band of the old gate\n")
    A("Real = production tracks over 60 acquisitions. Injection columns = steps of tracks "
      "attributed to injected bubbles, same tracker, same gate, same `min_track_length=15`.\n")
    A("| band (x old gate) | real lat:ax | inj lat:ax GAIN ON | inj lat:ax GAIN OFF |")
    A("|---|---|---|---|")
    for band in REAL:
        r = ratio(REAL[band]["lateral"], REAL[band]["axial"])
        on = T["gain_on"]["min15"]["per_band"].get(band, {})
        off = T["gain_off"]["min15"]["per_band"].get(band, {})
        A(f"| {band} | {r:.2f} | {on.get('ratio', float('nan')):.2f} | "
          f"{off.get('ratio', float('nan')):.2f} |")

    for arm, label in (("gain_on", "compounding gain ON (default)"),
                       ("gain_off", "compounding gain OFF (counterfactual)")):
        A(f"\n### Injection arm — {label}\n")
        A("| band | linked steps lateral | linked steps axial | lat:ax (min15) | lat:ax (min5) |")
        A("|---|---|---|---|---|")
        for band, e in T[arm]["min15"]["per_band"].items():
            e5 = T[arm]["min5"]["per_band"].get(band, {})
            A(f"| {band} | {e['lateral']} | {e['axial']} | {e.get('ratio', float('nan')):.2f} | "
              f"{e5.get('ratio', float('nan')):.2f} |")

    A("\n### Detection-level recovery (J3 units) from the same runs — the other half of the units fix\n")
    A("If the linked-step ratio and this ratio disagree, the difference is what LINKING adds.\n")
    A("| speed (mm/s) | band | GAIN ON lat | GAIN ON ax | GAIN ON lat:ax | GAIN OFF lat | "
      "GAIN OFF ax | GAIN OFF lat:ax |")
    A("|---|---|---|---|---|---|---|---|")
    for s in rep["speeds_mms"]:
        k = f"{s:g}"
        on = T["gain_on"]["detection"][k]
        off = T["gain_off"]["detection"][k]
        A(f"| {k} | {on['band']} | {on['lateral']:.3f} | {on['axial']:.3f} | "
          f"{on['ratio']:.2f} | {off['lateral']:.3f} | {off['axial']:.3f} | {off['ratio']:.2f} |")

    A("\n### Linked steps per speed (the raw counts behind the bands)\n")
    A("| speed | band | ON lat | ON ax | ON ratio | OFF lat | OFF ax | OFF ratio | "
      "compounding gain |")
    A("|---|---|---|---|---|---|---|---|---|")
    for s in rep["speeds_mms"]:
        k = f"{s:g}"
        on = T["gain_on"]["min15"]["per_speed"][k]
        off = T["gain_off"]["min15"]["per_speed"][k]
        g = rep["compounding_gain_at_speed"][k]
        A(f"| {k} | {on['band']} | {on['lateral']} | {on['axial']} | {on['ratio']:.2f} | "
          f"{off['lateral']} | {off['axial']} | {off['ratio']:.2f} | {g:.3f} |")
    return "\n".join(L)


if __name__ == "__main__":
    print(main(sys.argv[1] if len(sys.argv) > 1 else "outputs/inject_track.json"))
