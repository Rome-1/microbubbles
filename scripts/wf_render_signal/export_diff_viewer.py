"""Exporter for the 3D+time strategy-DIFF viewer (mb-y8i/diff-viewer).

Emits, under scripts/wf_render_signal/diff_viewer/data/, for each reconstruction
STRATEGY (gauss / graph[mb-ki9 win] / graph+incomp / raw):
  <id>.lines.bin  — v4 streamlines: 8 float32/vertex [x,y,z, speed_mm_frame, dirx,diry,dirz, conf01]
  <id>.grid.bin   — dense grid summary: occupancy(u8) + dir(f32*3) + speed(f32) + confidence(f32)
plus av.grid.bin (uint8 artery/vein labels, field-level) and manifest.json.

Design: docs/diff-viewer-spec.md (Opus-designed). The viewer classifies A-only /
B-only / shared per vertex at runtime from the occupancy grids (works for any pair).
Fleet-safe: numpy/scipy only, NO chrome. Run niced.
"""
from __future__ import annotations
import sys, os, json, struct
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from tractography_field import build_field, smooth_field, tractogram, GS, SP, ORG
from tractography_pde import regularize, streamlines

FD = "outputs/reference/wf/r2/signal/velocity-field/"
AV = "outputs/reference/wf/r2/signal/artery-vein/av_labels.npy"
OUT = "scripts/wf_render_signal/diff_viewer/data"
FRAME_HZ = 222.43
LINES_MAGIC, GRID_MAGIC = 0x554C4D54, 0x554C4D47
NSEED = 3500

STRATEGIES = [
    ("graph",        "Graph-Laplacian along-vessel (validated win)", "#20e0d0", False,
     lambda V, C: regularize(V, C, mode="graph", gamma=1.5, iters=200, dt=0.12, eps_iso=0.2)),
    ("gauss",        "Isotropic Gaussian (baseline)",                "#ff8a3d", False,
     lambda V, C: smooth_field(V, C, (2.5, 1.4, 2.5))),
    ("graph_incomp", "Graph + incompressibility (uses elevation)",   "#c78bff", True,
     lambda V, C: regularize(V, C, mode="graph+incomp", gamma=1.5, iters=200, dt=0.12, eps_iso=0.2, kappa=0.5)),
    ("raw",          "Raw unregularized field",                      "#8a8a8a", False,
     lambda V, C: (V, C)),
]


def vidx(p):
    """mm point(s) -> integer voxel index/indices, clipped to grid."""
    return np.clip(np.floor((p - ORG) / SP).astype(int), 0, np.array(GS) - 1)


def write_lines_bin(path, lines, Cr):
    cmax = float(np.percentile(Cr[Cr > 0], 99)) + 1e-9
    n = len(lines)
    total = int(sum(len(p) for p, _ in lines))
    allpts = np.concatenate([p for p, _ in lines]) if lines else np.zeros((0, 3))
    bmin = allpts.min(0) if total else np.zeros(3); bmax = allpts.max(0) if total else np.zeros(3)
    maxspd = float(max((s.max() for _, s in lines), default=0.0))
    with open(path, "wb") as f:
        hdr = struct.pack("<IIIIf", LINES_MAGIC, 4, n, total, maxspd)
        hdr += struct.pack("<6f", *bmin, *bmax)
        f.write(hdr + b"\x00" * (64 - len(hdr)))
        off = 0
        offtab = bytearray()
        for p, _ in lines:
            offtab += struct.pack("<II", off, len(p)); off += len(p)
        f.write(offtab)
        for p, s in lines:
            p = np.asarray(p, float)
            diffs = np.zeros_like(p)                          # central difference (robust to duplicate pts)
            diffs[1:-1] = p[2:] - p[:-2]; diffs[0] = p[1] - p[0]; diffs[-1] = p[-1] - p[-2]
            nrm = np.linalg.norm(diffs, axis=1)
            good = nrm > 1e-9
            d = np.zeros_like(p)
            d[good] = diffs[good] / nrm[good, None]
            if not good.all() and good.any():                # forward-fill degenerate tangents
                fill = np.maximum.accumulate(np.where(good, np.arange(len(good)), -1))
                fill[fill < 0] = np.where(good)[0][0]
                d = d[fill]
            iv = vidx(p)
            conf = np.clip(Cr[iv[:, 0], iv[:, 1], iv[:, 2]] / cmax, 0, 1)
            rec = np.column_stack([p, s, d, conf]).astype("<f4")
            f.write(rec.tobytes())
    return n, total


def write_grid_bin(path, occ, Vr, Cr):
    spd = np.linalg.norm(Vr, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.nan_to_num(Vr / spd[..., None])
    cmax = float(np.percentile(Cr[Cr > 0], 99)) + 1e-9
    conf = np.clip(Cr / cmax, 0, 1)
    nx, ny, nz = GS
    with open(path, "wb") as f:
        hdr = struct.pack("<IIIIII", GRID_MAGIC, 1, nx, ny, nz, int(occ.sum()))
        f.write(hdr + b"\x00" * (64 - len(hdr)))
        f.write(occ.astype(np.uint8).tobytes())              # C-order (numpy default)
        f.write(d.astype("<f4").tobytes())
        f.write(spd.astype("<f4").tobytes())
        f.write(conf.astype("<f4").tobytes())


def main():
    os.makedirs(OUT, exist_ok=True)
    S = np.load(FD + "samples.npz")
    V, C = build_field(S["mids"], S["vels"])
    try:
        pde_metrics = json.load(open(FD + "splithalf_pde.json"))
    except FileNotFoundError:
        pde_metrics = {}
    manifest = {
        "schema": "ulm-diff-viewer/1",
        "grid": {"shape": list(GS), "spacing_mm": [round(float(x), 4) for x in SP],
                 "origin_mm": [round(float(x), 3) for x in ORG]},
        "frame_rate_hz": FRAME_HZ, "speed_scale_max_mm_s": 38.0,
        "axes": {"x": "lateral", "y": "elevation - 2.77x coarser, partly synthesized", "z": "depth"},
        "default_pair": ["graph", "gauss"], "av_grid": "av.grid.bin", "strategies": [],
        "honesty_notes": [
            "A-only / B-only = present-in-one-strategy occupancy, NOT ground truth; A-only may be smoothing hallucination.",
            "Opacity encodes per-vertex confidence (count). Elevation axis is coarse and partly synthesized.",
            "Flow particles & pulsatility illustrate direction & RELATIVE speed - not literal bubble paths or cardiac timing.",
        ],
    }
    dice = {"gauss": 0.609, "graph": 0.715, "graph_incomp": 0.713, "raw": None}
    cosm = {"gauss": 0.844, "graph": 0.952, "graph_incomp": 0.956, "raw": None}
    for sid, label, hue, uses_inc, fn in STRATEGIES:
        Vr, Cr = fn(V, C)
        lines = streamlines(Vr, Cr, nseed=NSEED, min_len_mm=4.0)
        occ, _, _ = tractogram(Vr, Cr)
        n, total = write_lines_bin(f"{OUT}/{sid}.lines.bin", lines, Cr)
        write_grid_bin(f"{OUT}/{sid}.grid.bin", occ, Vr, Cr)
        manifest["strategies"].append({
            "id": sid, "label": label, "kind": "field-streamlines", "uses_incompressibility": uses_inc,
            "lines": f"{sid}.lines.bin", "grid": f"{sid}.grid.bin", "diff_hue": hue,
            "metrics": {"splithalf_dice": dice[sid], "direction_cosine": cosm[sid],
                        "n_streamlines": n, "occ_vox": int(occ.sum())},
        })
        print(f"{sid:13s} lines n={n:5d} pts={total:7d} occ={int(occ.sum()):6d}")
    # artery/vein grid (single uint8 block)
    lbl = np.load(AV).astype(np.uint8)
    with open(f"{OUT}/av.grid.bin", "wb") as f:
        hdr = struct.pack("<IIIIII", GRID_MAGIC, 1, GS[0], GS[1], GS[2], int((lbl > 0).sum()))
        f.write(hdr + b"\x00" * (64 - len(hdr)))
        f.write(lbl.tobytes())
    json.dump(manifest, open(f"{OUT}/manifest.json", "w"), indent=2)
    print("wrote manifest.json + av.grid.bin ->", OUT)


if __name__ == "__main__":
    main()
