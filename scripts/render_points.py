"""Faithful points render of a shipped track-viewer tracks.bin.

Mimics the shipped WebGL viewer: renders each localization as a POINT (not a
line), colored by smoothed speed via Jet at the viewer's vmax (u_maxSpeed=0.1713
mm/frame == 38 mm/s @222Hz), projected down the elevation (Y) axis = the shipped
default camera (looks along +Y, up=Z), composited on the viewer clear color.

Usage: python3 render_points.py <tracks.bin> <out.png> [--vmax MMPF] [--gain G] [--psf S]
"""
import sys, struct
import numpy as np
from scipy.ndimage import gaussian_filter
from matplotlib import cm
from PIL import Image

def parse_bin(path):
    b = open(path, "rb").read()
    magic, ver, n_tracks, total_points = struct.unpack_from("<IIII", b, 0)
    max_speed = struct.unpack_from("<f", b, 16)[0]
    bmin = struct.unpack_from("<3f", b, 20)
    bmax = struct.unpack_from("<3f", b, 32)
    assert magic == 0x554C4D54, f"bad magic {magic:#x}"
    off = 64
    table = np.frombuffer(b, dtype=np.uint32, count=n_tracks * 2, offset=off).reshape(n_tracks, 2)
    off += n_tracks * 8
    pts = np.frombuffer(b, dtype=np.float32, count=total_points * 6, offset=off).reshape(total_points, 6)
    # columns: x, y, z, frame, speed, intensity
    return dict(n_tracks=n_tracks, total_points=total_points, max_speed=max_speed,
                bmin=np.array(bmin), bmax=np.array(bmax), table=table, pts=pts)

def render(binpath, outpath, vmax=0.1713, gain=1.6, psf=0.7, W=1600, H=1050,
           hax=0, vax=2, flip_v=True, margin=0.06):
    d = parse_bin(binpath)
    pts = d["pts"]
    x = pts[:, hax]; z = pts[:, vax]; speed = pts[:, 4]
    # world extent from bounds (keep aspect via data range)
    xmin, xmax = d["bmin"][hax], d["bmax"][hax]
    zmin, zmax = d["bmin"][vax], d["bmax"][vax]
    xr = xmax - xmin; zr = zmax - zmin
    # fit into canvas preserving aspect ratio
    aspect_data = xr / zr
    aspect_canvas = (W * (1 - 2 * margin)) / (H * (1 - 2 * margin))
    if aspect_data > aspect_canvas:
        draw_w = W * (1 - 2 * margin); draw_h = draw_w / aspect_data
    else:
        draw_h = H * (1 - 2 * margin); draw_w = draw_h * aspect_data
    x0 = (W - draw_w) / 2; y0 = (H - draw_h) / 2
    px = x0 + (x - xmin) / xr * draw_w
    pz = (z - zmin) / zr * draw_h
    if flip_v:
        pz = draw_h - pz
    py = y0 + pz
    ix = np.round(px).astype(int); iy = np.round(py).astype(int)
    ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
    ix, iy, speed = ix[ok], iy[ok], speed[ok]
    t = np.clip(speed / vmax, 0, 1)
    colors = cm.get_cmap("jet")(t)[:, :3].astype(np.float32)  # (N,3)
    flat = iy * W + ix
    acc = np.zeros((H * W, 3), dtype=np.float32)
    for c in range(3):
        np.add.at(acc[:, c], flat, colors[:, c])
    acc = acc.reshape(H, W, 3)
    if psf > 0:
        for c in range(3):
            acc[:, :, c] = gaussian_filter(acc[:, :, c], psf)
    # tonemap (soft saturation), then floor to viewer clear color 0x010104
    img = 1.0 - np.exp(-acc * gain)
    bg = np.array([1, 1, 4], np.float32) / 255.0
    img = bg[None, None, :] + img * (1 - bg[None, None, :])
    img = np.clip(img, 0, 1)
    Image.fromarray((img * 255).astype(np.uint8)).save(outpath)
    print(f"rendered {ok.sum()}/{len(pts)} pts -> {outpath}  "
          f"(vmax={vmax} mmpf, speed p50={np.median(speed):.4f} p99={np.percentile(speed,99):.4f} max={speed.max():.4f})")

if __name__ == "__main__":
    args = sys.argv[1:]
    binpath, outpath = args[0], args[1]
    kw = {}
    for i, a in enumerate(args):
        if a == "--vmax": kw["vmax"] = float(args[i+1])
        if a == "--gain": kw["gain"] = float(args[i+1])
        if a == "--psf": kw["psf"] = float(args[i+1])
    render(binpath, outpath, **kw)
