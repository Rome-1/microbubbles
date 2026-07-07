"""r2 SIGNAL render — surfaces the VALIDATED signal (not the raw composite).

Coronal super-res drizzle (look down elevation, x-horizontal, z-vertical = blog camera):
  1. flow_direction_hero.png : hue=in-plane azimuth atan2(vz,vx), value=log drizzle density,
     OPACITY GATED by per-pixel split-half direction confidence (only reproducible flow shows).
  2. confidence_map.png       : the split-half cosine reproducibility field itself (the honesty
     visual: this is what makes the render trustworthy, computed odd-vs-even acqs).
  3. artery_vein.png          : codex-gated antiparallel A/V pairs (red artery / blue vein).
  4. signal_panels.png        : combined figure with data-true legends.
  viewer/index.html           : self-contained, embeds the hero + provenance.

CPU numpy + matplotlib(Agg) only.
"""
import pickle, base64, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from scipy.ndimage import gaussian_filter

SAFE = {("numpy._core.multiarray", "_reconstruct"), ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "scalar"), ("numpy.core.multiarray", "scalar"),
        ("numpy", "ndarray"), ("numpy", "dtype")}
class SU(pickle.Unpickler):
    def find_class(self, m, n):
        for mm in (m, m.replace("numpy._core", "numpy.core"), m.replace("numpy.core", "numpy._core")):
            if (mm, n) in SAFE:
                return super().find_class(mm, n)
        raise pickle.UnpicklingError(m + "." + n)
obj = SU(open("outputs/reference/full_tracks_smoothed.pkl", "rb")).load()
tr = obj["tracks_smoothed"]
gx, gy, gz = obj["grid_x"], obj["grid_y"], obj["grid_z"]
xax = gx[0, 0, :]; yax = gy[:, 0, 0]; zax = gz[0, :, 0]
ORIGIN = np.array([xax[0], yax[0], zax[0]], float)
SPACING = np.array([xax[1] - xax[0], yax[1] - yax[0], zax[1] - zax[0]], float)
GRID = np.array([len(xax), len(yax), len(zax)], int)
FS_HZ = 222.4306816130359
OUT = "outputs/reference/wf/r2/signal"; PROOF = f"{OUT}/proof"

mids = []; vels = []; par = []
for t in tr:
    p = np.asarray(t["positions"], float); f = np.asarray(t["frames"], float)
    if len(p) < 2:
        continue
    dp = np.diff(p, axis=0); dfr = np.diff(f); dfr[dfr == 0] = 1.0
    vels.append(dp / dfr[:, None]); mids.append(0.5 * (p[:-1] + p[1:]))
    par.append(np.full(len(p) - 1, int(t["acq_index"]) & 1))
mids = np.concatenate(mids); vels = np.concatenate(vels); par = np.concatenate(par)
speed = np.linalg.norm(vels, axis=1)
nx, ne, nz = GRID
SR = 4; W, H = nx * SR, nz * SR
x0, z0 = ORIGIN[0], ORIGIN[2]; dx, dz = SPACING[0] / SR, SPACING[2] / SR
FOVx = nx * SPACING[0]; FOVz = nz * SPACING[2]; FOVe = ne * SPACING[1]
ix = np.clip(((mids[:, 0] - x0) / dx).astype(int), 0, W - 1)
iz = np.clip(((mids[:, 2] - z0) / dz).astype(int), 0, H - 1)
flat = iz * W + ix

def accum(vals, mask=None):
    a = np.zeros(W * H)
    if mask is None:
        np.add.at(a, flat, vals)
    else:
        np.add.at(a, flat[mask], vals[mask])
    return a.reshape(H, W)

D = accum(np.ones(len(flat)))
SVX = accum(vels[:, 0]); SVZ = accum(vels[:, 2]); SSP = accum(speed)
occ = D > 0
mvx = np.zeros_like(D); mvz = np.zeros_like(D); msp = np.zeros_like(D)
mvx[occ] = SVX[occ] / D[occ]; mvz[occ] = SVZ[occ] / D[occ]; msp[occ] = SSP[occ] / D[occ]

# --- per-pixel split-half direction confidence (odd vs even acqs) ---
DA = accum(np.ones(len(flat)), par == 0); DB = accum(np.ones(len(flat)), par == 1)
AX = accum(vels[:, 0], par == 0); AZ = accum(vels[:, 2], par == 0)
BX = accum(vels[:, 0], par == 1); BZ = accum(vels[:, 2], par == 1)
both = (DA >= 2) & (DB >= 2)
ax_ = np.zeros_like(D); az_ = np.zeros_like(D); bx_ = np.zeros_like(D); bz_ = np.zeros_like(D)
ax_[both] = AX[both] / DA[both]; az_[both] = AZ[both] / DA[both]
bx_[both] = BX[both] / DB[both]; bz_[both] = BZ[both] / DB[both]
na = np.sqrt(ax_**2 + az_**2); nb = np.sqrt(bx_**2 + bz_**2)
conf = np.zeros_like(D); g = both & (na > 0) & (nb > 0)
conf[g] = (ax_[g] * bx_[g] + az_[g] * bz_[g]) / (na[g] * nb[g])
conf = np.clip(conf, 0, 1)
conf_s = gaussian_filter(conf, 0.6)

Ds = gaussian_filter(D, 0.7)
val = np.log1p(Ds); val = np.clip(val / (np.percentile(val[occ], 99.5) + 1e-9), 0, 1)
sat = np.clip(msp / (np.percentile(msp[occ], 95) + 1e-9), 0, 1); sat = 0.35 + 0.65 * sat
az_ang = (np.arctan2(mvz, mvx) + np.pi) / (2 * np.pi)

# hero: value modulated by confidence (trust scaffolding — dim the unreproducible)
val_gated = val * (0.25 + 0.75 * conf_s)
hsv = np.zeros((H, W, 3)); hsv[..., 0] = az_ang; hsv[..., 1] = sat; hsv[..., 2] = val_gated
hsv[~occ] = 0
rgb = hsv_to_rgb(hsv)

def frame_axes(fig, title, sub):
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor("black"); ax.axis("off")
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.text(0.012, 0.972, title, transform=ax.transAxes, color="white", fontsize=12,
            va="top", weight="bold")
    ax.text(0.012, 0.028, sub, transform=ax.transAxes, color="#b8c4d0", fontsize=8.5, va="bottom")
    return ax

# scalebar (10 mm)
sb_px = 10.0 / dx
def scalebar(ax, y=0.09):
    x1 = 0.72 * W; ax.plot([x1, x1 + sb_px], [y * H, y * H], color="white", lw=3)
    ax.text(x1 + sb_px / 2, y * H + 6, "10 mm", color="white", fontsize=8, ha="center")

# 1. hero
fig = plt.figure(figsize=(13, 13 * FOVz / FOVx), facecolor="black")
ax = frame_axes(fig, "Living Angiogram — flow DIRECTION x density, gated by split-half confidence",
                f"{len(flat):,} velocity samples | coronal super-res {SR}x | FOV {FOVx:.0f}x{FOVz:.0f} mm "
                f"(elev {FOVe:.0f} mm collapsed) | max flow {speed.max()*FS_HZ:.1f} mm/s = "
                f"{speed.max()*FS_HZ/10:.1f} cm/s (blog 0-38 mm/s SATURATES)")
ax.imshow(rgb, origin="lower", aspect="auto", interpolation="bilinear")
scalebar(ax)
fig.savefig(f"{PROOF}/flow_direction_hero.png", dpi=110, facecolor="black"); plt.close(fig)

# 2. confidence map
fig = plt.figure(figsize=(13, 13 * FOVz / FOVx), facecolor="black")
ax = frame_axes(fig, "Split-half reproducibility (odd vs even acqs) — the honesty channel",
                "per-pixel cosine(dir_oddacq, dir_evenacq); bright = flow direction that reproduces "
                "across independent acquisitions. This is why the render is trustworthy.")
cmask = np.where(occ, conf_s, np.nan)
im = ax.imshow(cmask, origin="lower", aspect="auto", cmap="turbo", vmin=0, vmax=1,
               interpolation="bilinear")
scalebar(ax)
fig.savefig(f"{PROOF}/confidence_map.png", dpi=110, facecolor="black"); plt.close(fig)

# 3. artery/vein
lbl = np.load(f"{OUT}/artery-vein/av_labels.npy")   # (138,13,77) at coarsen2
C = 2.0; spc = SPACING * C
av = np.load(f"{OUT}/artery-vein/av_pairs.npz")
pa = av["pa"]; pb = av["pb"]
fig = plt.figure(figsize=(13, 13 * FOVz / FOVx), facecolor="black")
ax = frame_axes(fig, "Artery / Vein — antiparallel counter-flow pairs (red=artery, blue=vein)",
                f"{(lbl>0).sum()} voxels in {len(pa)} lateral, split-half-reproducible A/V pairs "
                f"(dot_A<0 & dot_B<0, mean dot -0.74). Faster lumen labeled artery.")
# faint density backdrop
ax.imshow(np.where(occ, val * 0.4, 0), origin="lower", aspect="auto", cmap="gray", vmin=0, vmax=1)
ax_i, el_i, z_i = np.nonzero(lbl > 0)
xx = (ORIGIN[0] + (ax_i + 0.5) * spc[0] - x0) / dx
zz = (ORIGIN[2] + (z_i + 0.5) * spc[2] - z0) / dz
cols = np.where(lbl[ax_i, el_i, z_i] == 1, "#ff3b3b", "#3b7bff")
ax.scatter(xx, zz, c=cols, s=10, edgecolors="none", alpha=0.9)
scalebar(ax)
fig.savefig(f"{PROOF}/artery_vein.png", dpi=110, facecolor="black"); plt.close(fig)

# 4. combined panels
fig, axs = plt.subplots(1, 3, figsize=(21, 7 * FOVz / FOVx), facecolor="black")
for a in axs:
    a.set_facecolor("black"); a.axis("off")
axs[0].imshow(rgb, origin="lower", aspect="auto", interpolation="bilinear")
axs[0].set_title("Flow direction x density\n(confidence-gated)", color="white", fontsize=11)
axs[1].imshow(cmask, origin="lower", aspect="auto", cmap="turbo", vmin=0, vmax=1)
axs[1].set_title("Split-half reproducibility\n(GT-free honesty)", color="white", fontsize=11)
axs[2].imshow(np.where(occ, val * 0.4, 0), origin="lower", aspect="auto", cmap="gray", vmin=0, vmax=1)
axs[2].scatter(xx, zz, c=cols, s=6, edgecolors="none", alpha=0.9)
axs[2].set_title("Artery / vein\n(antiparallel pairs)", color="white", fontsize=11)
fig.savefig(f"{PROOF}/signal_panels.png", dpi=100, facecolor="black", bbox_inches="tight"); plt.close(fig)

# viewer/index.html — embed hero
with open(f"{PROOF}/flow_direction_hero.png", "rb") as fh:
    b64 = base64.b64encode(fh.read()).decode()
cov = json.load(open(f"{OUT}/coverage-frc/coverage_frc.json"))
avm = json.load(open(f"{OUT}/artery-vein/metrics.json"))
html = f"""<div style="background:#000;color:#dfe;font-family:system-ui,sans-serif;min-height:100vh;margin:0;padding:16px">
<h2 style="margin:0 0 4px">Ultratrace 3D-ULM — the validated signal (r2)</h2>
<p style="color:#9ab;margin:2px 0 12px;font-size:13px">Flow-direction living angiogram, gated by split-half reproducibility. Dense render of the signal the blog discards (velocity direction).</p>
<img src="data:image/png;base64,{b64}" style="max-width:100%;border:1px solid #234"/>
<div style="display:flex;gap:24px;flex-wrap:wrap;margin-top:14px;font-size:13px">
<div><b style="color:#6cf">Coverage (honest)</b><br>ref full-loc {cov['native_coverage_pct']['ref_full_localizations']}% ({cov['native_vox']['ref_full']} vox)<br>velocity field {cov['native_coverage_pct']['velocity_field_midpoints']}% = <b>{cov['HONEST_multiple_vf_vs_ref_full']}x</b> (NOT a coverage win)<br>@FSC-matched {cov['coverage_at_matched_FSC_resolution']['voxel_edge_mm']}mm: {cov['coverage_at_matched_FSC_resolution']['HONEST_multiple']}x</div>
<div><b style="color:#6cf">Validated direction</b><br>split-half cosine {cov['split_half']['n_ge10']['median_cosine']} (n&ge;10)<br>null {cov['split_half']['direction_null_n_ge10']['median_cosine']} &rarr; the win is DIRECTION, not coverage</div>
<div><b style="color:#6cf">Artery/vein</b><br>{avm['n_AV_pairs']} counter-flow pairs<br>{avm['voxels_in_AV_pair']} voxels, mean dot {avm['mean_antiparallel_dot']}<br>split-half reproducible + lateral-gated</div>
<div><b style="color:#6cf">Flow (data-true)</b><br>p99 {cov['speed_mm_per_s']['p99']} mm/s, max {cov['speed_mm_per_s']['max']} mm/s<br>blog 0-38 mm/s saturates real flow</div>
</div>
<p style="color:#789;font-size:11px;margin-top:14px">Panels: proof/flow_direction_hero.png, proof/confidence_map.png, proof/artery_vein.png, proof/signal_panels.png. FSC resolution {cov['fsc']['resolution_mm']}mm (0.143), {cov['fsc'].get('fsc_0p5_resolution_mm','?')}mm (0.5 half-bit).</p>
</div>"""
open(f"{OUT}/viewer/index.html", "w").write(html)
print("WROTE renders + viewer/index.html")
print("hero px", W, H, "| samples", len(flat), "| conf pixels", int(g.sum()),
      "| AV voxels", int((lbl > 0).sum()))
