"""r2 ARTERY/VEIN — antiparallel-adjacency pairing (codex-designed, improved over r1).

Improvements over r1 (per codex gpt-5.5 core-algorithm design):
  * LATERAL-adjacency gate: the connecting vector r_hat between the two voxels must be
    ~perpendicular to BOTH flow directions (|d.r_hat| < 0.5). This rejects a single
    vessel's own curvature (head-to-tail) being miscounted as an antiparallel A/V pair.
  * SPLIT-HALF-REPRODUCIBLE antiparallel: require dot_A<0 AND dot_B<0 (both independent
    acq halves see the counter-flow), not just the pooled direction. Strong GT-free gate.

CPU numpy only. Writes outputs/reference/wf/r2/signal/artery-vein/.
"""
import pickle, json, numpy as np
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
OUT = "outputs/reference/wf/r2/signal/artery-vein"

# velocity samples + acq parity
mids = []; vels = []; par = []
for t in tr:
    p = np.asarray(t["positions"], float); f = np.asarray(t["frames"], float)
    if len(p) < 2:
        continue
    dp = np.diff(p, axis=0); dfr = np.diff(f); dfr[dfr == 0] = 1.0
    mids.append(0.5 * (p[:-1] + p[1:])); vels.append(dp / dfr[:, None])
    par.append(np.full(len(p) - 1, int(t["acq_index"]) & 1))
mids = np.concatenate(mids); vels = np.concatenate(vels); par = np.concatenate(par)
speed = np.linalg.norm(vels, axis=1)

COARSEN = 2.0
sp = SPACING * COARSEN
dims = tuple((GRID / COARSEN + 0.5).astype(int)); nx, ne, nz = dims
def accum(mask, weight):
    idx = np.floor((mids[mask] - ORIGIN) / sp + 0.5).astype(np.int64)
    ok = np.all((idx >= 0) & (idx < np.array(dims)), axis=1)
    idx = idx[ok]; w = weight[mask][ok]
    s = np.zeros((nx, ne, nz, w.shape[1] if w.ndim > 1 else 1))
    c = np.zeros((nx, ne, nz))
    if w.ndim == 1:
        np.add.at(c, (idx[:, 0], idx[:, 1], idx[:, 2]), 1.0)
        np.add.at(s[..., 0], (idx[:, 0], idx[:, 1], idx[:, 2]), w)
        return s[..., 0], c
    for a in range(w.shape[1]):
        np.add.at(s[..., a], (idx[:, 0], idx[:, 1], idx[:, 2]), w[:, a])
    np.add.at(c, (idx[:, 0], idx[:, 1], idx[:, 2]), 1.0)
    return s, c

svF, cF = accum(np.ones(len(vels), bool), vels)
svA, cA = accum(par == 0, vels); svB, cB = accum(par == 1, vels)
spdF, _ = accum(np.ones(len(vels), bool), speed)   # sum speed for mean-speed labeling

def unit_dir(sv, c):
    m = np.zeros_like(sv); nz_ = c > 0
    m[nz_] = sv[nz_] / c[nz_, None]
    n = np.linalg.norm(m, axis=-1)
    d = np.full_like(m, np.nan); g = n > 1e-12
    d[g] = m[g] / n[g, None]
    return d
dF = unit_dir(svF, cF); dA = unit_dir(svA, cA); dB = unit_dir(svB, cB)
mean_speed = np.zeros((nx, ne, nz)); occ = cF > 0
mean_speed[occ] = spdF[occ] / cF[occ]

MIN_COUNT = 3
occupied = (cA >= MIN_COUNT) & (cB >= MIN_COUNT) & np.isfinite(dF).all(-1) \
    & np.isfinite(dA).all(-1) & np.isfinite(dB).all(-1)
print("occupied voxels (cA,cB>=3):", int(occupied.sum()))

# forward 13-neighborhood
offsets = []
for ox in (-1, 0, 1):
    for oy in (-1, 0, 1):
        for oz in (-1, 0, 1):
            if (ox, oy, oz) == (0, 0, 0):
                continue
            if ox > 0 or (ox == 0 and oy > 0) or (ox == 0 and oy == 0 and oz > 0):
                offsets.append((ox, oy, oz))

def pairs(lateral_gate):
    pa = []; pb = []; pd = []
    for off in offsets:
        sa = []; sb = []; st_a = []
        for n_, o in zip(dims, off):
            if o >= 0:
                sa.append(slice(0, n_ - o)); sb.append(slice(o, n_)); st_a.append(0)
            else:
                sa.append(slice(-o, n_)); sb.append(slice(0, n_ + o)); st_a.append(-o)
        sa = tuple(sa); sb = tuple(sb)
        occ2 = occupied[sa] & occupied[sb]
        da, db = dF[sa], dF[sb]
        dotF = np.einsum("...i,...i->...", da, db)
        dotA = np.einsum("...i,...i->...", dA[sa], dA[sb])
        dotB = np.einsum("...i,...i->...", dB[sa], dB[sb])
        r = np.asarray(off, float) * sp; rhat = r / np.linalg.norm(r)
        if lateral_gate:
            lat = (np.abs(np.einsum("...i,i->...", da, rhat)) <= 0.5) \
                & (np.abs(np.einsum("...i,i->...", db, rhat)) <= 0.5)
        else:
            lat = np.ones_like(dotF, bool)
        mask = occ2 & (dotF < 0) & (dotA < 0) & (dotB < 0) & lat
        loc = np.argwhere(mask)
        if loc.size:
            pa.append(loc + np.asarray(st_a))
            pb.append(loc + np.asarray(st_a) + np.asarray(off))
            pd.append(dotF[mask])
    if pa:
        return np.vstack(pa), np.vstack(pb), np.concatenate(pd)
    return np.zeros((0, 3), int), np.zeros((0, 3), int), np.zeros(0)

pa_lat, pb_lat, pd_lat = pairs(True)
pa_nolat, pb_nolat, _ = pairs(False)
n_occ_adj = 0
for off in offsets:      # total occupied-adjacent pair count (denominator)
    sa = []; sb = []
    for n_, o in zip(dims, off):
        if o >= 0:
            sa.append(slice(0, n_ - o)); sb.append(slice(o, n_))
        else:
            sa.append(slice(-o, n_)); sb.append(slice(0, n_ + o))
    n_occ_adj += int((occupied[tuple(sa)] & occupied[tuple(sb)]).sum())

# label artery/vein within each lateral pair: faster mean-speed = artery
lbl = np.zeros((nx, ne, nz), np.int8)   # 0 unknown, 1 artery, 2 vein
av_vox = set()
for a, b in zip(pa_lat, pb_lat):
    sa = mean_speed[tuple(a)]; sb = mean_speed[tuple(b)]
    fa, fv = (a, b) if sa >= sb else (b, a)
    lbl[tuple(fa)] = 1; lbl[tuple(fv)] = 2
    av_vox.add(tuple(a)); av_vox.add(tuple(b))
np.save(f"{OUT}/av_labels.npy", lbl)
# also save the pair voxel lists + directions for the render
np.savez_compressed(f"{OUT}/av_pairs.npz", pa=pa_lat, pb=pb_lat, dot=pd_lat,
                    dir_full=dF.astype(np.float32), mean_speed=mean_speed.astype(np.float32))

out = dict(
    method="antiparallel-adjacency, LATERAL-gated + split-half-reproducible (codex-designed)",
    coarsen=COARSEN, dims=[int(v) for v in dims], min_count=MIN_COUNT,
    occupied_voxels=int(occupied.sum()),
    occupied_adjacent_pairs=int(n_occ_adj),
    antiparallel_splithalf_reproducible=dict(
        without_lateral_gate=int(len(pa_nolat)),
        with_lateral_gate=int(len(pa_lat)),
        note="lateral gate removes single-vessel-curvature head-to-tail antiparallels; "
             "both require dot_A<0 AND dot_B<0 (independent acq halves agree)"),
    n_AV_pairs=int(len(pa_lat)),
    voxels_in_AV_pair=int(len(av_vox)),
    arteries=int((lbl == 1).sum()), veins=int((lbl == 2).sum()),
    mean_antiparallel_dot=round(float(pd_lat.mean()), 4) if len(pd_lat) else None,
    antiparallel_frac_of_occ_adj=round(len(pa_lat) / max(n_occ_adj, 1), 4),
    gt_free_validation="each reported A/V pair is antiparallel INDEPENDENTLY in both odd-acq and "
                       "even-acq direction fields (dot_A<0 & dot_B<0) AND laterally adjacent "
                       "(connecting vector perpendicular to both lumina) -> not noise, not curvature",
)
json.dump(out, open(f"{OUT}/metrics.json", "w"), indent=2)
print(json.dumps(out, indent=2))
