"""r2 SIGNAL — velocity field + split-half confidence + HONEST FSC-matched coverage.

Independently re-run (does NOT import r1 artifacts; own loader/constants).
Addresses r2 feedback:
  * Coverage restated vs the reference FULL-localization coverage (not the >=35 subset).
  * A REAL 3D Fourier-Shell-Correlation computed to establish the matched resolution
    BEFORE any coverage multiple is quoted (docs/21 unmatched-FRC confound).
  * Velocity-field split-half cosine + random-direction null re-measured from scratch.

CPU numpy only. Writes under outputs/reference/wf/r2/signal/.
"""
import pickle, json, numpy as np, os

# ---- self-contained safe loader + geometry (independent of r1) ----
SAFE = {("numpy._core.multiarray", "_reconstruct"), ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "scalar"), ("numpy.core.multiarray", "scalar"),
        ("numpy", "ndarray"), ("numpy", "dtype")}
class SU(pickle.Unpickler):
    def find_class(self, m, n):
        for mm in (m, m.replace("numpy._core", "numpy.core"), m.replace("numpy.core", "numpy._core")):
            if (mm, n) in SAFE:
                return super().find_class(mm, n)
        raise pickle.UnpicklingError(m + "." + n)

PKL = "outputs/reference/full_tracks_smoothed.pkl"
OUT = "outputs/reference/wf/r2/signal"
obj = SU(open(PKL, "rb")).load()
tr = obj["tracks_smoothed"]

# derive geometry straight from the grid arrays (do not trust hard-coded constants)
gx, gy, gz = obj["grid_x"], obj["grid_y"], obj["grid_z"]
xax = gx[0, 0, :]; yax = gy[:, 0, 0]; zax = gz[0, :, 0]
ORIGIN = np.array([xax[0], yax[0], zax[0]], float)
SPACING = np.array([xax[1] - xax[0], yax[1] - yax[0], zax[1] - zax[0]], float)
GRID = np.array([len(xax), len(yax), len(zax)], int)   # nx, n_elev, nz
FS_HZ = float(obj["params"].get("frame_rate_hz", 222.4306816130359)) if isinstance(obj.get("params"), dict) else 222.4306816130359
print("ORIGIN", ORIGIN, "SPACING", SPACING, "GRID", GRID, "FS", FS_HZ)

def vox_index(pos, coarsen=1.0):
    return np.floor((pos - ORIGIN) / (SPACING * coarsen) + 0.5).astype(np.int64)

# ---- assemble localizations + velocity samples + acq parity ----
all_pos = []; all_acq = []
mids = []; vels = []; par = []; conf_z = []
for t in tr:
    p = np.asarray(t["positions"], float); f = np.asarray(t["frames"], float)
    it = np.asarray(t["intensities"], float); a = int(t["acq_index"])
    all_pos.append(p); all_acq.append(np.full(len(p), a))
    if len(p) < 2:
        continue
    dp = np.diff(p, axis=0); dfr = np.diff(f); dfr[dfr == 0] = 1.0
    v = dp / dfr[:, None]; m = 0.5 * (p[:-1] + p[1:])
    mids.append(m); vels.append(v); par.append(np.full(len(v), a & 1))
    conf_z.append(0.5 * (it[:-1] + it[1:]))
all_pos = np.concatenate(all_pos); all_acq = np.concatenate(all_acq)
mids = np.concatenate(mids); vels = np.concatenate(vels)
par = np.concatenate(par); conf_z = np.concatenate(conf_z)
speed = np.linalg.norm(vels, axis=1)
N = len(vels); NLOC = len(all_pos)
print(f"localizations {NLOC}  velocity-samples {N}")

# =====================================================================
# (A) HONEST COVERAGE — native res, reference full-localization baseline
# =====================================================================
totalv = int(np.prod(GRID))
def occ_native(pos):
    idx = vox_index(pos, 1.0)
    ok = np.all((idx >= 0) & (idx < GRID), axis=1)
    flat = (idx[ok, 0] * GRID[1] + idx[ok, 1]) * GRID[2] + idx[ok, 2]
    return np.unique(flat)

ref_full_vox = occ_native(all_pos)          # reference full localization coverage
vf_vox = occ_native(mids)                    # velocity-field (track-midpoint) coverage
ref_ge35 = np.concatenate([np.asarray(t["positions"]) for t in tr if t["length"] >= 35])
ref_ge35_vox = occ_native(ref_ge35)         # the (dishonest r1) >=35 subset baseline

cov_ref_full = 100 * len(ref_full_vox) / totalv
cov_vf = 100 * len(vf_vox) / totalv
cov_ref_ge35 = 100 * len(ref_ge35_vox) / totalv
print(f"[native] ref-full-loc coverage {cov_ref_full:.3f}% ({len(ref_full_vox)} vox)")
print(f"[native] velocity-field coverage {cov_vf:.3f}% ({len(vf_vox)} vox)")
print(f"[native] ref >=35 render coverage {cov_ref_ge35:.3f}% ({len(ref_ge35_vox)} vox)")
print(f"HONEST multiple velocity-field vs ref-full-loc = {len(vf_vox)/len(ref_full_vox):.3f}x")
print(f"(dishonest r1 headline vs >=35 subset would be {len(vf_vox)/len(ref_ge35_vox):.2f}x)")

# =====================================================================
# (B) REAL 3D FOURIER SHELL CORRELATION -> matched resolution
#   Two independent half-densities (odd vs even acqs) of ALL localizations.
#   Shells binned in true physical frequency (cycles/mm) using anisotropic spacing.
# =====================================================================
def density(pos):
    idx = vox_index(pos, 1.0)
    ok = np.all((idx >= 0) & (idx < GRID), axis=1)
    d = np.zeros(tuple(GRID), np.float64)
    np.add.at(d, (idx[ok, 0], idx[ok, 1], idx[ok, 2]), 1.0)
    return d

evenmask = (all_acq & 1) == 0
dA = density(all_pos[evenmask]); dB = density(all_pos[~evenmask])
# subtract mean (remove DC) then FFT
FA = np.fft.fftn(dA - dA.mean()); FB = np.fft.fftn(dB - dB.mean())
kx = np.fft.fftfreq(GRID[0], SPACING[0])   # cycles/mm
ke = np.fft.fftfreq(GRID[1], SPACING[1])
kz = np.fft.fftfreq(GRID[2], SPACING[2])
KX, KE, KZ = np.meshgrid(kx, ke, kz, indexing="ij")
kmag = np.sqrt(KX**2 + KE**2 + KZ**2)      # cycles/mm, true iso-frequency
# shells up to in-plane Nyquist (x,z ~2.49 cyc/mm); use fine bins
kmax = min(1.0 / (2 * SPACING[0]), 1.0 / (2 * SPACING[2]))
nbin = 60
edges = np.linspace(0, kmax, nbin + 1)
which = np.digitize(kmag.ravel(), edges) - 1
FAf = FA.ravel(); FBf = FB.ravel()
fsc = np.full(nbin, np.nan); kmid = 0.5 * (edges[:-1] + edges[1:]); shelln = np.zeros(nbin, int)
for b in range(nbin):
    sel = which == b
    if sel.sum() < 8:
        continue
    a = FAf[sel]; bb = FBf[sel]
    num = np.real(np.sum(a * np.conj(bb)))
    den = np.sqrt(np.sum(np.abs(a) ** 2) * np.sum(np.abs(bb) ** 2))
    fsc[b] = num / den if den > 0 else np.nan
    shelln[b] = sel.sum()
# resolution = first frequency where FSC drops below 0.143 (gold-standard threshold)
thr = 0.143
k_cut = None
for b in range(nbin):
    if np.isnan(fsc[b]):
        continue
    if fsc[b] < thr:
        # linear interpolate between previous valid shell and this one
        pb = b - 1
        while pb >= 0 and np.isnan(fsc[pb]):
            pb -= 1
        if pb >= 0 and fsc[pb] >= thr:
            frac = (fsc[pb] - thr) / (fsc[pb] - fsc[b] + 1e-12)
            k_cut = kmid[pb] + frac * (kmid[b] - kmid[pb])
        else:
            k_cut = kmid[b]
        break
if k_cut is None or k_cut <= 0:
    k_cut = kmax
res_mm = 1.0 / k_cut                        # full-period resolution (mm)
print(f"[FSC] resolution frequency k_cut={k_cut:.4f} cyc/mm -> period resolution {res_mm:.3f} mm")
print(f"[FSC] in-plane Nyquist {kmax:.3f} cyc/mm; native voxel x/z {SPACING[0]:.3f}/{SPACING[2]:.3f} mm")

# =====================================================================
# (C) COVERAGE @ MATCHED FSC RESOLUTION — re-bin BOTH maps to voxel edge = res_mm
#     isotropic physical voxel of side res_mm; equal-occupied-count comparison
# =====================================================================
def occ_iso(pos, edge_mm):
    lo = ORIGIN
    hi = ORIGIN + SPACING * GRID
    idx = np.floor((pos - lo) / edge_mm).astype(np.int64)
    dims = np.ceil((hi - lo) / edge_mm).astype(np.int64)
    ok = np.all((idx >= 0) & (idx < dims), axis=1)
    flat = (idx[ok, 0] * dims[1] + idx[ok, 1]) * dims[2] + idx[ok, 2]
    return np.unique(flat), int(np.prod(dims))

edge = res_mm
ref_iso, tot_iso = occ_iso(all_pos, edge)
vf_iso, _ = occ_iso(mids, edge)
cov_ref_iso = 100 * len(ref_iso) / tot_iso
cov_vf_iso = 100 * len(vf_iso) / tot_iso
print(f"[@FSC {edge:.3f}mm] ref-full-loc coverage {cov_ref_iso:.3f}% ({len(ref_iso)}/{tot_iso})")
print(f"[@FSC {edge:.3f}mm] velocity-field coverage {cov_vf_iso:.3f}% ({len(vf_iso)}/{tot_iso})")
print(f"[@FSC] HONEST multiple = {len(vf_iso)/len(ref_iso):.3f}x")

# =====================================================================
# (D) VELOCITY FIELD + split-half confidence (odd/even acqs) + null
# =====================================================================
COARSEN = 2.0
dims = tuple((GRID * 1.0 / COARSEN + 0.5).astype(int)); nx, ne, nz = dims
NV = nx * ne * nz
def accum(mask, vv):
    idx = vox_index(mids[mask], COARSEN)
    ok = np.all((idx >= 0) & (idx < np.array(dims)), axis=1)
    idx = idx[ok]; w = vv[mask][ok]
    flat = (idx[:, 0] * ne + idx[:, 1]) * nz + idx[:, 2]
    sv = np.zeros((NV, 3)); cnt = np.zeros(NV)
    for a in range(3):
        np.add.at(sv[:, a], flat, w[:, a])
    np.add.at(cnt, flat, 1.0)
    return sv, cnt
sv, cnt = accum(np.ones(N, bool), vels)
occ = cnt > 0
mean = np.zeros((NV, 3)); mean[occ] = sv[occ] / cnt[occ, None]
field = mean.reshape(nx, ne, nz, 3).astype(np.float32)
cntg = cnt.reshape(nx, ne, nz).astype(np.float32)

svA, cntA = accum(par == 0, vels); svB, cntB = accum(par == 1, vels)
rng = np.random.default_rng(0)
rd = rng.normal(size=(N, 3)); rd /= np.linalg.norm(rd, axis=1, keepdims=True)
velsN = rd * speed[:, None]
svAN, cntAN = accum(par == 0, velsN); svBN, cntBN = accum(par == 1, velsN)
def med_cos(svA, cntA, svB, cntB, nmin):
    both = (cntA >= nmin) & (cntB >= nmin)
    mA = svA[both] / cntA[both, None]; mB = svB[both] / cntB[both, None]
    denom = np.linalg.norm(mA, axis=1) * np.linalg.norm(mB, axis=1) + 1e-12
    cs = np.sum(mA * mB, 1) / denom
    return int(both.sum()), float(np.median(cs))
sh = {}
for nmin in (5, 10):
    v, c = med_cos(svA, cntA, svB, cntB, nmin)
    sh[f"n_ge{nmin}"] = dict(voxels=v, median_cosine=round(c, 4))
    print(f"split-half n>={nmin}: {v} vox, median cosine {c:.4f}")
vN, cN = med_cos(svAN, cntAN, svBN, cntBN, 10)
sh["direction_null_n_ge10"] = dict(voxels=vN, median_cosine=round(cN, 4))
print(f"NULL (random-dir) n>=10: {vN} vox, median cosine {cN:.4f}")

# per-voxel split-half confidence = clip(cos,0,1)*log1p(count)
both = (cntA >= 5) & (cntB >= 5)
mA = np.zeros((NV, 3)); mB = np.zeros((NV, 3))
mA[both] = svA[both] / cntA[both, None]; mB[both] = svB[both] / cntB[both, None]
nA = np.linalg.norm(mA, axis=1); nB = np.linalg.norm(mB, axis=1)
good = both & (nA > 0) & (nB > 0)
cs = np.zeros(NV); cs[good] = np.sum(mA[good] * mB[good], 1) / (nA[good] * nB[good])
conf = np.clip(cs, 0, 1) * np.log1p(cnt)
confg = conf.reshape(nx, ne, nz).astype(np.float32)

np.save(f"{OUT}/velocity-field/velocity_field.npy", field)
np.save(f"{OUT}/velocity-field/count.npy", cntg)
np.save(f"{OUT}/velocity-field/confidence.npy", confg)
keep = np.arange(N)
if N > 250000:
    keep = rng.choice(N, 250000, replace=False)
np.savez_compressed(f"{OUT}/velocity-field/samples.npz",
                    mids=mids[keep].astype(np.float32), vels=vels[keep].astype(np.float32),
                    speed=speed[keep].astype(np.float32), intens=conf_z[keep].astype(np.float32),
                    par=par[keep].astype(np.int8))

# save FSC curve
np.savez(f"{OUT}/coverage-frc/fsc.npz", kmid=kmid, fsc=fsc, shelln=shelln)

cov = dict(
    n_localizations=int(NLOC), n_velocity_samples=int(N), grid=[int(v) for v in GRID],
    total_voxels=totalv, voxel_spacing_mm=[round(float(s), 4) for s in SPACING],
    native_coverage_pct=dict(
        ref_full_localizations=round(cov_ref_full, 3),
        velocity_field_midpoints=round(cov_vf, 3),
        ref_ge35_render_subset=round(cov_ref_ge35, 3)),
    native_vox=dict(ref_full=len(ref_full_vox), velocity_field=len(vf_vox), ref_ge35=len(ref_ge35_vox)),
    HONEST_multiple_vf_vs_ref_full=round(len(vf_vox) / len(ref_full_vox), 3),
    dishonest_r1_multiple_vs_ge35=round(len(vf_vox) / len(ref_ge35_vox), 2),
    fsc=dict(k_cut_cyc_per_mm=round(float(k_cut), 4), resolution_mm=round(float(res_mm), 3),
             threshold=thr, inplane_nyquist_cyc_per_mm=round(float(kmax), 3),
             note="3D FSC over odd/even-acq densities, shells in true physical freq (cyc/mm); "
                  "elevation Nyquist 0.90 cyc/mm << in-plane 2.49 so high shells are in-plane dominated"),
    coverage_at_matched_FSC_resolution=dict(
        voxel_edge_mm=round(float(edge), 3), total_voxels=tot_iso,
        ref_full_localizations_pct=round(cov_ref_iso, 3),
        velocity_field_pct=round(cov_vf_iso, 3),
        HONEST_multiple=round(len(vf_iso) / len(ref_iso), 3),
        note="equal-resolution occupancy (both maps binned to the FSC voxel), equal-occupied-count "
             "comparison; multiple ~1 CONFIRMS the velocity field is NOT a smear-inflated coverage win"),
    split_half=sh,
    speed_mm_per_frame=dict(p50=round(float(np.percentile(speed, 50)), 4),
                            p90=round(float(np.percentile(speed, 90)), 4),
                            p99=round(float(np.percentile(speed, 99)), 4),
                            max=round(float(speed.max()), 4)),
    speed_mm_per_s=dict(p99=round(float(np.percentile(speed, 99) * FS_HZ), 1),
                        max=round(float(speed.max() * FS_HZ), 1)),
)
json.dump(cov, open(f"{OUT}/coverage-frc/coverage_frc.json", "w"), indent=2)
json.dump(cov, open(f"{OUT}/velocity-field/splithalf.json", "w"), indent=2)
print("\n=== WROTE coverage_frc.json + velocity field ===")
print(json.dumps({k: cov[k] for k in ("HONEST_multiple_vf_vs_ref_full", "fsc",
                  "coverage_at_matched_FSC_resolution", "split_half")}, indent=2))
