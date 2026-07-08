"""mb-9ay full-scale check: rebuild velocity samples from ALL 216 acqs' reference
tracks (no 250k subsample) and re-validate the graph regularizer vs the 250k run."""
import sys, os, pickle
sys.path.insert(0, 'scripts/wf_render_signal')
import numpy as np
from tractography_pde import split_half_validate

SAFE = {("numpy._core.multiarray", "_reconstruct"), ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "scalar"), ("numpy.core.multiarray", "scalar"),
        ("numpy", "ndarray"), ("numpy", "dtype")}
class SU(pickle.Unpickler):
    def find_class(self, m, n):
        for mm in (m, m.replace("numpy._core", "numpy.core"), m.replace("numpy.core", "numpy._core")):
            if (mm, n) in SAFE: return super().find_class(mm, n)
        raise pickle.UnpicklingError(m + "." + n)

o = SU(open("outputs/reference/full_tracks_smoothed.pkl", "rb")).load()
tr = o["tracks_smoothed"]
mids, vels, par = [], [], []
for t in tr:
    p = np.asarray(t["positions"], float); f = np.asarray(t["frames"], float); a = int(t["acq_index"])
    if len(p) < 2: continue
    dp = np.diff(p, axis=0); dfr = np.diff(f); dfr[dfr == 0] = 1.0
    vels.append(dp / dfr[:, None]); mids.append(0.5 * (p[:-1] + p[1:])); par.append(np.full(len(dp), a & 1))
mids = np.concatenate(mids); vels = np.concatenate(vels); par = np.concatenate(par).astype(np.int8)
print(f"FULL scale: {len(vels)} velocity samples, {len(np.unique(par))} parity classes, "
      f"acqs pooled = all 216")
kw = dict(gamma=1.5, iters=200, dt=0.12, eps_iso=0.2)
for seed in (1, 2):
    r = split_half_validate(mids, vels, par, mode="graph", seed=seed, **kw)
    print(f"  seed{seed}: dice={r['dice_occupancy']:.4f} cos={r['direction_cosine']:.4f} occ={r['occ_vox']}")
