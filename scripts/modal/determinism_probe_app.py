"""Determinism probe for the adaptive SVD clutter filter (mb-ivt / upstream issue #2).

Question this answers: upstream closed alephneuro/microbubbles#2 by merging PR #4, which
runs the eigendecomposition in complex128. Does that actually make the DETECTIONS
deterministic, and is it the whole root cause?

Three self-contained variants of ``spectral_centroid_cutoff`` are compared on the SAME
real beamformed acquisition (no dependency on the fork's own svd.py, so the A/B is
unambiguous):

  v0_shipped  complex64 Gram + eigh, mode score |rfft(u.real)|^2      (pre-#4 upstream)
  v1_pr4      complex128 Gram + eigh, mode score |rfft(u.real)|^2     (upstream after #4)
  v2_ours     complex128 Gram + eigh, mode score |fft(u)|^2 over |f|  (our fork, 4bfaf78)

and three perturbations that model the three ways an identical input can produce a
different answer in practice:

  threads     BLAS reduction order (OMP threads 1 vs N) -- real cross-machine jitter
  epsilon     1e-6 relative input perturbation -- float32-epsilon proxy for that jitter
  phase       each eigenvector rotated by an arbitrary unit phase -- a Hermitian
              eigensolver defines eigenvectors only up to a phase, and LAPACK and
              cuSOLVER pick different ones. Invariant for v2 by construction; the
              measurement is how far v0/v1 move.

Each perturbation is scored at the cutoff level AND at the detection level (the number
Rome reported upstream: ~85% of detections differ between runs).

SPEND: CPU-only, one cropped acquisition, minutes. Verify teardown after the run.
"""

import modal

PROJECT = "microbubbles"
DATA_ROOT = f"/root/data/{PROJECT}"
vol = modal.Volume.from_name("research")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("h5py==3.11.0", "numpy<2.0", "scipy==1.13.1", "threadpoolctl==3.5.0")
    .add_local_dir("ultratrace_ulm", remote_path="/workspace/ultratrace_ulm")
)

app = modal.App(f"{PROJECT}-determinism")

PROBE = r'''
import numpy as np

def gram_c64(x):
    return x @ x.conj().T

def gram_c128(x, chunk=300_000):
    n = int(x.shape[0])
    g = np.zeros((n, n), dtype=np.complex128)
    for s0 in range(0, x.shape[1], chunk):
        mc = x[:, s0:s0 + chunk]
        g += (mc @ mc.conj().T).astype(np.complex128)
    return g

def modes(matrix, precision):
    """Mean-subtracted temporal Gram -> eigenvectors sorted by descending eigenvalue."""
    if precision == "c64":
        x = matrix - matrix.mean(axis=0, keepdims=True)
        cov = gram_c64(x)
    else:
        x = np.asarray(matrix, dtype=np.complex128)
        x = x - x.mean(axis=0, keepdims=True)
        cov = gram_c128(x)
    evals, u = np.linalg.eigh(cov)
    return u[:, np.argsort(evals)[::-1]]

def score_realpart(u, frame_rate_hz):
    """Shipped score: |rfft(u.real)|^2 -- PHASE-DEPENDENT."""
    n = int(u.shape[0])
    freqs = np.fft.rfftfreq(n, d=1.0 / frame_rate_hz)
    centroid = np.zeros(n)
    for i in range(n):
        spectrum = np.abs(np.fft.rfft(u[:, i].real)) ** 2
        spectrum[0] = 0.0
        total = spectrum.sum()
        if total > 0:
            centroid[i] = float(np.sum(freqs * spectrum) / total)
    return centroid

def score_complex(u, frame_rate_hz):
    """Our score: |fft(u)|^2 over |f| -- PHASE-INVARIANT."""
    n = int(u.shape[0])
    freqs = np.fft.fftfreq(n, d=1.0 / frame_rate_hz)
    spec = np.abs(np.fft.fft(u, axis=0)) ** 2
    spec[0, :] = 0.0
    total = spec.sum(axis=0)
    return (np.abs(freqs)[:, None] * spec).sum(axis=0) / np.where(total > 0, total, 1.0)

def cutoff_from_centroid(centroid, tissue_freq_hz, n_frames):
    above = np.where(centroid > tissue_freq_hz)[0]
    return int(above[0]) if len(above) else max(1, round(n_frames * 0.1))

VARIANTS = {
    "v0_shipped": ("c64", score_realpart),
    "v1_pr4":     ("c128", score_realpart),
    "v2_ours":    ("c128", score_complex),
}

def cutoff(matrix, variant, frame_rate_hz, tissue_freq_hz=100.0, phase_seed=None):
    precision, scorer = VARIANTS[variant]
    u = modes(matrix, precision)
    if phase_seed is not None:
        rng = np.random.default_rng(phase_seed)
        ph = np.exp(1j * rng.uniform(0, 2 * np.pi, size=u.shape[1]))
        u = u * ph[None, :]
    return cutoff_from_centroid(scorer(u, frame_rate_hz), tissue_freq_hz, int(matrix.shape[0])), u

def project(matrix, u, low):
    """The 'fast' covariance projection at the given low cutoff -> |filtered| volume.

    Held at complex64 for EVERY variant on purpose: the projection precision is a
    separate (already-measured) effect (mb-a0a, docs/ideation/15), and fixing it here
    isolates what this probe is asking about -- the cutoff and the eigenvectors.
    """
    uc = np.asarray(u[:, low:], dtype=np.complex64)
    return np.abs(uc @ (uc.conj().T @ matrix)).astype(np.float32)
'''


@app.function(image=image, timeout=3 * 3600, memory=131072, cpu=16.0,
              volumes={"/root/data": vol})
def centroids(acq_h5: str = "beamformed/base60_refs/acq_0000.h5",
              crop: bool = False, frame_rate: float = 222.0,
              tissue_freq_hz: float = 100.0) -> dict:
    """Cutoff-only pass on the FULL acquisition: dumps the per-mode centroid curve
    so the cutoff's margin over ``tissue_freq_hz`` is visible, not just the integer
    it lands on. Cheap (no detection), so it can run on the uncropped matrix."""
    import json
    import sys

    sys.path.insert(0, "/workspace")
    import numpy as np
    from threadpoolctl import threadpool_limits

    from ultratrace_ulm.h5_io import open_h5, acq_keys, load_compound

    ns: dict = {}
    exec(PROBE, ns)

    vol.reload()
    with open_h5(f"{DATA_ROOT}/{acq_h5}") as h5:
        aid = acq_keys(h5)[0]
        comp = load_compound(h5, aid)
    if crop:
        comp = comp[:, :, 60:180, 100:280]
    n_frames = int(comp.shape[0])
    matrix = np.asarray(comp, dtype=np.complex64).reshape(n_frames, -1)
    del comp
    print(f"[centroids] acq={aid} matrix={matrix.shape}", flush=True)

    out: dict = {"acq": str(aid), "n_frames": n_frames, "n_vox": int(matrix.shape[1]),
                 "tissue_freq_hz": tissue_freq_hz, "variants": {}}
    for variant in ("v0_shipped", "v1_pr4", "v2_ours"):
        precision, scorer = ns["VARIANTS"][variant]
        rec: dict = {}
        for tag, threads, seed, phase in (("baseline", 16, None, None),
                                          ("threads1", 1, None, None),
                                          ("eps1e-6", 16, 7, None),
                                          ("phase_a", 16, None, 11),
                                          ("phase_b", 16, None, 12)):
            m = matrix
            if seed is not None:
                rng = np.random.default_rng(seed)
                m = (matrix * (1.0 + 1e-6 * rng.standard_normal(matrix.shape).astype(np.float32))
                     ).astype(np.complex64)
            with threadpool_limits(limits=threads):
                u = ns["modes"](m, precision)
                if phase is not None:
                    rng = np.random.default_rng(phase)
                    u = u * np.exp(1j * rng.uniform(0, 2 * np.pi, size=u.shape[1]))[None, :]
                c = scorer(u, frame_rate)
            low = ns["cutoff_from_centroid"](c, tissue_freq_hz, n_frames)
            above = np.where(c > tissue_freq_hz)[0]
            rec[tag] = {"cutoff": int(low), "fallback": bool(len(above) == 0),
                        "centroid_max_hz": float(c.max()),
                        "n_modes_above": int(len(above)),
                        "centroid_head": [round(float(v), 3) for v in c[:12]],
                        "centroid_near_70": [round(float(v), 3) for v in c[65:80]]}
            print(f"[{variant}] {tag}: cutoff={low} fallback={rec[tag]['fallback']} "
                  f"max={rec[tag]['centroid_max_hz']:.1f}Hz n_above={len(above)}", flush=True)
        out["variants"][variant] = rec

    dest = f"{DATA_ROOT}/determinism_centroids.json"
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=2)
    vol.commit()
    return out


@app.function(image=image, timeout=3 * 3600, memory=131072, cpu=16.0,
              volumes={"/root/data": vol})
def centroid_survey(acqs: str = "0,55,111,222", frame_rate: float = 222.0,
                    tissue_freq_hz: float = 100.0) -> dict:
    """Does the adaptive cutoff EVER fire on this dataset, or only on acq 0?

    Runs the shipped score over several acquisitions (full volume, baseline config only)
    and reports the highest mode centroid against the tissue boundary."""
    import json
    import sys

    sys.path.insert(0, "/workspace")
    import numpy as np

    from ultratrace_ulm.h5_io import open_h5, acq_keys, load_compound

    ns: dict = {}
    exec(PROBE, ns)
    vol.reload()

    out: dict = {"tissue_freq_hz": tissue_freq_hz, "acqs": {}}
    for acq in [int(a) for a in acqs.split(",") if a.strip()]:
        path = f"{DATA_ROOT}/beamformed/base60_refs/acq_{acq:04d}.h5"
        with open_h5(path) as h5:
            aid = acq_keys(h5)[0]
            comp = load_compound(h5, aid)
        n_frames = int(comp.shape[0])
        matrix = np.asarray(comp, dtype=np.complex64).reshape(n_frames, -1)
        del comp
        rec = {}
        for variant in ("v0_shipped", "v2_ours"):
            precision, scorer = ns["VARIANTS"][variant]
            c = scorer(ns["modes"](matrix, precision), frame_rate)
            above = np.where(c > tissue_freq_hz)[0]
            rec[variant] = {"cutoff": ns["cutoff_from_centroid"](c, tissue_freq_hz, n_frames),
                            "fallback": bool(len(above) == 0),
                            "centroid_max_hz": round(float(c.max()), 2),
                            "n_modes_above": int(len(above))}
        del matrix
        out["acqs"][str(acq)] = rec
        print(f"[acq {acq}] {rec}", flush=True)

    with open(f"{DATA_ROOT}/determinism_centroid_survey.json", "w") as fh:
        json.dump(out, fh, indent=2)
    vol.commit()
    return out


@app.function(image=image, timeout=3 * 3600, memory=65536, cpu=16.0,
              volumes={"/root/data": vol})
def probe(acq_h5: str = "beamformed/base60_refs/acq_0000.h5",
          crop: bool = True, frame_rate: float = 222.0,
          sigma_threshold: float = 2.0, min_distance: int = 2,
          smoothing_sigma: float = 1.0, detections: bool = True) -> dict:
    import json
    import sys
    import time

    sys.path.insert(0, "/workspace")
    import numpy as np
    from threadpoolctl import threadpool_limits

    from ultratrace_ulm.h5_io import open_h5, acq_keys, load_compound, grid_arrays
    from ultratrace_ulm.tracking import detect_batch

    ns: dict = {}
    exec(PROBE, ns)

    vol.reload()
    path = f"{DATA_ROOT}/{acq_h5}"
    with open_h5(path) as h5:
        aid = acq_keys(h5)[0]
        comp = load_compound(h5, aid)
        gx, gy, gz = grid_arrays(h5, aid)
    if crop:
        comp = comp[:, :, 60:180, 100:280]
        gx, gy, gz = (g[:, 60:180, 100:280] for g in (gx, gy, gz))
    n_frames = int(comp.shape[0])
    spatial_shape = comp.shape[1:]
    matrix = np.asarray(comp, dtype=np.complex64).reshape(n_frames, -1)
    print(f"[probe] acq={aid} shape={comp.shape} matrix={matrix.shape}", flush=True)
    # mm travelled per unit step along each INDEX axis (elev, z, x) of the grid,
    # so detection displacements can be reported in mm rather than voxels.
    steps = []
    for a in range(3):
        d = np.sqrt(sum(np.diff(g.astype(np.float64), axis=a) ** 2 for g in (gx, gy, gz)))
        steps.append(float(np.median(d)) if d.size else 1.0)
    vox_mm = np.array(steps, dtype=np.float64)
    print(f"[probe] voxel step mm (elev,z,x) = {vox_mm}", flush=True)

    def detect(u, low):
        mag = ns["project"](matrix, u, low)
        mag = mag.reshape((n_frames,) + tuple(spatial_shape))
        batch = detect_batch(mag, sigma_threshold=sigma_threshold,
                             min_distance=min_distance, smoothing_sigma=smoothing_sigma)
        pts = []
        for f, frame in enumerate(batch):
            pix = frame[0]
            if len(pix):
                pts.append(np.column_stack([np.full(len(pix), f, np.int32), pix]))
        return np.concatenate(pts, 0) if pts else np.zeros((0, 4), np.int32)

    def compare(a, b):
        """Exact-match overlap + median NN distance (mm) of b's points to a's, per frame."""
        if len(a) == 0 or len(b) == 0:
            return {"n_a": int(len(a)), "n_b": int(len(b)), "overlap": None, "median_nn_mm": None}
        sa = {tuple(int(v) for v in row) for row in a}
        sb = {tuple(int(v) for v in row) for row in b}
        inter = len(sa & sb)
        by_frame: dict = {}
        for row in a:
            by_frame.setdefault(int(row[0]), []).append(row[1:])
        nn = []
        for row in b:
            ref = by_frame.get(int(row[0]))
            if not ref:
                continue
            d = (np.asarray(ref, np.float64) - np.asarray(row[1:], np.float64)) * vox_mm
            nn.append(float(np.sqrt((d ** 2).sum(axis=1)).min()))
        return {"n_a": int(len(a)), "n_b": int(len(b)),
                "overlap": inter / max(len(sa), len(sb)),
                "median_nn_mm": float(np.median(nn)) if nn else None}

    out: dict = {"acq": str(aid), "shape": list(map(int, comp.shape)),
                 "frame_rate_hz": frame_rate, "variants": {}}

    for variant in ("v0_shipped", "v1_pr4", "v2_ours"):
        t0 = time.time()
        rec: dict = {}
        # --- baseline (16 threads) ---
        with threadpool_limits(limits=16):
            low_base, u_base = ns["cutoff"](matrix, variant, frame_rate)
        # --- BLAS reduction-order perturbation (1 thread) ---
        with threadpool_limits(limits=1):
            low_thr, u_thr = ns["cutoff"](matrix, variant, frame_rate)
        # --- float32-epsilon input perturbation ---
        rng = np.random.default_rng(7)
        pert = (matrix * (1.0 + 1e-6 * rng.standard_normal(matrix.shape).astype(np.float32))
                ).astype(np.complex64)
        with threadpool_limits(limits=16):
            low_eps, u_eps = ns["cutoff"](pert, variant, frame_rate)
        # --- eigenvector phase convention (LAPACK vs cuSOLVER) ---
        with threadpool_limits(limits=16):
            low_ph, u_ph = ns["cutoff"](matrix, variant, frame_rate, phase_seed=11)
            low_ph2, _ = ns["cutoff"](matrix, variant, frame_rate, phase_seed=12)

        rec["cutoff"] = {"baseline": low_base, "threads1": low_thr, "eps1e-6": low_eps,
                         "phase_a": low_ph, "phase_b": low_ph2}
        print(f"[{variant}] cutoffs {rec['cutoff']}  ({time.time()-t0:.0f}s)", flush=True)

        if detections:
            d_base = detect(u_base, low_base)
            rec["detections"] = {"baseline_n": int(len(d_base))}
            for tag, (u_p, low_p) in {"threads1": (u_thr, low_thr),
                                      "eps1e-6": (u_eps, low_eps),
                                      "phase": (u_ph, low_ph)}.items():
                rec["detections"][tag] = compare(d_base, detect(u_p, low_p))
                print(f"[{variant}] {tag}: {rec['detections'][tag]}", flush=True)
        out["variants"][variant] = rec

    dest = f"{DATA_ROOT}/determinism_probe.json"
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=2)
    vol.commit()
    print(json.dumps(out, indent=2), flush=True)
    return out
