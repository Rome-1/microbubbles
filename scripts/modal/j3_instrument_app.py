"""J3 (bead mb-fmj): the injection + null instrument, and the joint operating-point sweep.

WHAT THIS RUNS
--------------
1. Measures the complex PSF from isolated bright detections on a real acquisition.
2. Injects synthetic bubbles with known trajectories into the beamformed COMPLEX
   volume (several independent realizations), spanning 0-130 mm/s including the
   88.97 mm/s compounding null, four depth bands and six SNRs.
3. Builds null data that cannot contain bubbles.
4. Sweeps {rank24, knee, knee+spatial-correlation} x {with/without the MP high
   cutoff} x {per-elev, per-(elev,z-band), spatial TGC} = 18 operating points,
   and scores injection recovery **at matched detection density** on the
   un-injected volume, cross-checked at matched false-alarm rate on the nulls.

WHY IT DOES NOT RE-BEAMFORM
---------------------------
``microbubbles/corrected_bf_1.h5`` survived the full-run cleanup: 1.9 GiB, acq 0
of the corrected 216-acquisition export, produced by ``recreate_app.recreate``
with ``--spatial-tgc`` at elev 25 / coarseness 0.5. Its provenance is known
exactly, so re-beamforming would spend GPU-minutes to reproduce a file we
already have. ``--bf-path`` points elsewhere if a second acquisition is wanted
for replication.

    modal run scripts/modal/j3_instrument_app.py::sweep
"""

from __future__ import annotations

import json
import sys
import time

import modal

PROJECT = "microbubbles"
DATA_ROOT = f"/root/data/{PROJECT}"
REPO = "/home/rome/gt/microbubbles/crew/cajal/ultratrace_ulm"

vol = modal.Volume.from_name("research")

gpu_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("h5py==3.11.0", "numpy<2.0", "scipy==1.13.1", "cupy-cuda12x==13.3.0")
    # The local package, not the pinned GitHub SHA: inject.py and opsweep.py are
    # this bead's own code and do not exist upstream.
    .add_local_dir(REPO, remote_path="/workspace/ultratrace_ulm")
)

# Beamforming needs the MACH kernel and the upstream package; the sweep does not,
# so it is a separate, heavier image used only to mint a replication acquisition.
POSTFIX_SHA = "193900631ad7eabdce1e39b7f3ecae7d7d1c88ad"
bf_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .pip_install("h5py==3.11.0", "numpy<2.0", "scipy==1.13.1", "tqdm==4.66.4",
                 "torch==2.4.1", "cupy-cuda12x==13.3.0", "mach-beamform")
    .pip_install(
        f"ultratrace-ulm-pipeline @ git+https://github.com/alephneuro/microbubbles.git@{POSTFIX_SHA}",
        extra_options="--no-deps",
    )
)

app = modal.App(f"{PROJECT}-j3-instrument")


@app.function(image=bf_image, gpu="A100-80GB", timeout=3600, memory=262144, cpu=16.0,
              volumes={"/root/data": vol})
def beamform_acq(acq_start: int = 1, out: str = "j3_bf_acq1.h5") -> dict:
    """Mint one more beamformed acquisition, for replicating the sweep's verdict.

    Same settings as the 216-acquisition corrected run (``--spatial-tgc``, 25
    elevation planes, coarseness 0.5), so the replication acquisition is
    processed identically to the one already on the volume rather than being a
    second, subtly different experiment.
    """
    import subprocess

    h5 = f"{DATA_ROOT}/sanitized_neutral_ultratrace_216.h5"
    dst = f"{DATA_ROOT}/{out}"
    cmd = ["ultratrace-ulm", "beamform", "--input", h5, "--output", dst,
           "--acq-start", str(acq_start), "--num-acqs", "1", "--spatial-tgc"]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    vol.commit()
    return {"out": out, "acq_start": acq_start}

# The 18 operating points. Filter arms are outer because they share a Gram.
FILTER_SPECS = [
    (f"{m}{'+mp' if h else ''}", m, h)
    for m in ("rank24", "knee", "knee_spatial")
    for h in (False, True)
]
NORMS = ("per_elev", "per_elev_zband", "spatial_tgc")

# The density the sane-threshold pipeline runs at: base60's ~23.6 k localizations
# per 240-frame acquisition, i.e. ~98/frame, and the ~34/frame the reference-
# matched run sits at. Sweeping a few makes the ranking's dependence on the
# budget visible rather than assumed.
TARGET_DENSITIES = (10.0, 34.0, 98.0)
PRIMARY_DENSITY = 34.0


def _load(path: str, acq: int):
    import h5py
    import numpy as np

    with h5py.File(path, "r") as h5:
        keys = sorted(h5["acquisitions"].keys(), key=int)
        grp = h5[f"acquisitions/{keys[acq]}"]
        meta = grp["meta"] if "meta" in grp else grp
        comp = np.asarray(meta["compound_image"], dtype=np.complex64)
        g = meta["grid"]
        gz, gy, gx = np.asarray(g["z"]), np.asarray(g["y"]), np.asarray(g["x"])
    d_z = float(abs(gz[1, 0, 0] - gz[0, 0, 0])) * 1000.0
    d_x = float(abs(gx[0, 0, 1] - gx[0, 0, 0])) * 1000.0
    d_e = float(abs(gy[0, 1, 0] - gy[0, 0, 0])) * 1000.0
    return comp, dict(d_elev_mm=d_e, d_z_mm=d_z, d_x_mm=d_x, z0_mm=float(gz[0, 0, 0]) * 1000.0)


@app.function(
    image=gpu_image,
    gpu="A100-80GB",
    timeout=4 * 3600,
    memory=196608,
    cpu=8.0,
    volumes={"/root/data": vol},
)
def sweep(
    bf_path: str = "corrected_bf_1.h5",
    acq: int = 0,
    n_realizations: int = 6,
    n_bubbles: int = 300,
    lifetime: int = 24,
    n_z_bands: int = 8,
    out: str = "j3_sweep.json",
) -> dict:
    sys.path.insert(0, "/workspace")
    import cupy as cp
    import numpy as np

    from ultratrace_ulm import opsweep
    from ultratrace_ulm.inject import (
        Geometry,
        NULL_DESCRIPTIONS,
        estimate_complex_psf,
        inject_bubbles,
        make_bubble_plan,
        null_block_shuffle,
        null_phase_surrogate,
        null_quiet_crop,
        reference_noise_sigma,
    )
    from ultratrace_ulm.gpu_svd import filter_svd_3d_gpu

    t0 = time.time()
    comp, spacing = _load(f"{DATA_ROOT}/{bf_path}", acq)
    geom = Geometry(frame_rate_hz=222.43, **spacing)
    n_f = int(comp.shape[0])
    report: dict = {
        "bf_path": bf_path, "acq": acq, "shape": list(comp.shape), "spacing_mm": spacing,
        "null_descriptions": NULL_DESCRIPTIONS, "target_densities": list(TARGET_DENSITIES),
        "primary_density_per_frame": PRIMARY_DENSITY,
    }
    print(f"[j3] loaded {comp.shape} {comp.dtype} spacing={spacing}", flush=True)

    # ---- 1. Measure the PSF on a baseline-filtered COMPLEX volume ----------- #
    base_complex = filter_svd_3d_gpu(comp, low_cutoff=0.1, method="fast")
    base_mag = np.abs(base_complex).astype(np.float32)
    zb = opsweep.zscore_field(cp.asarray(base_mag), mode="per_elev", xp=cp)
    pk = opsweep.find_peaks(zb, min_distance=2, floor=3.0, xp=cp)
    del zb
    cp.get_default_memory_pool().free_all_blocks()
    thr_psf = opsweep.threshold_for_count(pk, int(PRIMARY_DENSITY * n_f))
    sel = pk.at(thr_psf)
    dets = []
    bounds = np.searchsorted(sel.frame, np.arange(n_f + 1))
    for f in range(n_f):
        lo, hi = int(bounds[f]), int(bounds[f + 1])
        dets.append((
            np.stack([sel.e[lo:hi], sel.z[lo:hi], sel.x[lo:hi]], axis=1).astype(np.int32),
            np.ones(hi - lo, np.float32), sel.zscore[lo:hi],
        ))
    psf = estimate_complex_psf(base_complex, dets, patch_radius=(3, 6, 6))
    report["psf"] = {
        "n_patches": psf.n_patches, "k_axial_rad_per_voxel": psf.k_axial,
        "radius": list(psf.radius), "psf_mining_threshold": float(thr_psf),
        # A -3 dB extent read off the measured kernel, per axis, in mm. The
        # elevation number is the one to watch: it is the weak axis.
        "fwhm_mm": _fwhm(np.abs(psf.kernel()), geom),
    }
    print(f"[j3] PSF: {report['psf']}", flush=True)
    del base_complex, base_mag, pk, sel
    cp.get_default_memory_pool().free_all_blocks()

    # ---- 2. Reference noise, injection, nulls ------------------------------ #
    sigma, band_edges = reference_noise_sigma(comp, ref_rank=8, n_z_bands=n_z_bands)
    report["reference_sigma_by_zband"] = sigma.mean(axis=0).tolist()

    volumes: dict[str, object] = {"clean": comp}
    truths: dict[str, dict] = {}
    for r in range(int(n_realizations)):
        plan = make_bubble_plan(comp.shape, geom, n_bubbles=n_bubbles, lifetime=lifetime,
                                n_z_bands=n_z_bands, seed=100 + r)
        inj, truth = inject_bubbles(comp, plan, psf, geom, sigma, band_edges)
        volumes[f"inj{r}"] = inj
        truths[f"inj{r}"] = truth
        print(f"[j3] realization {r}: {truth['frame'].size} bubble-frames "
              f"({truth['frame'].size / n_f:.1f}/frame)", flush=True)
    report["injection"] = {
        "n_realizations": int(n_realizations), "n_bubbles_per_realization": int(n_bubbles),
        "lifetime": int(lifetime),
        "bubble_frames_per_frame": {k: float(v["frame"].size) / n_f for k, v in truths.items()},
        "compounding_gain_applied": True,
    }

    qc_ref = np.abs(filter_svd_3d_gpu(comp, low_cutoff=0.1, method="fast")).astype(np.float32)
    volumes["null_quiet_crop"], qc_origin = null_quiet_crop(comp, qc_ref, crop=(64, 96))
    del qc_ref
    volumes["null_block1"] = null_block_shuffle(comp, block=1, seed=1)
    volumes["null_block20"] = null_block_shuffle(comp, block=20, seed=2)
    volumes["null_phase"] = null_phase_surrogate(comp, seed=3)
    report["null_quiet_crop_origin_zx"] = list(qc_origin)

    # ---- 3. The sweep ------------------------------------------------------ #
    # Pass 1 fixes each arm's operating point on the CLEAN volume; passes 2 and 3
    # only ever apply thresholds decided there. No arm is allowed to pick its own.
    voxel_mm3 = float(np.prod(geom.voxel_mm))
    results: dict[str, dict] = {}
    order = ["clean"] + [k for k in volumes if k != "clean"]
    for vname in order:
        v = volumes[vname]
        t1 = time.time()
        for fname, cut, mag in opsweep.svd_filter_bank(v, FILTER_SPECS, xp=cp, to_host=False):
            for norm in NORMS:
                key = f"{fname}|{norm}"
                z = opsweep.zscore_field(mag, mode=norm, n_z_bands=n_z_bands,
                                         voxel_mm=tuple(geom.voxel_mm), xp=cp)
                peaks = opsweep.find_peaks(z, min_distance=2, floor=2.0, xp=cp)
                del z
                cp.get_default_memory_pool().free_all_blocks()
                slot = results.setdefault(key, {"cutoffs": cut, "norm": norm, "filter": fname})
                # The knee is data-driven, so each volume resolves its own rank.
                # Record them: if an injected volume's rank differs from clean's,
                # the matched-density guarantee is only approximate and the
                # reader needs to know rather than guess.
                slot.setdefault("cutoffs_by_volume", {})[vname] = cut
                if vname == "clean":
                    slot["thresholds"] = {
                        f"{d:g}": opsweep.threshold_for_count(peaks, int(d * n_f))
                        for d in TARGET_DENSITIES
                    }
                    slot["clean_peaks_above_floor"] = int(peaks.zscore.size)
                elif vname.startswith("inj"):
                    rec = slot.setdefault("recovery", {})
                    for d in TARGET_DENSITIES:
                        t = slot["thresholds"][f"{d:g}"]
                        if not np.isfinite(t):
                            continue
                        hit = opsweep.match_peaks_to_truth(truths[vname], peaks.at(t))
                        acc = rec.setdefault(f"{d:g}", {"hit": [], "truth": {}})
                        acc["hit"].append(hit)
                        for kk, vv in truths[vname].items():
                            acc["truth"].setdefault(kk, []).append(vv)
                else:
                    fa = slot.setdefault("false_alarms", {})
                    for d in TARGET_DENSITIES:
                        t = slot["thresholds"][f"{d:g}"]
                        n_hit = int(np.count_nonzero(peaks.zscore >= t)) if np.isfinite(t) else 0
                        mm3 = peaks.n_voxels * voxel_mm3
                        fa.setdefault(f"{d:g}", {})[vname] = {
                            "per_frame": n_hit / n_f,
                            "per_frame_per_cm3": n_hit / n_f / (mm3 / 1000.0),
                        }
            del mag
            cp.get_default_memory_pool().free_all_blocks()
        print(f"[j3] {vname}: 18 configs in {time.time() - t1:.0f}s", flush=True)
        if vname != "clean":
            volumes[vname] = None

    # ---- 4. Score ---------------------------------------------------------- #
    for key, slot in results.items():
        rec = slot.pop("recovery", {})
        slot["recovery"] = {}
        for d, acc in rec.items():
            hit = np.concatenate(acc["hit"])
            truth = {k: np.concatenate(v) for k, v in acc["truth"].items()}
            slot["recovery"][d] = opsweep.recovery_table(truth, hit)

    report["results"] = results
    report["elapsed_s"] = time.time() - t0
    report["ranking"] = _rank(results, str(int(PRIMARY_DENSITY)) if PRIMARY_DENSITY.is_integer()
                              else f"{PRIMARY_DENSITY:g}")
    path = f"{DATA_ROOT}/{out}"
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    vol.commit()
    print(f"[j3] wrote {path} in {report['elapsed_s']:.0f}s", flush=True)
    print(json.dumps(report["ranking"], indent=2, default=float), flush=True)
    return report["ranking"]


def _fwhm(kernel, geom):
    import numpy as np

    out = {}
    for ax, name, d in ((0, "elev", geom.d_elev_mm), (1, "z", geom.d_z_mm), (2, "x", geom.d_x_mm)):
        prof = kernel.max(axis=tuple(i for i in range(3) if i != ax))
        prof = prof / max(float(prof.max()), 1e-30)
        above = np.flatnonzero(prof >= 0.5)
        out[name] = float((above[-1] - above[0] + 1) * d) if above.size else float("nan")
    return out


def _rank(results, density_key):
    import numpy as np

    rows = []
    for key, slot in results.items():
        r = slot.get("recovery", {}).get(density_key)
        if not r:
            continue
        rows.append({
            "config": key,
            "low": slot["cutoffs"]["low"],
            "high_remove": slot["cutoffs"]["high_remove"],
            "recovery": r["recovery"],
            "depth_nonuniformity": r.get("depth_nonuniformity", float("nan")),
            "n_truth": r["n_truth"],
        })
    rows.sort(key=lambda x: -x["recovery"])
    if rows:
        base = next((x for x in rows if x["config"] == "rank24|per_elev"), rows[-1])
        for x in rows:
            x["vs_status_quo"] = x["recovery"] - base["recovery"]
            n = max(x["n_truth"], 1)
            # Binomial SE on the difference, treating the two arms as independent
            # (they are not -- same injected bubbles -- so this is CONSERVATIVE).
            se = np.sqrt(2 * x["recovery"] * (1 - x["recovery"]) / n)
            x["vs_status_quo_se"] = float(se)
    return {"density_per_frame": density_key, "rows": rows}
