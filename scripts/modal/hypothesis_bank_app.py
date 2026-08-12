"""J2 (mb-8t2) — motion-phase hypothesis-bank compounding.

The pipeline coherently sums 4 steered transmits per frame. That sum is not a neutral SNR
average: a scatterer moving axially at v advances the round-trip phase by
`dphi = 4*pi*v/(lambda*PRF)` between consecutive transmits, so the 4-phasor sum has gain
`|sin(2*dphi) / (4*sin(dphi/2))|`. With lambda = 0.800 mm (tx 2.0 MHz, c 1600 m/s) and
PRF = 4*FR = 889.7 Hz that is a NULL at `lambda*FR/2 = 88.97 mm/s` axial, and -1.7 to -7.5 dB
across the bulk of the measured speed range.

The same 88.97 mm/s is, by algebra, the tracker's 2-voxel/frame gate wall (voxel = lambda/4,
so 2 voxels/frame = lambda*FR/2). Gate-censoring and compounding-annihilation are therefore
confounded in every track-level statistic and separate only by axis: the null suppresses
AXIAL motion only. This experiment is the only way to size the annihilated axial tail, because
no downstream measurement can see detections that were never made.

Rather than the single-angle path (which would throw away the compound's sidelobe suppression
and, as codex correctly objected, treats four different angle operators as one uniform movie),
this re-sums the per-angle stack under a bank of conjugate phase ramps:

    compound_k = sum_n  bf_n * exp(-i * n * dphi(v_k))

and detects on the per-voxel max over the bank, recording the argmax as a coarse per-detection
axial-velocity prior the tracker can later consume.

Outputs, all GT-free:
  * the measured coherent-gain-vs-speed curve against the parameter-free physics prediction;
  * a census of detections found under a moving hypothesis but absent from the standard
    compound (the size of the annihilated tail), filtered by frame-to-frame persistence;
  * the argmax velocity map, checkable later against independently tracked step velocities.

Cost: the DAS is done once per acquisition; each extra hypothesis is a weighted sum plus a
magnitude and a running max over a ~2 GB array. Budget ~$10 for 3 acquisitions.
"""

import modal

PROJECT = "microbubbles"
DATA_ROOT = f"/root/data/{PROJECT}"
H5 = f"{DATA_ROOT}/sanitized_neutral_ultratrace_216.h5"

vol = modal.Volume.from_name("research")

gpu_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("h5py==3.11.0", "numpy<2.0", "scipy==1.13.1", "tqdm==4.66.4",
                 "torch==2.4.1", "cupy-cuda12x==13.3.0", "mach-beamform")
    .add_local_dir("ultratrace_ulm", remote_path="/workspace/ultratrace_ulm")
)

app = modal.App(f"{PROJECT}-hypothesis-bank")


@app.function(image=gpu_image, gpu="A100-80GB", timeout=4 * 3600, memory=262144, cpu=16.0,
              volumes={"/root/data": vol})
def bank_v3(n_acqs: int = 2, acq_start: int = 0, sigma_threshold: float = 2.0,
            min_distance: int = 2, smoothing_sigma: float = 1.0) -> dict:
    """The coherence test done correctly (mb-8t2).

    bank_v2 took its detection coordinates from the POST-SVD volume but evaluated the
    coherence ratio on the RAW pre-SVD per-angle field, where tissue clutter sits 25-40 dB
    above the bubble. So it measured the 4-angle coherence of CLUTTER, which is static and
    therefore flat in velocity by construction -- the flat 0.52 was guaranteed regardless of
    what the bubble component does. That result is withdrawn; it refutes nothing about bubbles.

    Here each angle stack is SVD-filtered on its own with the same adaptive cutoff BEFORE the
    ratio is formed, so the ratio is evaluated in the subspace the detector actually works in.

    PRE-REGISTERED, so this cannot be re-interpreted after the fact:
      * flat at ~0.5 across velocity bands  -> the compounding-null line is dead. Abandon the
        bank, and flip `compounding_gain_on` to default False in inject.py, since a refuted
        notch hand-applied to injections biases every future axial recovery number downward.
      * a sweep, or structure correlated with the velocity pick -> the bubble component is
        point-like and velocity-coherent, and the bank earns one properly-powered rebuild.
    """
    import json
    import sys
    import time

    sys.path.insert(0, "/workspace")
    import h5py
    import numpy as np
    from scipy.ndimage import gaussian_filter, maximum_filter

    from ultratrace_ulm.beamform_core import beamform_iq
    from ultratrace_ulm.beamform_mach import _load_acq, _load_neutral_config
    from ultratrace_ulm.svd import filter_svd_3d

    class _Opts:
        elev_planes = 25
        z_coarseness = 0.5
        x_coarseness = 0.5
        large_fov = True
        xlarge_fov = False
        row_index = None
        mean_subtract_channels = True

    vol.reload()
    out = []
    with h5py.File(H5, "r") as h5:
        config = _load_neutral_config(h5, _Opts())
        attrs = dict(h5["config"].attrs)
        lam_mm = float(config.speed_of_sound_m_s) / float(config.tx_freq_hz) * 1000.0
        fr = float(attrs.get("frame_rate_hz", 222.4306816130359))
        prf = 4.0 * fr
        ids = [int(k) for k in sorted(h5["acquisitions"].keys(), key=int)][acq_start:acq_start + n_acqs]

        for aid in ids:
            t0 = time.time()
            iq, txd, txd_e = _load_acq(h5, aid)
            angles, _ = beamform_iq(iq, txd, txd_e, config, return_angles=True)
            n_ang = angles.shape[0]

            # per-angle SVD filtering, same adaptive cutoff, complex output retained
            filt = np.stack([filter_svd_3d(angles[a], method="adaptive", frame_rate_hz=fr)
                             for a in range(n_ang)], axis=0).astype(np.complex64)
            del angles

            # detections from the standard (summed-then-filtered) path, unchanged
            std_mag = np.abs(filter_svd_3d(filt.sum(axis=0), method="adaptive",
                                           frame_rate_hz=fr)).astype(np.float32)
            sm = gaussian_filter(std_mag, sigma=(0, 0, smoothing_sigma, smoothing_sigma))
            n_elev = sm.shape[1]
            mean = np.zeros(n_elev, np.float32); sd = np.ones(n_elev, np.float32)
            for e in range(n_elev):
                pos = sm[:, e][sm[:, e] > 0]
                if pos.size:
                    mean[e] = pos.mean(); s = pos.std(); sd[e] = s if s > 1e-10 else 1.0
            z = (sm - mean[None, :, None, None]) / (sd[None, :, None, None] + 1e-10)
            fs = 2 * int(min_distance) + 1
            pk = (z == maximum_filter(z, size=(1, fs, fs, fs))) & (z > sigma_threshold)
            idx = tuple(np.array(np.where(pk)))
            zs = z[pk]

            pa = filt[(slice(None),) + idx]
            ratio = (np.abs(pa.sum(axis=0)) / np.maximum(np.abs(pa).sum(axis=0), 1e-20))
            # per-detection axial phase slope across the 4 transmits -> implied velocity
            ph = np.angle(pa[1:] * np.conj(pa[:-1]))          # (A-1, N) inter-transmit phase
            slope = np.angle(np.exp(1j * ph).mean(axis=0))
            v_implied = slope * (lam_mm / 1000.0) * prf / (4.0 * np.pi) * 1000.0

            bands = []
            for lo, hi in ((0, 20), (20, 40), (40, 60), (60, 80), (80, 100), (100, 131)):
                m = (np.abs(v_implied) >= lo) & (np.abs(v_implied) < hi)
                if m.sum() < 50:
                    continue
                vm = float(np.median(np.abs(v_implied[m])))
                d = 4.0 * np.pi * (vm / 1000.0) / ((lam_mm / 1000.0) * prf)
                pred = abs(np.sin(2 * d) / (4 * np.sin(d / 2))) if d else 1.0
                bands.append({"band": [lo, hi], "n": int(m.sum()), "v_median": round(vm, 1),
                              "observed_ratio": round(float(np.median(ratio[m])), 4),
                              "predicted_ratio": round(float(pred), 4)})

            # export per-detection coherence + coordinates so the curve can be re-binned
            # against INDEPENDENTLY tracked step velocity. The in-run binning uses velocity
            # implied from the inter-angle phase slope, which is derived from the same phasors
            # as the ratio -- noise inflates both, so that curve cannot stand on its own.
            np.savez_compressed(
                f"{DATA_ROOT}/coherence_acq{aid}.npz",
                frame=idx[0].astype(np.int32), elev=idx[1].astype(np.int32),
                z=idx[2].astype(np.int32), x=idx[3].astype(np.int32),
                ratio=ratio.astype(np.float32), zscore=zs.astype(np.float32),
                v_implied=v_implied.astype(np.float32))

            rec = {"acq": str(aid), "n_detections": int(len(zs)),
                   "ratio_median_all": round(float(np.median(ratio)), 4),
                   "ratio_p10": round(float(np.percentile(ratio, 10)), 4),
                   "ratio_p90": round(float(np.percentile(ratio, 90)), 4),
                   "speckle_value_1_over_sqrtA": round(float(1 / np.sqrt(n_ang)), 4),
                   "bands": bands, "seconds": round(time.time() - t0, 1)}
            print("[acq %s] " % aid + json.dumps(rec, indent=2), flush=True)
            out.append(rec)
            del filt, std_mag, sm, z
            vol.commit()

    rep = {"lambda_mm": round(lam_mm, 4), "acqs": out}
    with open(f"{DATA_ROOT}/hypothesis_bank_v3.json", "w") as fh:
        json.dump(rep, fh, indent=2)
    vol.commit()
    print("BANK_V3 " + json.dumps(rep, indent=2), flush=True)
    return rep


@app.function(image=gpu_image, gpu="A100-80GB", timeout=4 * 3600, memory=262144, cpu=16.0,
              volumes={"/root/data": vol})
def bank_v2(n_acqs: int = 2, acq_start: int = 0, v_max: float = 130.0, n_hyp: int = 13,
            sigma_threshold: float = 2.0, min_distance: int = 2,
            smoothing_sigma: float = 1.0) -> dict:
    """Corrected J2. The first run compared arms at a FIXED 2-sigma threshold, but taking a
    per-voxel max over 13 correlated hypotheses raises the noise floor and inflates the
    z-score denominator, so the same threshold means different things in each arm. It
    reported 81k "new" detections while finding FEWER total than the baseline, and 93% of
    voxels preferring a non-zero velocity -- the signature of fitting noise, not recovering
    signal. That result is discarded.

    Three changes:

      CONTROL     a bank containing only v=0 must reproduce the standard compound EXACTLY
                  (the ramp is all-ones, so the weighted sum is the plain sum). Asserted
                  before anything else runs. This is the correctness check the first run
                  lacked.

      DENSITY-    detections are compared at MATCHED COUNT per frame (top-N by z, N from the
      MATCHED     standard arm) rather than at a matched threshold, so "which detections
                  change" is separated from "how the noise floor moved".

      PHYSICS     the parameter-free diagnostic that was missing: at each standard detection,
      CURVE       measure the coherence ratio |sum_n bf_n| / sum_n |bf_n| and the bank's
                  argmax velocity, then compare the binned ratio against the predicted
                  |sin(2*dphi)/(4*sin(dphi/2))|. Theory fixes this curve with no free
                  parameters -- it either lands on the data or it does not.
    """
    import json
    import sys
    import time

    sys.path.insert(0, "/workspace")
    import h5py
    import numpy as np
    from scipy.ndimage import gaussian_filter, maximum_filter

    from ultratrace_ulm.beamform_core import beamform_iq
    from ultratrace_ulm.beamform_mach import _load_acq, _load_neutral_config
    from ultratrace_ulm.svd import filter_svd_3d

    class _Opts:
        elev_planes = 25
        z_coarseness = 0.5
        x_coarseness = 0.5
        large_fov = True
        xlarge_fov = False
        row_index = None
        mean_subtract_channels = True

    vol.reload()
    out_acqs = []
    with h5py.File(H5, "r") as h5:
        config = _load_neutral_config(h5, _Opts())
        attrs = dict(h5["config"].attrs)
        lam_mm = float(config.speed_of_sound_m_s) / float(config.tx_freq_hz) * 1000.0
        frame_rate = float(attrs.get("frame_rate_hz", 222.4306816130359))
        prf = 4.0 * frame_rate
        velocities = np.linspace(-v_max, v_max, n_hyp)
        acq_ids = [int(k) for k in sorted(h5["acquisitions"].keys(), key=int)][acq_start:acq_start + n_acqs]
        print(f"[cfg] lambda={lam_mm:.3f} mm null={lam_mm*frame_rate/2:.2f} mm/s", flush=True)

        def dphi_of(v):
            return 4.0 * np.pi * (v / 1000.0) / ((lam_mm / 1000.0) * prf)

        def detect_counts(mag):
            sm = gaussian_filter(mag, sigma=(0, 0, smoothing_sigma, smoothing_sigma))
            n_elev = sm.shape[1]
            mean = np.zeros(n_elev, np.float32); sd = np.ones(n_elev, np.float32)
            for e in range(n_elev):
                pos = sm[:, e][sm[:, e] > 0]
                if pos.size:
                    mean[e] = pos.mean(); s = pos.std(); sd[e] = s if s > 1e-10 else 1.0
            z = (sm - mean[None, :, None, None]) / (sd[None, :, None, None] + 1e-10)
            fs = 2 * int(min_distance) + 1
            pk = (z == maximum_filter(z, size=(1, fs, fs, fs))) & (z > sigma_threshold)
            c = np.array(np.where(pk))
            return c, z[pk].astype(np.float32)

        for aid in acq_ids:
            t0 = time.time()
            iq, txd, txd_e = _load_acq(h5, aid)
            angles, _ = beamform_iq(iq, txd, txd_e, config, return_angles=True)
            n_ang = angles.shape[0]
            std = angles.sum(axis=0)

            # ---- CONTROL: v=0 bank must be the standard compound, exactly ----
            zero = np.tensordot(np.ones(n_ang, np.complex64), angles, axes=(0, 0))
            ctrl_max = float(np.abs(zero - std).max())
            print(f"[acq {aid}] v=0 control max|diff| = {ctrl_max:.3e}", flush=True)
            del zero
            if ctrl_max > 1e-3:
                raise RuntimeError(f"v=0 bank does not reproduce the compound ({ctrl_max})")

            mag_std = np.abs(filter_svd_3d(std, method="adaptive",
                                           frame_rate_hz=frame_rate)).astype(np.float32)
            c_std, z_std = detect_counts(mag_std)
            del std

            # ---- PHYSICS CURVE at the standard detections ----
            idx = tuple(c_std)
            per_angle = angles[(slice(None),) + idx]                    # (A, N)
            coh = np.abs(per_angle.sum(axis=0))
            inc = np.abs(per_angle).sum(axis=0)
            ratio = (coh / np.maximum(inc, 1e-20)).astype(np.float32)
            del per_angle

            # ---- bank ----
            best = None; best_i = None
            for k, v in enumerate(velocities):
                ramp = np.exp(-1j * dphi_of(v) * np.arange(n_ang)).astype(np.complex64)
                m = np.abs(filter_svd_3d(np.tensordot(ramp, angles, axes=(0, 0)),
                                         method="adaptive",
                                         frame_rate_hz=frame_rate)).astype(np.float32)
                if best is None:
                    best = m; best_i = np.zeros(m.shape, np.int8)
                else:
                    t = m > best
                    best = np.where(t, m, best); best_i = np.where(t, np.int8(k), best_i)
                    del t
                del m
            del angles
            v_at_det = velocities[best_i[idx]]

            # predicted vs observed coherence ratio, binned by the bank's own velocity pick
            curve = []
            for lo, hi in ((0, 20), (20, 40), (40, 60), (60, 80), (80, 100), (100, 131)):
                m = (np.abs(v_at_det) >= lo) & (np.abs(v_at_det) < hi)
                if m.sum() < 50:
                    continue
                vm = float(np.median(np.abs(v_at_det[m])))
                d = dphi_of(vm)
                pred = abs(np.sin(2 * d) / (4 * np.sin(d / 2))) if d else 1.0
                curve.append({"band": [lo, hi], "n": int(m.sum()),
                              "v_median": round(vm, 1),
                              "observed_ratio": round(float(np.median(ratio[m])), 4),
                              "predicted_ratio": round(float(pred), 4)})

            # ---- DENSITY-MATCHED detection comparison ----
            c_bank, z_bank = detect_counts(best)
            n_frames = int(best.shape[0])
            keep_bank = []
            for f in range(n_frames):
                ns = int((c_std[0] == f).sum())
                ib = np.where(c_bank[0] == f)[0]
                if ns and len(ib):
                    keep_bank.append(ib[np.argsort(z_bank[ib])[::-1][:ns]])
            keep_bank = np.concatenate(keep_bank) if keep_bank else np.zeros(0, int)
            s_std = set(map(tuple, c_std.T.tolist()))
            s_bank = set(map(tuple, c_bank[:, keep_bank].T.tolist()))
            new = s_bank - s_std
            new_arr = np.array(sorted(new), dtype=np.int64) if new else np.zeros((0, 4), np.int64)
            v_new = velocities[best_i[tuple(new_arr.T)]] if len(new_arr) else np.zeros(0)

            rec = {"acq": str(aid),
                   "v0_control_max_abs_diff": ctrl_max,
                   "n_std": int(c_std.shape[1]),
                   "n_bank_matched": int(len(s_bank)),
                   "n_changed_at_matched_density": int(len(new_arr)),
                   "changed_frac": round(float(len(new_arr) / max(1, c_std.shape[1])), 4),
                   "changed_v_median": round(float(np.median(np.abs(v_new))), 1) if len(v_new) else None,
                   "argmax_zero_frac_at_detections": round(float((np.abs(v_at_det) < 1e-6).mean()), 4),
                   "coherence_curve": curve,
                   "seconds": round(time.time() - t0, 1)}
            print(f"[acq {aid}] " + json.dumps(rec, indent=2), flush=True)
            out_acqs.append(rec)
            del best, best_i, mag_std
            vol.commit()

    out = {"lambda_mm": round(lam_mm, 4), "predicted_null_mm_s": round(lam_mm * frame_rate / 2, 2),
           "velocities_mm_s": [round(float(v), 1) for v in velocities], "acqs": out_acqs}
    with open(f"{DATA_ROOT}/hypothesis_bank_v2.json", "w") as fh:
        json.dump(out, fh, indent=2)
    vol.commit()
    print("BANK_V2 " + json.dumps(out, indent=2), flush=True)
    return out


@app.function(image=gpu_image, gpu="A100-80GB", timeout=4 * 3600, memory=262144, cpu=16.0,
              volumes={"/root/data": vol})
def bank(n_acqs: int = 2, acq_start: int = 0, v_max: float = 130.0, n_hyp: int = 13,
         sigma_threshold: float = 2.0, min_distance: int = 2,
         smoothing_sigma: float = 1.0) -> dict:
    import json
    import sys
    import time

    sys.path.insert(0, "/workspace")
    import h5py
    import numpy as np
    from scipy.ndimage import gaussian_filter, maximum_filter

    from ultratrace_ulm.beamform_core import NeutralConfig, beamform_iq
    from ultratrace_ulm.svd import filter_svd_3d

    vol.reload()
    out_acqs = []

    from ultratrace_ulm.beamform_mach import _load_acq, _load_neutral_config

    class _Opts:                                  # the knobs _load_neutral_config overrides
        elev_planes = 25
        z_coarseness = 0.5
        x_coarseness = 0.5
        large_fov = True
        xlarge_fov = False
        row_index = None
        mean_subtract_channels = True

    with h5py.File(H5, "r") as h5:
        config = _load_neutral_config(h5, _Opts())
        attrs = dict(h5["config"].attrs)
        c = float(config.speed_of_sound_m_s)
        f0 = float(config.tx_freq_hz)
        lam_mm = c / f0 * 1000.0
        acq_ids = [int(k) for k in sorted(h5["acquisitions"].keys(), key=int)][acq_start:acq_start + n_acqs]
        frame_rate = float(attrs.get("frame_rate_hz", 222.4306816130359))
        prf = 4.0 * frame_rate
        print(f"[cfg] lambda={lam_mm:.3f} mm  FR={frame_rate:.3f} Hz  PRF={prf:.1f} Hz  "
              f"null at {lam_mm*frame_rate/2:.2f} mm/s", flush=True)

        velocities = np.linspace(-v_max, v_max, n_hyp)

        for aid in acq_ids:
            t0 = time.time()
            iq, txd, txd_e = _load_acq(h5, aid)
            angles, grid = beamform_iq(iq, txd, txd_e, config, return_angles=True)
            n_ang = angles.shape[0]
            print(f"[acq {aid}] per-angle stack {angles.shape} in {time.time()-t0:.0f}s", flush=True)

            # ---- standard coherent compound (the baseline the pipeline uses) ----
            std = angles.sum(axis=0)
            mag_std = np.abs(filter_svd_3d(std, method="adaptive",
                                           frame_rate_hz=frame_rate)).astype(np.float32)
            del std

            # ---- hypothesis bank ------------------------------------------------
            best_mag = None
            best_idx = None
            for k, v in enumerate(velocities):
                dphi = 4.0 * np.pi * (v / 1000.0) / ((lam_mm / 1000.0) * prf)
                ramp = np.exp(-1j * dphi * np.arange(n_ang)).astype(np.complex64)
                comp = np.tensordot(ramp, angles, axes=(0, 0))
                m = np.abs(filter_svd_3d(comp, method="adaptive",
                                         frame_rate_hz=frame_rate)).astype(np.float32)
                del comp
                if best_mag is None:
                    best_mag = m
                    best_idx = np.zeros(m.shape, np.int8)
                else:
                    take = m > best_mag
                    best_mag = np.where(take, m, best_mag)
                    best_idx = np.where(take, np.int8(k), best_idx)
                    del take
                del m
                print(f"[acq {aid}]  hyp {k+1}/{n_hyp} v={v:+.0f} mm/s", flush=True)
            del angles

            # ---- detect on both, same detector ---------------------------------
            def detect(mag):
                sm = gaussian_filter(mag, sigma=(0, 0, smoothing_sigma, smoothing_sigma))
                n_elev = sm.shape[1]
                mean = np.zeros(n_elev, np.float32)
                std_ = np.ones(n_elev, np.float32)
                for e in range(n_elev):
                    v_ = sm[:, e]
                    pos = v_[v_ > 0]
                    if pos.size:
                        mean[e] = pos.mean()
                        s = pos.std()
                        std_[e] = s if s > 1e-10 else 1.0
                z = (sm - mean[None, :, None, None]) / (std_[None, :, None, None] + 1e-10)
                fs = 2 * int(min_distance) + 1
                pk = (z == maximum_filter(z, size=(1, fs, fs, fs))) & (z > sigma_threshold)
                return np.array(np.where(pk)), z[pk].astype(np.float32)

            c_std, z_std = detect(mag_std)
            c_bank, z_bank = detect(best_mag)

            # detections present in the bank but absent from the standard compound
            key = lambda c: set(map(tuple, c.T.tolist()))
            s_std, s_bank = key(c_std), key(c_bank)
            new = s_bank - s_std
            # persistence: does a new detection have a bank neighbour in an adjacent frame?
            new_arr = np.array(sorted(new), dtype=np.int64) if new else np.zeros((0, 4), np.int64)
            persist = 0
            if len(new_arr):
                by_f = {}
                for row in c_bank.T:
                    by_f.setdefault(int(row[0]), []).append(row[1:])
                for row in new_arr:
                    nb = by_f.get(int(row[0]) + 1, [])
                    if any(abs(int(q[0]) - row[1]) <= 1 and abs(int(q[1]) - row[2]) <= 2
                           and abs(int(q[2]) - row[3]) <= 2 for q in nb):
                        persist += 1
            # argmax velocity at the new detections
            vsel = velocities[best_idx[tuple(new_arr.T)]] if len(new_arr) else np.zeros(0)

            rec = {
                "acq": str(aid),
                "n_std": int(c_std.shape[1]), "n_bank": int(c_bank.shape[1]),
                "n_new": int(len(new_arr)),
                "n_new_persistent": int(persist),
                "new_frac_of_std": round(float(len(new_arr) / max(1, c_std.shape[1])), 3),
                "new_hyp_velocity_median": round(float(np.median(np.abs(vsel))), 1) if len(vsel) else None,
                "new_hyp_velocity_p90": round(float(np.percentile(np.abs(vsel), 90)), 1) if len(vsel) else None,
                "bank_argmax_nonzero_frac": round(float((best_idx != n_hyp // 2).mean()), 4),
                "seconds": round(time.time() - t0, 1),
            }
            print(f"[acq {aid}] {rec}", flush=True)
            out_acqs.append(rec)
            np.savez_compressed(f"{DATA_ROOT}/bank_acq{aid}.npz",
                                velocities=velocities,
                                new_detections=new_arr.astype(np.int32),
                                new_hyp_velocity=vsel.astype(np.float32))
            del mag_std, best_mag, best_idx
            vol.commit()

    out = {"velocities_mm_s": [round(float(v), 1) for v in velocities],
           "lambda_mm": round(lam_mm, 4), "frame_rate_hz": frame_rate,
           "predicted_null_mm_s": round(lam_mm * frame_rate / 2, 2),
           "acqs": out_acqs}
    with open(f"{DATA_ROOT}/hypothesis_bank_report.json", "w") as fh:
        json.dump(out, fh, indent=2)
    vol.commit()
    print("BANK_REPORT " + json.dumps(out, indent=2), flush=True)
    return out
