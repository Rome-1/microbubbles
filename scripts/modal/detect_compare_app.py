"""Detect-only comparison: run the FORK's own detection path with the shipped c64
SVD (static copy, no git-revert timing) on the first n acqs of a beamformed h5, and
save raw detections. Compared locally against the pristine package's detections to
prove fork-c64 == pristine-c64 at the detection stage.
"""
import modal

PROJECT = "microbubbles"
DATA_ROOT = f"/root/data/{PROJECT}"
vol = modal.Volume.from_name("research")

# Static c64 fork copy lives in the scratchpad; add_local_dir uploads it as-is.
SCR = "/tmp/claude-1000/-home-rome-gt-microbubbles-crew-cajal/66fb2222-e5d5-461e-950f-473915e99636/scratchpad"
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("h5py==3.11.0", "numpy<2.0", "scipy==1.13.1", "tqdm==4.66.4")
    .add_local_dir(f"{SCR}/fork_c64_ulm", remote_path="/workspace/ultratrace_ulm")
)

app = modal.App(f"{PROJECT}-detectcmp")


@app.function(image=image, timeout=2 * 3600, memory=98304, cpu=16.0,
              volumes={"/root/data": vol})
def fork_detect(beamformed: str = "pristine_out/beamformed.h5", n: int = 1,
                out: str = "fork_c64_det_1.npz", frame_rate: float = 222.0) -> dict:
    import sys, os
    from pathlib import Path
    sys.path.insert(0, "/workspace")
    import numpy as np
    from ultratrace_ulm.h5_io import open_h5, acq_keys
    from ultratrace_ulm.tracking import TrackingOptions, detect_localize_acq, _iter_compounds_h5

    vol.reload()
    bf = f"{DATA_ROOT}/{beamformed}"
    with open_h5(bf) as h5:
        ids = acq_keys(h5)[:n]
    opts = TrackingOptions(
        beamformed_path=Path(bf), tracks_path=Path(f"{DATA_ROOT}/_detcmp.pkl"),
        svd_method="adaptive", knee_filter=True,
        tissue_freq_hz=100.0, temporal_sigma=0.0, filter_method="svd", svd_low_cutoff=0.1,
        sigma_threshold=2.0, min_distance=2, smoothing_sigma=1.0, subpixel="centroid",
        window_size=5, frame_rate_hz=frame_rate)
    pos, ints, zs, aidx = [], [], [], []
    for order, (aid, comp, gx, gy, gz) in enumerate(_iter_compounds_h5(Path(bf), ids)):
        d = detect_localize_acq(comp, gx, gy, gz, opts)   # filter_fn/detect_fn None => CPU c64
        p = d["positions_mm"]
        pos.append(p); ints.append(d["intensities"]); zs.append(d.get("zscores", np.zeros(len(p))))
        aidx.append(np.full(len(p), order, dtype=np.int32))
        print(f"[fork_detect] order={order} aid={aid} ndet={len(p)}", flush=True)
    P = np.concatenate(pos, 0) if pos else np.zeros((0, 3), np.float32)
    outp = f"{DATA_ROOT}/{out}"
    np.savez(outp, positions_mm=P,
             intensities=np.concatenate(ints) if ints else np.zeros(0, np.float32),
             zscores=np.concatenate(zs) if zs else np.zeros(0, np.float32),
             acq_index=np.concatenate(aidx) if aidx else np.zeros(0, np.int32))
    vol.commit()
    rep = {"out": outp, "n": n, "total_detections": int(len(P)), "per_acq": [int(len(p)) for p in pos]}
    print(rep); return rep


@app.local_entrypoint()
def main(n: int = 1):
    out = f"fork_c64_det_{n}.npz"
    print(fork_detect.remote(n=n, out=out))
