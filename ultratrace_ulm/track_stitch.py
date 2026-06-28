"""Post-hoc track stitching (agreed idea #1; mb-crr.19).

The dominant track-quality problem is fragmentation (median ~6 frames). The
Kalman+Hungarian linker is frame-greedy with a small ``max_gap``; a bubble that
fades for a few frames or crosses another's gate starts a NEW track. This is a
pure-CPU second pass over the SAVED track list that RE-LINKS fragment ends to
fragment starts across larger gaps, using gap-extrapolated position + velocity
agreement + intensity continuity, via one global ``linear_sum_assignment``.

No GPU, no re-detection, infinitely sweepable. Anisotropic tolerance: tight in
z/x (reliable lateral+depth), loose in elevation (y, synthesized from 1 physical
row → untrustworthy). Validate with bench.py + odd/even split-half reproducibility;
guard against the merge-two-real-bubbles failure (Opus cross-check note: a real
stitch should make split-half renders agree MORE and must not degrade straightness
or mean_speed).

Positions are (x, y, z) mm with y = elevation.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def _end_state(track: dict, k: int = 4):
    """Last position, outgoing velocity (mean over last k segments), last frame,
    representative intensity of a track."""
    pos = np.asarray(track["positions"], dtype=np.float64)
    fr = np.asarray(track["frames"], dtype=np.float64)
    inten = track.get("intensities")
    n = len(pos)
    kk = min(k, n - 1) if n > 1 else 0
    if kk >= 1:
        dp = pos[-1] - pos[-1 - kk]
        df = max(fr[-1] - fr[-1 - kk], 1.0)
        vel = dp / df
    else:
        vel = np.zeros(3)
    ii = float(np.median(inten[-kk - 1:])) if inten is not None and len(inten) else 0.0
    return pos[-1], vel, fr[-1], ii


def _start_state(track: dict, k: int = 4):
    pos = np.asarray(track["positions"], dtype=np.float64)
    fr = np.asarray(track["frames"], dtype=np.float64)
    inten = track.get("intensities")
    n = len(pos)
    kk = min(k, n - 1) if n > 1 else 0
    if kk >= 1:
        dp = pos[kk] - pos[0]
        df = max(fr[kk] - fr[0], 1.0)
        vel = dp / df
    else:
        vel = np.zeros(3)
    ii = float(np.median(inten[: kk + 1])) if inten is not None and len(inten) else 0.0
    return pos[0], vel, fr[0], ii


def stitch_tracks(
    tracks: list[dict],
    max_gap: int = 12,
    tol_lateral_mm: float = 0.6,
    tol_elev_mm: float = 1.0,  # Codex cross-check: 2.0mm (~3 elev planes) false-merged
    # distinct elevation-separated bubbles. 1.0mm (~2 planes) allows real jitter,
    # rejects distinct ones. Final value swept in the bake-off (split-half arbiter).
    vel_cos_min: float = 0.8,  # lit-grounded (docs/literature): 0.3 over-merged
    # (straightness 0.88->0.55); >=0.8 default, 0.7 permissive edge, avoid 0.3.
    intensity_log_tol: float = 1.5,
    max_cost: float = 1.0,
) -> list[dict]:
    """Greedy-optimal one-pass stitch. Returns a NEW track list with stitched
    chains merged (positions/frames/intensities concatenated, re-sorted by frame).

    A link end_i→start_j is allowed only if: 1 ≤ (start_frame_j − end_frame_i) ≤
    max_gap; the gap-extrapolated endpoint of i lands within an anisotropic tol of
    start_j (tight lateral z/x, loose elevation y); outgoing/incoming velocity
    directions agree (cosine ≥ vel_cos_min, when both move); and log-intensity is
    continuous. Cost = normalized anisotropic distance (+ small velocity/intensity
    penalties); links above max_cost are rejected even if assigned.
    """
    n = len(tracks)
    if n < 2:
        return [dict(t) for t in tracks]

    ends = [_end_state(t) for t in tracks]
    starts = [_start_state(t) for t in tracks]
    # axis weights: x,z lateral (index 0,2) tight; y elevation (index 1) loose.
    w = np.array([1.0 / tol_lateral_mm, 1.0 / tol_elev_mm, 1.0 / tol_lateral_mm])

    INF = 1e6
    cost = np.full((n, n), INF, dtype=np.float64)
    for i, (epos, evel, efr, eii) in enumerate(ends):
        for j, (spos, svel, sfr, sii) in enumerate(starts):
            if i == j:
                continue
            gap = sfr - efr
            if gap < 1 or gap > max_gap:
                continue
            pred = epos + evel * gap
            d = (spos - pred) * w
            dist = float(np.sqrt(np.sum(d * d)))
            if dist > 1.0:  # outside the anisotropic gate (normalized)
                continue
            # velocity direction agreement (only when both meaningfully move)
            es, ss = np.linalg.norm(evel), np.linalg.norm(svel)
            if es > 1e-6 and ss > 1e-6:
                cosv = float(np.dot(evel, svel) / (es * ss))
                if cosv < vel_cos_min:
                    continue
                vpen = 0.3 * (1.0 - cosv)
            else:
                vpen = 0.0
            # intensity continuity (log scale; skip if missing)
            if eii > 0 and sii > 0:
                ilog = abs(np.log(sii) - np.log(eii))
                if ilog > intensity_log_tol:
                    continue
                ipen = 0.2 * (ilog / intensity_log_tol)
            else:
                ipen = 0.0
            cost[i, j] = dist + vpen + ipen

    rows, cols = linear_sum_assignment(cost)
    # successor[i] = j if i stitches to j
    succ = {}
    pred_of = {}
    for r, c in zip(rows, cols):
        if cost[r, c] < max_cost:
            succ[r] = c
            pred_of[c] = r

    # Build chains: heads are tracks that are nobody's successor.
    used = set()
    out = []
    for i in range(n):
        if i in pred_of:
            continue  # not a chain head
        chain = [i]
        cur = i
        while cur in succ and succ[cur] not in used:
            nxt = succ[cur]
            if nxt in chain:  # safety: no cycles
                break
            chain.append(nxt)
            cur = nxt
        for c in chain:
            used.add(c)
        out.append(_merge_chain([tracks[c] for c in chain]))
    return out


def _merge_chain(chain: list[dict]) -> dict:
    if len(chain) == 1:
        return dict(chain[0])
    pos = np.concatenate([np.asarray(t["positions"], dtype=np.float32) for t in chain])
    fr = np.concatenate([np.asarray(t["frames"], dtype=np.float32) for t in chain])
    has_int = all("intensities" in t and t["intensities"] is not None and len(t["intensities"]) for t in chain)
    order = np.argsort(fr, kind="stable")
    merged = dict(chain[0])
    merged["positions"] = pos[order]
    merged["frames"] = fr[order]
    merged["length"] = int(len(fr))
    merged["stitched_from"] = int(len(chain))
    if has_int:
        ii = np.concatenate([np.asarray(t["intensities"], dtype=np.float32) for t in chain])
        merged["intensities"] = ii[order]
    return merged


def stitch_pickle_data(data: dict, key: str = "tracks_smoothed", **kwargs) -> dict:
    """Stitch the tracks in a loaded pickle dict; returns it with stitched tracks
    under the same key (and 'tracks' updated)."""
    tk = key if key in data and data[key] else "tracks"
    stitched = stitch_tracks(list(data.get(tk, [])), **kwargs)
    out = dict(data)
    out["tracks"] = stitched
    out["tracks_smoothed"] = stitched
    out.setdefault("params", {})
    out["params"] = {**out.get("params", {}), "stitched": True, "stitch_kwargs": kwargs}
    return out
