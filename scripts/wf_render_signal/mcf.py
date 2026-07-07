"""Motion-confirmed dense reconstruction by min-cost-flow (CPU numpy).

Two-stage Zhang-2008 tracker on the reference acq-0 detection harness:
  Stage 1: constant-velocity Kalman gap=1 mutual-NN tracklets.
  Stage 2: successive-shortest-paths min-cost-flow over tracklet nodes with a
           2nd-order velocity-predicted transition gate (anisotropic box AND a
           Euclidean anti-teleport ceiling). beta is the single continuity knob
           that replaces the min_length>=35 cliff; it is calibrated against a
           frame-scramble null.

Pure numpy + stdlib. Self-contained MCF primitive with negative-edge support.
"""
from __future__ import annotations
import numpy as np
from collections import deque

VBOX = np.array([0.4008175182481752, 1.1093333333333333, 0.4015686274509804])
EUCLID_CEIL = 0.761
Z0 = 6.0
# Kalman noise (mm) — elevation axis is 2.77x coarser
R_DIAG = np.array([0.04, 0.12, 0.04]) ** 2
Q_DIAG = np.array([0.04, 0.12, 0.04, 0.03, 0.08, 0.03]) ** 2
CHI2_SPLIT = 11.34   # chi-square_3,0.99
DV = np.array([0.20, 0.55, 0.20])
LAM_GAP = 0.35
LAM_VEL = 0.25


# ---------------------------------------------------------------- MCF primitive
class MinCostFlow:
    """Successive-shortest-paths min-cost-flow with negative edges (SPFA)."""
    def __init__(self, n):
        self.n = n
        self.head = [-1] * n
        self.to = []; self.cap = []; self.cost = []; self.nxt = []

    def add(self, u, v, cap, cost):
        # forward
        self.to.append(v); self.cap.append(cap); self.cost.append(cost)
        self.nxt.append(self.head[u]); self.head[u] = len(self.to) - 1
        # reverse
        self.to.append(u); self.cap.append(0.0); self.cost.append(-cost)
        self.nxt.append(self.head[v]); self.head[v] = len(self.to) - 1

    def _spfa(self, s, t):
        INF = float("inf")
        dist = [INF] * self.n
        inq = [False] * self.n
        pe = [-1] * self.n
        dist[s] = 0.0
        q = deque([s]); inq[s] = True
        while q:
            u = q.popleft(); inq[u] = False
            du = dist[u]
            e = self.head[u]
            while e != -1:
                if self.cap[e] > 1e-9:
                    v = self.to[e]; nd = du + self.cost[e]
                    if nd < dist[v] - 1e-12:
                        dist[v] = nd; pe[v] = e
                        if not inq[v]:
                            inq[v] = True
                            q.append(v)
                e = self.nxt[e]
        return dist, pe

    def solve(self, s, t, stop_at_zero=True):
        """Augment 1 unit along the shortest s->t path until cost >= 0."""
        total_cost = 0.0; flow = 0
        paths = []
        while True:
            dist, pe = self._spfa(s, t)
            if dist[t] == float("inf"):
                break
            if stop_at_zero and dist[t] >= -1e-9:
                break
            # trace path, min residual = 1 (unit caps)
            path_edges = []; v = t
            while v != s:
                e = pe[v]; path_edges.append(e); v = self.to[e ^ 1]
            for e in path_edges:
                self.cap[e] -= 1.0; self.cap[e ^ 1] += 1.0
            total_cost += dist[t]; flow += 1
        return total_cost, flow, paths

    def forward_flow_paths(self, s, t):
        """Decompose the FINAL flow into node-disjoint s->t paths by following
        forward edges that carry flow (cap dropped 1->0). Correct even when the
        SSP augmenting paths used reverse edges. Returns list of edge-id lists."""
        # adjacency of flow-carrying forward edges (even ids), per node
        out = {}
        e = 0
        while e < len(self.to):
            if (e % 2 == 0) and self.cap[e] < 0.5:      # forward edge with flow=1
                out.setdefault(self.to[e ^ 1], []).append(e)
            e += 2
        used = set()
        paths = []
        for e0 in list(out.get(s, [])):
            if e0 in used:
                continue
            path = []; node = s; edge = e0
            while True:
                path.append(edge); used.add(edge)
                node = self.to[edge]
                if node == t:
                    break
                nxts = [ee for ee in out.get(node, []) if ee not in used]
                if not nxts:
                    break
                edge = nxts[0]
            if self.to[path[-1]] == t:
                paths.append(path)
        return paths


# ------------------------------------------------------------- Stage 1 tracklets
def build_tracklets(pos, frame, z):
    """Constant-velocity Kalman gap=1 mutual-NN tracklets.

    Returns list of tracklets; each is dict(idx=[det indices], frames, pos,
    entry_pos, entry_frame, exit_pos, exit_vel, exit_frame, zsum_reward).
    """
    order = np.argsort(frame, kind="stable")
    frames_u = np.unique(frame)
    by_frame = {int(f): order[frame[order] == f] for f in frames_u}

    gate = 0.65 * VBOX
    H = np.zeros((3, 6)); H[0, 0] = H[1, 1] = H[2, 2] = 1.0
    R = np.diag(R_DIAG); Q = np.diag(Q_DIAG)

    # active tracklet state
    tracklets = []            # finalized
    active = []               # list of dicts with kalman state
    P0 = np.diag([0.04, 0.12, 0.04, 0.5, 0.5, 0.5]) ** 1  # velocity unknown

    def start(di):
        x = np.zeros(6); x[:3] = pos[di]
        active.append(dict(idx=[di], x=x, P=P0.copy(), last_f=int(frame[di]),
                           n=1))

    fu = [int(f) for f in frames_u]
    MAXGAP1 = 2                          # allow up to 1 missing frame within a tracklet
    # seed with first frame
    for di in by_frame[fu[0]]:
        start(int(di))

    for fi in range(1, len(fu)):
        f = fu[fi]
        dets = by_frame[f]
        # finalize actives whose gap to f would exceed MAXGAP1
        keep = []
        for a in active:
            if f - a["last_f"] > MAXGAP1:
                tracklets.append(a)
            else:
                keep.append(a)
        active = keep
        if not len(active):
            for di in dets:
                start(int(di))
            continue
        dp = pos[dets]                       # (M,3)
        F = np.eye(6); F[0, 3] = F[1, 4] = F[2, 5] = 1.0
        A = len(active); M = len(dets)
        # predict each active forward by its own gap = f - last_f (1 or 2)
        preds = np.zeros((A, 3)); Sinvs = []; last_p = np.zeros((A, 3)); gaps = np.zeros(A)
        for ai, a in enumerate(active):
            gap = f - a["last_f"]; gaps[ai] = gap
            xp = a["x"].copy(); Pp = a["P"].copy()
            for _ in range(gap):
                xp = F @ xp; Pp = F @ Pp @ F.T + Q
            a["_xp"] = xp; a["_Pp"] = Pp
            preds[ai] = H @ xp
            Sinvs.append(np.linalg.inv(H @ Pp @ H.T + R))
            last_p[ai] = pos[a["idx"][-1]]
        cost = np.full((A, M), np.inf)
        for ai in range(A):
            innov = dp - preds[ai]
            m2 = np.einsum("mi,ij,mj->m", innov, Sinvs[ai], innov)
            g = gaps[ai]
            box_ok = np.all(np.abs(dp - last_p[ai]) <= VBOX * g + 1e-9, axis=1)
            eu = np.linalg.norm(dp - last_p[ai], axis=1) / g
            ok = (m2 <= CHI2_SPLIT) & box_ok & (eu <= EUCLID_CEIL + 1e-9)
            cost[ai][ok] = m2[ok]
        # greedy mutual NN (prefer gap=1 by adding a small gap penalty)
        flat = [(cost[ai, mi] + 2.0 * (gaps[ai] - 1), ai, mi)
                for ai in range(A) for mi in range(M) if np.isfinite(cost[ai, mi])]
        flat.sort()
        matched_a = set(); used_m = set()
        for c, ai, mi in flat:
            if ai in matched_a or mi in used_m:
                continue
            matched_a.add(ai); used_m.add(mi)
            a = active[ai]; di = int(dets[mi])
            xp = a["_xp"]; Pp = a["_Pp"]
            innov = pos[di] - H @ xp; S = H @ Pp @ H.T + R
            K = Pp @ H.T @ np.linalg.inv(S)
            a["x"] = xp + K @ innov; a["P"] = (np.eye(6) - K @ H) @ Pp
            a["idx"].append(di); a["last_f"] = f; a["n"] += 1
        # unmatched dets -> new tracklets (unmatched actives stay active for gap-2)
        for mi in range(M):
            if mi not in used_m:
                start(int(dets[mi]))
    tracklets.extend(active)

    # finalize geometry
    out = []
    for a in tracklets:
        idx = np.array(a["idx"])
        fr = frame[idx]; ps = pos[idx]
        srt = np.argsort(fr); idx = idx[srt]; fr = fr[srt]; ps = ps[srt]
        if len(ps) >= 2:
            # robust exit velocity: mean step over last up-to-5 points (less noisy
            # than a single last-pair difference), and entry velocity over first 5
            k = min(5, len(ps))
            dfe = np.diff(fr[-k:]); dfe[dfe == 0] = 1
            ev = np.mean(np.diff(ps[-k:], axis=0) / dfe[:, None], axis=0)
            dfb = np.diff(fr[:k]); dfb[dfb == 0] = 1
            env = np.mean(np.diff(ps[:k], axis=0) / dfb[:, None], axis=0)
        else:
            ev = np.zeros(3); env = np.zeros(3)
        P = np.clip(1.0 / (1.0 + np.exp(-0.9 * (z[idx] - Z0))), 1e-4, 1 - 1e-4)
        reward = float(np.sum(-np.log(P / (1 - P))))   # negative for high-z
        reward = float(np.clip(reward, -1e9, 1e9))
        out.append(dict(idx=idx, frames=fr, pos=ps,
                        entry_pos=ps[0], entry_frame=int(fr[0]),
                        exit_pos=ps[-1], exit_frame=int(fr[-1]),
                        exit_vel=ev, entry_vel=env,
                        obs_cost=reward, maxz=float(z[idx].max())))
    return out


# --------------------------------------------------------------- Stage 2 MCF
def link_tracklets(tracklets, beta=3.0):
    T = len(tracklets)
    # node ids: 0=S, 1=Sink, then i_in=2+2i, i_out=3+2i
    S, SINK = 0, 1
    n = 2 + 2 * T
    g = MinCostFlow(n)
    def i_in(i): return 2 + 2 * i
    def i_out(i): return 3 + 2 * i
    for i, tk in enumerate(tracklets):
        g.add(S, i_in(i), 1.0, beta)              # birth
        g.add(i_out(i), SINK, 1.0, beta)          # death
        g.add(i_in(i), i_out(i), 1.0, tk["obs_cost"])  # observation reward
    # transition edges (2nd-order gated)
    entry_f = np.array([tk["entry_frame"] for tk in tracklets])
    exit_f = np.array([tk["exit_frame"] for tk in tracklets])
    exit_p = np.array([tk["exit_pos"] for tk in tracklets])
    exit_v = np.array([tk["exit_vel"] for tk in tracklets])
    entry_v = np.array([tk["entry_vel"] for tk in tracklets])
    entry_p = np.array([tk["entry_pos"] for tk in tracklets])
    Sig = np.diag((VBOX / 2.0) ** 2)
    Sinv = np.linalg.inv(Sig)
    n_edges = 0
    order = np.argsort(exit_f)
    for i in order:
        gap = entry_f - exit_f[i]
        cand = np.where((gap >= 1) & (gap <= 10))[0]
        if not len(cand):
            continue
        d = entry_p[cand] - exit_p[i]
        g_ = gap[cand]
        # HARD gates: anisotropic box AND Euclidean anti-teleport ceiling
        box_ok = np.all(np.abs(d) <= VBOX * g_[:, None] + 1e-9, axis=1)
        eu_ok = (np.linalg.norm(d, axis=1) / g_) <= EUCLID_CEIL + 1e-9
        # SOFT 2nd-order prior: predicted-position residual within 2*box (loose;
        # the hard gates above already bound teleports). Priced in the cost.
        pred = exit_p[i] + exit_v[i] * g_[:, None]
        vres = entry_p[cand] - pred
        vres_ok = np.all(np.abs(vres) <= 2.0 * VBOX * g_[:, None] + 1e-9, axis=1)
        ok = box_ok & eu_ok & vres_ok
        for k in np.where(ok)[0]:
            j = int(cand[k])
            r = vres[k]
            maha = 0.5 * float(r @ Sinv @ r)
            dv = (entry_v[j] - exit_v[i]) / DV
            c = maha + LAM_GAP * (g_[k] - 1) + LAM_VEL * float(dv @ dv)
            g.add(i_out(i), i_in(j), 1.0, c)
            n_edges += 1
    total_cost, flow, _ = g.solve(S, SINK)
    paths = g.forward_flow_paths(S, SINK)   # decompose FINAL flow, forward-only

    # each path: S -> i_in -> i_out -> (i_in -> i_out)* -> SINK. The observation
    # edges (i_in=2+2i -> i_out=3+2i) name the tracklets in this track, in order.
    tracks = []
    for path in paths:
        seq = []
        for e in path:
            u = g.to[e ^ 1]; v = g.to[e]
            if u >= 2 and v >= 2 and (u % 2 == 0) and (v == u + 1):
                seq.append((u - 2) // 2)
        if seq:
            tracks.append(seq)
    return tracks, dict(n_tracklets=T, n_trans_edges=n_edges,
                        total_cost=total_cost, n_tracks=len(tracks))


def assemble(tracklets, track_tk_lists, z):
    out = []
    for seq in track_tk_lists:
        idx = np.concatenate([tracklets[i]["idx"] for i in seq])
        fr = np.concatenate([tracklets[i]["frames"] for i in seq])
        ps = np.concatenate([tracklets[i]["pos"] for i in seq])
        srt = np.argsort(fr)
        idx, fr, ps = idx[srt], fr[srt], ps[srt]
        conf = float(np.mean(1.0 / (1.0 + np.exp(-(z[idx] - Z0)))))
        out.append(dict(det_idx=idx, frames=fr, positions=ps, confidence=conf,
                        length=len(fr)))
    return out


def run(pos, frame, z, beta=3.0, persist_z=10.0):
    tks_all = build_tracklets(pos, frame, z)
    # Stage-0 persistence pre-gate: a lone detection with no gap=1 neighbour is
    # NOT motion-confirmed -> drop it unless it is very bright (z>=persist_z).
    tks = [t for t in tks_all if len(t["idx"]) >= 2 or t["maxz"] >= persist_z]
    seqs, diag = link_tracklets(tks, beta=beta)
    tracks = assemble(tks, seqs, z)
    diag["n_tracklets_prepruned"] = len(tks_all)
    return tracks, tks, diag


if __name__ == "__main__":
    # self-test on synthetic linear tracks + noise
    rng = np.random.default_rng(0)
    pos = []; frame = []; z = []
    for k in range(5):
        p0 = rng.uniform([-20, -5, 12], [20, 5, 38])
        v = rng.uniform(-0.2, 0.2, 3)
        for f in range(40):
            pos.append(p0 + v * f + rng.normal(0, 0.02, 3)); frame.append(f); z.append(12)
    for f in range(40):                    # noise
        for _ in range(8):
            pos.append(rng.uniform([-25, -6, 11], [25, 6, 40])); frame.append(f); z.append(5.2)
    pos = np.array(pos); frame = np.array(frame); z = np.array(z)
    tracks, tks, diag = run(pos, frame, z, beta=3.0)
    lens = sorted([t["length"] for t in tracks], reverse=True)
    print("self-test:", diag, "| top track lengths:", lens[:8])
