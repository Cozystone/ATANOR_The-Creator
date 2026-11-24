"""D25 (atanor_os/GRAIN_THEORY.md, registered before running): is the value's knowledge there at place grain but not at muscle grain?
Judge-side measurements on V3's saved lives; no new being.
  PYTHONPATH=. python -m tools.d25_grain OUT.json 9101_0 ..."""
import json, os, sys, time
from collections import deque
import numpy as np, torch
from scipy.stats import spearmanr
from tools.need_fork import t_room, place, DEV, DEPTH
from tools.a1_channel import V2
from tools.a1_relief import relief_arm, shuffled
from atanor_os.emerge.room_body import RoomBody
from atanor_os.emerge.body_contract import Organs
from atanor_os.emerge.mind import Mind

ROOT = "a1cache_many_relief_sleep"
SPANS = (1, 3, 10, 30, 100)


def dist_map(rows, target):
    g = np.array([list(r) for r in rows]); H, W = g.shape; d = np.full((H, W), -1, int); q = deque()
    for y, x in zip(*np.where(g == target)):
        d[y, x] = 0; q.append((y, x))
    while q:
        y, x = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            y2, x2 = y + dy, x + dx
            if 0 <= y2 < H and 0 <= x2 < W and g[y2, x2] != "#" and d[y2, x2] < 0:
                d[y2, x2] = d[y, x] + 1; q.append((y2, x2))
    return d


@torch.no_grad()
def stretch(ck, rows, info, seed, n=64, beats=3000):
    """The V3 mind living on, frozen (no learning), with the judge's position beside every sensed state."""
    torch.manual_seed(seed + 77)
    B = RoomBody(n, rows, drain=V2["drain"], seed=seed, device=DEV, regrow=V2["regrow"]); O = Organs(B.c, n, dev=DEV, depth=DEPTH)
    Mi = Mind(O, B.c, n, dev=DEV, value_arm="relief"); Mi.load(ck); Mi.wo.on = True
    B.w.store[:] = 0.75; place(B, info); every = torch.ones(n, dtype=torch.bool, device=DEV); O.reset(every); Mi.begin(every); x = O(B.raw())
    X, Pz, St, Lf = [], [], [], []; A = []; life = np.zeros(n, int); fills = np.zeros(2, int)
    for t in range(beats):
        X.append(x.cpu()); Pz.append(B.w.pos.cpu().numpy().copy()); St.append(B.w.store.cpu().numpy().copy()); Lf.append(life.copy())
        act = Mi.choose(x); A.append(act.cpu().numpy().copy()); f = B.step(act); fills += f.cpu().numpy().astype(int).sum(0); x2 = O(B.raw()); Mi.prev = (x, act); Mi.cut = torch.zeros(n, dtype=torch.bool, device=DEV); x = x2
        if B.dead.any():
            dd = B.dead.clone(); idx = torch.nonzero(dd)[:, 0]; B.w.dead[idx] = False; B.w.store[idx] = 0.75; place(B, info, which=idx); O.reset(dd); Mi.begin(dd); x = O(B.raw())
            life[idx.cpu().numpy()] += 1
    stretch.acts = np.stack(A)                                       # each body's own action per beat (D27)
    return torch.stack(X), np.stack(Pz), np.stack(St), np.stack(Lf), Mi, fills


def zones(P, info):
    left = P[..., 0] < info["x0"] - 2.5; right = P[..., 0] > info["x0"] + 2.5; mid = ~(left | right)
    return np.where(left, -1, np.where(right, 1, 0)), mid & (P[..., 1] < 3.0)                      # arm side (-1 / 0 / 1), at the junction


def pieces(P, Lf, info, rows):
    """(d) leaving an arm, and spring-to-spring trips, per 1,000 body-beats."""
    side, _ = zones(P, info); T, n = side.shape; leave = 0; trips = 0; dw, df = dist_map(rows, "w"), dist_map(rows, "f")
    cy, cx = np.clip(P[..., 1].astype(int), 0, dw.shape[0] - 1), np.clip(P[..., 0].astype(int), 0, dw.shape[1] - 1)
    at = np.where(dw[cy, cx] <= 1, 1, np.where(df[cy, cx] <= 1, 2, 0))                              # beside the water spring / the food spring
    for i in range(n):
        same = Lf[1:, i] == Lf[:-1, i]; leave += int(((side[:-1, i] != 0) & (side[1:, i] == 0) & same).sum()); last = 0; lid = -1
        for t in range(T):
            if Lf[t, i] != lid:
                lid = Lf[t, i]; last = 0
            if at[t, i] and at[t, i] != last:
                trips += int(last != 0); last = at[t, i]
    k = T * n / 1000.0
    return dict(leave_arm_per_1000=leave / k, spring_to_spring_per_1000=trips / k)


@torch.no_grad()
def scores(arm, X, ins, need, bs=8192):
    out = []
    for j in range(0, len(X), bs):
        x = X[j:j + bs].to(DEV).clone(); x[:, ins[2]], x[:, ins[1]] = (0.30, 0.95) if need == 0 else (0.95, 0.30); out.append(arm.score(x).float().cpu())            # inside order: (constant, food, water) - checked against the stores
    return torch.cat(out).numpy()


def rowcorr(a, b):
    a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / (np.sqrt((a * a).sum(1) * (b * b).sum(1)) + 1e-12)


def main():
    V2["on"] = True; RoomBody.default_cues = True; res = []; t0 = time.time(); out = sys.argv[1]
    for lay in sys.argv[2:]:
        seed, mirror = map(int, lay.split("_")); d = f"{ROOT}/{lay}"; rows, info = t_room(seed, bool(mirror)); sd = seed * 10 + mirror; ck = os.path.join(d, "mind.pt")
        X, P, St, Lf, Mi, fills = stretch(ck, rows, info, sd); T, n, dx_ = X.shape; ins_ = Mi.o.inside; ins = list(range(dx_))[ins_] if isinstance(ins_, slice) else [int(v) for v in ins_]; M = Mi.M; dx = Mi.net.dx
        chk = float(max(np.abs(X[:, :, ins[2]].numpy() - St[..., 0]).max(), np.abs(X[:, :, ins[1]].numpy() - St[..., 1]).max())); print(lay, "stretch", X.shape, "fills", fills.tolist(), "inside == store check", round(chk, 4), round(time.time() - t0), flush=True)
        ring = torch.load(os.path.join(d, "life_ring.pt"), weights_only=False); ring["bx2_true"], ring["bd_true"] = ring["bx2"], ring["bd"]
        import pickle; pk = os.path.join(d, "d25_arms.pt"); have = os.path.exists(pk); saved = torch.load(pk, weights_only=False) if have else {}
        arms = {"life": Mi.wo, "A": saved["A"] if have else relief_arm(ring, dx, ins_, M, sd, updates=10000, gamma=0.99, sync=25)}; print(lay, "A trained", round(time.time() - t0), flush=True)
        arms["B"] = saved["B"] if have else relief_arm(ring, dx, ins_, M, sd + 1000, updates=10000, gamma=0.99, sync=25)
        arms["shuf"] = saved["shuf"] if have else relief_arm(shuffled(ring, ins_, sd), dx, ins_, M, sd + 2000, updates=10000, gamma=0.99, sync=25)
        if not have:
            for a_ in ("A", "B", "shuf"):
                arms[a_].bx = arms[a_].bx[:1].clone(); arms[a_].bx2 = arms[a_].bx2[:1].clone()
            torch.save({k: arms[k] for k in ("A", "B", "shuf")}, pk)
        print(lay, "arms trained", round(time.time() - t0), flush=True)
        side, junc = zones(P, info); wl = -1 if info["water_left"] else 1; dmap = {0: dist_map(rows, "w"), 1: dist_map(rows, "f")}
        cy, cx = np.clip(P[..., 1].astype(int), 0, dmap[0].shape[0] - 1), np.clip(P[..., 0].astype(int), 0, dmap[0].shape[1] - 1)
        rec = dict(layout=lay, fills=fills.tolist(), inside_check=chk, pieces=pieces(P, Lf, info, rows), needs={})
        Xf = X.reshape(T * n, dx_)
        for need in (0, 1):
            S = {k: scores(a, Xf, ins, need) for k, a in arms.items()}                               # [T*n, M] each
            need_side = wl if need == 0 else -wl; dist = dmap[need][cy, cx].reshape(-1).astype(float)
            zs = side.reshape(-1); jz = junc.reshape(-1)
            zone = {"stem": (zs == 0) & ~jz, "junction": jz, "needed arm": zs == need_side, "other arm": zs == -need_side}
            r = dict(share={k: float(v.mean()) for k, v in zone.items()})
            # (a) per-beat choice signal: agreement of two seeds across the combos, and the size of the gap
            for pair in (("A", "B"), ("A", "life"), ("A", "shuf")):
                rc = rowcorr(S[pair[0]], S[pair[1]]); r["beat " + "-".join(pair)] = dict(all=float(rc.mean()), **{k: float(rc[v].mean()) if v.any() else None for k, v in zone.items()})
            V = {k: s.mean(1) for k, s in S.items()}; Vx = {k: s.max(1) for k, s in S.items()}
            gap = (S["A"].max(1) - np.median(S["A"], 1)); r["gap over |V| (A)"] = float(np.median(gap) / (np.median(np.abs(V["A"])) + 1e-9)); r["|V| (A)"] = float(np.median(np.abs(V["A"])))
            # (b) reliability across a span of the body's own lived steps
            r["span"] = {}
            for k in SPANS:
                ok = (Lf[k:] == Lf[:-k]).reshape(-1); sp = {}
                for nm, vv in (("mean", V), ("max", Vx)):
                    dA = (vv["A"].reshape(T, n)[k:] - vv["A"].reshape(T, n)[:-k]).reshape(-1)[ok]; dB = (vv["B"].reshape(T, n)[k:] - vv["B"].reshape(T, n)[:-k]).reshape(-1)[ok]
                    dL = (vv["life"].reshape(T, n)[k:] - vv["life"].reshape(T, n)[:-k]).reshape(-1)[ok]
                    sp[nm] = dict(A_B=float(np.corrcoef(dA, dB)[0, 1]), A_life=float(np.corrcoef(dA, dL)[0, 1]))
                r["span"][str(k)] = sp
            # (c) does the value know the way (true path distance to the needed spring)
            r["way"] = {}
            for nm in arms:
                w = dict(all=float(spearmanr(V[nm], -dist)[0]))
                for zk, zv in zone.items():
                    w[zk] = float(spearmanr(V[nm][zv], -dist[zv])[0]) if zv.sum() > 200 and np.ptp(dist[zv]) > 0 else None
                r["way"][nm] = w
            rec["needs"]["thirsty" if need == 0 else "hungry"] = r
            print(lay, "thirsty" if need == 0 else "hungry", "| beat A-B", {k: (None if v is None else round(v, 2)) for k, v in r["beat A-B"].items()}, "| span A-B(mean)", {k: round(v["mean"]["A_B"], 2) for k, v in r["span"].items()},
                  "| way A", {k: (None if v is None else round(v, 2)) for k, v in r["way"]["A"].items()}, "| way shuf", round(r["way"]["shuf"]["all"], 2), "| way life", round(r["way"]["life"]["all"], 2), round(time.time() - t0), flush=True)
        print(lay, "pieces", rec["pieces"], flush=True); res.append(rec); json.dump(dict(protocol="atanor_os/GRAIN_THEORY.md (D25)", rows=res), open(out, "w"), indent=1)
        del arms, Mi; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
