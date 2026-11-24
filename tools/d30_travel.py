"""D30 (atanor_os/GRAIN_THEORY.md, registered before running): a judge-driven traveller's stretch sensed through the being's organs; then the being-side maps.
  PYTHONPATH=. python -m tools.d30_travel OUT.json 9101_0 ..."""
import json, math, os, sys, time
import numpy as np, torch
from scipy.stats import spearmanr
from tools.need_fork import t_room, place, DEV, DEPTH
from tools.a1_channel import V2
from tools import d25_grain as D25
from tools import d29_landmarks as D29
from tools.d26_map import kmeans
from tools.d27_place import code
from tools.d28_shortcut import value
from atanor_os.emerge.room_body import RoomBody
from atanor_os.emerge.body_contract import Organs
from atanor_os.emerge.mind import Mind

SAVE = "d30"


@torch.no_grad()
def travel(ck, rows, info, seed, n=64, beats=3000):
    rng = np.random.default_rng(seed + 5); torch.manual_seed(seed + 5)
    B = RoomBody(n, rows, drain=V2["drain"], seed=seed, device=DEV, regrow=V2["regrow"]); O = Organs(B.c, n, dev=DEV, depth=DEPTH); Mi = Mind(O, B.c, n, dev=DEV, value_arm="relief"); Mi.load(ck)
    legs, neck, hand = B.legs.cpu().numpy(), B.neck.cpu().numpy(), B.hand.cpu().numpy(); M = len(legs)

    def pick(lg, nk, hd):
        m = np.nonzero((legs == lg) & (neck == nk) & (hand == hd))[0]
        return int(m[0]) if len(m) else int(np.nonzero((legs == lg) & (neck == nk))[0][0])
    FWD = {h: pick(1, 0, h) for h in (0, 2)}; LEFT = {h: pick(0, 1, h) for h in (0, 2)}; RIGHT = {h: pick(0, 2, h) for h in (0, 2)}
    g = np.array([list(r) for r in rows]); free = [(y, x) for y in range(g.shape[0]) for x in range(g.shape[1]) if g[y, x] != "#"]
    spots = {"w": tuple(int(v) for v in np.argwhere(g == "w")[0]), "f": tuple(int(v) for v in np.argwhere(g == "f")[0]), "s": (int(info["ys"]), int(info["x0"]))}

    def field(cell):
        rr = [("".join("x" if (yy, xx) == cell else ch for xx, ch in enumerate(row))) for yy, row in enumerate(rows)]; return D25.dist_map(rr, "x")
    fields = {}; tgt = [None] * n; left = np.zeros(n, int)
    B.w.store[:] = 0.75; place(B, info); every = torch.ones(n, dtype=torch.bool, device=DEV); O.reset(every); x = O(B.raw())
    X, Pz, St, Lf, A = [], [], [], [], []; life = np.zeros(n, int)
    for t in range(beats):
        X.append(x.cpu()); pos = B.w.pos.cpu().numpy().copy(); Pz.append(pos); St.append(B.w.store.cpu().numpy().copy()); Lf.append(life.copy()); head = B.w.head.cpu().numpy(); act = np.zeros(n, int)
        for i in range(n):
            cy, cx = int(pos[i, 1]), int(pos[i, 0])
            if tgt[i] is None or left[i] <= 0 or fields[tgt[i]][cy, cx] <= 1:
                c = [spots["w"], spots["f"], spots["s"], free[rng.integers(len(free))]][rng.integers(4)]; tgt[i] = c; left[i] = 250
                if c not in fields:
                    fields[c] = field(c)
            left[i] -= 1; F = fields[tgt[i]]; hd = 2 if rng.random() < 0.3 else 0
            if rng.random() < 0.2:
                act[i] = rng.integers(M); continue
            best = min(((F[cy + dy, cx + dx], dy, dx) for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)) if F[cy + dy, cx + dx] >= 0), default=None)
            if best is None:
                act[i] = rng.integers(M); continue
            want = math.atan2(best[1] + (cy + 0.5 - pos[i, 1]) * 0.5, best[2] + (cx + 0.5 - pos[i, 0]) * 0.5); d = (want - head[i] + math.pi) % (2 * math.pi) - math.pi
            act[i] = FWD[hd] if abs(d) < math.radians(20) else (RIGHT[hd] if d > 0 else LEFT[hd])
        a = torch.as_tensor(act, device=DEV); A.append(act.copy()); B.step(a); x = O(B.raw())
        if B.dead.any():
            dd = B.dead.clone(); idx = torch.nonzero(dd)[:, 0]; B.w.dead[idx] = False; B.w.store[idx] = 0.75; place(B, info, which=idx); O.reset(dd); x = O(B.raw()); life[idx.cpu().numpy()] += 1
            for i in idx.cpu().numpy():
                tgt[i] = None
    ins_ = Mi.o.inside; ins = list(range(x.shape[1]))[ins_] if isinstance(ins_, slice) else [int(v) for v in ins_]
    return dict(X=torch.stack(X), P=np.stack(Pz), St=np.stack(St), Lf=np.stack(Lf), ins=ins, acts=np.stack(A), M=int(M))


def main():
    V2["on"] = True; RoomBody.default_cues = True; out = sys.argv[1]; os.makedirs(SAVE, exist_ok=True); res = []; t0 = time.time()
    for lay in sys.argv[2:]:
        seed, mirror = map(int, lay.split("_")); rows, info = t_room(seed, bool(mirror)); sd = seed * 10 + mirror; pk = f"{SAVE}/{lay}.pt"
        if not os.path.exists(pk):
            torch.save(travel(f"{D25.ROOT}/{lay}/mind.pt", rows, info, sd), pk)
        z = torch.load(pk, weights_only=False); P, St, Lf = z["P"], z["St"], z["Lf"]; T, n = Lf.shape
        rise = np.zeros((T, n, 2), bool); rise[:-1] = ((St[1:] - St[:-1]) > 1e-4) & (Lf[1:] == Lf[:-1])[..., None]
        g = np.array([list(r) for r in rows]); cy, cx = np.clip(P[..., 1].astype(int), 0, g.shape[0] - 1), np.clip(P[..., 0].astype(int), 0, g.shape[1] - 1)
        cell = (cy * 100 + cx).reshape(-1); cells, ci = np.unique(cell, return_inverse=True); k = len(cells); Dc = [D25.dist_map(rows, c)[cells // 100, cells % 100].astype(float) for c in "wf"]

        def per_cell(U, kk, shuf=None):
            V = value(U.reshape(T, n), Lf, rise, kk, shuffle_seed=shuf); return [round(float(spearmanr(np.bincount(ci, weights=V[U, c]) / np.bincount(ci), -Dc[c])[0]), 2) for c in (0, 1)]
        rng = np.random.default_rng(1); rec = dict(layout=lay, deaths=int(Lf[-1].sum()), rises=rise.reshape(-1, 2).sum(0).tolist(), ceiling=dict(clean=per_cell(ci, k)))
        for eps in (0.002, 0.01, 0.03):
            U = ci.copy(); m = rng.random(len(U)) < eps; U[m] = rng.integers(0, k, m.sum()); rec["ceiling"][f"{eps} wrong"] = per_cell(U, k)
        same = (Lf[1:] == Lf[:-1]).reshape(-1); a = ci.reshape(T, n)[:-1].reshape(-1)[same]; b = ci.reshape(T, n)[1:].reshape(-1)[same]; C = np.zeros((k, k)); np.add.at(C, (a, b), 1); C = C + C.T; np.fill_diagonal(C, 0)
        rec["least_crossed_link"] = int(C.max(1).min()); rec["median_link"] = int(np.median(C.max(1))); print(lay, "travel", {q: rec[q] for q in ("deaths", "rises", "ceiling", "least_crossed_link", "median_link")}, round(time.time() - t0), flush=True)
        X, ins, Acts, M = z["X"], z["ins"], z["acts"], z["M"]; dx = X.shape[2]; Xo = X.clone(); Xo[:, :, ins] = 0; mu, sg = Xo.reshape(-1, dx).mean(0), Xo.reshape(-1, dx).std(0) + 1e-6; Xo = (Xo - mu) / sg
        Z, _, _ = code("one", Xo, Acts, Lf, M, sd); feats = {"raw": Xo.reshape(T * n, dx).to(DEV), "learned": Z.reshape(T * n, -1)}; rec["maps"] = {}
        for fn, Fm in feats.items():
            for nm, U, kk in [(f"{fn} landmarks 512", D29.landmarks(Fm, 512, sd), 512), (f"{fn} landmarks 128", D29.landmarks(Fm, 128, sd), 128), (f"{fn} clusters 64", kmeans(Fm, 64, sd).cpu().numpy(), 64)]:
                rec["maps"][nm] = dict(map=per_cell(U, kk), shuffled=per_cell(U, kk, shuf=sd)); print(lay, nm, rec["maps"][nm], round(time.time() - t0), flush=True)
        res.append(rec); json.dump(dict(protocol="atanor_os/GRAIN_THEORY.md (D30)", rows=res), open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
