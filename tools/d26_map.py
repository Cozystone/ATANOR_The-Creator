"""D26 (atanor_os/GRAIN_THEORY.md, registered before running): does a stitched map made from the being's own lived steps know the way?
  PYTHONPATH=. python -m tools.d26_map OUT.json 9101_0 ..."""
import json, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import spearmanr
from tools.need_fork import t_room, DEV
from tools.a1_channel import V2
from tools.d25_grain import stretch, zones, dist_map, ROOT
from atanor_os.emerge.room_body import RoomBody

K = 64; GAMMA = 0.99; SAVE = "d26"


def kmeans(Z, k, seed, iters=25):
    g = torch.Generator(device=Z.device).manual_seed(seed); C = Z[torch.randperm(len(Z), generator=g, device=Z.device)[:k]].clone()
    for _ in range(iters):
        a = torch.cat([torch.cdist(Z[j:j + 16384], C).argmin(1) for j in range(0, len(Z), 16384)])
        for u in range(k):
            m = a == u
            if m.any():
                C[u] = Z[m].mean(0)
    return a


def place_code(Xo, Lf, seed, steps=3000, d=32):
    """Time contrast on the being's own steps: states a few beats apart in one life are near (InfoNCE)."""
    torch.manual_seed(seed); T, n, dx = Xo.shape; net = nn.Sequential(nn.Linear(dx, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, d)).to(DEV)
    opt = torch.optim.Adam(net.parameters(), 1e-3); X = Xo.to(DEV); L = torch.as_tensor(Lf, device=DEV)
    for s in range(steps):
        t = torch.randint(0, T - 8, (1024,), device=DEV); i = torch.randint(0, n, (1024,), device=DEV); dt = torch.randint(1, 9, (1024,), device=DEV)
        ok = L[t, i] == L[t + dt, i]; t, i, dt = t[ok], i[ok], dt[ok]
        a = F.normalize(net(X[t, i]), dim=1); p = F.normalize(net(X[t + dt, i]), dim=1); loss = F.cross_entropy(a @ p.T / 0.1, torch.arange(len(a), device=DEV))
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return torch.cat([F.normalize(net(X[t0]), dim=1) for t0 in range(T)]).reshape(T, n, d), float(loss)


def map_value(U, Lf, rise, k, sym, shuffle_seed=None):
    """U [T, n] unit of each lived state; rise [T, n, 2] the store rose on that beat. Returns V [k, 2]: value over places."""
    same = (Lf[1:] == Lf[:-1]); a, b = U[:-1][same], U[1:][same]; C = np.zeros((k, k)); np.add.at(C, (a, b), 1)
    if sym:
        C = C + C.T
    Tm = C / np.maximum(C.sum(1, keepdims=True), 1); r = np.zeros((k, 2)); cnt = np.bincount(U.reshape(-1), minlength=k).astype(float)
    for c in (0, 1):
        r[:, c] = np.bincount(U.reshape(-1), weights=rise[..., c].reshape(-1).astype(float), minlength=k) / np.maximum(cnt, 1)
    if shuffle_seed is not None:
        r = r[np.random.default_rng(shuffle_seed).permutation(k)]
    return np.linalg.solve(np.eye(k) - GAMMA * Tm, r)


def main():
    V2["on"] = True; RoomBody.default_cues = True; res = []; t0 = time.time(); out = sys.argv[1]; os.makedirs(SAVE, exist_ok=True)
    for lay in sys.argv[2:]:
        seed, mirror = map(int, lay.split("_")); rows, info = t_room(seed, bool(mirror)); sd = seed * 10 + mirror; pk = f"{SAVE}/{lay}.pt"
        if os.path.exists(pk):
            z = torch.load(pk, weights_only=False); X, P, St, Lf, ins = z["X"], z["P"], z["St"], z["Lf"], z["ins"]
        else:
            X, P, St, Lf, Mi, fills = stretch(f"{ROOT}/{lay}/mind.pt", rows, info, sd); ins_ = Mi.o.inside; ins = list(range(X.shape[2]))[ins_] if isinstance(ins_, slice) else [int(v) for v in ins_]
            torch.save(dict(X=X, P=P, St=St, Lf=Lf, ins=ins), pk); del Mi; torch.cuda.empty_cache()
        T, n, dx = X.shape; print(lay, "stretch", X.shape, round(time.time() - t0), flush=True)
        rise = np.zeros((T, n, 2), bool); rise[:-1] = ((St[1:] - St[:-1]) > 1e-4) & (Lf[1:] == Lf[:-1])[..., None]                 # St channels: (water, food)
        Xo = X.clone(); Xo[:, :, ins] = 0; mu, sg = Xo.reshape(-1, dx).mean(0), Xo.reshape(-1, dx).std(0) + 1e-6; Xo = (Xo - mu) / sg
        Z, loss = place_code(Xo, Lf, sd); print(lay, "place code loss", round(loss, 3), round(time.time() - t0), flush=True)
        units = {"learned": kmeans(Z.reshape(T * n, -1), K, sd).reshape(T, n).cpu().numpy(), "raw": kmeans(Xo.reshape(T * n, dx).to(DEV), K, sd).reshape(T, n).cpu().numpy()}
        g = np.array([list(r) for r in rows]); cy, cx = np.clip(P[..., 1].astype(int), 0, g.shape[0] - 1), np.clip(P[..., 0].astype(int), 0, g.shape[1] - 1)
        cell = cy * g.shape[1] + cx; _, cell = np.unique(cell, return_inverse=True); units["true cells"] = cell.reshape(T, n)
        side, junc = zones(P, info); wl = -1 if info["water_left"] else 1; dmap = {0: dist_map(rows, "w"), 1: dist_map(rows, "f")}
        # judge-side purity of the place units: spread of true position inside a unit
        rec = dict(layout=lay, rises=rise.reshape(-1, 2).sum(0).tolist(), units={})
        for nm, U in units.items():
            k = int(U.max()) + 1; pos = P.reshape(-1, 2); u = U.reshape(-1)
            spread = float(np.mean([np.sqrt(((pos[u == q] - pos[u == q].mean(0)) ** 2).sum(1).mean()) for q in range(k) if (u == q).sum() > 5]))
            r = dict(k=k, spread_cells=spread)
            for sym in (False, True):
                for shuf in (False, True):
                    V = map_value(U, Lf, rise, k, sym, shuffle_seed=sd if shuf else None); key = ("sym" if sym else "lived") + ("-shuf" if shuf else ""); r[key] = {}
                    for need in (0, 1):
                        v = V[u, need]; dist = dmap[need][cy, cx].reshape(-1).astype(float); other = (side.reshape(-1) == -(wl if need == 0 else -wl))
                        r[key]["thirsty" if need == 0 else "hungry"] = dict(all=float(spearmanr(v, -dist)[0]), other_arm=float(spearmanr(v[other], -dist[other])[0]) if other.sum() > 200 and np.ptp(v[other]) > 0 else None)
            rec["units"][nm] = r
            print(lay, nm, "k", k, "spread", round(spread, 2), {kk: {q: (round(w["all"], 2), None if w["other_arm"] is None else round(w["other_arm"], 2)) for q, w in vv.items()} for kk, vv in r.items() if isinstance(vv, dict)}, round(time.time() - t0), flush=True)
        res.append(rec); json.dump(dict(protocol="atanor_os/GRAIN_THEORY.md (D26)", rows=res), open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
