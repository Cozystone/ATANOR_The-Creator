"""D27 (atanor_os/GRAIN_THEORY.md, registered before running): which place sense makes the stitched map work.
  PYTHONPATH=. python -m tools.d27_place OUT.json 9101_0 ..."""
import json, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import spearmanr
from tools.need_fork import t_room, DEV
from tools.a1_channel import V2
from tools import d25_grain as D25
from tools.d26_map import kmeans, map_value, K
from atanor_os.emerge.room_body import RoomBody

SAVE = "d27"; W = 16


class Past(nn.Module):                       # the sensed states at t, t-4, t-8, t-12 together
    def __init__(self, dx, M, d=32):
        super().__init__(); self.f = nn.Sequential(nn.Linear(4 * dx, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, d))

    def forward(self, xs, acts):             # xs [B, W, dx] oldest first
        return self.f(torch.cat([xs[:, -1], xs[:, -5], xs[:, -9], xs[:, -13]], 1))


class One(nn.Module):
    def __init__(self, dx, M, d=32):
        super().__init__(); self.f = nn.Sequential(nn.Linear(dx, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, d))

    def forward(self, xs, acts):
        return self.f(xs[:, -1])


class Carried(nn.Module):                    # a small recurrent net over the last 16 sensed states and the body's own actions
    def __init__(self, dx, M, d=32):
        super().__init__(); self.e = nn.Embedding(M, 16); self.g = nn.GRU(dx + 16, 128, batch_first=True); self.o = nn.Linear(128, d)

    def forward(self, xs, acts):             # acts [B, W]: the action taken on arriving at each state (previous beat's action)
        return self.o(self.g(torch.cat([xs, self.e(acts)], 2))[0][:, -1])


def windows(X, A, t, i):
    idx = t[:, None] + torch.arange(-W + 1, 1, device=DEV)[None]                                     # [B, W]
    return X[idx, i[:, None]], A[(idx - 1).clamp(min=0), i[:, None]]


def code(kind, Xo, Acts, Lf, M, seed, steps=3000):
    torch.manual_seed(seed); T, n, dx = Xo.shape; net = {"one": One, "past": Past, "carried": Carried}[kind](dx, M).to(DEV); opt = torch.optim.Adam(net.parameters(), 1e-3)
    X = Xo.to(DEV); A = torch.as_tensor(Acts, device=DEV).long(); L = torch.as_tensor(Lf, device=DEV)
    for s in range(steps):
        t = torch.randint(W, T - 8, (768,), device=DEV); i = torch.randint(0, n, (768,), device=DEV); dt = torch.randint(1, 9, (768,), device=DEV)
        ok = (L[t - W + 1, i] == L[t + dt, i]); t, i, dt = t[ok], i[ok], dt[ok]
        a = F.normalize(net(*windows(X, A, t, i)), dim=1); p = F.normalize(net(*windows(X, A, t + dt, i)), dim=1); loss = F.cross_entropy(a @ p.T / 0.1, torch.arange(len(a), device=DEV))
        opt.zero_grad(); loss.backward(); opt.step()
    Z = torch.zeros(T, n, 32, device=DEV); valid = np.zeros((T, n), bool)
    with torch.no_grad():
        for t0 in range(W, T):
            t = torch.full((n,), t0, device=DEV); i = torch.arange(n, device=DEV); Z[t0] = F.normalize(net(*windows(X, A, t, i)), dim=1); valid[t0] = (Lf[t0 - W + 1] == Lf[t0])
    return Z, valid, float(loss)


def main():
    V2["on"] = True; RoomBody.default_cues = True; res = []; t0 = time.time(); out = sys.argv[1]; os.makedirs(SAVE, exist_ok=True)
    for lay in sys.argv[2:]:
        seed, mirror = map(int, lay.split("_")); rows, info = t_room(seed, bool(mirror)); sd = seed * 10 + mirror; pk = f"{SAVE}/{lay}.pt"
        if os.path.exists(pk):
            z = torch.load(pk, weights_only=False)
        else:
            X, P, St, Lf, Mi, fills = D25.stretch(f"{D25.ROOT}/{lay}/mind.pt", rows, info, sd); ins_ = Mi.o.inside
            z = dict(X=X, P=P, St=St, Lf=Lf, ins=list(range(X.shape[2]))[ins_] if isinstance(ins_, slice) else [int(v) for v in ins_], acts=D25.stretch.acts, M=int(Mi.M)); torch.save(z, pk); del Mi; torch.cuda.empty_cache()
        X, P, St, Lf, ins, Acts, M = z["X"], z["P"], z["St"], z["Lf"], z["ins"], z["acts"], z["M"]; T, n, dx = X.shape; print(lay, "stretch", X.shape, "actions", Acts.shape, round(time.time() - t0), flush=True)
        rise = np.zeros((T, n, 2), bool); rise[:-1] = ((St[1:] - St[:-1]) > 1e-4) & (Lf[1:] == Lf[:-1])[..., None]
        Xo = X.clone(); Xo[:, :, ins] = 0; mu, sg = Xo.reshape(-1, dx).mean(0), Xo.reshape(-1, dx).std(0) + 1e-6; Xo = (Xo - mu) / sg
        g = np.array([list(r) for r in rows]); cy, cx = np.clip(P[..., 1].astype(int), 0, g.shape[0] - 1), np.clip(P[..., 0].astype(int), 0, g.shape[1] - 1)
        side, junc = D25.zones(P, info); wl = -1 if info["water_left"] else 1; dmap = {0: D25.dist_map(rows, "w"), 1: D25.dist_map(rows, "f")}; rec = dict(layout=lay, codes={})
        for kind in ("one", "past", "carried"):
            Z, valid, loss = code(kind, Xo, Acts, Lf, M, sd); vf = valid.reshape(-1)
            U = np.full(T * n, K, int); U[vf] = kmeans(Z.reshape(T * n, -1)[torch.as_tensor(vf, device=DEV)], K, sd).cpu().numpy(); U = U.reshape(T, n)       # unit K = 'just born' (no past yet)
            u = U.reshape(-1); pos = P.reshape(-1, 2); sf = side.reshape(-1)
            spread = float(np.mean([np.sqrt(((pos[u == q] - pos[u == q].mean(0)) ** 2).sum(1).mean()) for q in range(K) if (u == q).sum() > 5]))
            mix = 0
            for q in range(K):
                m = (u == q) & (sf != 0)
                if m.sum():
                    mix += min((sf[m] == -1).sum(), (sf[m] == 1).sum())
            r = dict(loss=loss, spread_cells=spread, arm_mix=float(mix / max(1, ((sf != 0) & vf).sum())))
            for sym in (False, True):
                for shuf in (False, True):
                    V = map_value(U, Lf, rise, K + 1, sym, shuffle_seed=sd if shuf else None); key = ("sym" if sym else "lived") + ("-shuf" if shuf else ""); r[key] = {}
                    for need in (0, 1):
                        v = V[u, need]; dist = dmap[need][cy, cx].reshape(-1).astype(float); other = (sf == -(wl if need == 0 else -wl)) & vf
                        r[key]["thirsty" if need == 0 else "hungry"] = dict(all=float(spearmanr(v[vf], -dist[vf])[0]), other_arm=float(spearmanr(v[other], -dist[other])[0]) if other.sum() > 200 and np.ptp(v[other]) > 0 else None)
            rec["codes"][kind] = r
            print(lay, kind, "loss", round(loss, 2), "spread", round(spread, 2), "arm mix", round(r["arm_mix"], 3), {kk: {q: (round(w["all"], 2), None if w["other_arm"] is None else round(w["other_arm"], 2)) for q, w in vv.items()} for kk, vv in r.items() if isinstance(vv, dict)}, round(time.time() - t0), flush=True)
        res.append(rec); json.dump(dict(protocol="atanor_os/GRAIN_THEORY.md (D27)", rows=res), open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
