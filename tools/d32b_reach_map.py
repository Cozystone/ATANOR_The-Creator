"""D32b (atanor_os/GRAIN_THEORY.md, registered before running): goal as a place unit, near, handed on along the stitched map.
  PYTHONPATH=. python -m tools.d32b_reach_map OUT.json 9101_0 ..."""
import json, math, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from tools.need_fork import t_room, DEV, DEPTH, K
from tools.a1_channel import V2, junction
from tools.a1_sf import paths_of
from tools import d25_grain as D25
from tools.d26_map import kmeans
from tools.d28_shortcut import value
from atanor_os.emerge.room_body import RoomBody
from atanor_os.emerge.body_contract import Organs

TRAVEL = "d30"; KU = 64


def place_net(X, Lf, seed, steps=3000):
    """Time contrast on one sensed state: states 1 - 8 beats apart in one life are the same place."""
    torch.manual_seed(seed); T, n, d = X.shape; net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 32)).to(DEV); opt = torch.optim.Adam(net.parameters(), 1e-3)
    L = torch.as_tensor(Lf, device=DEV)
    for s in range(steps):
        t = torch.randint(0, T - 8, (768,), device=DEV); i = torch.randint(0, n, (768,), device=DEV); dt = torch.randint(1, 9, (768,), device=DEV)
        ok = L[t, i] == L[t + dt, i]; t, i, dt = t[ok], i[ok], dt[ok]
        a = F.normalize(net(X[t, i]), dim=1); p = F.normalize(net(X[t + dt, i]), dim=1); loss = F.cross_entropy(a @ p.T / 0.1, torch.arange(len(a), device=DEV))
        opt.zero_grad(); loss.backward(); opt.step()
    return net.eval()


def skill(X, U, A, Lf, M, seed, updates=6000, batch=512, far=40):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); T, n, d = X.shape
    net = nn.Sequential(nn.Linear(d + KU, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, M)).to(DEV); opt = torch.optim.Adam(net.parameters(), 3e-4); le = 0.0
    for u in range(updates):
        t = rng.integers(0, T - 1, batch); i = rng.integers(0, n, batch); t2 = np.minimum(t + rng.integers(1, far + 1, batch), T - 1); ok = Lf[t, i] == Lf[t2, i]; t, i, t2 = t[ok], i[ok], t2[ok]
        loss = F.cross_entropy(net(torch.cat([X[t, i], F.one_hot(U[t2, i], KU).float()], 1)), A[t, i]); opt.zero_grad(); loss.backward(); opt.step()
        if u >= updates - 200:
            le += float(loss.detach()) / 200
    return net.eval(), le


@torch.no_grad()
def reach(sk, enc, cent, nxt, mu, sd, out, organs_state, rows, info, seed, P, start, which, T=400, eps=0.1):
    """nxt[c][u] = the unit to go to next from unit u when inside c is low. which: 'right' | 'wrong' (the other inside's map) | 'shuffled'."""
    torch.manual_seed(seed + 13); n = 2 * K
    B = RoomBody(n, rows, drain=0.0, seed=seed, device=DEV, regrow=V2["regrow"]); O = Organs(B.c, n, dev=DEV, depth=DEPTH); O.load(organs_state)
    if start == "junction":
        junction(B, info)
    elif start == "stem":
        B.w.pos[:, 0], B.w.pos[:, 1], B.w.head[:] = info["x0"], info["ys"], -math.pi / 2
    else:
        at = {sp: cells[-1] for cells, sp in P.values()}
        for i in range(n):
            px, py, h = at["f"] if i < K else at["w"]; B.w.pos[i, 0], B.w.pos[i, 1], B.w.head[i] = px, py, h
    B.w.store[:K, 0], B.w.store[:K, 1] = 0.30, 0.95; B.w.store[K:, 0], B.w.store[K:, 1] = 0.95, 0.30
    O.reset(torch.ones(n, dtype=torch.bool, device=DEV)); x = O(B.raw()); st0 = B.w.store.clone(); M = sk[-1].out_features
    need = torch.as_tensor([0] * K + [1] * K, device=DEV); tab = {"right": nxt["map"], "wrong": nxt["map"].flip(0), "shuffled": nxt["shuf"]}[which]
    for t in range(T):
        xo = (x[:, out] - mu) / sd; u = (F.normalize(enc(xo), dim=1) @ cent.T).argmax(1); g = tab[need, u]
        a = sk(torch.cat([xo, F.one_hot(g, KU).float()], 1)).argmax(1); a = torch.where(torch.rand(n, device=DEV) < eps, torch.randint(0, M, (n,), device=DEV), a); B.step(a); x = O(B.raw())
    r = (B.w.store - st0).cpu().numpy(); got = np.r_[r[:K, 0], r[K:, 1]] > 0.05
    return [float(got[:K].mean()), float(got[K:].mean())]


def next_unit(Un, Lf, rise, shuffle_seed=None, share=0.02):
    V = value(Un, Lf, rise, KU, shuffle_seed=shuffle_seed)                                   # [KU, 2]
    same = Lf[1:] == Lf[:-1]; a, b = Un[:-1][same], Un[1:][same]; C = np.zeros((KU, KU)); np.add.at(C, (a, b), 1); C = C + C.T
    nb = C >= share * np.maximum(C.sum(1, keepdims=True), 1); np.fill_diagonal(nb, True)
    return torch.as_tensor(np.stack([np.where(nb, V[None, :, c], -np.inf).argmax(1) for c in (0, 1)]), device=DEV)   # [2, KU]


def main():
    V2["on"] = True; RoomBody.default_cues = True; outp = sys.argv[1]; res = []; t0 = time.time()
    for lay in sys.argv[2:]:
        seed, mirror = map(int, lay.split("_")); assert not 7701 <= seed <= 7712; rows, info = t_room(seed, bool(mirror)); sd_ = seed * 10 + mirror
        z = torch.load(f"{TRAVEL}/{lay}.pt", weights_only=False); st = torch.load(f"{D25.ROOT}/{lay}/mind.pt", map_location=DEV, weights_only=False)
        out = [i for i in range(z["X"].shape[2]) if i not in set(z["ins"])]; P = paths_of(rows, info); St, Lf = z["St"], z["Lf"]; T, n = Lf.shape
        X = z["X"][:, :, out].to(DEV); mu, sd = X.reshape(-1, len(out)).mean(0), X.reshape(-1, len(out)).std(0) + 1e-3; X = (X - mu) / sd; A = torch.as_tensor(z["acts"]).to(DEV)
        rise = np.zeros((T, n, 2), bool); rise[:-1] = ((St[1:] - St[:-1]) > 1e-4) & (Lf[1:] == Lf[:-1])[..., None]
        with torch.enable_grad():
            enc = place_net(X, Lf, sd_)
        with torch.no_grad():
            Z = F.normalize(enc(X.reshape(T * n, -1)), dim=1); U0 = kmeans(Z, KU, sd_); cent = F.normalize(torch.stack([Z[U0 == k].mean(0) if (U0 == k).any() else Z[0] * 0 for k in range(KU)]), dim=1)
            U = (Z @ cent.T).argmax(1).reshape(T, n)
        Un = U.cpu().numpy(); nxt = dict(map=next_unit(Un, Lf, rise), shuf=next_unit(Un, Lf, rise, shuffle_seed=sd_))
        with torch.enable_grad():
            sk, le = skill(X, U, A, Lf, int(z["M"]), sd_)
        rec = dict(layout=lay, loss_end=round(le, 3), chance_loss=round(math.log(z["M"]), 3), units_used=int(len(np.unique(Un))), reach={})
        for which in ("right", "wrong", "shuffled"):
            rec["reach"][which] = {s: reach(sk, enc, cent, nxt, mu, sd, out, st["organs"], rows, info, sd_, P, s, which) for s in ("junction", "far", "stem")}
        print(lay, "loss", rec["loss_end"], "of", rec["chance_loss"], "units", rec["units_used"], "|", rec["reach"], round(time.time() - t0), flush=True)
        res.append(rec); json.dump(dict(protocol="atanor_os/GRAIN_THEORY.md (D32b)", rows=res), open(outp, "w"), indent=1)


if __name__ == "__main__":
    main()
