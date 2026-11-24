"""D32 (atanor_os/GRAIN_THEORY.md, registered before running): a skill of going to a remembered place, learned from trips by hindsight (no reward).
  PYTHONPATH=. python -m tools.d32_reach OUT.json 9101_0 ..."""
import json, math, os, sys, time
import numpy as np, torch, torch.nn as nn
from tools.need_fork import t_room, DEV, DEPTH, K
from tools.a1_channel import V2, junction
from tools.a1_sf import paths_of
from tools import d25_grain as D25
from atanor_os.emerge.room_body import RoomBody
from atanor_os.emerge.body_contract import Organs

TRAVEL = "d30"


def train(z, out, seed, updates=6000, batch=512, far=200):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    X = z["X"][:, :, out].to(DEV); A = torch.as_tensor(z["acts"]).to(DEV); Lf = z["Lf"]; T, n, d = X.shape; M = int(z["M"])
    mu, sd = X.reshape(-1, d).mean(0), X.reshape(-1, d).std(0) + 1e-3
    net = nn.Sequential(nn.Linear(2 * d, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, M)).to(DEV); opt = torch.optim.Adam(net.parameters(), 3e-4)
    loss_end = 0.0
    for u in range(updates):
        t = rng.integers(0, T - 1, batch); i = rng.integers(0, n, batch); k = rng.integers(1, far + 1, batch); t2 = np.minimum(t + k, T - 1)
        ok = Lf[t, i] == Lf[t2, i]                                         # the later state must belong to the same life
        t, i, t2 = t[ok], i[ok], t2[ok]
        x = (X[t, i] - mu) / sd; g = (X[t2, i] - mu) / sd
        loss = nn.functional.cross_entropy(net(torch.cat([x, g], 1)), A[t, i]); opt.zero_grad(); loss.backward(); opt.step()
        if u >= updates - 200:
            loss_end += float(loss) / 200
    return net.eval(), mu, sd, loss_end


def relief_memories(z, out):
    St, Lf = z["St"], z["Lf"]; X = z["X"]; mem = []
    for c in (0, 1):
        t, i = np.nonzero((St[1:, :, c] - St[:-1, :, c] > 0.01) & (Lf[1:] == Lf[:-1]))
        mem.append(X[t + 1, i][:, out].to(DEV))                           # outside senses at the beat after that inside rose
    return mem


@torch.no_grad()
def reach(net, mu, sd, out, mem, organs_state, rows, info, seed, P, start, goal, T=400, eps=0.1):
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
    O.reset(torch.ones(n, dtype=torch.bool, device=DEV)); x = O(B.raw()); st0 = B.w.store.clone(); M = net[-1].out_features
    need = [0] * K + [1] * K; G = torch.zeros(n, len(out), device=DEV)
    if goal != "zero":
        for i in range(n):
            m = mem[need[i] if goal == "right" else 1 - need[i]]; G[i] = (m[torch.randint(0, len(m), (1,), device=DEV)[0]] - mu) / sd
    for t in range(T):
        a = net(torch.cat([(x[:, out] - mu) / sd, G], 1)).argmax(1); a = torch.where(torch.rand(n, device=DEV) < eps, torch.randint(0, M, (n,), device=DEV), a); B.step(a); x = O(B.raw())
    r = (B.w.store - st0).cpu().numpy(); got = np.r_[r[:K, 0], r[K:, 1]] > 0.05
    return [float(got[:K].mean()), float(got[K:].mean())]


def main():
    V2["on"] = True; RoomBody.default_cues = True; outp = sys.argv[1]; res = []; t0 = time.time()
    for lay in sys.argv[2:]:
        seed, mirror = map(int, lay.split("_")); assert not 7701 <= seed <= 7712; rows, info = t_room(seed, bool(mirror)); sd_ = seed * 10 + mirror
        z = torch.load(f"{TRAVEL}/{lay}.pt", weights_only=False); st = torch.load(f"{D25.ROOT}/{lay}/mind.pt", map_location=DEV, weights_only=False)
        out = [i for i in range(z["X"].shape[2]) if i not in set(z["ins"])]; P = paths_of(rows, info)
        with torch.enable_grad():
            net, mu, sd, le = train(z, out, sd_)
        mem = relief_memories(z, out); rec = dict(layout=lay, loss_end=round(le, 3), chance_loss=round(math.log(z["M"]), 3), memories=[len(m) for m in mem], reach={})
        for goal in ("right", "wrong", "zero"):
            rec["reach"][goal] = {s: reach(net, mu, sd, out, mem, st["organs"], rows, info, sd_, P, s, goal) for s in ("junction", "far", "stem")}
        print(lay, "loss", rec["loss_end"], "of", rec["chance_loss"], "memories", rec["memories"], "|", rec["reach"], round(time.time() - t0), flush=True)
        res.append(rec); json.dump(dict(protocol="atanor_os/GRAIN_THEORY.md (D32)", rows=res), open(outp, "w"), indent=1)


if __name__ == "__main__":
    main()
