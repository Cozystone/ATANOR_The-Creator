"""D33b (atanor_os/P1_PATH_PROTOCOL.md, registered before running): the D33 skill, handed on along a map built from the being's own play.
  PYTHONPATH=. python -m tools.d33b_map_reach OUT.json 9101_0 ..."""
import json, math, sys, time
import numpy as np, torch
from tools.need_fork import t_room, DEV, K
from tools.a1_channel import V2, junction
from tools.a1_sf import paths_of
from tools import d25_grain as D25
from tools.d26_map import kmeans
from tools import d33_play_reach as D
from atanor_os.emerge.room_body import RoomBody

KU, GAMMA = 64, 0.99


def own_map(plays, seed):
    """Units from the place codes of all play beats; transitions within play; per-inside value over units; neighbour sets."""
    G = torch.cat([p[0][:, :, -D.NP:-2] for p in plays], 0)                 # [T, n, 24]
    Rr = torch.cat([p[2] for p in plays], 0)                                # [T, n, 2]
    T, n, _ = G.shape; U = kmeans(G.reshape(-1, D.NP - 2), KU, seed).reshape(T, n)
    cent = torch.stack([G.reshape(-1, D.NP - 2)[(U == k).reshape(-1)].mean(0) if (U == k).any() else torch.zeros(D.NP - 2, device=DEV) for k in range(KU)])
    Un = U.cpu().numpy(); C = np.zeros((KU, KU)); o = 0
    for p in plays:                                                         # transitions inside each play only (D33c2: the seams between plays linked
        L = p[0].shape[0]; np.add.at(C, (Un[o:o + L - 1].reshape(-1), Un[o + 1:o + L].reshape(-1)), 1); o += L   # every last place to the birth place)
    C = C + C.T; Tm = C / np.maximum(C.sum(1, keepdims=True), 1); cnt = np.bincount(Un.reshape(-1), minlength=KU).astype(float)
    r = np.stack([np.bincount(Un.reshape(-1), weights=Rr[..., c].reshape(-1).float().cpu().numpy(), minlength=KU) / np.maximum(cnt, 1) for c in (0, 1)], 1)
    nb = (C > 0) | np.eye(KU, dtype=bool)
    return cent, Tm, r, nb


def next_table(Tm, r, nb, shuffle_seed=None):
    if shuffle_seed is not None:
        r = r[np.random.default_rng(shuffle_seed).permutation(len(r))]
    V = np.linalg.solve(np.eye(KU) - GAMMA * Tm, r)
    return torch.as_tensor(np.stack([np.where(nb, V[None, :, c], -np.inf).argmax(1) for c in (0, 1)]), device=DEV)   # [2, KU]


@torch.no_grad()
def test_map(skill, cent, nxt, rows, info, seed, organs_state, P, start, inside, flip=False, T=400, eps=0.1):
    torch.manual_seed(seed + 13); n = 2 * K; B, O = D.body(rows, n, seed, organs_state); M = len(B.legs)
    if start == "junction":
        junction(B, info)
    elif start == "stem":
        B.w.pos[:, 0], B.w.pos[:, 1], B.w.head[:] = info["x0"], info["ys"], -math.pi / 2
    else:
        at = {sp: cells[-1] for cells, sp in P.values()}
        for i in range(n):
            px, py, h = at["f"] if i < K else at["w"]; B.w.pos[i, 0], B.w.pos[i, 1], B.w.head[i] = px, py, h
    O.reset(torch.ones(n, dtype=torch.bool, device=DEV)); D.set_organ(B, O, info)
    B.w.store[:K, 0], B.w.store[:K, 1] = 0.30, 0.95; B.w.store[K:, 0], B.w.store[K:, 1] = 0.95, 0.30; st0 = B.w.store.clone(); x = O(B.raw())
    need = torch.as_tensor([0] * K + [1] * K, device=DEV); need = 1 - need if flip else need
    for t in range(T):
        u = torch.cdist(x[:, -D.NP:-2], cent).argmin(1); g = cent[nxt[need, u]]
        xx = x.clone(); xx[:, inside] = 0; a = torch.multinomial(torch.softmax(skill(xx, g), 1), 1)[:, 0]
        a = torch.where(torch.rand(n, device=DEV) < eps, torch.randint(0, M, (n,), device=DEV), a); B.step(a); x = O(B.raw())
    r = (B.w.store - st0).cpu().numpy(); got = np.r_[r[:K, 0], r[K:, 1]] > 0.05
    return [float(got[:K].mean()), float(got[K:].mean())]


def main():
    V2["on"] = True; RoomBody.default_cues = True; outp = sys.argv[1]; res = []; t0 = time.time()
    for lay in sys.argv[2:]:
        seed, mirror = map(int, lay.split("_")); assert not 7701 <= seed <= 7712; rows, info = t_room(seed, bool(mirror)); sd = seed * 10 + mirror
        st = torch.load(f"{D25.ROOT}/{lay}/mind.pt", map_location=DEV, weights_only=False)["organs"]; P = paths_of(rows, info); inside = list(range(3))
        plays = [D.play(rows, info, sd, st, None)]; X0 = plays[0][0]; dx = X0.shape[2]
        skill = D.Skill(dx, 90).to(DEV); Xf = X0.reshape(-1, dx); skill.mu.data = Xf.mean(0); skill.sd.data = Xf.std(0) + 1e-3
        for rnd in range(6):                                                  # as D33: babbling, then 5 rounds of goal practice, retrained each round
            with torch.enable_grad():
                le = D.train(skill, plays, inside)
            if rnd < 5:
                pos = torch.cat([p[0] for p in plays])[:, :, -D.NP:-2].reshape(-1, D.NP - 2); plays.append(D.play(rows, info, sd + 100 * (rnd + 1), st, skill, pool=pos[torch.randperm(len(pos), device=DEV)[:20000]]))
        cent, Tm, r, nb = own_map(plays, sd); rec = dict(layout=lay, loss=round(le, 3), units_with_relief=[int((r[:, c] > 0).sum()) for c in (0, 1)], reach={})
        tabs = {"map": (next_table(Tm, r, nb), False), "other inside's map": (next_table(Tm, r, nb), True), "shuffled map": (next_table(Tm, r, nb, shuffle_seed=sd), False)}
        for nm, (tab, flip) in tabs.items():
            rec["reach"][nm] = {s: test_map(skill, cent, tab, rows, info, sd, st, P, s, inside, flip=flip) for s in ("junction", "far", "stem")}
        print(lay, "loss", rec["loss"], "relief units", rec["units_with_relief"], "|", rec["reach"], round(time.time() - t0), flush=True)
        res.append(rec); json.dump(dict(protocol="atanor_os/P1_PATH_PROTOCOL.md (D33b)", rows=res), open(outp, "w"), indent=1); del plays; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
