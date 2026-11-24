"""D33b follow-up diagnostic (no change to the being): why nothing arrives from the wrong spring. One layout: play + practice as D33b, own map,
then (a) the map's own chain from the unit of the wrong spring: does following the table reach a unit where the needed inside rises?
(b) the bodies' path from the wrong spring: distance moved, units visited, beats with goal unit == own unit.
  PYTHONPATH=. python -m tools.d33c_far_diag OUT.json 9101_0"""
import json, math, sys
import numpy as np, torch
from tools.need_fork import t_room, DEV, K
from tools.a1_channel import V2
from tools.a1_sf import paths_of
from tools import d25_grain as D25
from tools import d33_play_reach as D
from tools import d33b_map_reach as DB
from atanor_os.emerge.room_body import RoomBody


@torch.no_grad()
def far_trace(skill, cent, nxt, rows, info, seed, st, P, inside, T=400):
    torch.manual_seed(seed + 13); n = 2 * K; B, O = D.body(rows, n, seed, st); M = len(B.legs)
    at = {sp: cells[-1] for cells, sp in P.values()}
    for i in range(n):
        px, py, h = at["f"] if i < K else at["w"]; B.w.pos[i, 0], B.w.pos[i, 1], B.w.head[i] = px, py, h
    O.reset(torch.ones(n, dtype=torch.bool, device=DEV)); D.set_organ(B, O, info); x = O(B.raw()); p0 = B.w.pos.clone()
    need = torch.as_tensor([0] * K + [1] * K, device=DEV); same = 0; far = torch.zeros(n, device=DEV); seen = [set() for _ in range(n)]
    start_u = torch.cdist(x[:, -D.NP:-2], cent).argmin(1)
    for t in range(T):
        u = torch.cdist(x[:, -D.NP:-2], cent).argmin(1); g_u = nxt[need, u]; same += int((g_u == u).sum())
        for i, v in enumerate(u.tolist()):
            seen[i].add(v)
        xx = x.clone(); xx[:, inside] = 0; a = torch.multinomial(torch.softmax(skill(xx, cent[g_u]), 1), 1)[:, 0]
        a = torch.where(torch.rand(n, device=DEV) < 0.1, torch.randint(0, M, (n,), device=DEV), a); B.step(a); x = O(B.raw()); far = torch.maximum(far, (B.w.pos - p0).norm(dim=1))
    return dict(start_units=sorted(set(start_u.tolist())), goal_is_own_unit=same / (T * n), max_distance_cells=[round(float(far[:K].mean()), 2), round(float(far[K:].mean()), 2)],
                units_visited=round(float(np.mean([len(s) for s in seen])), 1))


def chain(nxt, start, r, c, steps=64):
    u = start; path = [u]
    for _ in range(steps):
        v = int(nxt[c, u]); path.append(v)
        if r[v, c] > 0 or v == u:
            break
        u = v
    return dict(path=path, ends_where_inside_rises=bool(r[path[-1], c] > 0), stuck=path[-1] == path[-2])


def main():
    V2["on"] = True; RoomBody.default_cues = True; lay = sys.argv[2]; seed, mirror = map(int, lay.split("_")); rows, info = t_room(seed, bool(mirror)); sd = seed * 10 + mirror
    st = torch.load(f"{D25.ROOT}/{lay}/mind.pt", map_location=DEV, weights_only=False)["organs"]; P = paths_of(rows, info); inside = list(range(3))
    plays = [D.play(rows, info, sd, st, None)]; X0 = plays[0][0]; dx = X0.shape[2]
    skill = D.Skill(dx, 90).to(DEV); Xf = X0.reshape(-1, dx); skill.mu.data = Xf.mean(0); skill.sd.data = Xf.std(0) + 1e-3
    for rnd in range(6):
        with torch.enable_grad():
            D.train(skill, plays, inside)
        if rnd < 5:
            pos = torch.cat([p[0] for p in plays])[:, :, -D.NP:-2].reshape(-1, D.NP - 2); plays.append(D.play(rows, info, sd + 100 * (rnd + 1), st, skill, pool=pos[torch.randperm(len(pos), device=DEV)[:20000]]))
    cent, Tm, r, nb = DB.own_map(plays, sd); nxt = DB.next_table(Tm, r, nb)
    tr = far_trace(skill, cent, nxt, rows, info, sd, st, P, inside)
    ch = {f"from unit {u}": {need: chain(nxt.cpu().numpy(), u, r, c) for c, need in ((0, "water"), (1, "food"))} for u in tr["start_units"]}
    G = torch.cat([p[0][:, :, -D.NP:-2] for p in plays]); Rr = torch.cat([p[2] for p in plays]); fr = {}
    for c, nm in ((0, "water"), (1, "food")):
        u = torch.cdist(G[Rr[..., c]], cent).argmin(1); fr[nm] = torch.bincount(u, minlength=DB.KU).nonzero()[:, 0].tolist()
    out = dict(layout=lay, trace=tr, chains=ch, relief_units=fr, unit_play_counts_at_start=[int(np.bincount(torch.cdist(G.reshape(-1, D.NP - 2), cent).argmin(1).cpu().numpy(), minlength=DB.KU)[u]) for u in tr["start_units"]])
    print(json.dumps(out, indent=1)); json.dump(out, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
