"""D22 probe (registered in atanor_os/A1_LATENT_PROTOCOL.md before running): follow the relief-share score greedily (its best combo every beat; no pull,
hold, try or softmax) from the junction and from the wrong spring. Judge-only probe: is the knowledge good enough to get there?
  PYTHONPATH=. python -m tools.a1_greedy OUT.json 9101_0 ..."""
import json, os, sys, time
import numpy as np, torch
from tools.need_fork import t_room, DEV, DEPTH, K
from tools.a1_channel import V2, junction
from tools.a1_relief import relief_arm, shuffled
from tools.a1_sf import paths_of
from tools.a1_hold import zone
from atanor_os.emerge.room_body import RoomBody
from atanor_os.emerge.body_contract import Organs
from atanor_os.emerge.mind import Mind


@torch.no_grad()
def greedy(arm, organs_state, rows, info, seed, P, start, T=400, eps=0.0):
    torch.manual_seed(seed + 11); n = 2 * K
    B = RoomBody(n, rows, drain=0.0, seed=seed, device=DEV, regrow=V2["regrow"]); O = Organs(B.c, n, dev=DEV, depth=DEPTH); O.load(organs_state)
    if start == "junction":
        junction(B, info)
    else:
        at = {sp: cells[-1] for cells, sp in P.values()}
        for i in range(n):
            px, py, h = at["f"] if i < K else at["w"]; B.w.pos[i, 0], B.w.pos[i, 1], B.w.head[i] = px, py, h
    B.w.store[:K, 0], B.w.store[:K, 1] = 0.30, 0.95; B.w.store[K:, 0], B.w.store[K:, 1] = 0.95, 0.30
    O.reset(torch.ones(n, dtype=torch.bool, device=DEV)); x = O(B.raw()); st0 = B.w.store.clone(); entry = np.zeros(n, int); M = arm.m
    for t in range(T):
        a = arm.v(x).argmax(1)
        if eps:
            a = torch.where(torch.rand(n, device=DEV) < eps, torch.randint(0, M, (n,), device=DEV), a)
        B.step(a); x = O(B.raw()); side, _ = zone(B, info); entry = np.where((entry == 0) & (side != 0), side, entry)
    r = (B.w.store - st0).cpu().numpy(); got = np.r_[r[:K, 0], r[K:, 1]] > 0.05
    need_side = np.array([-1 if info["water_left"] else 1] * K + [1 if info["water_left"] else -1] * K); ch = entry != 0
    return dict(recover=[float(got[:K].mean()), float(got[K:].mean())], first_match=float((entry[ch] == need_side[ch]).mean()) if ch.any() else None, entered=float(ch.mean()))


if __name__ == "__main__":
    V2["on"] = True; RoomBody.default_cues = True; res = []; t0 = time.time(); out = sys.argv[1]
    for lay in sys.argv[2:]:
        seed, mirror = map(int, lay.split("_")); d = f"a1cache_many/{lay}"; rows, info = t_room(seed, bool(mirror)); sd = seed * 10 + mirror
        ring = torch.load(os.path.join(d, "life_ring.pt"), weights_only=False); ring["bx2_true"], ring["bd_true"] = ring["bx2"], ring["bd"]
        st = torch.load(os.path.join(d, "mind.pt"), map_location=DEV, weights_only=False)
        c = RoomBody(1, rows, drain=0.0, device=DEV).c; tmp = Mind(Organs(c, 1, dev=DEV, depth=DEPTH), c, 1, dev=DEV, worth=0); dx, ins, M = tmp.net.dx, tmp.o.inside, tmp.M; del tmp
        P = paths_of(rows, info); rec = dict(layout=lay)
        for nm, rg in (("relief", ring), ("relief-shuf", shuffled(ring, ins, sd))):
            arm = relief_arm(rg, dx, ins, M, sd)
            rec[nm] = {f"{s} eps{e}": greedy(arm, st["organs"], rows, info, sd, P, s, eps=e) for s in ("junction", "far") for e in (0.0, 0.1)}
            print(lay, nm, {k: (v["recover"], None if v["first_match"] is None else round(v["first_match"], 2)) for k, v in rec[nm].items()}, round(time.time() - t0), flush=True)
        res.append(rec); json.dump(dict(protocol="atanor_os/A1_LATENT_PROTOCOL.md (D22)", rows=res), open(out, "w"), indent=1)
