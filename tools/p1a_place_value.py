"""P1a (atanor_os/P1_PATH_PROTOCOL.md, registered before running): with an innate place sense (path integration, grid-like code), does the being's own
value learn the way from its own experience, and steer from afar? Offline, as D31.
In the room felt motion is exact, so the integrated place equals the true place relative to where the life began; it is computed from that here.
The stretches hold no heading, so this probe gives the place part of the code only (24 numbers), not the heading (said in the result).
  PYTHONPATH=. python -m tools.p1a_place_value OUT.json 9101_0 ..."""
import json, math, os, sys, time
import numpy as np, torch
from scipy.stats import spearmanr
from tools.need_fork import t_room, DEV, DEPTH, K
from tools.a1_channel import V2, junction
from tools.a1_relief import relief_arm, shuffled
from tools.a1_sf import paths_of
from tools import d25_grain as D25
from tools.d31_travel_value import ring_of
from atanor_os.emerge.room_body import RoomBody
from atanor_os.emerge.body_contract import Organs
from atanor_os.emerge.mind import Mind

OWN = "d27"; TRAVEL = "d30"
DIRS = np.array([[math.cos(a), math.sin(a)] for a in (0.0, math.pi / 3, 2 * math.pi / 3)]); SCALES = (2.0, 4.0, 8.0, 16.0)


def grid(p, origin):
    """p [..., 2] places (cells) -> [..., 24] grid-like code relative to where the life began."""
    q = (np.asarray(p, np.float64) - origin) @ DIRS.T                               # [..., 3]
    ph = np.concatenate([2 * math.pi * q / s for s in SCALES], -1)                   # [..., 12]
    return np.concatenate([np.cos(ph), np.sin(ph)], -1).astype(np.float32)


def with_code(z, origin):
    z = dict(z); z["X"] = torch.cat([z["X"], torch.as_tensor(grid(z["P"], origin))], 2); return z


@torch.no_grad()
def greedy(arm, organs_state, rows, info, seed, P, start, origin, code=True, T=400, eps=0.1):
    torch.manual_seed(seed + 11); n = 2 * K
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
    O.reset(torch.ones(n, dtype=torch.bool, device=DEV)); st0 = B.w.store.clone(); M = arm.m

    def sense():
        x = O(B.raw()); return torch.cat([x, torch.as_tensor(grid(B.w.pos.cpu().numpy(), origin), device=DEV)], 1) if code else x
    x = sense()
    for t in range(T):
        a = arm.v(x).argmax(1); a = torch.where(torch.rand(n, device=DEV) < eps, torch.randint(0, M, (n,), device=DEV), a); B.step(a); x = sense()
    r = (B.w.store - st0).cpu().numpy(); got = np.r_[r[:K, 0], r[K:, 1]] > 0.05
    return [float(got[:K].mean()), float(got[K:].mean())]


def main():
    V2["on"] = True; RoomBody.default_cues = True; out = sys.argv[1]; res = []; t0 = time.time(); U = int(os.environ.get("P1_UPDATES", 10000))
    for lay in sys.argv[2:]:
        seed, mirror = map(int, lay.split("_")); assert not 7701 <= seed <= 7712; rows, info = t_room(seed, bool(mirror)); sd = seed * 10 + mirror
        st = torch.load(f"{D25.ROOT}/{lay}/mind.pt", map_location=DEV, weights_only=False); origin = np.array([info["x0"], info["ys"]], np.float64)
        c = RoomBody(1, rows, drain=0.0, device=DEV).c; tmp = Mind(Organs(c, 1, dev=DEV, depth=DEPTH), c, 1, dev=DEV, worth=0); dx, ins_, M = tmp.net.dx, tmp.o.inside, tmp.M; del tmp
        P = paths_of(rows, info); own = torch.load(f"{OWN}/{lay}.pt", weights_only=False); X, Pz, ins = own["X"], own["P"], own["ins"]; T, n, _ = X.shape
        g = np.array([list(r) for r in rows]); cy, cx = np.clip(Pz[..., 1].astype(int), 0, g.shape[0] - 1), np.clip(Pz[..., 0].astype(int), 0, g.shape[1] - 1)
        cell = (cy * 100 + cx).reshape(-1); cells, ci = np.unique(cell, return_inverse=True); Dc = [D25.dist_map(rows, q)[cells // 100, cells % 100].astype(float) for q in "wf"]
        Xc = torch.cat([X, torch.as_tensor(grid(Pz, origin))], 2).reshape(T * n, -1)
        rec = dict(layout=lay, arms={})
        for src, path in (("own", OWN), ("travel", TRAVEL)):
            ring = ring_of(with_code(torch.load(f"{path}/{lay}.pt", weights_only=False), origin))
            for nm, rg, s_ in ((src, ring, sd), (src + "-shuf", shuffled(ring, ins_, sd), sd + 2000)):
                arm = relief_arm(rg, dx + 24, ins_, M, s_, updates=U, gamma=0.99, sync=25); way = {}
                for need, q in ((0, "thirsty"), (1, "hungry")):
                    V = D25.scores(arm, Xc, ins, need).mean(1); vc = np.bincount(ci, weights=V) / np.bincount(ci)
                    way[q] = dict(per_state=round(float(spearmanr(V, -Dc[need][ci])[0]), 2), per_cell=round(float(spearmanr(vc, -Dc[need])[0]), 2))
                r = dict(way=way, greedy={s: greedy(arm, st["organs"], rows, info, sd, P, s, origin) for s in ("junction", "far", "stem")})
                rec["arms"][nm] = r; print(lay, nm, "| way", way, "| greedy+10%", r["greedy"], round(time.time() - t0), flush=True)
                del arm; torch.cuda.empty_cache()
        res.append(rec); json.dump(dict(protocol="atanor_os/P1_PATH_PROTOCOL.md (P1a)", rows=res), open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
