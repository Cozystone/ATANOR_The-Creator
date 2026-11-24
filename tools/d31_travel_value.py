"""D31 (atanor_os/GRAIN_THEORY.md, registered before running): the being's own value trained on a traveller's experience - does it learn the way and steer?
  PYTHONPATH=. python -m tools.d31_travel_value OUT.json 9101_0 ..."""
import json, math, os, sys, time
import numpy as np, torch
from scipy.stats import spearmanr
from tools.need_fork import t_room, DEV, DEPTH, K
from tools.a1_channel import V2, side_sets, fork, junction
from tools.a1_relief import relief_arm, shuffled, far_fork
from tools.a1_sf import paths_of
from tools import d25_grain as D25
from atanor_os.emerge.room_body import RoomBody
from atanor_os.emerge.body_contract import Organs
from atanor_os.emerge.mind import Mind

TRAVEL = "d30"; OWN = "d27"


def ring_of(z):
    X, A, Lf = z["X"], torch.as_tensor(z["acts"]), z["Lf"]; T, n, dx = X.shape; same = torch.as_tensor(Lf[1:] == Lf[:-1])
    bx = X[:-1].reshape(-1, dx); bx2 = X[1:].reshape(-1, dx); ba = A[:-1].reshape(-1).long(); ba2 = torch.where(same, A[1:], torch.full_like(A[1:], -1)).reshape(-1).long()
    bd = (~same).reshape(-1).float()                                   # a change of life between two beats = the body died on that step
    r = dict(bx=bx, bx2=bx2, ba=ba, bd=bd, ba2=ba2); r["bx2_true"], r["bd_true"] = r["bx2"], r["bd"]
    return r


@torch.no_grad()
def greedy_from(arm, organs_state, rows, info, seed, P, start, T=400, eps=0.1):
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
    O.reset(torch.ones(n, dtype=torch.bool, device=DEV)); x = O(B.raw()); st0 = B.w.store.clone(); M = arm.m
    for t in range(T):
        a = arm.v(x).argmax(1); a = torch.where(torch.rand(n, device=DEV) < eps, torch.randint(0, M, (n,), device=DEV), a); B.step(a); x = O(B.raw())
    r = (B.w.store - st0).cpu().numpy(); got = np.r_[r[:K, 0], r[K:, 1]] > 0.05
    return [float(got[:K].mean()), float(got[K:].mean())]


def main():
    V2["on"] = True; RoomBody.default_cues = True; out = sys.argv[1]; res = []; t0 = time.time()
    for lay in sys.argv[2:]:
        seed, mirror = map(int, lay.split("_")); rows, info = t_room(seed, bool(mirror)); sd = seed * 10 + mirror; ck = f"{D25.ROOT}/{lay}/mind.pt"
        ring = ring_of(torch.load(f"{TRAVEL}/{lay}.pt", weights_only=False)); st = torch.load(ck, map_location=DEV, weights_only=False)
        c = RoomBody(1, rows, drain=0.0, device=DEV).c; tmp = Mind(Organs(c, 1, dev=DEV, depth=DEPTH), c, 1, dev=DEV, worth=0); dx, ins_, M = tmp.net.dx, tmp.o.inside, tmp.M; del tmp
        L, R = side_sets(rows, info, seed); P = paths_of(rows, info); own = torch.load(f"{OWN}/{lay}.pt", weights_only=False); X, Pz, ins = own["X"], own["P"], own["ins"]; T, n, dxx = X.shape
        g = np.array([list(r) for r in rows]); cy, cx = np.clip(Pz[..., 1].astype(int), 0, g.shape[0] - 1), np.clip(Pz[..., 0].astype(int), 0, g.shape[1] - 1)
        cell = (cy * 100 + cx).reshape(-1); cells, ci = np.unique(cell, return_inverse=True); Dc = [D25.dist_map(rows, q)[cells // 100, cells % 100].astype(float) for q in "wf"]
        rec = dict(layout=lay, ring=len(ring["ba"]), arms={})
        for nm, rg, s_ in (("travel", ring, sd), ("travel-shuf", shuffled(ring, ins_, sd), sd + 2000)):
            arm = relief_arm(rg, dx, ins_, M, s_, updates=int(os.environ.get("D31_UPDATES", 10000)), gamma=0.99, sync=25); r = {}
            way = {}
            for need, q in ((0, "thirsty"), (1, "hungry")):
                V = D25.scores(arm, X.reshape(T * n, dxx), ins, need).mean(1); vc = np.bincount(ci, weights=V) / np.bincount(ci)
                way[q] = dict(per_state=round(float(spearmanr(V, -Dc[need][ci])[0]), 2), per_cell=round(float(spearmanr(vc, -Dc[need])[0]), 2))
            r["way"] = way
            r["greedy"] = {s: greedy_from(arm, st["organs"], rows, info, sd, P, s) for s in ("junction", "far", "stem")}
            fo = fork(ck, arm, rows, info, sd, L, R)
            r["being"] = dict(junction_first=fo["first_match"], junction_recover=fo["recover_share"], far=far_fork(ck, arm, rows, info, sd, P)["recover"], stem=far_fork(ck, arm, rows, info, sd, P, start="stem")["recover"])
            rec["arms"][nm] = r; print(lay, nm, "| way", way, "| greedy+10%", r["greedy"], "| being", {k: [round(float(v), 2) for v in vv] for k, vv in r["being"].items()}, round(time.time() - t0), flush=True)
            del arm; torch.cuda.empty_cache()
        res.append(rec); json.dump(dict(protocol="atanor_os/GRAIN_THEORY.md (D31)", rows=res), open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
