"""D12 (atanor_os/A1_SF_DIAG.md, registered before running): arm 3 (successor features x the current need) with its per-channel futures bootstrapped
from argmax S (as before) or from the combo really done next (SARSA). Offline on the cached lived rings; value map along both paths; A1 v2 fork.
  PYTHONPATH=. python -m tools.a1_sf OUT.json"""
import argparse, json, math, os, time
import numpy as np, torch
from tools.need_fork import t_room, DEV, DEPTH
from tools.a1_channel import V2, KEEP, side_sets, fork
from tools.a1_option import CACHE
from atanor_os.emerge.room_body import RoomBody
from atanor_os.emerge.body_contract import Organs
from atanor_os.emerge.mind import Mind
from atanor_os.emerge.channel_worth import ChannelWorth


def paths_of(rows, info):
    g = np.array([list(r) for r in rows]); out = {}
    for side in (-1, 1):
        x = int(info["x0"]); cells = []
        while g[2, x + side] != "#":
            x += side; cells.append((x + 0.5, 2.0, math.pi if side < 0 else 0.0))
        y = 2
        while g[y + 1, x] not in "#wf":
            y += 1; cells.append((x + 0.5, y + 0.5, math.pi / 2))
        out[side] = (cells, g[y + 1, x])
    return out


def arm3(ring, dx, ins, M, target, seed, updates=2000):
    torch.manual_seed(seed)
    a = ChannelWorth(dx, ins, lam=2.0, m=M, keep=KEEP, d=128, mode="bedrock").to(DEV); a.target = target
    X, X2, A, D = [ring[q].to(DEV) for q in ("bx", "bx2", "ba", "bd")]; k = len(A)
    a.see(X, X2, D > 0.5, a=A)
    joined = torch.zeros(k, dtype=torch.bool, device=DEV); joined[:-1] = ((X[1:] - X2[:-1]).abs().max(1).values < 1e-5) & (D[:-1] < 0.5)
    nxt = torch.cat([A[1:], A[-1:]]); a.ba2[:k] = torch.where(joined, nxt, torch.full_like(nxt, -1))
    for _ in range(updates // 4):
        a.learn(n=4)
    return a


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); a_ = ap.parse_args(); V2["on"] = True; res = []; t0 = time.time()
    for lay in sorted(os.listdir(CACHE)):
        seed, mirror = map(int, lay.split("_")); d = os.path.join(CACHE, lay); rows, info = t_room(seed, bool(mirror)); sd = seed * 10 + mirror
        ring = torch.load(os.path.join(d, "life_ring.pt"), weights_only=False); ck = os.path.join(d, "mind.pt")
        c = RoomBody(1, rows, drain=0.0, device=DEV).c; tmp = Mind(Organs(c, 1, dev=DEV, depth=DEPTH), c, 1, dev=DEV, worth=0); dx, ins, M = tmp.net.dx, tmp.o.inside, tmp.M; del tmp
        L, R = side_sets(rows, info, seed); P = paths_of(rows, info); rec = dict(layout=lay, arms={})
        for target in ("max", "sarsa"):
            arm = arm3(ring, dx, ins, M, target, sd); r = {}
            with torch.no_grad():
                j = torch.randint(0, len(ring["ba"]), (2048,)); xs = ring["bx"][j].to(DEV); _, psi = arm.P(xs)
                r["psi_mean"] = float(psi.mean()); r["saturated_share"] = float(((xs[:, ins][:, None, :] + psi) >= 1).float().mean())
            gaps = {}
            for side, (cells, spring) in P.items():
                n = 2 * len(cells); B = RoomBody(n, rows, drain=0.0, device=DEV, regrow=V2["regrow"]); O = Organs(B.c, n, dev=DEV, depth=DEPTH)
                for k, (px, py, h) in enumerate(cells * 2):
                    B.w.pos[k, 0], B.w.pos[k, 1], B.w.head[k] = px, py, h
                B.w.store[:len(cells), 0], B.w.store[:len(cells), 1] = 0.30, 0.95; B.w.store[len(cells):, 0], B.w.store[len(cells):, 1] = 0.95, 0.30
                O.reset(torch.ones(n, dtype=torch.bool, device=DEV)); x = O(B.raw())
                with torch.no_grad():
                    s = arm.v(x).max(1).values.cpu().numpy()
                gaps[spring] = dict(thirsty=s[:len(cells)].tolist(), hungry=s[len(cells):].tolist())
            r["map"] = gaps
            r["needed_gap_end"] = dict(thirsty=gaps["w"]["thirsty"][-1] - gaps["f"]["thirsty"][-1], hungry=gaps["f"]["hungry"][-1] - gaps["w"]["hungry"][-1])
            r["score_spread"] = float(np.ptp(gaps["w"]["thirsty"] + gaps["f"]["thirsty"] + gaps["w"]["hungry"] + gaps["f"]["hungry"]))
            fo = fork(ck, arm, rows, info, sd, L, R); r["fork"] = {k: fo[k] for k in ("value_pref_right", "first_match", "recover_share")}
            rec["arms"][target] = r
            print(lay, target, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k in ("psi_mean", "saturated_share", "score_spread", "needed_gap_end")}, r["fork"], round(time.time() - t0), flush=True)
        res.append(rec); json.dump(dict(protocol="atanor_os/A1_SF_DIAG.md", rows=res), open(a_.out, "w"), indent=1)


if __name__ == "__main__":
    main()
