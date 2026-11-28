"""L1 (atanor_os/P1_PATH_PROTOCOL.md, registered before running): a life with the going organ (atanor_os/emerge/going.py).
Protected childhood (3,000 beats: babbling, then practice toward places from its own place memory), then ordinary life with drain, the mind choosing
with the lean of the lack. At the end, the being as it is, judge-put at the wrong spring and the stem with one inside low, choosing on its own.
  PYTHONPATH=. python -m tools.l1_life OUT.json 9101_0 ... [--W 1.0] [--tag X]"""
import argparse, json, math, os, time
import numpy as np, torch
from tools.need_fork import t_room, place, DEV, DEPTH, BEING, K
from tools.a1_channel import V2
from tools.a1_sf import paths_of
from tools import d25_grain as D25
from tools.d33_play_reach import set_organ
from atanor_os.emerge.room_body import RoomBody
from atanor_os.emerge.body_contract import Organs, Channel
from atanor_os.emerge.mind import Mind
from atanor_os.emerge.going import Going

SAVE = "l1"


def make(rows, n, seed, drain):
    B = RoomBody(n, rows, drain=drain, seed=seed, device=DEV, regrow=V2["regrow"]); B.c.sensors[2] = Channel("vector", (3,), motion=True); return B


@torch.no_grad()
def forks(Mi, O, rows, info, seed, P, T=400):
    """The being as it is (no learning during the test), judge-put with one inside low; returns recovery of the needed store."""
    out = {}; n = Mi.n
    for start in ("far", "stem"):
        torch.manual_seed(seed + 21); B = make(rows, n, seed, 0.0); h = n // 2
        if start == "stem":
            B.w.pos[:, 0], B.w.pos[:, 1], B.w.head[:] = info["x0"], info["ys"], -math.pi / 2
        else:
            at = {sp: cells[-1] for cells, sp in P.values()}
            for i in range(n):
                px, py, hh = at["f"] if i < h else at["w"]; B.w.pos[i, 0], B.w.pos[i, 1], B.w.head[i] = px, py, hh
        B.w.store[:h, 0], B.w.store[:h, 1] = 0.30, 0.95; B.w.store[h:, 0], B.w.store[h:, 1] = 0.95, 0.30
        every = torch.ones(n, dtype=torch.bool, device=DEV); O.reset(every); set_organ(B, O, info); Mi.begin(every); Mi.prev = None
        st0 = B.w.store.clone(); x = O(B.raw()); src = np.zeros(5, int)
        for t in range(T):
            a = Mi.choose(x); src += np.bincount(Mi.src.cpu().numpy(), minlength=5); B.step(a); x = O(B.raw())
        r = (B.w.store - st0).cpu().numpy(); got = np.r_[r[:h, 0], r[h:, 1]] > 0.05
        out[start] = [float(got[:h].mean()), float(got[h:].mean())]; out[start + "_src"] = (src / src.sum()).round(3).tolist()
    return out


def life(rows, info, seed, W, n=64, beats=12000, child=3000, babble=1000, keep_last=3000):
    torch.manual_seed(seed); rng = np.random.default_rng(seed + 9)
    B = make(rows, n, seed, V2["drain"]); O = Organs(B.c, n, dev=DEV, depth=DEPTH); Mi = Mind(O, B.c, n, dev=DEV, value_arm="relief"); Mi.grow(BEING); Mi.wo.on = True
    Mi.going = Going(O.dx, Mi.M, n, O.inside, DEV, W=W); G = Mi.going
    B.w.store[:] = 0.75; place(B, info); every = torch.ones(n, dtype=torch.bool, device=DEV); O.reset(every); Mi.begin(every); x = O(B.raw())
    per = beats // 4; Q = [dict(fills=[0, 0], need_fills=[0, 0], deaths=0, src=np.zeros(5, int), stem=0, beats=0, skill_loss=[]) for _ in range(4)]
    X, Pz, St, Lf, A = [], [], [], [], []; lifeid = np.zeros(n, int); goal = None; left = torch.zeros(n, device=DEV)
    for t in range(beats):
        q = Q[min(t // per, 3)]; s0 = B.w.store.clone(); pos = B.w.pos.cpu().numpy().copy()
        q["stem"] += int(((np.abs(pos[:, 0] - info["x0"]) <= 2.5) & (pos[:, 1] >= 3.0)).sum()); q["beats"] += n
        if t < babble:                                                    # childhood, babbling
            act = torch.randint(0, Mi.M, (n,), device=DEV)
        elif t < child:                                                   # childhood, practice toward places from its own place memory
            u = G.unit(G.place(x)); arrive = goal is not None and (u == goal)
            new = (left <= 0) | (arrive if goal is not None else torch.ones(n, dtype=torch.bool, device=DEV))
            pick = torch.randint(0, G.used if G.born else G.KU, (n,), device=DEV); goal = pick if goal is None else torch.where(new, pick, goal); left = torch.where(new, torch.full_like(left, 200), left - 1)
            act = torch.multinomial(G.toward(x, G.C[goal]).exp(), 1)[:, 0]; act = torch.where(torch.rand(n, device=DEV) < 0.1, torch.randint(0, Mi.M, (n,), device=DEV), act)
        else:
            act = Mi.choose(x); q["src"] += np.bincount(Mi.src.cpu().numpy(), minlength=5)
        if t >= beats - keep_last:
            X.append(x.cpu()); Pz.append(pos); St.append(s0.cpu().numpy()); Lf.append(lifeid.copy()); A.append(act.cpu().numpy().copy())
        alive0 = ~B.dead.clone(); f = B.step(act)
        for c in (0, 1):
            m = f[:, c].bool(); q["fills"][c] += int(m.sum()); q["need_fills"][c] += int((m & (s0[:, c] < s0[:, 1 - c])).sum())
        died = alive0 & B.dead; x2 = O(B.raw()); Mi.after(x, act, x2, died, B.dead & ~died); x = x2; q["deaths"] += int(died.sum())
        if t < child:                                                     # protected: the stores do not drain (what a fill gave is felt, then held at .75)
            B.w.store[:] = 0.75; B.w.dead[:] = False
        if (t + 1) % 1000 == 0:                                           # sleep
            q["skill_loss"].append(round(G.learn() or 0.0, 3))
            if t >= child:
                with torch.enable_grad():
                    Mi.wo.learn(n=max(1, 2500 // max(1, getattr(Mi.wo, 'per_call', 1))))
        if B.dead.any():
            dd = B.dead.clone(); idx = torch.nonzero(dd)[:, 0]; B.w.dead[idx] = False; B.w.store[idx] = 0.75; place(B, info, which=idx); O.reset(dd); Mi.begin(dd); B.felt[:] = 0; x = O(B.raw()); lifeid[idx.cpu().numpy()] += 1   # L1d: the re-read after rebirth feels no motion (the drift bug); HC1: the life counter sat inside this comment and never ran
    for q in Q:
        q["src"] = q["src"].tolist(); q["time_in_stem"] = q.pop("stem") / max(1, q.pop("beats"))
    ins_ = Mi.o.inside; ins = list(range(x.shape[1]))[ins_] if isinstance(ins_, slice) else [int(v) for v in ins_]
    return Mi, O, Q, dict(X=torch.stack(X), P=np.stack(Pz), St=np.stack(St), Lf=np.stack(Lf), ins=ins, acts=np.stack(A), M=int(Mi.M))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("lays", nargs="+"); ap.add_argument("--W", type=float, default=1.0); ap.add_argument("--tag", default=""); ap.add_argument("--beats", type=int, default=12000)
    a = ap.parse_args(); V2["on"] = True; RoomBody.default_cues = True; res = []; t0 = time.time(); d = f"{SAVE}{a.tag}/W{a.W}"; os.makedirs(d, exist_ok=True)
    for lay in a.lays:
        seed, mirror = map(int, lay.split("_")); assert not 7701 <= seed <= 7712; rows, info = t_room(seed, bool(mirror)); sd = seed * 10 + mirror
        child = min(3000, a.beats // 4); Mi, O, Q, z = life(rows, info, sd, a.W, beats=a.beats, child=child, babble=child // 3, keep_last=min(3000, a.beats // 4))
        torch.save(z, f"{d}/{lay}.pt"); torch.save(Mi.going.state(), f"{d}/{lay}_going.pt"); Mi.save(f"{d}/{lay}_mind.pt")
        pc = D25.pieces(z["P"], z["Lf"], info, rows); n = z["Lf"].shape[1]; fk = forks(Mi, O, rows, info, sd, paths_of(rows, info))
        rec = dict(layout=lay, W=a.W, quarters=Q, last_quarter=dict(deaths_per_body=Q[3]["deaths"] / n, need_fills_per_body=sum(Q[3]["need_fills"]) / n), travel_last=pc, forks=fk)
        print(lay, "W", a.W, "life", [(q["need_fills"], q["fills"], q["deaths"], round(q["time_in_stem"], 2), q["skill_loss"]) for q in Q], "| travel", {k: round(v, 2) for k, v in pc.items()},
              "| forks", fk, "| last quarter by", [round(v / max(1, sum(Q[3]["src"])), 2) for v in Q[3]["src"]], round(time.time() - t0), flush=True)
        res.append(rec); json.dump(dict(protocol="atanor_os/P1_PATH_PROTOCOL.md (L1)", rows=res), open(a.out, "w"), indent=1); del Mi; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
