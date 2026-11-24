"""E1 (atanor_os/E1_ROAM_PROTOCOL.md, registered before running): the V3 recipe living with the place-unfamiliarity pull (Mind.roam).
  PYTHONPATH=. python -m tools.e1_roam OUT.json 9101_0 [--roam 2.0] [--beats 12000]"""
import argparse, json, os, sys, time
import numpy as np, torch
from tools.need_fork import t_room, place, DEV, DEPTH, BEING
from tools.a1_channel import V2
from tools import d25_grain as D25
from atanor_os.emerge.room_body import RoomBody
from atanor_os.emerge.body_contract import Organs
from atanor_os.emerge.mind import Mind

SAVE = "e1"


def life(rows, info, seed, roam, n=64, beats=12000, keep_last=3000, sleep_every=3000, sleep_updates=2500):
    torch.manual_seed(seed)
    B = RoomBody(n, rows, drain=V2["drain"], seed=seed, device=DEV, regrow=V2["regrow"]); O = Organs(B.c, n, dev=DEV, depth=DEPTH)
    Mi = Mind(O, B.c, n, dev=DEV, value_arm="relief"); Mi.load(BEING); Mi.wo.on = True; Mi.roam = roam
    B.w.store[:] = 0.75; place(B, info); every = torch.ones(n, dtype=torch.bool, device=DEV); O.reset(every); Mi.begin(every); x = O(B.raw())
    per = beats // 4; Q = [dict(fills=[0, 0], need_fills=[0, 0], deaths=0, src=np.zeros(5, int), stem=0, beats=0) for _ in range(4)]
    X, Pz, St, Lf, A = [], [], [], [], []; lifeid = np.zeros(n, int)
    for t in range(beats):
        q = Q[min(t // per, 3)]; s0 = B.w.store.clone(); pos = B.w.pos.cpu().numpy().copy()
        q["stem"] += int(((np.abs(pos[:, 0] - info["x0"]) <= 2.5) & (pos[:, 1] >= 3.0)).sum()); q["beats"] += n
        act = Mi.choose(x); q["src"] += np.bincount(Mi.src.cpu().numpy(), minlength=5)
        if t >= beats - keep_last:
            X.append(x.cpu()); Pz.append(pos); St.append(s0.cpu().numpy()); Lf.append(lifeid.copy()); A.append(act.cpu().numpy().copy())
        alive0 = ~B.dead.clone(); f = B.step(act)
        for c in (0, 1):
            m = f[:, c].bool(); q["fills"][c] += int(m.sum()); q["need_fills"][c] += int((m & (s0[:, c] < s0[:, 1 - c])).sum())
        died = alive0 & B.dead; x2 = O(B.raw()); Mi.after(x, act, x2, died, B.dead & ~died); x = x2; q["deaths"] += int(died.sum())
        if sleep_every and (t + 1) % sleep_every == 0:
            with torch.enable_grad():
                Mi.wo.learn(n=max(1, sleep_updates // max(1, getattr(Mi.wo, 'per_call', 1))))
        if B.dead.any():
            dd = B.dead.clone(); idx = torch.nonzero(dd)[:, 0]; B.w.dead[idx] = False; B.w.store[idx] = 0.75; place(B, info, which=idx); O.reset(dd); Mi.begin(dd); x = O(B.raw()); lifeid[idx.cpu().numpy()] += 1
    for q in Q:
        q["src"] = q["src"].tolist(); q["time_in_stem"] = q.pop("stem") / max(1, q.pop("beats"))
    ins_ = Mi.o.inside; ins = list(range(x.shape[1]))[ins_] if isinstance(ins_, slice) else [int(v) for v in ins_]
    return Mi, Q, dict(X=torch.stack(X), P=np.stack(Pz), St=np.stack(St), Lf=np.stack(Lf), ins=ins, acts=np.stack(A), M=int(Mi.M))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("lays", nargs="+"); ap.add_argument("--roam", type=float, default=2.0); ap.add_argument("--beats", type=int, default=12000); ap.add_argument("--tag", default="")
    a = ap.parse_args(); V2["on"] = True; RoomBody.default_cues = True; res = []; t0 = time.time(); d = SAVE + a.tag; os.makedirs(d, exist_ok=True)
    for lay in a.lays:
        seed, mirror = map(int, lay.split("_")); assert not 7701 <= seed <= 7712; rows, info = t_room(seed, bool(mirror)); sd = seed * 10 + mirror
        Mi, Q, z = life(rows, info, sd, a.roam, beats=a.beats, keep_last=min(3000, a.beats)); torch.save(z, f"{d}/{lay}.pt"); Mi.save(f"{d}/{lay}_mind.pt")
        pc = D25.pieces(z["P"], z["Lf"], info, rows); n = z["Lf"].shape[1]
        rec = dict(layout=lay, roam=a.roam, quarters=Q, last_quarter=dict(deaths_per_body=Q[3]["deaths"] / n, need_fills_per_body=sum(Q[3]["need_fills"]) / n, time_in_stem=Q[3]["time_in_stem"]), travel_last_3000=pc)
        print(lay, "roam", a.roam, "life", [(q["need_fills"], q["fills"], q["deaths"], round(q["time_in_stem"], 2)) for q in Q], "| travel", {k: round(v, 2) for k, v in pc.items()}, round(time.time() - t0), flush=True)
        res.append(rec); json.dump(dict(protocol="atanor_os/E1_ROAM_PROTOCOL.md", rows=res), open(a.out, "w"), indent=1); del Mi; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
