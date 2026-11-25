"""N1 (atanor_os/N1_NURTURE_PROTOCOL.md, registered before running): a nurtured childhood in the room, after the human way.
In arms -> walking after (weaning) -> leaving -> alone. The caregiver is the world's (judge-driven), seen by the young's own eye; nothing is added to the
mind's choosing. Control: a caregiver that stays where life starts and feeds as much.
  PYTHONPATH=. python -m tools.n1_nurture OUT.json 9101_0 ... [--arm nurture|still] [--beats 12000] [--tag X]"""
import argparse, json, math, os, time
import numpy as np, torch
from tools.need_fork import t_room, place, DEV, DEPTH, BEING
from tools.a1_channel import V2
from tools import d25_grain as D25
from atanor_os.emerge.room_body import RoomBody
from atanor_os.emerge.body_contract import Organs, Channel
from atanor_os.emerge.mind import Mind

SAVE = "n1"
FAR = -100.0                                                           # a caregiver that is not there is drawn nowhere


class Carer:
    """The caregivers (one per young), in their own copy of the room: each goes to the spring of its lower inside, drinks or eats, goes on."""

    def __init__(self, rows, info, n, seed):
        self.B = RoomBody(n, rows, drain=V2["drain"], seed=seed + 1, device=DEV, regrow=V2["regrow"]); self.n = n; self.rng = np.random.default_rng(seed + 3)
        B = self.B; legs, neck, hand = B.legs.cpu().numpy(), B.neck.cpu().numpy(), B.hand.cpu().numpy()
        pick = lambda lg, nk, hd: int(np.nonzero((legs == lg) & (neck == nk) & (hand == hd))[0][0])
        self.FWD, self.LEFT, self.RIGHT, self.DO, self.STILL = pick(1, 0, 0), pick(0, 1, 0), pick(0, 2, 0), pick(0, 0, 2), pick(0, 0, 0)
        self.F = {c: D25.dist_map(rows, c) for c in "wf"}; B.w.store[:] = 0.75; place(B, info)

    def step(self, wait):
        """wait [n] bool: stand still this beat (the young is too far behind). Returns the fills [n, 2] of this beat."""
        B = self.B; pos = B.w.pos.cpu().numpy(); head = B.w.head.cpu().numpy(); st = B.w.store.cpu().numpy(); act = np.full(self.n, self.STILL)
        for i in range(self.n):
            if wait[i]:
                continue
            F = self.F["w" if st[i, 0] < st[i, 1] else "f"]; cy, cx = int(pos[i, 1]), int(pos[i, 0])
            if F[cy, cx] <= 1:
                act[i] = self.DO; continue
            best = min(((F[cy + dy, cx + dx], dy, dx) for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)) if F[cy + dy, cx + dx] >= 0), default=None)
            if best is None or self.rng.random() < 0.05:
                act[i] = self.rng.integers(len(B.legs)); continue
            want = math.atan2(best[1] + (cy + 0.5 - pos[i, 1]) * 0.5, best[2] + (cx + 0.5 - pos[i, 0]) * 0.5); d = (want - head[i] + math.pi) % (2 * math.pi) - math.pi
            act[i] = self.FWD if abs(d) < math.radians(20) else (self.RIGHT if d > 0 else self.LEFT)
        f = B.step(torch.as_tensor(act, device=DEV))
        if B.dead.any():
            dd = B.dead.clone(); idx = torch.nonzero(dd)[:, 0]; B.w.dead[idx] = False; B.w.store[idx] = 0.75
        return f


def stage_of(t, beats):
    q = t / beats
    return "arms" if q < 1 / 8 else "after" if q < 1 / 2 else "leaving" if q < 3 / 4 else "alone"


def life(rows, info, seed, arm, path=False, n=64, beats=12000, keep_last=3000, sleep_every=3000, sleep_updates=2500):
    torch.manual_seed(seed); rng = np.random.default_rng(seed + 9)
    B = RoomBody(n, rows, drain=V2["drain"], seed=seed, device=DEV, regrow=V2["regrow"])
    if path:                                                             # P1: the body declares its own felt motion; the being's innate path organ reads it
        B.c.sensors[2] = Channel("vector", (3,), motion=True)
    O = Organs(B.c, n, dev=DEV, depth=DEPTH); Mi = Mind(O, B.c, n, dev=DEV, value_arm="relief")
    grown = Mi.grow(BEING) if path else Mi.load(BEING)
    Mi.wo.on = True; C = Carer(rows, info, n, seed)
    B.w.store[:] = 0.75; place(B, info); start = B.w.pos.clone(); start_h = B.w.head.clone()
    every = torch.ones(n, dtype=torch.bool, device=DEV); O.reset(every); Mi.begin(every)
    per = beats // 4; Q = [dict(fills=[0, 0], need_fills=[0, 0], fed=[0.0, 0.0], deaths=0, src=np.zeros(5, int), near=0, seen=0, beats=0) for _ in range(4)]
    X, Pz, St, Lf, A = [], [], [], [], []; lifeid = np.zeros(n, int); x = None
    for t in range(beats):
        stg = stage_of(t, beats); q = Q[min(t // per, 3)]; tq = t / beats
        here = 0.0 if arm == "alone" else {"arms": 1.0, "after": 1.0, "leaving": 1.0 - (tq - 0.5) / 0.25, "alone": 0.0}[stg]
        show = torch.as_tensor(rng.random(n) < here, device=DEV)
        kin = C.B.w.pos.clone() if arm == "nurture" else start.clone()
        if stg == "arms":
            kin[:] = FAR                                                 # in arms the young does not see the one who carries it, it sees what is carried past
        if x is None:
            B.kin = torch.where(show[:, None], kin, torch.full_like(kin, FAR)); C.prev_cs = C.B.w.store.clone(); x = O(B.raw())
        s0 = B.w.store.clone(); pos = B.w.pos.cpu().numpy().copy()
        act = Mi.choose(x); q["src"] += np.bincount(Mi.src.cpu().numpy(), minlength=5)
        if t >= beats - keep_last:
            X.append(x.cpu()); Pz.append(pos); St.append(s0.cpu().numpy()); Lf.append(lifeid.copy()); A.append(act.cpu().numpy().copy())
        gap = (B.w.pos - kin).norm(dim=1)
        if stg in ("after", "leaving"):
            q["near"] += int(((gap < 3) & show).sum()); q["seen"] += int(show.sum())
        wait = ((gap > 3) & (stg == "after")).cpu().numpy() if arm == "nurture" else np.zeros(n, bool)   # control: the caregiver lives and feeds as much, unseen; the young sees one standing where life starts
        cf = C.step(wait)                                                # the caregiver's own fills this beat (it lives either way)
        alive0 = ~B.dead.clone()
        if stg == "arms" and arm != "alone":                             # carried: at the caregiver's place and heading (control: held where life starts)
            pre, preh = B.w.pos.clone(), B.w.head.clone()
            if arm == "nurture":
                B.w.pos[:] = C.B.w.pos; B.w.head[:] = C.B.w.head
            else:
                B.w.pos[:] = start; B.w.head[:] = start_h
            f = B.step(torch.full_like(act, C.STILL))                    # the young's own gestures move nothing while carried
            dp = B.w.pos - pre; c_, s_ = torch.cos(preh), torch.sin(preh)        # being carried is felt (the inner ear feels it), as the body's own motion
            B.felt = torch.stack([dp[:, 0] * c_ + dp[:, 1] * s_, -dp[:, 0] * s_ + dp[:, 1] * c_, ((B.w.head - preh + math.pi) % (2 * math.pi)) - math.pi], 1)
        else:
            f = B.step(act)
        feed = 0.0 if arm == "alone" else {"arms": 1.0, "after": 1.0 - (tq - 0.125) / 0.375, "leaving": 0.0, "alone": 0.0}[stg]
        if feed > 0:                                                     # fed: what the caregiver took, the young gets (weaned over 'after')
            got = (C.B.w.store - C.prev_cs).clamp(min=0) * feed
            B.w.store[:] = torch.where(alive0[:, None], (B.w.store + got).clamp(max=1.0), B.w.store)
            for c in (0, 1):
                q["fed"][c] += float(got[:, c].sum())
        C.prev_cs = C.B.w.store.clone()
        for c in (0, 1):
            m = f[:, c].bool(); q["fills"][c] += int(m.sum()); q["need_fills"][c] += int((m & (s0[:, c] < s0[:, 1 - c])).sum())
        q["beats"] += n
        kin = C.B.w.pos.clone() if arm == "nurture" else start.clone()
        if stg == "arms":
            kin[:] = FAR
        B.kin = torch.where(show[:, None], kin, torch.full_like(kin, FAR))
        died = alive0 & B.dead; x2 = O(B.raw()); Mi.after(x, act, x2, died, B.dead & ~died); x = x2; q["deaths"] += int(died.sum())
        if sleep_every and (t + 1) % sleep_every == 0:
            with torch.enable_grad():
                Mi.wo.learn(n=max(1, sleep_updates // max(1, getattr(Mi.wo, 'per_call', 1))))
        if B.dead.any():                                                 # born again: beside the caregiver while it is there, where life starts when alone
            dd = B.dead.clone(); idx = torch.nonzero(dd)[:, 0]; B.w.dead[idx] = False; B.w.store[idx] = 0.75; place(B, info, which=idx)
            if stg != "alone" and arm == "nurture":
                B.w.pos[idx] = C.B.w.pos[idx]
            O.reset(dd); Mi.begin(dd); B.felt[:] = 0; x = O(B.raw()); lifeid[idx.cpu().numpy()] += 1   # felt zeroed: the drift fix (P1_PATH_PROTOCOL L1d)
    for q in Q:
        q["src"] = q["src"].tolist(); q["near_share"] = q.pop("near") / max(1, q.pop("seen")); q.pop("beats")
    ins_ = Mi.o.inside; ins = list(range(x.shape[1]))[ins_] if isinstance(ins_, slice) else [int(v) for v in ins_]
    return Mi, Q, dict(X=torch.stack(X), P=np.stack(Pz), St=np.stack(St), Lf=np.stack(Lf), ins=ins, acts=np.stack(A), M=int(Mi.M))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("lays", nargs="+"); ap.add_argument("--arm", default="nurture"); ap.add_argument("--beats", type=int, default=12000); ap.add_argument("--tag", default=""); ap.add_argument("--path", action="store_true")
    a = ap.parse_args(); V2["on"] = True; RoomBody.default_cues = True; res = []; t0 = time.time(); d = f"{SAVE}{a.tag}/{a.arm}{'_path' if a.path else ''}"; os.makedirs(d, exist_ok=True)
    for lay in a.lays:
        seed, mirror = map(int, lay.split("_")); assert not 7701 <= seed <= 7712; rows, info = t_room(seed, bool(mirror)); sd = seed * 10 + mirror
        Mi, Q, z = life(rows, info, sd, a.arm, path=a.path, beats=a.beats, keep_last=min(3000, a.beats // 4)); torch.save(z, f"{d}/{lay}.pt"); Mi.save(f"{d}/{lay}_mind.pt")
        pc = D25.pieces(z["P"], z["Lf"], info, rows); n = z["Lf"].shape[1]
        rec = dict(layout=lay, arm=a.arm, path=a.path, quarters=Q, alone=dict(deaths_per_body=Q[3]["deaths"] / n, need_fills_per_body=sum(Q[3]["need_fills"]) / n), travel_alone=pc)
        print(lay, a.arm, "path" if a.path else "", "life", [(q["need_fills"], q["fills"], [round(v) for v in q["fed"]], q["deaths"], round(q["near_share"], 2)) for q in Q], "| travel alone", {k: round(v, 2) for k, v in pc.items()}, round(time.time() - t0), flush=True)
        res.append(rec); json.dump(dict(protocol="atanor_os/N1_NURTURE_PROTOCOL.md", rows=res), open(a.out, "w"), indent=1); del Mi; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
