"""LV2 the two-lever device as a world of the room body - and the whole room being in it, unmodified (founder 2026-11-28: "an integrated
being; do not cut organs out for a test - it will leave unused what it does not need"). The same body contract as the room (same eyes, skin,
legs, neck, hand, the same two insides); only the world differs: a closed cell with no springs. The hand's two gestures are the two levers
(hand 1 = the first lever, hand 2 = the second). One lever brings, 8 beats later, a rise (+.30) of whichever inside is lower; the other,
14 beats later, a fall (-.10) of both; the device is busy for 16 beats after a press. Phase 1: the levers are linked (either hand gesture
moves both). Phase 2: apart. Phase 3: the wiring is swapped, nothing looks different. Deaths refill the body; the being goes on.
Compare with tools/lever2.py (the common organ alone: 24 / 24 found the cause).
  PYTHONPATH=. python -m tools.lever_world OUT.json --being s3_e --eye 1.1 [--bodies 16 --view 1]"""
import argparse, json, time
import numpy as np, torch
from tools.need_fork import place, DEV
from tools.a1_channel import V2
from tools.s4_kin import being, rebirth, reread
from atanor_os.emerge.room_body import RoomBody
from atanor_os.emerge.body_contract import Channel

S = 16
RISE_AT, FALL_AT, BUSY, RISE, FALL = 8, 14, 16, 0.30, 0.10


def layout():
    g = np.full((S, S), "#")
    g[7:10, 7:10] = "."                                                  # the cell it lives in (3 x 3), closed
    g[1, 1] = "w"; g[1, 14] = "f"                                        # the room world needs its springs to exist; they are walled off and out of reach
    return ["".join(r) for r in g], dict(x0=8.5, ys=8.5, layout="lever", mirror=False)


class LeverBody(RoomBody):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        n, d = self.n, self.dev
        self.busy = torch.zeros(n, dtype=torch.long, device=d); self.rise_in = torch.full((n,), -1, dtype=torch.long, device=d); self.fall_in = torch.full((n,), -1, dtype=torch.long, device=d)
        self.phase = 0; self.pressed = torch.zeros(n, dtype=torch.long, device=d)       # judge-side: which lever(s) this beat (0 none, 1 first, 2 second, 3 both)

    @torch.no_grad()
    def step(self, act, dt=None):
        f = super().step(act, dt); w = self.w; alive = ~w.dead
        hd = self.hand[act]; press = (hd != 0) & (self.busy == 0) & alive
        good = 1 if self.phase < 2 else 2                                 # which hand gesture is the lever that brings the rise (hidden)
        if self.phase == 0:
            r_, fl_ = press, press; self.pressed = torch.where(press, torch.full_like(hd, 3), torch.zeros_like(hd))
        else:
            r_, fl_ = press & (hd == good), press & (hd != good); self.pressed = torch.where(press, hd, torch.zeros_like(hd))
        self.rise_in = torch.where(r_, torch.full_like(self.rise_in, RISE_AT), self.rise_in); self.fall_in = torch.where(fl_, torch.full_like(self.fall_in, FALL_AT), self.fall_in)
        self.busy = torch.where(press, torch.full_like(self.busy, BUSY), (self.busy - 1).clamp(min=0))
        rise_now = (self.rise_in == 0) & alive; fall_now = (self.fall_in == 0) & alive
        if bool(rise_now.any()):
            low = w.store.argmin(1); idx = torch.nonzero(rise_now)[:, 0]
            w.store[idx, low[idx]] = (w.store[idx, low[idx]] + RISE).clamp(max=1.0); f[idx, low[idx]] = True
        if bool(fall_now.any()):
            w.store[fall_now] = w.store[fall_now] - FALL
        self.rise_in = torch.where(self.rise_in >= 0, self.rise_in - 1, self.rise_in); self.fall_in = torch.where(self.fall_in >= 0, self.fall_in - 1, self.fall_in)
        w.dead |= (w.store <= 0).any(1)
        return f


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out", nargs="?", default=""); ap.add_argument("--being", required=True); ap.add_argument("--eye", type=float, default=0.0)
    ap.add_argument("--bodies", type=int, default=16); ap.add_argument("--t1", type=int, default=1500); ap.add_argument("--t2", type=int, default=5000); ap.add_argument("--t3", type=int, default=9000)
    ap.add_argument("--seed", type=int, default=9601); ap.add_argument("--view", type=int, default=0)
    ap.add_argument("--cut", type=int, default=1); ap.add_argument("--motor", default="", choices=["", "choice", "shuffle", "xshuffle", "anyact"]); ap.add_argument("--motor-src", default="")
    a = ap.parse_args(); V2["on"] = True
    if a.eye:
        RoomBody.default_eye = a.eye
    RoomBody.default_cues = True; t0 = time.time(); n = a.bodies; assert not 7701 <= a.seed <= 7712
    rows, info = layout(); torch.manual_seed(a.seed)
    B = LeverBody(n, rows, drain=1 / 50.0, seed=a.seed, device=DEV, regrow=None); B.c.sensors[2] = Channel("vector", (3,), motion=True)   # .004 a beat, as tools/lever2.py
    O, Mi = being(B, n, a.being); w = B.w
    every = torch.ones(n, dtype=torch.bool, device=DEV); w.store[:] = 0.75; place(B, info); O.reset(every); Mi.begin(every); Mi.prev = None
    x = O(B.raw()); push = None
    probe = None                                                           # MC1: Astra's movement candidate (her class, imported from her checkout; None = the being as it was)
    if a.motor:
        import importlib.util
        sp_ = importlib.util.spec_from_file_location("motor_src", a.motor_src); mm_ = importlib.util.module_from_spec(sp_); sp_.loader.exec_module(mm_)
        probe = mm_.MotorProposal(torch.tensor(B.combos, device=DEV), n); G_ = Mi.going
        defi = lambda x_: 1 - x_[:, [G_.ins[2], G_.ins[1]]].clamp(0, 1)
        Mi.motor_candidate = lambda x_, pick, due, allowed: probe.propose(pick, defi(x_), due, allowed)
        ring = torch.zeros(n, 256, 2, device=DEV); gsh = torch.Generator(); gsh.manual_seed(a.seed + 1); rows_ = torch.arange(n, device=DEV)
    if a.view:
        from tools.lever2 import serve
        push = serve(8780)
    deaths = [0, 0, 0]; picks = [[0, 0, 0, 0] for _ in range(3)]; late = [[0, 0, 0, 0] for _ in range(3)]; src = [[0] * 5 for _ in range(3)]; moved = [0.0, 0.0, 0.0]
    T = [a.t1, a.t2, a.t3]; t = 0
    # SH1 shadow (Astra 714123a; read-only, changes no behaviour): the same lived record given to a second gut that keeps "the movement it
    # actually began, and what its insides did 1-20 beats later" instead of "the place it set out for". Kinds: 1 first hand, 2 second hand, 3 a walk begun.
    SH = dict(hist=torch.zeros(n, 21, dtype=torch.long, device=DEV), acc=torch.zeros(4, 21, device=DEV), cnt=torch.zeros(4, 21, device=DEV), prev_hand=torch.zeros(n, dtype=torch.long, device=DEV),
              prev_leg=torch.zeros(n, dtype=torch.bool, device=DEV), snaps=[])
    while a.view or t < a.t3:
        tt = t % a.t3; ph = 0 if tt < a.t1 else (1 if tt < a.t2 else 2); B.phase = ph
        p0 = w.pos.clone(); act = Mi.choose(x); al = ~B.dead.clone(); f = B.step(act); x2 = O(B.raw()); died = al & B.dead
        deaths[ph] += int(died.sum()); pr = B.pressed
        hd_ = B.hand[act]; lg_ = B.legs[act] != 0; on = torch.where((hd_ != 0) & (hd_ != SH["prev_hand"]), hd_, torch.where(lg_ & ~SH["prev_leg"] & (hd_ == 0), torch.full_like(hd_, 3), torch.zeros_like(hd_)))
        SH["prev_hand"], SH["prev_leg"] = hd_.clone(), lg_.clone(); SH["prev_hand"][died] = 0; SH["prev_leg"][died] = False   # HC1: a new life's first onset was missed
        dl = (w.store.sum(1) - s_prev.sum(1)) if "s_prev" in dir() else torch.zeros(n, device=DEV); s_prev = w.store.clone()
        SH["hist"] = torch.cat([on[:, None], SH["hist"][:, :-1]], 1); SH["hist"][died] = 0
        SH["acc"] *= 0.9995; SH["cnt"] *= 0.9995
        for k_ in (1, 2, 3):
            m_ = (SH["hist"] == k_) & (~died)[:, None]
            if bool(m_.any()):
                SH["acc"][k_] += (m_.float() * dl[:, None]).sum(0); SH["cnt"][k_] += m_.float().sum(0)
        if (t + 1) % 250 == 0:
            net = (SH["acc"] / SH["cnt"].clamp(min=1e-6))[:, 1:].sum(1)
            SH["snaps"].append(dict(t=t + 1, phase=ph, first=round(float(net[1]), 4), second=round(float(net[2]), 4), walk=round(float(net[3]), 4), n=[int(SH["cnt"][k_, 1]) for k_ in (1, 2, 3)]))
        for v in (1, 2, 3):
            c_ = int((pr == v).sum()); picks[ph][v] += c_
            if tt >= T[ph] - (T[ph] - (0 if ph == 0 else T[ph - 1])) // 3:
                late[ph][v] += c_
        picks[ph][0] += int((pr == 0).sum())
        for k in range(5):
            src[ph][k] += int((Mi.src == k).sum())
        moved[ph] += float((w.pos - p0).norm(dim=1).sum())
        if push is not None:
            st = w.store[0].clamp(0, 1).tolist(); pv = int(pr[0]); G = Mi.going
            push(dict(t=t, phase=ph, store=st, act={0: 0, 3: 1, 1: 2, 2: 3}[pv], beat=int(BUSY - B.busy[0]) if int(B.busy[0]) else 99, rise=bool(B.rise_in[0] >= 0 and B.rise_in[0] < 2),
                      fall=bool(B.fall_in[0] >= 0 and B.fall_in[0] < 2), conf=0.0, called=False, deaths=deaths, picks=[[r[0], r[3], r[1], r[2]] for r in picks], good=1 if ph < 2 else 2,
                      first_single=None, recovered=None,
                      gut=[dict(a=0, d=[round(float(getattr(G, "eye_err", -1)), 2), round(float(getattr(G, "mem_err", -1)), 2)], n=int(Mi.src[0]))]))
            time.sleep(0.03)
        if probe is not None:
            dlt = defi(x2) - defi(x)
            if a.motor == "shuffle":                                       # the same body's change from 21-255 beats back: same numbers, no time pairing with what it did
                ring[:, t % 256] = dlt; k_ = torch.randint(21, 256, (n,), generator=gsh).to(DEV); s_ = t - k_
                dlt = torch.where((s_ >= 0)[:, None], ring[rows_, s_ % 256], torch.zeros_like(dlt))
            if a.motor == "xshuffle":                                      # MC1b: another body's change at the same beat
                dlt = dlt.roll(int(torch.randint(1, n, (1,), generator=gsh)), 0)
            if a.motor == "anyact":                                        # MC1c: the organ is told a gesture drawn at random, not the one made; candidates stay the ones really made
                v_ = probe.visited.clone(); probe.record(torch.randint(0, len(B.combos), (n,), generator=gsh).to(DEV), dlt, B.dead.clone()); probe.visited = v_; probe.visited[act] = True
            else:
                probe.record(act, dlt, B.dead.clone())
        Mi.after(x, act, x2, died, B.dead & ~died); x = x2
        if B.dead.any():
            cut_ = B.dead.clone()
            dd_ = rebirth(B, O, Mi, info)
            if a.cut:                                                      # Astra 2cef2a5: rebirth clears the dead flags, so keep the mask first; a reborn body must not inherit the device's state
                B.busy[cut_] = 0; B.rise_in[cut_] = -1; B.fall_in[cut_] = -1
            else:
                B.busy[:] = torch.where(B.dead, torch.zeros_like(B.busy), B.busy)
            x = reread(B, O, x, dd_)
        t += 1
        if t % 1000 == 0:
            Mi.going.learn()
            if hasattr(Mi, "rest"):
                Mi.rest()                                                      # RL1: the rest - what the waking beats owed is learned now
            with torch.enable_grad():
                Mi.wo.learn(n=max(1, 2500 // max(1, getattr(Mi.wo, "per_call", 1))))
    share = lambda r, good: round(r[good] / max(r[1] + r[2], 1), 3)
    r = dict(protocol="atanor_os/WORLD_A_PROTOCOL.md LV2", quiet=dict(rel=getattr(Mi, "q_rel", None), checked=getattr(Mi, "q_n", None), turns_not_scored=getattr(Mi, "q_turn", 0)), being=a.being, bodies=n, seed=a.seed, deaths_by_phase=deaths, presses_by_phase=[dict(none=p[0], first=p[1], second=p[2], both=p[3]) for p in picks],
             causal_share_last_third=[None, share(late[1], 1), share(late[2], 2)], who_chose_by_phase=[dict(pull=s[0], value=s[1], random=s[2], held=s[3], tried=s[4]) for s in src],
             cells_walked_by_phase=[round(m, 1) for m in moved], gut=dict(asked=getattr(Mi, "g_asked", 0), known=getattr(Mi, "g_known", 0), journeys=getattr(Mi, "g_ends", 0)),
             remaps=getattr(Mi.going, "remaps", 0), life_cut=a.cut, motor=(dict(arm=a.motor, replaced=probe.changed, rechoices=probe.opportunities) if probe is not None else None), seconds=round(time.time() - t0), shadow_movement_gut=SH["snaps"], place_gut_actions=(int((Mi.gut.fast_action >= 0).sum()) if getattr(Mi, "gut", None) is not None else None))
    print(json.dumps(r), flush=True)
    if a.out:
        json.dump(r, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
