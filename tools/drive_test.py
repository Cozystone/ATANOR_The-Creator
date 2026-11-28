"""DR1 body-driving tests (founder 2026-11-28: "is it not intelligent, or can it not drive its body?"). Judge side only; nothing is added
to the being. An open 16 x 16 room, food in the middle, water far in a corner. A trial = a birth 2.5 cells from the food, lacking food
(food inside .4, water .9), the food at a set bearing from where it faces (0, +-45, +-90, +-135, 180 degrees); the trial ends at the
first food fill or after `cap` beats. Finding and remembering are not needed - only turning to it, stepping up and using the hand.
  test 1 (does it know its body): each beat, its own forecaster's felt motion for the gesture it made against what its body then felt;
          and over all its gestures, whether the foreseen turn / step has the right sign.
  test 2 (can it drive): arms  own    - the being as it is
                               told   - the judge gives the true direction to the food as its homing content, at full weight (never a pass)
                               script - the judge drives the same body (turn until aimed, step until in reach, hand): the body's best
  PYTHONPATH=. python -m tools.drive_test OUT.json --being s3_e --eye 1.1 --arm own"""
import argparse, json, math, time
import numpy as np, torch
from tools.need_fork import DEV
from tools.a1_channel import V2
from tools.l1_life import make
from tools.s4_kin import being
from atanor_os.emerge.room_body import RoomBody
from atanor_os.emerge.room3d_gpu import FOOD

S = 16
BEAR = [0, 45, -45, 90, -90, 135, -135, 180]


def layout():
    g = np.full((S, S), "#"); g[1:15, 1:15] = "."; g[8, 8] = "f"; g[1, 1] = "w"
    return ["".join(r) for r in g]


def wrap(a):
    return torch.remainder(a + math.pi, 2 * math.pi) - math.pi


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--being", required=True); ap.add_argument("--eye", type=float, default=0.0)
    ap.add_argument("--arm", default="own", choices=["own", "told", "script"]); ap.add_argument("--frame", type=int, default=1); ap.add_argument("--rounds", type=int, default=12); ap.add_argument("--cap", type=int, default=120)
    ap.add_argument("--warm", type=int, default=100); ap.add_argument("--bodies", type=int, default=64); ap.add_argument("--seed", type=int, default=9701)
    a = ap.parse_args(); V2["on"] = True; assert not 7701 <= a.seed <= 7712
    if a.eye:
        RoomBody.default_eye = a.eye
    RoomBody.default_cues = True; t0 = time.time(); n = a.bodies; torch.manual_seed(a.seed); rng = np.random.RandomState(a.seed)
    B = make(layout(), n, a.seed, V2["drain"]); O, Mi = being(B, n, a.being); w = B.w; G_ = Mi.going
    every = torch.ones(n, dtype=torch.bool, device=DEV); fc = w.src[:, 1].float() + 0.5
    legs, neck, hand = B.legs, B.neck, B.hand; idx = {tuple(c): i for i, c in enumerate(B.combos)}
    A_L, A_R, A_F, A_DO = idx[(0, 1, 0)], idx[(0, 2, 0)], idx[(1, 0, 0)], idx[(0, 0, 2)]
    bear_of = torch.tensor([math.radians(BEAR[i % len(BEAR)]) for i in range(n)], device=DEV)

    def birth():
        phi = torch.tensor(rng.uniform(-math.pi, math.pi, n), device=DEV, dtype=torch.float32)
        w.pos[:, 0] = fc[:, 0] + 2.5 * torch.cos(phi); w.pos[:, 1] = fc[:, 1] + 2.5 * torch.sin(phi)
        to_food = torch.atan2(fc[:, 1] - w.pos[:, 1], fc[:, 0] - w.pos[:, 0]); w.head[:] = torch.remainder(to_food - bear_of, 2 * math.pi)   # the food stands at `bear_of` from where it faces
        w.dead[:] = False; w.store[:, 0] = 0.9; w.store[:, 1] = 0.4; B.felt[:] = 0
        O.reset(every)
        if a.frame:                                                         # HC1 (2026-11-28 audit): the judge put the body down, so its path organ is told where (one fixed frame: the food, heading -90 deg) -
            hb = -math.pi / 2; d_ = w.pos - fc; c_, s_ = math.cos(hb), math.sin(hb)   # without this every round's zeroed path organ meant a different world spot for the same place unit
            O.path.p = torch.stack([d_[:, 0] * c_ + d_[:, 1] * s_, -d_[:, 0] * s_ + d_[:, 1] * c_], 1); O.path.h = (w.head - hb + math.pi) % (2 * math.pi) - math.pi
        Mi.begin(every); Mi.prev = None
        return O(B.raw())

    def bearing():
        v = fc - w.pos; return wrap(torch.atan2(v[:, 1], v[:, 0]) - w.head), v.norm(dim=1)

    if a.arm == "told":
        OC = [0]

        def oracle(c):
            OC[0] += 1; tgt = torch.where((c == 1)[:, None], fc, w.src[:, 0].float() + 0.5); v = tgt - w.pos; co, si = torch.cos(w.head), torch.sin(w.head)
            return v[:, 0] * co + v[:, 1] * si, -v[:, 0] * si + v[:, 1] * co
        G_.oracle = oracle
    place_ = lambda: None
    x = birth()
    for t in range(a.warm):                                              # a short life here first (its beat count passes 50; nothing is scored)
        act = Mi.choose(x); al = ~B.dead.clone(); B.step(act); x2 = O(B.raw()); died = al & B.dead; Mi.after(x, act, x2, died, B.dead & ~died); x = x2
        w.store[:] = w.store.clamp(min=0.3)
    E = dict(f=[], s=[], t=[], cls=dict(turn=[], step=[], still=[])); SG = [0, 0, 0, 0]; trials = []
    for r in range(a.rounds):
        x = birth(); done = torch.zeros(n, dtype=torch.bool, device=DEV); t_fill = torch.full((n,), -1, device=DEV); t_aim = torch.full((n,), -1, device=DEV); t_face = torch.full((n,), -1, device=DEV)
        turn_to = torch.zeros(n, device=DEV); turn_away = torch.zeros(n, device=DEV); hand_n = torch.zeros(n, device=DEV); still_n = torch.zeros(n, device=DEV)
        for t in range(a.cap):
            b0, d0 = bearing()
            if a.arm == "script":
                k0, dd0 = w.cast(w.head[:, None], step=0.02, far=3.0); inreach = (k0[:, 0] == FOOD) & (dd0[:, 0] <= 1.2)
                act = torch.where(inreach, torch.full((n,), A_DO, device=DEV), torch.where(b0.abs() > math.radians(15), torch.where(b0 > 0, torch.full((n,), A_R, device=DEV), torch.full((n,), A_L, device=DEV)), torch.full((n,), A_F, device=DEV)))
                Mi.choose(x)                                             # the being still feels and learns; the judge's hand moves the body
            else:
                act = Mi.choose(x)
            if a.arm != "script" and getattr(G_, "net", None) is not None and r >= 1:
                fo_all = G_._foreseen(x)                                  # [n, M, 3] forward, side, turn - its own forecast for every gesture
                fo = fo_all.gather(1, act[:, None, None].expand(-1, 1, 3))[:, 0]
                if t % 10 == 0:                                          # over all gestures: is the sign of the foreseen turn / step right?
                    tn = torch.where(neck == 1, -1.0, torch.where(neck == 2, 1.0, 0.0)); fw = torch.where((legs == 1) | (legs == 5), 1.0, torch.where(legs == 2, -1.0, 0.0))
                    m_t = tn != 0; m_f = fw != 0
                    SG[0] += int((torch.sign(fo_all[:, m_t, 2]) == tn[m_t][None]).sum()); SG[1] += int(m_t.sum()) * n
                    SG[2] += int((torch.sign(fo_all[:, m_f, 0]) == fw[m_f][None]).sum()); SG[3] += int(m_f.sum()) * n
            else:
                fo = None
            al = ~B.dead.clone(); f = B.step(act); x2 = O(B.raw()); died = al & B.dead
            if fo is not None:
                live = ~done & ~died; e = (fo - B.felt).abs()
                E["f"] += e[live, 0].tolist()[:8]; E["s"] += e[live, 1].tolist()[:8]; E["t"] += e[live, 2].tolist()[:8]
                isturn = (neck[act] == 1) | (neck[act] == 2); isstep = legs[act] != 0
                for nm, m in (("turn", isturn & ~isstep), ("step", isstep & ~isturn), ("still", ~isturn & ~isstep)):
                    m = m & live
                    if bool(m.any()):
                        E["cls"][nm].append([float(fo[m, 0].mean()), float(B.felt[m, 0].mean()), float(fo[m, 2].abs().mean()), float(B.felt[m, 2].abs().mean()), int(m.sum())])
            b1, d1 = bearing(); live = ~done
            tsgn = torch.where(neck[act] == 1, -1.0, torch.where(neck[act] == 2, 1.0, 0.0)); off = b0.abs() > math.radians(15)
            turn_to += (live & off & (tsgn * torch.sign(b0) > 0)).float(); turn_away += (live & off & (tsgn * torch.sign(b0) < 0)).float()
            hand_n += (live & (hand[act] == 2)).float(); still_n += (live & (legs[act] == 0) & (tsgn == 0) & (hand[act] != 2)).float()
            k1, dd1 = w.cast(w.head[:, None], step=0.02, far=3.0); face = (k1[:, 0] == FOOD) & (dd1[:, 0] <= 1.2)
            t_aim = torch.where(live & (t_aim < 0) & (b1.abs() <= math.radians(15)), torch.full_like(t_aim, t + 1), t_aim)
            t_face = torch.where(live & (t_face < 0) & (face | f[:, 1].bool()), torch.full_like(t_face, t + 1), t_face)
            got = live & f[:, 1].bool(); t_fill = torch.where(got, torch.full_like(t_fill, t + 1), t_fill); done |= got | died
            Mi.after(x, act, x2, died, B.dead & ~died); x = x2
            w.store[:, 0] = w.store[:, 0].clamp(min=0.5)                # the trial is about food; water is kept from running out
            if bool(done.all()):
                break
        for i in range(n):
            trials.append(dict(bear=BEAR[i % len(BEAR)], fill=float(t_fill[i]), aim=float(t_aim[i]), face=float(t_face[i]), turn_to=float(turn_to[i]), turn_away=float(turn_away[i]), hand=float(hand_n[i]), still=float(still_n[i])))
    md = lambda v: float(np.median(v)) if len(v) else None
    by = {}
    third = len(trials) // 3; thirds = [round(float(np.mean([q["fill"] > 0 for q in trials[i * third:(i + 1) * third]])), 3) for i in range(3)]
    for b_ in sorted(set(abs(x_) for x_ in BEAR)):
        T = [q for q in trials if abs(q["bear"]) == b_]; fl = [q["fill"] for q in T if q["fill"] > 0]
        by[str(b_)] = dict(trials=len(T), filled_share=round(len(fl) / max(len(T), 1), 3), fill_beats_median=md(fl), aimed_share=round(float(np.mean([q["aim"] > 0 for q in T])), 3), aim_beats_median=md([q["aim"] for q in T if q["aim"] > 0]),
                           faced_in_reach_share=round(float(np.mean([q["face"] > 0 for q in T])), 3), turns_toward_mean=round(float(np.mean([q["turn_to"] for q in T])), 2), turns_away_mean=round(float(np.mean([q["turn_away"] for q in T])), 2),
                           hand_beats_mean=round(float(np.mean([q["hand"] for q in T])), 2), still_beats_mean=round(float(np.mean([q["still"] for q in T])), 2))
    fl = [q["fill"] for q in trials if q["fill"] > 0]
    cls = {nm: (dict(foreseen_forward=round(float(np.average([c[0] for c in v], weights=[c[4] for c in v])), 3), felt_forward=round(float(np.average([c[1] for c in v], weights=[c[4] for c in v])), 3),
                     foreseen_turn_abs=round(float(np.average([c[2] for c in v], weights=[c[4] for c in v])), 3), felt_turn_abs=round(float(np.average([c[3] for c in v], weights=[c[4] for c in v])), 3), beats=int(sum(c[4] for c in v))) if v else None)
           for nm, v in E["cls"].items()}
    res = dict(protocol="atanor_os/WORLD_A_PROTOCOL.md DR1", arm=a.arm, being=a.being, bodies=n, rounds=a.rounds, cap=a.cap, seed=a.seed,
               all=dict(trials=len(trials), filled_share=round(len(fl) / max(len(trials), 1), 3), fill_beats_median=md(fl), filled_share_by_third=thirds), by_bearing=by,
               body_knowledge=(dict(abs_error_median=dict(forward=md(E["f"]), side=md(E["s"]), turn=md(E["t"])), abs_error_mean=dict(forward=round(float(np.mean(E["f"])), 3), side=round(float(np.mean(E["s"])), 3), turn=round(float(np.mean(E["t"])), 3)),
                                    by_gesture_kind=cls, sign_right_share=dict(turn=round(SG[0] / max(SG[1], 1), 3), step=round(SG[2] / max(SG[3], 1), 3))) if E["f"] else None),
               oracle_calls=(OC[0] if a.arm == "told" else None), seconds=round(time.time() - t0))
    print(json.dumps(res), flush=True); json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
