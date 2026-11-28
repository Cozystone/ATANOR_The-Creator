"""S5 spatial tests (founder 2026-11-27: "is it good at space? build the shortcut and detour tests now"). Judge side only; nothing is added
to the being. World: a 16 x 16 room with a central block; food in the middle of the top side, water in the middle of the bottom side, the
birth point below the block. Two ways around the block (left, right). After `--learn` beats of living, the world changes for every body
(and for its rebirths):
  shortcut - a corridor opens straight through the block (Tolman): is the new short way taken, and how soon?
  detour   - the side the bodies used more to reach food is walled off: do they go round the other side, or keep pushing / circling?
Read-outs per phase: deaths, hungry -> food waits, which way food journeys came (left / right / through the gap), and for the shortcut
the beats until the first food journey through the gap.
  PYTHONPATH=. python -m tools.s5_space OUT.json --being s3_e --eye 1.1 --test shortcut"""
import argparse, json, time
import numpy as np, torch
from tools.need_fork import place, DEV
from tools.a1_channel import V2
from tools.l1_life import make
from tools.s4_kin import being, rebirth
from atanor_os.emerge.room_body import RoomBody
from atanor_os.emerge.room3d_gpu import FLOOR, WALL

S = 16
GAP = [(7, y) for y in range(4, 12)] + [(8, y) for y in range(4, 12)]      # (x, y) cells of the shortcut through the block


def layout():
    g = np.full((S, S), "#")
    g[1:15, 1:15] = "."
    g[4:12, 3:13] = "#"                                                    # the central block (rows 4-11, columns 3-12)
    g[1, 7] = "f"; g[14, 7] = "w"
    rows = ["".join(r) for r in g]
    return rows, dict(x0=8.0, ys=13.0, layout="s5", mirror=False)


def layout_strong():
    """Strong detour (Astra's failing case; Copilot 2026-11-27): the body is born inside a pocket that opens only away from the food.
    Food straight ahead beyond the pocket's wall; the only way is first away from it (down, out of the pocket), then round a side."""
    g = np.full((S, S), "#")
    g[1:15, 1:15] = "."
    g[6, 5:11] = "#"; g[6:11, 5] = "#"; g[6:11, 10] = "#"                  # the pocket: top wall and two arms, open at the bottom (y = 11)
    g[1, 7] = "f"; g[14, 7] = "w"
    return ["".join(r) for r in g], dict(x0=8.0, ys=9.0, layout="s5strong", mirror=False)


def side(x, y):
    """which way a body is going round: left of the block, right of it, or inside the gap"""
    if 3 <= y <= 12 and x < 3:
        return 0
    if 3 <= y <= 12 and x >= 13:
        return 1
    if 4 <= y <= 11 and 7 <= x <= 8:
        return 2
    return -1


def set_cells(w, cells, kind):
    for (x, y) in cells:
        w.map[:, x, y] = kind; w.base[x, y] = kind


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--being", required=True); ap.add_argument("--eye", type=float, default=0.0)
    ap.add_argument("--test", choices=["shortcut", "detour", "strong"], required=True); ap.add_argument("--learn", type=int, default=3000); ap.add_argument("--after", type=int, default=2000)
    ap.add_argument("--chain", type=int, default=0)
    ap.add_argument("--cond", default="c", choices=["a", "b", "c"]); ap.add_argument("--seed", type=int, default=9501); ap.add_argument("--bodies", type=int, default=64)
    a = ap.parse_args(); V2["on"] = True
    if a.eye:
        RoomBody.default_eye = a.eye
    RoomBody.default_cues = True; t0 = time.time(); n = a.bodies
    assert not 7701 <= a.seed <= 7712
    rows, info = (layout_strong() if a.test == "strong" else layout()); torch.manual_seed(a.seed)
    pocket_lack = [0, 0]; exits = []; born_t = torch.zeros(n, dtype=torch.long, device=DEV); out_ = torch.zeros(n, dtype=torch.bool, device=DEV)
    B = make(rows, n, a.seed, V2["drain"]); O, Mi = being(B, n, a.being); w = B.w
    every = torch.ones(n, dtype=torch.bool, device=DEV); B.w.store[:] = 0.75; place(B, info); O.reset(every); Mi.begin(every); Mi.prev = None
    x = O(B.raw())
    ph = lambda: dict(deaths=0, waits=[], ways=[0, 0, 0], beats=0)
    P = [ph(), ph()]; since = torch.full((n,), -1, dtype=torch.long, device=DEV); last_side = torch.full((n,), -1, dtype=torch.long, device=DEV)
    first_gap = None; blocked = None
    CH = None                                                              # PK3b-chain: one meeting per body, first beat of each stage after the food moves
    for t in range(a.learn + a.after):
        k = 0 if t < a.learn else 1
        if a.test == "strong" and a.cond != "c" and t in (0, a.learn):      # Astra's three conditions for the pocket room (2026-11-28)
            from atanor_os.emerge.room3d_gpu import FOOD
            if a.cond == "a":                                              # a: it has eaten at that very place - first through a gap in the pocket's top, then the gap closes
                set_cells(w, [(7, 6), (8, 6)], FLOOR if t == 0 else WALL)
            else:                                                          # b: the place is new, the look is known - first the food stands elsewhere (left, by the pocket's mouth), then at the top
                c_new = (1, 11) if t == 0 else (7, 1); old_ = w.src[:, 1].clone()
                w.map[torch.arange(n, device=DEV), old_[:, 0], old_[:, 1]] = FLOOR; w.base[int(old_[0, 0]), int(old_[0, 1])] = FLOOR
                w.map[:, c_new[0], c_new[1]] = FOOD; w.base[c_new[0], c_new[1]] = FOOD; w.src[:, 1] = torch.tensor(c_new, device=DEV); w.spots[FOOD] = torch.tensor([c_new], device=DEV)
        if t == a.learn and a.test != "strong":                            # the world changes, for every body and its rebirths
            if a.test == "shortcut":
                set_cells(w, GAP, FLOOR)
            else:
                blocked = 0 if P[0]["ways"][0] >= P[0]["ways"][1] else 1  # wall off the side used more for food
                xs = (1, 2) if blocked == 0 else (13, 14)
                set_cells(w, [(xx, yy) for xx in xs for yy in (7, 8)], WALL)
            here = w.pos.long(); inside = w.map[torch.arange(n, device=DEV), here[:, 0].clamp(0, S - 1), here[:, 1].clamp(0, S - 1)] == WALL
            if bool(inside.any()):                                         # a body standing where a wall now is: put it back at the birth point
                place(B, info, torch.nonzero(inside)[:, 0])
        act = Mi.choose(x); al = ~B.dead.clone(); f = B.step(act); x2 = O(B.raw()); died = al & B.dead
        P[k]["deaths"] += int(died.sum()); P[k]["beats"] += 1
        cx, cy = w.pos[:, 0].long().clamp(0, S - 1).tolist(), w.pos[:, 1].long().clamp(0, S - 1).tolist()
        for i in range(n):
            sd = side(cx[i], cy[i])
            if sd >= 0:
                last_side[i] = sd
        start = (w.store[:, 1] < 0.7) & ~died & (since < 0); since = torch.where(start, torch.full_like(since, t), since)
        got = f[:, 1].bool() & (since >= 0)
        for i in torch.nonzero(got)[:, 0].tolist():
            P[k]["waits"].append(t - int(since[i])); s_ = int(last_side[i])
            if s_ >= 0:
                P[k]["ways"][s_] += 1
                if s_ == 2 and first_gap is None:
                    first_gap = t - a.learn
        since = torch.where(got | died, torch.full_like(since, -1), since)
        last_side = torch.where(f[:, 1].bool() | died, torch.full_like(last_side, -1), last_side)
        if a.test == "strong":                                             # judge side: beats in the pocket's top half while lacking food; beats from birth to leaving the pocket
            px, py = w.pos[:, 0], w.pos[:, 1]; inp = (px > 6) & (px < 10) & (py > 7) & (py < 11); hungry = (w.store[:, 1] < 0.7) & ~died
            pocket_lack[0] += int((inp & (py < 9) & hungry).sum()); pocket_lack[1] += int(hungry.sum())
            left = ~inp & ~out_ & ~died
            for i in torch.nonzero(left)[:, 0].tolist():
                exits.append((t, t - int(born_t[i])))
            out_ = out_ | left
            born_t = torch.where(died, torch.full_like(born_t, t + 1), born_t); out_ = out_ & ~died
        Mi.after(x, act, x2, died, B.dead & ~died); x = x2
        if a.chain and t >= a.learn:
            import math
            from atanor_os.emerge.room3d_gpu import FOOD as _FOOD
            G_ = Mi.going
            if CH is None:
                CH = dict(st=torch.full((n, 7), -1, dtype=torch.long, device=DEV), live=torch.ones(n, dtype=torch.bool, device=DEV), dead=torch.full((n,), -1, dtype=torch.long, device=DEV), cerr=[])
                old_food = torch.tensor([1.5, 11.5], device=DEV)
            fc = w.src[:, 1].float() + 0.5; dfood = (w.pos - fc).norm(dim=1)
            cols = B.cols[:: max(1, len(B.cols) // 32)]; kind_, _ = w.cast(w.head[:, None] + cols[None], step=0.05)
            now = [(kind_ == _FOOD).any(1)]
            pw = getattr(G_, "wp_prev", None); hw = getattr(G_, "wh_prev", None)
            def to_world(q):                                               # a point of its own frame, turned into the world by where it is and faces now
                r = q - pw; d_ = w.head - hw; c_, s_ = torch.cos(d_), torch.sin(d_)
                return w.pos + torch.stack([c_ * r[:, 0] - s_ * r[:, 1], s_ * r[:, 0] + c_ * r[:, 1]], 1)
            L_ = getattr(G_, "_lg", None); claim = torch.zeros(n, dtype=torch.bool, device=DEV); cd = None
            if L_ is not None and pw is not None and hw is not None and len(pw) == n:
                cd = (to_world(L_["q"][:, 1]) - fc).norm(dim=1); claim = L_["set"][:, 1] & (cd < 2.0)
            now.append(claim)
            cand = torch.zeros(n, dtype=torch.bool, device=DEV); cold = torch.zeros(n, dtype=torch.bool, device=DEV)
            for nm in ("_t_plan", "_t_lean"):
                tg = getattr(G_, nm, None)
                if tg is not None and pw is not None and hw is not None and len(tg[0]) == n:
                    wq = to_world(tg[0]); cand |= tg[1] & ((wq - fc).norm(dim=1) < 2.0); cold |= tg[1] & ((wq - old_food).norm(dim=1) < 2.0)
            now += [cand, dfood < 2.5, dfood < 1.2, f[:, 1].bool(), cold]
            for j, m in enumerate(now):
                m = m & CH["live"] & (CH["st"][:, j] < 0); CH["st"][:, j] = torch.where(m, torch.full_like(CH["st"][:, j], t - a.learn), CH["st"][:, j])
            fl = f[:, 1].bool() & CH["live"]
            if cd is not None and bool(fl.any()):
                CH["cerr"] += cd[fl & L_["set"][:, 1]].tolist()
            dd = died & CH["live"]; CH["dead"] = torch.where(dd, torch.full_like(CH["dead"], t - a.learn), CH["dead"]); CH["live"] &= ~(dd | fl)
        if B.dead.any():
            rebirth(B, O, Mi, info); B.felt[:] = 0; x = O(B.raw())
        if (t + 1) % 1000 == 0:
            Mi.going.learn()
            if hasattr(Mi, "rest"):
                Mi.rest()                                                      # RL1: the rest - what the waking beats owed is learned now
            with torch.enable_grad():
                Mi.wo.learn(n=max(1, 2500 // max(1, getattr(Mi.wo, "per_call", 1))))
    md = lambda v: float(np.median(v)) if v else None
    chain = None
    if CH is not None:
        st = CH["st"].cpu().numpy(); names = ["sight", "claim", "candidate", "approach", "reach", "fill", "candidate_at_old_place"]
        chain = dict(bodies=n, died_before_fill=int((CH["dead"] >= 0).sum()), open_at_end=int(CH["live"].sum()),
                     each={nm: dict(n=int((st[:, j] >= 0).sum()), median_beat=md(st[st[:, j] >= 0, j].tolist())) for j, nm in enumerate(names)},
                     in_order=[int(np.all(st[:, :j + 1] >= 0, axis=1).sum()) for j in range(6)],
                     skipping_claim=[int(np.all(st[:, [c for c in range(j + 1) if c not in (1, 2)]] >= 0, axis=1).sum()) for j in range(6)],
                     died_last_stage={nm: int(((CH["dead"].cpu().numpy() >= 0) & (np.array([max([j for j in range(6) if st[i, j] >= 0], default=-1) for i in range(n)]) == j)).sum()) for j, nm in list(enumerate(names[:6])) + [(-1, "none")]},
                     claim_to_food_at_fill_median=md(CH["cerr"]))
    r = dict(chain=chain, cond=a.cond, ledger=Mi.going.ledger_report(), self_fix=dict(spot_used=getattr(Mi.going, "sf1_n", 0), spots=int((Mi.going.Rpn > 0).sum()) if getattr(Mi.going, "Rpn", None) is not None else 0, spot_move=round(float(getattr(Mi.going, "rp_move", -1)), 2)), quiet=dict(rel=getattr(Mi, "q_rel", None), checked=getattr(Mi, "q_n", None), turns_not_scored=getattr(Mi, "q_turn", 0)), rely=dict(eye_err=round(float(getattr(Mi.going, "eye_err", -1)), 2), mem_err=round(float(getattr(Mi.going, "mem_err", -1)), 2), eye_led=getattr(Mi.going, "rely_eye", 0), memory_led=getattr(Mi.going, "rely_mem", 0), seen_led=getattr(Mi.going, "sm3_n", 0)), gut=dict(asked=getattr(Mi, "g_asked", 0), known=getattr(Mi, "g_known", 0), differed=getattr(Mi, "g_diff", 0), journeys_ended=getattr(Mi, "g_ends", 0), usual_len=round(float(getattr(Mi, "g_len", 0)), 1)), seen_relief=dict(hit=round(float(getattr(Mi.going, "sr_hit", -1)), 3), samples=getattr(Mi.going, "sr_n", 0), writes=getattr(Mi.going, "sr_writes", 0)), seen_map=dict(err=round(float(getattr(Mi.going, "sm_err", -1)), 3), stops=getattr(Mi.going, "sm_n", 0), writes=getattr(Mi.going, "sm_writes", 0)), e1=getattr(Mi, "e1_n", 0), w0=(dict(beats=getattr(Mi, "w0_n", 0), spread=round(float(getattr(Mi, "w0_spread", 0)), 5), picks=(Mi.w0_pick.tolist() if getattr(Mi, "w0_pick", None) is not None else None))), test=a.test, seed=a.seed, being=a.being, learn=a.learn, after=a.after, blocked_side=blocked, first_gap_food_beats_after=first_gap,
             phases=[dict(deaths=p["deaths"], food_fills=len(p["waits"]), food_wait_median=md(p["waits"]), ways_left_right_gap=p["ways"]) for p in P],
             strong=(dict(share_of_hungry_beats_at_pocket_top=round(pocket_lack[0] / max(pocket_lack[1], 1), 3), exits=len(exits), exit_beats_median_first_half=md([e[1] for e in exits if e[0] < a.learn]), exit_beats_median_second_half=md([e[1] for e in exits if e[0] >= a.learn])) if a.test == "strong" else None), remaps=getattr(Mi.going, "remaps", 0), seconds=round(time.time() - t0))
    print(json.dumps(r), flush=True); json.dump(r, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
