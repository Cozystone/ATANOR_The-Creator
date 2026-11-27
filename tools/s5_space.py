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
    ap.add_argument("--seed", type=int, default=9501); ap.add_argument("--bodies", type=int, default=64)
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
    for t in range(a.learn + a.after):
        k = 0 if t < a.learn else 1
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
        if B.dead.any():
            rebirth(B, O, Mi, info); B.felt[:] = 0; x = O(B.raw())
        if (t + 1) % 1000 == 0:
            Mi.going.learn()
            with torch.enable_grad():
                Mi.wo.learn(n=max(1, 2500 // max(1, getattr(Mi.wo, "per_call", 1))))
    md = lambda v: float(np.median(v)) if v else None
    r = dict(rely=dict(eye_err=round(float(getattr(Mi.going, "eye_err", -1)), 2), mem_err=round(float(getattr(Mi.going, "mem_err", -1)), 2), eye_led=getattr(Mi.going, "rely_eye", 0), memory_led=getattr(Mi.going, "rely_mem", 0), seen_led=getattr(Mi.going, "sm3_n", 0)), gut=dict(asked=getattr(Mi, "g_asked", 0), known=getattr(Mi, "g_known", 0), differed=getattr(Mi, "g_diff", 0), journeys_ended=getattr(Mi, "g_ends", 0), usual_len=round(float(getattr(Mi, "g_len", 0)), 1)), seen_relief=dict(hit=round(float(getattr(Mi.going, "sr_hit", -1)), 3), samples=getattr(Mi.going, "sr_n", 0), writes=getattr(Mi.going, "sr_writes", 0)), seen_map=dict(err=round(float(getattr(Mi.going, "sm_err", -1)), 3), stops=getattr(Mi.going, "sm_n", 0), writes=getattr(Mi.going, "sm_writes", 0)), e1=getattr(Mi, "e1_n", 0), w0=(dict(beats=getattr(Mi, "w0_n", 0), spread=round(float(getattr(Mi, "w0_spread", 0)), 5), picks=(Mi.w0_pick.tolist() if getattr(Mi, "w0_pick", None) is not None else None))), test=a.test, seed=a.seed, being=a.being, learn=a.learn, after=a.after, blocked_side=blocked, first_gap_food_beats_after=first_gap,
             phases=[dict(deaths=p["deaths"], food_fills=len(p["waits"]), food_wait_median=md(p["waits"]), ways_left_right_gap=p["ways"]) for p in P],
             strong=(dict(share_of_hungry_beats_at_pocket_top=round(pocket_lack[0] / max(pocket_lack[1], 1), 3), exits=len(exits), exit_beats_median_first_half=md([e[1] for e in exits if e[0] < a.learn]), exit_beats_median_second_half=md([e[1] for e in exits if e[0] >= a.learn])) if a.test == "strong" else None), remaps=getattr(Mi.going, "remaps", 0), seconds=round(time.time() - t0))
    print(json.dumps(r), flush=True); json.dump(r, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
