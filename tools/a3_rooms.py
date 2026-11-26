"""World A3 (atanor_os/WORLD_A_PROTOCOL.md): two rooms joined by a corridor with a dead-end spur on each side; the being is born at the bottom
of a stem below the corridor's middle. Water in a corner of one room, food in a corner of the other; each spring is out of view until the body
is inside its room (the doorway is a two-cell passage up from the corridor). Reaching a spring takes a journey: up the stem, along the corridor
past the spur, up through the doorway, into the room's far corner. Read-out as A2-1 (per-body hunger -> fill waits, deaths) plus wrong turns
(beats spent in the spurs while lacking).
  PYTHONPATH=. python -m tools.a3_rooms OUT.json --being s3_a [--mirror 0 --beats 6000]"""
import argparse, json, time
import numpy as np, torch
from tools.need_fork import place, DEV
from tools.a1_channel import V2
from tools.l1_life import make
from tools.s4_kin import being, rebirth
from atanor_os.emerge.room_body import RoomBody

S = 16


def layout(mirror=False, both=False, seen=False):
    g = np.full((S, S), "#")
    g[1:5, 1:6] = "."; g[1:5, 10:15] = "."                 # two rooms (5 wide, 4 tall)
    g[5:7, 2:4] = "."; g[5:7, 12:14] = "."                 # doorways down from each room
    g[7:9, 1:15] = "."                                     # corridor
    g[9:12, 4:6] = "."; g[9:12, 10:12] = "."               # dead-end spurs below the corridor, between the stem and each doorway
    g[9:14, 7:9] = "."                                     # stem, birth at its bottom
    g[1, 1] = "w"; g[1, 14] = "f"
    if seen:                                               # A3-2: springs in line with the doorways (seen from the corridor looking up), not in the far corners
        g[1, 1] = "."; g[1, 14] = "."; g[1, 2] = "w"; g[1, 13] = "f"
    if both:                                               # A3-1 stage 1: each room also holds the other kind (near corners); a body's spring jumps between them when used
        g[4, 14] = "w"; g[4, 1] = "f"
    if mirror and seen:
        g[1, 2], g[1, 13] = "f", "w"
    elif mirror:
        g[1, 1], g[1, 14] = "f", "w"
        if both:
            g[4, 14], g[4, 1] = "f", "w"
    rows = ["".join(r) for r in g]
    info = dict(x0=8.0, ys=13.5, layout="a3", mirror=bool(mirror))
    spurs = [(x, y) for y in range(9, 12) for x in (4, 5, 10, 11)]
    return rows, info, spurs


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--being", required=True); ap.add_argument("--mirror", type=int, default=0)
    ap.add_argument("--beats", type=int, default=6000); ap.add_argument("--seed", type=int, default=9301); ap.add_argument("--staged", type=int, default=0); ap.add_argument("--seen", type=int, default=0)   # A3-1: beats of stage 1 (both kinds in both rooms), then the A3 layout
    a = ap.parse_args(); V2["on"] = True; RoomBody.default_cues = True; t0 = time.time(); n = 64
    assert not 7701 <= a.seed <= 7712
    rows, info, spurs = layout(bool(a.mirror), both=bool(a.staged), seen=bool(a.seen)); torch.manual_seed(a.seed)
    B = make(rows, n, a.seed, V2["drain"]); O, Mi = being(B, n, a.being); w = B.w
    every = torch.ones(n, dtype=torch.bool, device=DEV); B.w.store[:] = 0.75; place(B, info); O.reset(every); Mi.begin(every); Mi.prev = None
    x = O(B.raw()); spur = torch.zeros(S, S, dtype=torch.bool, device=DEV)
    for (sx, sy) in spurs:
        spur[sx, sy] = True
    since = torch.full((n, 2), -1, dtype=torch.long, device=DEV); waits = [[], []]; lost = [0, 0]; deaths = [0] * 4; need = [0] * 4; spur_lacking = [0, 0]
    from atanor_os.emerge.room3d_gpu import FLOOR, WATER, FOOD
    fixed = {WATER: (14, 1) if a.mirror else (1, 1), FOOD: (1, 1) if a.mirror else (14, 1)}
    side = [[0, 0], [0, 0]]                                              # stage 2: fills per channel at its fixed spring
    for t in range(a.beats):
        q = min(3, t * 4 // a.beats)
        if a.staged and t == a.staged:                                   # stage 2: one spring of each kind, the A3 layout (the world's change, for every body)
            for j, kind in enumerate((WATER, FOOD)):
                old = w.src[:, j].clone(); w.map[torch.arange(n, device=DEV), old[:, 0], old[:, 1]] = FLOOR
                c = fixed[kind]; w.map[:, c[0], c[1]] = kind; w.src[:, j] = torch.tensor(c, device=DEV); w.spots[kind] = torch.tensor([c], device=DEV)
        act = Mi.choose(x); al = ~B.dead.clone(); s0 = B.w.store.clone(); f = B.step(act); x2 = O(B.raw()); died = al & B.dead
        deaths[q] += int(died.sum()); need[q] += int(((f[:, 0].bool() & (s0[:, 0] < s0[:, 1])) | (f[:, 1].bool() & (s0[:, 1] < s0[:, 0]))).sum())
        cx, cy = w.pos[:, 0].long().clamp(0, S - 1), w.pos[:, 1].long().clamp(0, S - 1); insp = spur[cx, cy]; lacking = (w.store.min(1).values < 0.7) & ~died
        spur_lacking[0] += int((insp & lacking).sum()); spur_lacking[1] += int(lacking.sum())
        for j in range(2):                                               # j = the world's store order: 0 water, 1 food
            start = (w.store[:, j] < 0.7) & ~died & (since[:, j] < 0); since[:, j] = torch.where(start, torch.full_like(since[:, j], t), since[:, j])
            got = f[:, j].bool() & (since[:, j] >= 0)
            for i in torch.nonzero(got)[:, 0].tolist():
                waits[j].append(t - int(since[i, j]))
            lost[j] += int((died & (since[:, j] >= 0)).sum()); since[:, j] = torch.where(got | died, torch.full_like(since[:, j], -1), since[:, j])
        if a.staged and t >= a.staged:
            side[0][0] += int(f[:, 0].sum()); side[0][1] += int(f[:, 1].sum())
        Mi.after(x, act, x2, died, B.dead & ~died); x = x2
        if B.dead.any():
            rebirth(B, O, Mi, info); B.felt[:] = 0; x = O(B.raw())
        if (t + 1) % 1000 == 0:
            Mi.going.learn()
            with torch.enable_grad():
                Mi.wo.learn(n=max(1, 2500 // max(1, getattr(Mi.wo, "per_call", 1))))
    wt = lambda v: dict(n=len(v), median=float(np.median(v)) if v else None, first_half_median=float(np.median(v[:len(v) // 2])) if len(v) > 1 else None,
                        last_half_median=float(np.median(v[len(v) // 2:])) if len(v) > 1 else None)
    r = dict(seen=a.seen, staged=a.staged, stage2_fills=dict(water=side[0][0], food=side[0][1]), world="a3", mirror=a.mirror, seed=a.seed, being=a.being, beats=a.beats, deaths_by_quarter=deaths, need_fills_by_quarter=need,
             deaths_per_1000_first_quarter=round(1000 * deaths[0] / (n * a.beats / 4), 3), hunger_to_fill=dict(water=wt(waits[0]), food=wt(waits[1])),
             died_lacking=lost, share_of_lacking_beats_in_spurs=round(spur_lacking[0] / max(spur_lacking[1], 1), 4), seconds=round(time.time() - t0))
    print(json.dumps(r), flush=True); json.dump(r, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
