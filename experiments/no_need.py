"""With no survival condition at all (insides never drain, start full), what does the same being do? Does it go still, or keep moving / exploring?
Same being file (raised in varied rooms), same mind (learning on), rooms 1-3, 64 bodies, 900 ticks; compared with the usual drain.
Per 300-tick block: share of 'legs still' choices, distance walked per tick, cells visited (coverage of reachable floor), turning share, 'use' share,
and how often it still goes to springs and uses them (fills, even though full)."""
import sys, torch, numpy as np
from atanor_os.emerge.room_body import RoomBody
from atanor_os.emerge.body_contract import Organs
from atanor_os.emerge.mind import Mind
from atanor_os.emerge.room_vary import condition
from atanor_os.emerge import room3d_gpu as G
dev, N, T = "cuda", 64, 900
BEING = "being/host/atanor_born_in_rooms.pt"


def run(drain_on, room):
    torch.manual_seed(room)
    rows, drain = condition(room)
    B = RoomBody(N, rows, drain=(drain if drain_on else 0.0), seed=room, device=dev)
    if not drain_on:
        B.w.store[:] = 1.0
    O = Organs(B.c, N, dev=dev, depth="mc/mc_depth.pt"); Mi = Mind(O, B.c, N, dev=dev); Mi.load(BEING); Mi.wo.on = True
    legs = B.legs; neck = B.neck; hand = B.hand
    reach = int((B.w.base == G.FLOOR).sum())
    seen = torch.zeros(N, B.w.S, B.w.S, dtype=torch.bool, device=dev)
    x = O(B.raw()); rows_out = []
    blk = dict(still=0, turn=0, use=0, dist=0.0, fills=0, n=0)
    for t in range(T):
        act = Mi.choose(x); a0 = ~B.dead.clone(); p0 = B.w.pos.clone()
        f = B.step(act); died = a0 & B.dead; x2 = O(B.raw()); Mi.after(x, act, x2, died, B.dead & ~died); x = x2
        al = ~B.dead
        blk["still"] += int(((legs[act] == 0) & al).sum()); blk["turn"] += int((((neck[act] == 1) | (neck[act] == 2)) & al).sum()); blk["use"] += int(((hand[act] == 2) & al).sum())
        blk["dist"] += float((B.w.pos - p0).norm(dim=1)[al].sum()); blk["fills"] += int(f.any(1).sum()); blk["n"] += int(al.sum())
        cx = B.w.pos[:, 0].long().clamp(0, B.w.S - 1); cy = B.w.pos[:, 1].long().clamp(0, B.w.S - 1); seen[torch.arange(N, device=dev), cx, cy] = True
        if (t + 1) % 300 == 0:
            n = max(blk["n"], 1)
            rows_out.append(f"t{t + 1}: legs still {blk['still'] / n:.2f} | turning {blk['turn'] / n:.2f} | use {blk['use'] / n:.2f} | walked {blk['dist'] / n:.3f}/tick | "
                            f"floor visited {float(seen.sum((1, 2)).float().mean()) / reach:.2f} | fills {blk['fills']} | alive {float(al.float().mean()):.2f}")
            blk = dict(still=0, turn=0, use=0, dist=0.0, fills=0, n=0)
    return rows_out


for room in (1, 2, 3):
    for drain_on in (True, False):
        print(f"room {room}, {'usual hunger/thirst' if drain_on else 'NO survival condition (never drains)'}:", flush=True)
        for r in run(drain_on, room):
            print("   ", r, flush=True)
print("DONE")
