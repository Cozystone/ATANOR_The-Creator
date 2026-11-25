"""D37 (atanor_os/FEAR_PROTOCOL.md, registered before running): does the being foresee a deadly drop, and does that change what it does?
The room gets two drops cut into the walls beside the stem (seen as dark low rims; walked into, they kill). The being (S3-end, one mind, 64 bodies)
lives there as always - no rule about drops. Read each beat for bodies at an edge (within .9 cell of a drop or of a plain wall cell, facing it):
  value vote = its learned value's lean (Mind.lw_last) for the forward gestures minus the others; chosen forward or not.
Pit edge vs wall edge (a wall only blocks, a drop kills) separates foresight of death from a general dislike of walking at things.
  PYTHONPATH=. python -m tools.d37_ledge OUT.json --lay 9101_0 --being s3_a [--beats 8000]"""
import argparse, json, math, time
import numpy as np, torch
from tools.need_fork import t_room, place, DEV
from tools.a1_channel import V2
from tools.l1_life import make
from tools.s4_kin import being, rebirth
from atanor_os.emerge.room_body import RoomBody, G


def pit_rows(rows):
    """two drops in the walls beside the stem: left of it one row above the start, right of it two rows above (both on the stem)."""
    g = [list(r) for r in rows]; sy = next(y for y, r in enumerate(g) if "s" in r); sx = g[sy].index("s"); pits = []
    for dy, side in ((1, -1), (2, 1)):
        y = sy - dy; x = sx
        while 0 <= x < len(g[y]) and g[y][x] == ".":
            x += side
        if g[y][x] == "#" and g[y][x - side] == ".":
            g[y][x] = "p"; pits.append((x, y))
    return ["".join(r) for r in g], pits


def edges(B, cells):
    """[n] bool: within .9 of one of these cells' centres and facing it (35 degrees)."""
    if not cells:
        return torch.zeros(B.n, dtype=torch.bool, device=DEV)
    c = torch.tensor([[x + 0.5, y + 0.5] for x, y in cells], device=DEV); d = c[None] - B.w.pos[:, None]; dist = d.norm(dim=2)
    ang = torch.atan2(d[..., 1], d[..., 0]) - B.w.head[:, None]; face = torch.cos(ang) > math.cos(math.radians(35))
    return ((dist < 1.4) & face).any(1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--lay", default="9101_0"); ap.add_argument("--being", required=True)
    ap.add_argument("--beats", type=int, default=8000); ap.add_argument("--conflict", type=float, default=0.0); ap.add_argument("--dread", type=float, default=0.0); ap.add_argument("--view", action="store_true"); ap.add_argument("--earned-going", action="store_true")
    a = ap.parse_args(); V2["on"] = True; RoomBody.default_cues = True; t0 = time.time(); n = 64
    seed, mirror = map(int, a.lay.split("_")); assert not 7701 <= seed <= 7712; rows0, info = t_room(seed, bool(mirror)); rows, pits = pit_rows(rows0)
    sy = int(info["ys"]); walls = [(x, y) for y in range(sy - 4, sy + 1) for x in range(len(rows[y])) if rows[y][x] == "#" and any(
        0 <= x + d < len(rows[y]) and rows[y][x + d] == "." for d in (-1, 1)) and (x, y) not in pits]
    sd = seed * 10 + mirror + 37; torch.manual_seed(sd)
    B = make(rows, n, sd, V2["drain"]); O, Mi = being(B, n, a.being); Mi.conflict = a.conflict; Mi.record_parts = True; Mi.going.dread = a.dread
    if a.earned_going:
        from atanor_os.emerge.mind import Earned
        Mi.going_earned = Earned()
    every = torch.ones(n, dtype=torch.bool, device=DEV); B.w.store[:] = 0.75; place(B, info); O.reset(every); Mi.begin(every); Mi.prev = None
    lg = B.legs; FWD = (lg == 1) | (lg == 5)
    VIEW = None
    if a.view:                                                          # the room viewer (port 8768): body 0's own eye, the judge's map with the drops, the run's numbers
        from tools import room_live_view as VIEW
        VIEW.start()
    x = O(B.raw()); W = 1000; nw = a.beats // W; fatal_src = np.zeros(5, int); fparts = {}; fatal_ground = [0, 0, 0]
    rec = {k: np.zeros(nw) for k in ("pit_deaths", "other_deaths", "need", "pe_n", "pe_fwd", "pe_vote", "we_n", "we_fwd", "we_vote", "lw_sd", "g_n", "g_fwd")}
    for t in range(a.beats):
        k = t // W; pe = edges(B, pits); we = edges(B, walls) & ~pe
        act = Mi.choose(x); lw = Mi.lw_last; parts = Mi.parts
        sk = B.raw()[3]; gnd = sk[:, 9:15].max(1).values > 0.5; gnd_ahead = (sk[:, 9] > 0.5) | (sk[:, 13] > 0.5) | (sk[:, 14] > 0.5)   # the being feels missing ground near it / just ahead (D37f)
        rec["g_n"][k] += int(gnd.sum()); rec["g_fwd"][k] += int((gnd & FWD[act]).sum())
        if lw is not None:
            vote = lw[:, FWD].mean(1) - lw[:, ~FWD].mean(1); rec["lw_sd"][k] += float(lw.std(1).mean())
            rec["pe_vote"][k] += float(vote[pe].sum()); rec["we_vote"][k] += float(vote[we].sum())
        f_ = FWD[act]; rec["pe_n"][k] += int(pe.sum()); rec["we_n"][k] += int(we.sum()); rec["pe_fwd"][k] += int((f_ & pe).sum()); rec["we_fwd"][k] += int((f_ & we).sum())
        s0 = B.w.store.clone(); al = ~B.dead.clone(); f = B.step(act); died = al & B.dead; fell = getattr(B, "fell", torch.zeros_like(died)) & died
        if bool(fell.any()):
            fatal_ground[0] += int(fell.sum()); fatal_ground[1] += int((gnd & fell).sum()); fatal_ground[2] += int((gnd_ahead & fell).sum())
        if bool(fell.any()):                                            # who decided the fatal step: 0 strong pull, 1 value, 2 random, 3 hold, 4 try
            fatal_src += np.bincount(Mi.src[fell].cpu().numpy(), minlength=5)
            ia = act[fell]                                              # D37c: each score part for the fatal gesture, minus that part's mean over gestures
            for k_, v_ in parts.items():
                if k_ != "strong":
                    rel = (v_[fell].gather(1, ia[:, None])[:, 0] - v_[fell].mean(1)).cpu().numpy(); srcs = Mi.src[fell].cpu().numpy()
                    for s_, r_ in zip(srcs, rel):
                        fparts.setdefault(int(s_), {}).setdefault(k_, []).append(float(r_))
        rec["pit_deaths"][k] += int(fell.sum()); rec["other_deaths"][k] += int((died & ~fell).sum())
        rec["need"][k] += int(((f[:, 0].bool() & (s0[:, 0] < s0[:, 1])) | (f[:, 1].bool() & (s0[:, 1] < s0[:, 0]))).sum())
        raw2 = B.raw(); x2 = O(raw2); Mi.after(x, act, x2, died, B.dead & ~died); x = x2
        if VIEW is not None and t % 3 == 0:
            if bool(died[0]):
                VIEW.TRAIL.clear()
            names = ["strong pull", "value", "random", "hold", "try"]; k_ = t // W
            VIEW.push(raw2[-1][0].cpu().numpy(), rows, B.w.pos[0].tolist(), float(B.w.head[0]), B.w.store[0].tolist(), B.w.stock[0].tolist() if hasattr(B.w, "stock") else None,
                      f"D37 drops room {a.lay} - dread {a.dread} - earned lean {a.earned_going} - beat {t + 1} / {a.beats}",
                      f"body 0 decided by: {names[int(Mi.src[0])]}   stores (water, food): {B.w.store[0, 0]:.2f} {B.w.store[0, 1]:.2f}\n"
                      f"this 1,000 beats (64 bodies): drop deaths {int(rec['pit_deaths'][k_])}, other deaths {int(rec['other_deaths'][k_])}, need fills {int(rec['need'][k_])}\n"
                      f"all so far: drop deaths {int(rec['pit_deaths'].sum())}   (red cells = drops; green = body 0's trail)")
        if (t + 1) % 1000 == 0:
            Mi.going.learn()
            with torch.enable_grad():
                Mi.wo.learn(n=max(1, 2500 // max(1, getattr(Mi.wo, 'per_call', 1))))
        if B.dead.any():
            rebirth(B, O, Mi, info); B.felt[:] = 0; x = O(B.raw())
    win = []
    for k in range(nw):
        r = rec; sd_ = r["lw_sd"][k] / W if r["lw_sd"][k] else float("nan")
        win.append(dict(beats=(k + 1) * W, pit_deaths=int(r["pit_deaths"][k]), other_deaths=int(r["other_deaths"][k]), need=int(r["need"][k]),
                        pit_edge=int(r["pe_n"][k]), p_fwd_pit=round(r["pe_fwd"][k] / max(1, r["pe_n"][k]), 3), wall_edge=int(r["we_n"][k]),
                        p_fwd_wall=round(r["we_fwd"][k] / max(1, r["we_n"][k]), 3),
                        vote_pit=round(r["pe_vote"][k] / max(1, r["pe_n"][k]), 4), vote_wall=round(r["we_vote"][k] / max(1, r["we_n"][k]), 4), lw_sd=round(sd_, 4), ground_felt=int(r['g_n'][k]), p_fwd_ground=round(r['g_fwd'][k] / max(1, r['g_n'][k]), 3)))
    src = Mi.src_count.sum(0).tolist() if getattr(Mi, "src_count", None) is not None else None
    out = dict(layout=a.lay, being=a.being, fatal_ground_felt=fatal_ground, dread=a.dread, going_weight=(Mi.going_earned.weight() if getattr(Mi, 'going_earned', None) is not None else None), fatal_src=fatal_src.tolist(), fatal_parts={str(s_): {k_: [round(float(np.mean(v_)), 3), round(float(np.std(v_)), 3), len(v_)] for k_, v_ in d_.items()} for s_, d_ in fparts.items()}, earned=round(float(Mi.wo.earned()), 4), trust_oc=[round(float(z), 5) for z in Mi.wo.oc], conflict=a.conflict, conflict_calls=int(Mi.conflict_n), pits=pits, windows=win, decided_by=src, seconds=round(time.time() - t0))
    print(json.dumps(out), flush=True)
    try:
        old = json.load(open(a.out))["rows"]
    except (OSError, ValueError, KeyError):
        old = []
    json.dump(dict(protocol="atanor_os/FEAR_PROTOCOL.md (D37)", rows=old + [out]), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
