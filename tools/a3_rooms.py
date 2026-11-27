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
    ap.add_argument("--beats", type=int, default=6000); ap.add_argument("--seed", type=int, default=9301); ap.add_argument("--staged", type=int, default=0); ap.add_argument("--seen", type=int, default=0); ap.add_argument("--why", type=int, default=0); ap.add_argument("--lw-mode", default=""); ap.add_argument("--roam", type=float, default=0.0); ap.add_argument("--roam-trust", type=int, default=0); ap.add_argument("--eye", type=float, default=0.0); ap.add_argument("--oracle", type=int, default=0); ap.add_argument("--force", type=int, default=0)   # LW3: learned trust in the homing content   # gate 2 run 1 explained (judge side only): where its own goals are, whether bodies get there   # A3-1: beats of stage 1 (both kinds in both rooms), then the A3 layout
    a = ap.parse_args(); V2["on"] = True
    if a.eye:                                                             # gate 2 run 3: eye height of the being grown at it (s3_e: 1.1)
        RoomBody.default_eye = a.eye
    RoomBody.default_cues = True; t0 = time.time(); n = 64
    assert not 7701 <= a.seed <= 7712
    rows, info, spurs = layout(bool(a.mirror), both=bool(a.staged), seen=bool(a.seen)); torch.manual_seed(a.seed)
    B = make(rows, n, a.seed, V2["drain"]); O, Mi = being(B, n, a.being); w = B.w
    import os as _os
    if _os.environ.get("ATANOR_HOPS"):                                   # D50 in A3 (diagnostic): how far ahead on its own chain the homing lean aims
        Mi.going.hops = int(_os.environ["ATANOR_HOPS"])
    if a.roam > 0:                                                        # G2a diagnostic: E1b unfamiliarity in the score (Mind.roam_mode="score")
        Mi.roam = a.roam; Mi.roam_mode = "score"
        if a.roam_trust:                                                  # gate 2 run 2: the unfamiliarity's weight learned from its kept promises
            from atanor_os.emerge.mind import LearnedWeight
            Mi.learned_roam = LearnedWeight(w0=0.5, mode="reli")
    if a.lw_mode:
        from atanor_os.emerge.mind import LearnedWeight
        Mi.learned_home = LearnedWeight(w0=0.0, mode=a.lw_mode)
    every = torch.ones(n, dtype=torch.bool, device=DEV); B.w.store[:] = 0.75; place(B, info); O.reset(every); Mi.begin(every); Mi.prev = None
    x = O(B.raw()); spur = torch.zeros(S, S, dtype=torch.bool, device=DEV)
    for (sx, sy) in spurs:
        spur[sx, sy] = True
    since = torch.full((n, 2), -1, dtype=torch.long, device=DEV); waits = [[], []]; lost = [0, 0]; deaths = [0] * 4; need = [0] * 4; spur_lacking = [0, 0]
    from atanor_os.emerge.room3d_gpu import FLOOR, WATER, FOOD
    fixed = {WATER: (14, 1) if a.mirror else (1, 1), FOOD: (1, 1) if a.mirror else (14, 1)}
    if a.oracle:                                                          # G2L ladder (judge-side diagnostic, never a pass): 1 = the straight vector to the true spring of the
        import collections                                                # lacking kind; 2 = the vector to the next cell on the true shortest way around the walls
        gm = np.array([list(r_) for r_ in rows]); NX = torch.zeros(2, S, S, 2, device=DEV)
        for j, kind in enumerate((WATER, FOOD)):
            sx, sy = fixed[kind]; D = np.full((S, S), np.inf); q = collections.deque()
            for dx_, dy_ in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                x_, y_ = sx + dx_, sy + dy_
                if 0 <= x_ < S and 0 <= y_ < S and gm[y_, x_] == ".":
                    D[x_, y_] = 0; q.append((x_, y_))
            while q:
                i_, j_ = q.popleft()
                for dx_, dy_ in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    x_, y_ = i_ + dx_, j_ + dy_
                    if 0 <= x_ < S and 0 <= y_ < S and gm[y_, x_] == "." and D[x_, y_] == np.inf:
                        D[x_, y_] = D[i_, j_] + 1; q.append((x_, y_))
            for x_ in range(S):
                for y_ in range(S):
                    tx, ty = sx + 0.5, sy + 0.5
                    if a.oracle == 2 and np.isfinite(D[x_, y_]) and D[x_, y_] > 0:
                        best = None
                        for dx_, dy_ in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                            u_, v_ = x_ + dx_, y_ + dy_
                            if not (0 <= u_ < S and 0 <= v_ < S) or not np.isfinite(D[u_, v_]):
                                continue
                            if dx_ and dy_ and not (np.isfinite(D[x_ + dx_, y_]) and np.isfinite(D[x_, y_ + dy_])):
                                continue                                  # no cutting a wall's corner
                            if best is None or D[u_, v_] < best[0]:
                                best = (D[u_, v_], u_ + 0.5, v_ + 0.5)
                        if best is not None and best[0] < D[x_, y_]:
                            tx, ty = best[1], best[2]
                    NX[j, x_, y_, 0] = tx; NX[j, x_, y_, 1] = ty

        def oracle(c):
            cx_, cy_ = w.pos[:, 0].long().clamp(0, S - 1), w.pos[:, 1].long().clamp(0, S - 1)
            v_ = NX[c, cx_, cy_] - w.pos.float(); co, si = torch.cos(w.head), torch.sin(w.head)
            return v_[:, 0] * co + v_[:, 1] * si, -v_[:, 0] * si + v_[:, 1] * co
        Mi.going.oracle = oracle
    if _os.environ.get("ATANOR_B0", "0") == "oracle":                     # B0 ladder rung (judge-side, never a pass): the imagination walks the true map
        import collections
        gm = np.array([list(r_) for r_ in rows]); DW = torch.full((2, S, S), 99.0, device=DEV)
        for j, kind in enumerate((WATER, FOOD)):
            sx, sy = fixed[kind]; D = np.full((S, S), np.inf); q = collections.deque()
            for dx_, dy_ in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                x_, y_ = sx + dx_, sy + dy_
                if 0 <= x_ < S and 0 <= y_ < S and gm[y_, x_] == ".":
                    D[x_, y_] = 0; q.append((x_, y_))
            while q:
                i_, j_ = q.popleft()
                for dx_, dy_ in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    x_, y_ = i_ + dx_, j_ + dy_
                    if 0 <= x_ < S and 0 <= y_ < S and gm[y_, x_] == "." and D[x_, y_] == np.inf:
                        D[x_, y_] = D[i_, j_] + 1; q.append((x_, y_))
            D[~np.isfinite(D)] = 99.0; DW[j] = torch.tensor(D, device=DEV, dtype=torch.float32)

        def _field(c, xy, near=False):                                    # bilinear over cell centres (x + .5)
            if near:
                i_ = xy[..., 0].floor().long().clamp(0, S - 1); j_ = xy[..., 1].floor().long().clamp(0, S - 1)
                return DW[c.view(-1, *([1] * (xy.dim() - 2))).expand_as(i_), i_, j_]
            u_ = (xy[..., 0] - 0.5).clamp(0, S - 1.001); v_ = (xy[..., 1] - 0.5).clamp(0, S - 1.001)
            i0, j0 = u_.floor().long(), v_.floor().long(); a_, b_ = u_ - i0, v_ - j0
            cc = c.view(-1, *([1] * (xy.dim() - 2))).expand_as(i0)
            g_ = lambda i, j: DW[cc, i.clamp(0, S - 1), j.clamp(0, S - 1)]
            from atanor_os.emerge.going import _blend
            return _blend(((1 - a_) * (1 - b_), g_(i0, j0)), (a_ * (1 - b_), g_(i0 + 1, j0)), ((1 - a_) * b_, g_(i0, j0 + 1)), (a_ * b_, g_(i0 + 1, j0 + 1)))
        Mi.going.b0_true = dict(pos=lambda: w.pos.float(), head=lambda: w.head.float(), field=_field)
    side = [[0, 0], [0, 0]]                                              # stage 2: fills per channel at its fixed spring
    for t in range(a.beats):
        q = min(3, t * 4 // a.beats)
        if a.staged and t == a.staged:                                   # stage 2: one spring of each kind, the A3 layout (the world's change, for every body)
            for j, kind in enumerate((WATER, FOOD)):
                old = w.src[:, j].clone(); w.map[torch.arange(n, device=DEV), old[:, 0], old[:, 1]] = FLOOR
                c = fixed[kind]; w.map[:, c[0], c[1]] = kind; w.src[:, j] = torch.tensor(c, device=DEV); w.spots[kind] = torch.tensor([c], device=DEV)
        if a.force:                                                       # G2L rung 2b (judge-side probe, never a pass): every body chooses afresh every beat
            Mi.force = torch.ones(n, dtype=torch.bool, device=DEV)
        act = Mi.choose(x); al = ~B.dead.clone(); s0 = B.w.store.clone(); f = B.step(act); x2 = O(B.raw()); died = al & B.dead
        if a.why:
            G_ = Mi.going; KU = G_.KU
            if t == 0:
                upos = torch.zeros(KU, 2, device=DEV); useen = torch.zeros(KU, device=DEV); gcount = torch.zeros(KU, device=DEV); why = dict(held=0, at_goal=0, bumped_held=0, bumped_free=0, free=0)
            gp = getattr(Mi, "goal_prev", None); u0 = G_.unit(G_.place(x)); bump = w.felt[:, 2] > 0
            if gp is not None:
                hd = (gp >= 0) & al; why["held"] += int(hd.sum()); why["at_goal"] += int((hd & (u0 == gp)).sum()); why["bumped_held"] += int((hd & bump).sum())
                why["free"] += int((~hd & al).sum()); why["bumped_free"] += int((~hd & al & bump).sum())
                if bool(hd.any()):
                    gcount += torch.bincount(gp[hd], minlength=KU).float()
            if t == 0:
                dg = dict(none=0, right=0, wrong=0, unknown_place=0, near_right_spring=0, n=0)
            if gp is not None and bool(died.any()):                       # gate 2 run 3 explained: what each dying body was aiming at (judge side only)
                for i in torch.nonzero(died)[:, 0].tolist():
                    dg["n"] += 1; j = int(s0[i].argmin()); sp = torch.tensor(fixed[(WATER, FOOD)[j]], device=DEV, dtype=torch.float)
                    if float((w.pos[i].float() - sp).norm()) < 3.0:
                        dg["near_right_spring"] += 1
                    g_ = int(gp[i])
                    if g_ < 0:
                        dg["none"] += 1
                    elif useen[g_] == 0:
                        dg["unknown_place"] += 1
                    elif float((upos[g_] - sp).norm()) < 2.5:
                        dg["right"] += 1
                    else:
                        dg["wrong"] += 1
            with torch.no_grad():
                uu = G_.unit(G_.place(x2)); lv = ~B.dead
                if bool(lv.any()):
                    cnt = torch.bincount(uu[lv], minlength=KU).float(); sm = torch.zeros(KU, 2, device=DEV).index_add_(0, uu[lv], w.pos[lv].float()); hit = cnt > 0
                    upos = torch.where((hit & (useen == 0))[:, None], sm / cnt.clamp(min=1)[:, None], torch.where(hit[:, None], 0.99 * upos + 0.01 * sm / cnt.clamp(min=1)[:, None], upos)); useen += cnt
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
    whyr = None
    if a.why:
        G_ = Mi.going; rr = (G_.R / G_.Vn.clamp(min=1)[:, None]); top = torch.argsort(gcount, descending=True)[:5].tolist(); tot = float(gcount.sum())
        whyr = dict(why, death_goal=dg, goals=[dict(unit=u_, share=round(float(gcount[u_]) / max(tot, 1), 3), judge_cell=[round(float(v), 1) for v in upos[u_].tolist()], judge_seen=int(useen[u_]),
                    in_spur=bool(spur[int(upos[u_][0]), int(upos[u_][1])]) if useen[u_] > 0 else None, relief_rate=[round(float(v), 4) for v in rr[u_].tolist()],
                    V=[round(float(v), 3) for v in G_.V[u_].tolist()], homing_cell_own=[round(float(v), 2) for v in G_.Pu[u_].tolist()]) for u_ in top],
                    best_relief_rate_units=[[int(v) for v in torch.argsort(rr[:, c], descending=True)[:3].tolist()] for c in range(rr.shape[1])])
    r = dict(rely=dict(eye_err=round(float(getattr(Mi.going, "eye_err", -1)), 2), mem_err=round(float(getattr(Mi.going, "mem_err", -1)), 2), eye_led=getattr(Mi.going, "rely_eye", 0), memory_led=getattr(Mi.going, "rely_mem", 0), seen_led=getattr(Mi.going, "sm3_n", 0)), gut=dict(asked=getattr(Mi, "g_asked", 0), known=getattr(Mi, "g_known", 0), differed=getattr(Mi, "g_diff", 0), journeys_ended=getattr(Mi, "g_ends", 0), usual_len=round(float(getattr(Mi, "g_len", 0)), 1)), seen_relief=dict(hit=round(float(getattr(Mi.going, "sr_hit", -1)), 3), samples=getattr(Mi.going, "sr_n", 0), writes=getattr(Mi.going, "sr_writes", 0)), seen_map=dict(err=round(float(getattr(Mi.going, "sm_err", -1)), 3), stops=getattr(Mi.going, "sm_n", 0), writes=getattr(Mi.going, "sm_writes", 0)), e1=getattr(Mi, "e1_n", 0), w0=(dict(beats=getattr(Mi, "w0_n", 0), spread=round(float(getattr(Mi, "w0_spread", 0)), 5), picks=(Mi.w0_pick.tolist() if getattr(Mi, "w0_pick", None) is not None else None))), b0=dict(mode=_os.environ.get("ATANOR_B0", "0"), chose=getattr(Mi, "b0_n", 0), drain=round(float(getattr(Mi.going, "b0_dr", 0)), 5)), hops=Mi.going.hops, goal_beats=getattr(Mi.going, "n_goal", 0), noway_beats=getattr(Mi.going, "n_noway", 0), r2=(dict(reads=[int(v) for v in Mi.going.r2_reads[::100]], splits=int(Mi.going.r2.splits), moves=int(Mi.going.r2.moves), forgotten=int(Mi.going.r2.forgotten)) if getattr(Mi.going, "r2", None) is not None else None), trust_path=(Mi.learned_home.hist[::300] if getattr(Mi, "learned_home", None) is not None else None), trust_reliefs=getattr(getattr(Mi, "learned_home", None), "n_rel", None), force=a.force, remaps=getattr(Mi.going, "remaps", 0), oracle=a.oracle, eye=RoomBody.default_eye, mismatch_events=getattr(Mi, "mm_n", 0), roam_trust=(dict(end=round(float(Mi.learned_roam.w), 3), path=Mi.learned_roam.hist[::300], promises=getattr(Mi.learned_roam, 'n_arr', 0)) if getattr(Mi, 'learned_roam', None) is not None else None), roam=a.roam, trust_end=(round(float(Mi.learned_home.w), 3) if getattr(Mi, "learned_home", None) is not None else None), lw=(dict(mode=a.lw_mode, end=Mi.learned_home.w, path=Mi.learned_home.hist[::500], arrivals=getattr(Mi.learned_home, "n_arr", None)) if a.lw_mode else None), why=whyr, goal_held=getattr(Mi, 'goal_held', 0), goal_switch=getattr(Mi, 'goal_switch', 0), seen=a.seen, staged=a.staged, stage2_fills=dict(water=side[0][0], food=side[0][1]), world="a3", mirror=a.mirror, seed=a.seed, being=a.being, beats=a.beats, deaths_by_quarter=deaths, need_fills_by_quarter=need,
             deaths_per_1000_first_quarter=round(1000 * deaths[0] / (n * a.beats / 4), 3), hunger_to_fill=dict(water=wt(waits[0]), food=wt(waits[1])),
             died_lacking=lost, share_of_lacking_beats_in_spurs=round(spur_lacking[0] / max(spur_lacking[1], 1), 4), seconds=round(time.time() - t0))
    print(json.dumps(r), flush=True); json.dump(r, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
