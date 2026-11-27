"""World A2 (atanor_os/LOKI_IDEAL_SPEC.md section 5; registered in atanor_os/WORLD_A_PROTOCOL.md): food moves between three fixed dead-end corners
in a fixed cycle (the leg end it starts at -> top-left corner of the crossbar -> top-right corner), `--period` beats each, for every body at once.
It does not jump when eaten (one spot at a time). Water stays. Everything else as the room V2 world. Judge side: when food moved, how long until the
first fill at its new place, fills, deaths, where the being believes the food is (K1 read-out), and visits to the NEXT corner before food comes.
  PYTHONPATH=. python -m tools.a2_moving_food OUT.json --lay 9101_0 --being s3_a [--beats 6000 --period 300]"""
import argparse, json, time
import numpy as np, torch
from tools.need_fork import t_room, place, DEV
from tools.a1_channel import V2
from tools.l1_life import make
from tools.s4_kin import being, rebirth
from atanor_os.emerge.room_body import RoomBody
from atanor_os.emerge.room3d_gpu import FLOOR, FOOD


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--lay", default="9101_0"); ap.add_argument("--being", required=True)
    ap.add_argument("--beats", type=int, default=6000); ap.add_argument("--period", type=int, default=300); ap.add_argument("--still", type=int, default=0); ap.add_argument("--decay", type=float, default=0.0); ap.add_argument("--rdecay", type=float, default=0.0); ap.add_argument("--oracle-goal", type=int, default=0); ap.add_argument("--skill-every", type=int, default=1000); ap.add_argument("--homing", type=int, default=0); ap.add_argument("--own-goal", type=int, default=0); ap.add_argument("--rep", type=int, default=0); ap.add_argument("--view", type=int, default=0); ap.add_argument("--home-rem", type=int, default=0); ap.add_argument("--eye", type=float, default=0.0); ap.add_argument("--lw", type=float, default=-1.0); ap.add_argument("--lw-lr", type=float, default=1e-3); ap.add_argument("--lw-mode", default="beat"); ap.add_argument("--bodies", type=int, default=64); ap.add_argument("--circ", type=int, default=0); ap.add_argument("--hops", type=int, default=0); ap.add_argument("--save-trust", default=""); ap.add_argument("--save-place", default=""); ap.add_argument("--save-seen", default="")   # LW1: learned weight on the homing content from this start (-1 = off)   # view check (founder 2026-11-26): is food in the picture, by distance; near misses   # second start: seeds offset by 1000 x rep   # D48: the goal = the unit its own relief memory values most for the lacking channel   # D46: innate homing vector from path integration, read by its own body forecaster   # D45: how often the going skill learns from lived pieces   # D44 diagnostic: 1 = the true food corner as the going goal (earned weight), 2 = the same at full weight   # V1 variant: the going organ's fading per beat (0 = as it is, .9995)
    a = ap.parse_args(); V2["on"] = True; RoomBody.default_cues = True
    t0 = time.time(); n = a.bodies                                          # B1 (one body, founder 2026-11-27): 64 = as before
    if a.eye:
        RoomBody.default_eye = a.eye                                     # E-height (founder 2026-11-24 / 11-27): eye height in cells
    seed, mirror = map(int, a.lay.split("_")); assert not 7701 <= seed <= 7712; rows, info = t_room(seed, bool(mirror)); sd = seed * 10 + mirror + 71 + 1000 * a.rep
    torch.manual_seed(sd)
    g = np.array([list(r) for r in rows]); H, W = g.shape
    f0 = tuple(int(v) for v in np.argwhere(g == "f")[0][::-1])                    # (x, y)
    corners = [f0, (1, 1), (W - 2, 1)]
    for c in corners[1:]:
        assert g[c[1], c[0]] == ".", (c, g[c[1], c[0]])
    B = make(rows, n, sd, V2["drain"]); O, Mi = being(B, n, a.being); w = B.w; G = Mi.going
    if a.hops > 0:                                                       # D50: how far ahead along its own chain the homing lean aims (L1d: 4)
        G.hops = a.hops
    if a.decay:
        G.decay = a.decay
    if a.rdecay:
        G.rdecay = a.rdecay
    if a.home_rem:
        G.home_rem = True
    if a.lw >= 0:
        from atanor_os.emerge.mind import LearnedWeight
        Mi.learned_home = LearnedWeight(w0=a.lw, lr=a.lw_lr, mode=a.lw_mode)
    if a.homing:
        G.homing, G.organs, G.net = True, O, Mi.net
        G.homing_lean = a.homing == 2                                    # 2 = D47 (the going lean itself walks by the vector); 1 = only goals given through plan_goal

    def put(k):                                                          # food at corner k for every body (the world's clock, not a body's doing)
        c = corners[k]; old = w.src[:, 1].clone()
        w.map[torch.arange(n, device=DEV), old[:, 0], old[:, 1]] = FLOOR
        w.map[:, c[0], c[1]] = FOOD; w.src[:, 1] = torch.tensor(c, device=DEV); w.spots[FOOD] = torch.tensor([c], device=DEV)
    every = torch.ones(n, dtype=torch.bool, device=DEV); B.w.store[:] = 0.75; place(B, info); O.reset(every); Mi.begin(every); Mi.prev = None
    put(0); x = O(B.raw()); k = 0
    if a.oracle_goal:                                                   # D44 (judge's help, a diagnostic - never a pass): each corner's going unit, from its innate path code
        from tools.d34b_organ_parts import code_of
        cc = code_of([(c[0] + 0.5, c[1] + 0.5) for c in corners], info, DEV)
        from tools.a2_oracle import bfs
        openr = [r_.replace("f", ".").replace("w", ".") for r_ in rows]; DM = [bfs(openr, c) for c in corners]; appr = [0, 0, 0]; unit_err = []   # closer / same / farther on the true map
    KU = G.KU; upos = torch.zeros(KU, 2, device=DEV); useen = torch.zeros(KU, device=DEV)
    cyc, cur = [], None; deaths = 0; fills = [0, 0]; need = 0; trace = []
    since = torch.full((n,), -1, dtype=torch.long, device=DEV); waits, lost = [], 0   # A2-1: per body, beats from food store falling under .7 to its next food fill
    for t in range(a.beats):
        if not a.still and t % a.period == 0:
            k = (t // a.period) % 3; put(k)
            if cur is not None:
                cyc.append(cur)
            nxt = corners[(k + 1) % 3]
            cur = dict(t=t, corner=k, first_fill=None, fills=0, deaths=0, next_visits=0, at_next=nxt)
        if a.own_goal and t >= 50 and getattr(G, "V", None) is not None:   # D48: no judge - its own belief (relief memory's value per channel), BFS path, homing vector
            lack_, c_ = G.needs(x); Vc = G.V[:G.used].clone() if G.used < G.KU else G.V.clone()
            best = Vc.argmax(0)                                          # per channel (0 water, 1 food: the going organ's order)
            Mi.plan_goal = torch.where(lack_ > 0.3, best[c_], torch.full((n,), -1, device=DEV, dtype=torch.long))
            from atanor_os.emerge.mind import Earned
            if getattr(Mi, "plan_earned", None) is None or not getattr(Mi.plan_earned, "fixed", False):
                Mi.plan_earned = Earned(); Mi.plan_earned.weight = lambda: 1.0; Mi.plan_earned.fixed = True
        if a.oracle_goal and t >= 50:
            gu = G.unit(cc); food_low = (w.store[:, 1] < w.store[:, 0]) & (w.store[:, 1] < 0.7)
            Mi.plan_goal = torch.where(food_low, gu[k].expand(n), torch.full((n,), -1, device=DEV, dtype=torch.long))
            if a.oracle_goal == 2:
                from atanor_os.emerge.mind import Earned
                Mi.plan_earned = Earned(); Mi.plan_earned.weight = lambda: 1.0
        if a.oracle_goal and t >= 50:
            pos0 = w.pos.cpu().numpy(); had = (Mi.plan_goal >= 0).cpu().numpy() if Mi.plan_goal is not None else np.zeros(n, bool)
        if a.view:                                                       # judge side only: food block in any column of the 80-degree picture, by distance
            from atanor_os.emerge.room3d_gpu import FOV, PW
            offs = (torch.arange(PW, device=DEV) / (PW - 1) - 0.5) * FOV; kd, _ = w.cast(w.head[:, None] + offs[None], step=0.1)
            inview = (kd == FOOD).any(1); fc = w.src[:, 1].float() + 0.5; dd_ = (w.pos - fc).norm(dim=1)
            if t == 0:
                VB = [0.0, 1.5, 3.0, 6.0, 99.0]; vseen = np.zeros(4); vall = np.zeros(4); inzone = np.zeros(n, bool); zin = np.zeros(n); zview = np.zeros(n); miss = []; hit_ = []; dock = []
            b_ = np.digitize(dd_.cpu().numpy(), VB[1:-1]); iv = inview.cpu().numpy(); hg = (w.store[:, 1] < 0.7).cpu().numpy(); alv = (~B.dead).cpu().numpy()
            np.add.at(vall, b_[alv], 1); np.add.at(vseen, b_[alv & iv], 1)
            near = (dd_ <= 2.0).cpu().numpy() & hg & alv; far_ = (dd_ > 3.0).cpu().numpy()
            enter = near & ~inzone; zin[enter] = 0; zview[enter] = 0; inzone |= enter; zin[inzone & near] += 1; zview[inzone & near & iv] += 1
        act = Mi.choose(x); al = ~B.dead.clone(); s0 = B.w.store.clone(); f = B.step(act); x2 = O(B.raw()); died = al & B.dead
        if getattr(Mi, "learned_home", None) is not None:                # LW diagnostic: what each body's beat was (2 death, 1 a fill, 0 else)
            Mi.learned_home.kinds = torch.where(died, 2, torch.where(f.any(1), 1, 0))
            gp_ = getattr(Mi, "goal_prev", None)
            if gp_ is not None:
                if "gset" not in dir():
                    gset = torch.full((n,), -1, dtype=torch.long, device=DEV); glag = []
                start_ = (gp_ >= 0) & (gset < 0); gset = torch.where(start_, torch.full_like(gset, t), gset)
                for i in torch.nonzero(f[:, 1].bool() & (gset >= 0))[:, 0].tolist():
                    glag.append(t - int(gset[i]))
                gset = torch.where(f[:, 1].bool() | died | (gp_ < 0), torch.full_like(gset, -1), gset)
        if a.view:
            ate = f[:, 1].bool().cpu().numpy(); dn = died.cpu().numpy()
            for i in np.nonzero(inzone & ate)[0]:
                hit_.append(zview[i] / max(zin[i], 1)); dock.append(zin[i])
            for i in np.nonzero(inzone & ~ate & (far_ | dn))[0]:
                miss.append(zview[i] / max(zin[i], 1))
            inzone &= ~(ate | far_ | dn)
        if a.oracle_goal and t >= 50:
            pos1 = w.pos.cpu().numpy(); Dk = DM[k]
            for i in np.nonzero(had)[0]:
                d0 = Dk[int(pos0[i, 1]), int(pos0[i, 0])]; d1 = Dk[int(pos1[i, 1]), int(pos1[i, 0])]
                if d0 >= 0 and d1 >= 0:
                    appr[0 if d1 < d0 else (1 if d1 == d0 else 2)] += 1
            if t % 300 == 60:
                up = upos[gu[k]].cpu().numpy(); unit_err.append(float(np.hypot(up[0] - corners[k][0] - 0.5, up[1] - corners[k][1] - 0.5)))
        if a.circ:                                                       # C1-C7 (WORLD_A_PROTOCOL.md; judge side only): tools/walk_diag.py
            if "WD" not in dir():
                from tools.walk_diag import WalkDiag
                WD = WalkDiag(n, Mi.M)
            WD.step(w, B, Mi, G, O, x2, act, died)
        cxy = w.pos.long(); inblock = w.map[torch.arange(n, device=DEV), cxy[:, 0], cxy[:, 1]] != FLOOR   # world bug check (founder 2026-11-27): a body inside a block (food moved onto it)
        trapped_bb = locals().get("trapped_bb", 0) + int((inblock & ~B.dead).sum()); trapped_died = locals().get("trapped_died", 0) + int((inblock & died).sum())
        deaths += int(died.sum()); fills[0] += int(f[:, 0].sum()); fills[1] += int(f[:, 1].sum())
        need += int(((f[:, 0].bool() & (s0[:, 0] < s0[:, 1])) | (f[:, 1].bool() & (s0[:, 1] < s0[:, 0]))).sum())
        hungry = (w.store[:, 1] < 0.7) & ~died; start = hungry & (since < 0); since = torch.where(start, torch.full_like(since, t), since)
        got = f[:, 1].bool() & (since >= 0)
        for i in torch.nonzero(got)[:, 0].tolist():
            waits.append(t - int(since[i]))
        lost += int((died & (since >= 0)).sum()); since = torch.where(got | died, torch.full_like(since, -1), since)
        if cur is not None:
            cur["deaths"] += int(died.sum()); cur["fills"] += int(f[:, 1].sum())
            if cur["first_fill"] is None and bool(f[:, 1].any()):
                cur["first_fill"] = t - cur["t"]
            nx = torch.tensor(cur["at_next"], device=DEV, dtype=torch.float32) + 0.5
            cur["next_visits"] += int(((w.pos - nx).norm(dim=1) < 1.6).sum())
            if not a.still and (t % a.period) >= a.period - 100:          # anticipation (founder/Copilot 2026-11-27): body-beats near each corner in the last 100 beats before food moves
                cur.setdefault("late", [0, 0, 0])
                for ci, cc_ in enumerate(corners):
                    cur["late"][ci] += int(((w.pos - (torch.tensor(cc_, device=DEV, dtype=torch.float32) + 0.5)).norm(dim=1) < 1.6).sum())
        with torch.no_grad():                                            # judge side: where each place unit really is (as K1)
            u = G.unit(G.place(x2)); live = ~B.dead
            if bool(live.any()):
                cnt = torch.bincount(u[live], minlength=KU).float(); sm = torch.zeros(KU, 2, device=DEV).index_add_(0, u[live], w.pos[live].float()); hit = cnt > 0
                upos = torch.where((hit & (useen == 0))[:, None], sm / cnt.clamp(min=1)[:, None], torch.where(hit[:, None], 0.99 * upos + 0.01 * sm / cnt.clamp(min=1)[:, None], upos)); useen += cnt
        Mi.after(x, act, x2, died, B.dead & ~died); x = x2
        if B.dead.any():
            rebirth(B, O, Mi, info); B.felt[:] = 0; x = O(B.raw())
        if (t + 1) % a.skill_every == 0 and (t + 1) % 1000 != 0:
            G.learn()                                                   # D45: the going skill learns more often (the value keeps its 1,000-beat cadence)
        if (t + 1) % 1000 == 0:
            G.learn()
            with torch.enable_grad():
                Mi.wo.learn(n=max(1, 2500 // max(1, getattr(Mi.wo, "per_call", 1))))
        if (t + 1) % 50 == 0 and getattr(G, "V", None) is not None:
            V = G.V[:, 1].clone(); V[useen == 0] = -1e9; b = int(V.argmax()); p = upos[b].cpu().numpy()
            dist = [float(np.hypot(c[0] + 0.5 - p[0], c[1] + 0.5 - p[1])) for c in corners]
            trace.append(dict(t=t + 1, food_corner=k, believed_corner=int(np.argmin(dist)), believed_err=round(dist[k], 2)))
    if cur is not None:
        cyc.append(cur)
    lat = [c["first_fill"] if c["first_fill"] is not None else a.period for c in cyc]
    ant = None
    if not a.still:                                                      # same corner compared with itself: when it is NEXT vs when it is the other (neither current nor next)
        ant = {}
        for ci in range(3):
            nxt_ = [c_["late"][ci] for c_ in cyc if "late" in c_ and (c_["corner"] + 1) % 3 == ci]; oth = [c_["late"][ci] for c_ in cyc if "late" in c_ and (c_["corner"] + 2) % 3 == ci]
            ant[str(ci)] = dict(when_next=float(np.mean(nxt_)) if nxt_ else None, when_other=float(np.mean(oth)) if oth else None, n=(len(nxt_), len(oth)))
    third = max(1, len(lat) // 3)
    lwd = None
    if getattr(Mi, "learned_home", None) is not None:
        gl_ = locals().get("glag", [])
        lwd = dict(mode=a.lw_mode, arrivals=getattr(Mi.learned_home, "n_arr", None), events=getattr(Mi.learned_home, "n_ev", None), trace_T=round(getattr(Mi.learned_home, "T", 0.0), 1), split={k_: round(v_, 4) for k_, v_ in getattr(Mi.learned_home, "split", {}).items()}, goal_to_food=dict(n=len(gl_), median=float(np.median(gl_)) if gl_ else None, p90=float(np.percentile(gl_, 90)) if gl_ else None))
    ph = None
    if a.lw >= 0 and len(Mi.learned_home.hist) >= a.beats - 60:
        hh = np.array(Mi.learned_home.hist[-(a.beats - 60):]); tt = np.arange(60, a.beats)[:len(hh)] % a.period
        ph = [round(float(hh[(tt >= b0) & (tt < b0 + 50)].mean()), 3) for b0 in range(0, a.period, 50)]   # trust by beat within the food period (judge side)
    if a.save_place and getattr(G, "Pu", None) is not None:              # M1d: the place units' own path positions learned in this life
        torch.save(dict(Pu=G.Pu.cpu(), Pn=G.Pn.cpu(), layout=a.lay, beats=a.beats, bodies=n), a.save_place)
    if a.save_trust and getattr(Mi, "learned_home", None) is not None:   # M1a: write the trust's record learned in this life
        json.dump(dict(home=Mi.learned_home.state(), source=dict(layout=a.lay, beats=a.beats, bodies=n, rep=a.rep)), open(a.save_trust, "w"))
    if a.save_seen and getattr(G, "sm_net", None) is not None:           # SM1 / SM2: the eye's learning from this 64-body life, for the being file
        torch.save(G.seen_state(), a.save_seen)
    r = dict(gut=dict(asked=getattr(Mi, "g_asked", 0), known=getattr(Mi, "g_known", 0), differed=getattr(Mi, "g_diff", 0), journeys_ended=getattr(Mi, "g_ends", 0), usual_len=round(float(getattr(Mi, "g_len", 0)), 1)), seen_map=dict(err=round(float(getattr(G, "sm_err", -1)), 3), stops=getattr(G, "sm_n", 0), writes=getattr(G, "sm_writes", 0)), seen_relief=dict(hit=round(float(getattr(G, "sr_hit", -1)), 3), samples=getattr(G, "sr_n", 0), writes=getattr(G, "sr_writes", 0)), circ=(WD.report(G.hops) if a.circ else None), bodies=n, trust_end=(round(float(Mi.learned_home.w), 3) if getattr(Mi, "learned_home", None) is not None else None), trust_promises=getattr(getattr(Mi, "learned_home", None), "n_arr", None), mismatch_events=getattr(Mi, "mm_n", 0), trust_stalls=getattr(getattr(Mi, "learned_home", None), "n_stall", None), lw_phase=ph, lw_diag=lwd, trapped=dict(body_beats=locals().get('trapped_bb', 0), died_inside=locals().get('trapped_died', 0)), anticipation=ant, eye=RoomBody.default_eye, lw=(dict(start=a.lw, lr=a.lw_lr, end=Mi.learned_home.w, path=Mi.learned_home.hist[::500]) if a.lw >= 0 else None), goal_set_n=getattr(Mi, 'goal_set_n', 0), goal_unset=getattr(Mi, 'goal_unset', 0), goal_switch=getattr(Mi, 'goal_switch', 0), view=(dict(bins=["<1.5", "1.5-3", "3-6", ">6"], beats=vall.tolist(), food_in_view=(vseen / np.maximum(vall, 1)).round(3).tolist(),
                        near_then_ate=len(hit_), near_then_left=len(miss), view_share_when_ate=round(float(np.mean(hit_)), 3) if hit_ else None,
                        view_share_when_left=round(float(np.mean(miss)), 3) if miss else None, left_mostly_unseen=int(sum(m_ < 0.5 for m_ in miss)), dock_beats_median=(float(np.median(dock)) if dock else None), dock_beats_mean=(round(float(np.mean(dock)), 1) if dock else None)) if a.view else None), rep=a.rep, homing=a.homing, skill_every=a.skill_every, approach=(dict(closer=appr[0], same=appr[1], farther=appr[2]) if a.oracle_goal else None), goal_unit_cells_from_corner=(round(float(np.mean(unit_err)), 2) if a.oracle_goal and unit_err else None), oracle_goal=a.oracle_goal, plan_weight=(Mi.plan_earned.weight() if getattr(Mi, 'plan_earned', None) is not None else None), decay=G.decay, rdecay=getattr(G, 'rdecay', None), layout=a.lay, being=a.being, beats=a.beats, period=a.period, still=a.still, corners=corners, deaths=deaths, fills=fills, need_fills=need,
             hunger_to_food=dict(n=len(waits), median=float(np.median(waits)) if waits else None, mean=float(np.mean(waits)) if waits else None, first_half_median=float(np.median(waits[:len(waits) // 2])) if len(waits) > 1 else None, last_half_median=float(np.median(waits[len(waits) // 2:])) if len(waits) > 1 else None, died_hungry=lost),
             deaths_per_1000=round(1000 * deaths / (n * a.beats), 3), food_fills_per_1000=round(1000 * fills[1] / (n * a.beats), 2),
             latency_first_third=float(np.mean(lat[:third])), latency_last_third=float(np.mean(lat[-third:])),
             belief_on_food_corner=round(float(np.mean([q["believed_corner"] == q["food_corner"] for q in trace])), 3), cycles=cyc, trace=trace, seconds=round(time.time() - t0))
    print(json.dumps({k_: v for k_, v in r.items() if k_ not in ("cycles", "trace")}), flush=True)
    json.dump(r, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
