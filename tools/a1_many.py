"""D20 (atanor_os/A1_MANY_PROTOCOL.md, registered before running): one mind living in n bodies at once in the cued full-size T room.
  PYTHONPATH=. python -m tools.a1_many OUT.json --seeds 9101-9101 [--n 64] [--mirrors 0,1]"""
import argparse, copy, json, os, time
import numpy as np, torch
from tools.need_fork import t_room, place, DEV, DEPTH, BEING
from tools.a1_channel import V2, side_sets, fork
from atanor_os.emerge.room_body import RoomBody
from atanor_os.emerge.body_contract import Organs
from atanor_os.emerge.mind import Mind


def many_life(rows, info, seed, n, beats, quarters=4, value_arm="worth", sleep_every=0, sleep_updates=0, arbitrate=False):
    torch.manual_seed(seed)
    B = RoomBody(n, rows, drain=V2["drain"], seed=seed, device=DEV, regrow=V2["regrow"]); O = Organs(B.c, n, dev=DEV, depth=DEPTH)
    Mi = Mind(O, B.c, n, dev=DEV, value_arm=value_arm, arbitrate=arbitrate); Mi.load(BEING); Mi.wo.on = True
    B.w.store[:] = 0.75; place(B, info); every = torch.ones(n, dtype=torch.bool, device=DEV); O.reset(every); Mi.begin(every); x = O(B.raw())
    Q = [dict(fills=[0, 0], need_fills=[0, 0], deaths=0, src=np.zeros(5, int)) for _ in range(quarters)]; per = beats // quarters
    last_fill = np.zeros((n, 2), int); born = np.zeros(n, int); deaths_log = []                  # D24: judge-side cause of each death
    for t in range(beats):
        q = Q[min(t // per, quarters - 1)]; s0 = B.w.store.clone(); act = Mi.choose(x); q["src"] += np.bincount(Mi.src.cpu().numpy(), minlength=5)
        alive0 = ~B.dead.clone(); f = B.step(act)
        for c in (0, 1):
            m = f[:, c].bool(); q["fills"][c] += int(m.sum()); q["need_fills"][c] += int((m & (s0[:, c] < s0[:, 1 - c])).sum()); last_fill[m.cpu().numpy(), c] = t
        died = alive0 & B.dead; x2 = O(B.raw()); Mi.after(x, act, x2, died, B.dead & ~died); x = x2; q["deaths"] += int(died.sum())
        if died.any():
            st = s0.cpu().numpy(); ps = B.w.pos.cpu().numpy()
            for i in torch.nonzero(died)[:, 0].tolist():
                c = int(np.argmin(st[i])); side = -1 if ps[i, 0] < info["x0"] - 2.5 else (1 if ps[i, 0] > info["x0"] + 2.5 else 0)
                water_side = -1 if info["water_left"] else 1
                deaths_log.append(dict(t=t, lacked="water" if c == 0 else "food", other_store=float(st[i, 1 - c]), since_fill_lacked=int(t - max(last_fill[i, c], born[i])), age=int(t - born[i]),
                                       where=("stem/junction" if side == 0 else ("arm of the lacked spring" if side == (water_side if c == 0 else -water_side) else "arm of the other spring")), y=float(ps[i, 1])))
                born[i] = t + 1; last_fill[i] = t + 1
        if sleep_every and (t + 1) % sleep_every == 0:              # V3: rest while the mind replays its own ring (a world schedule like night)
            with torch.enable_grad():
                Mi.wo.learn(n=max(1, sleep_updates // max(1, getattr(Mi.wo, 'per_call', 1))))
        if B.dead.any():
            dd = B.dead.clone(); idx = torch.nonzero(dd)[:, 0]; B.w.dead[idx] = False; B.w.store[idx] = 0.75; place(B, info, which=idx); O.reset(dd); Mi.begin(dd); x = O(B.raw())
    for q in Q:
        q["src"] = q["src"].tolist()
    Q.append(dict(deaths_log=deaths_log))
    return Mi, Q


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--seeds", default="9101-9103"); ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--beats", type=int, default=12000); ap.add_argument("--mirrors", default="0,1"); ap.add_argument("--latent", action="store_true"); ap.add_argument("--value-arm", default="worth"); ap.add_argument("--far", action="store_true"); ap.add_argument("--sleep-every", type=int, default=0); ap.add_argument("--sleep-updates", type=int, default=0); ap.add_argument("--tag", default=""); ap.add_argument("--arbitrate", action="store_true"); a = ap.parse_args()
    s0, s1 = map(int, a.seeds.split("-")); assert not (7701 <= s0 <= 7712 or 7701 <= s1 <= 7712)
    V2["on"] = True; RoomBody.default_cues = True; res = []; t0 = time.time()
    for seed in range(s0, s1 + 1):
        for mirror in [bool(int(m)) for m in a.mirrors.split(",")]:
            rows, info = t_room(seed, mirror); sd = seed * 10 + mirror; lay = f"{seed}_{int(mirror)}"
            Mi, Q = many_life(rows, info, sd, a.n, a.beats, value_arm=a.value_arm, sleep_every=a.sleep_every, sleep_updates=a.sleep_updates, arbitrate=a.arbitrate); print(lay, "life", [(q["need_fills"], q["fills"], q["deaths"], q["src"][1]) for q in Q[:4]], round(time.time() - t0), flush=True)
            d = f"a1cache_many{'' if a.value_arm == 'worth' else '_' + a.value_arm}{a.tag}/{lay}"; os.makedirs(d, exist_ok=True); ck = os.path.join(d, "mind.pt"); Mi.save(ck)
            wo = Mi.wo; k = int(wo.bn); ring = dict(bx=wo.bx[:k].cpu(), bx2=wo.bx2[:k].cpu(), ba=wo.ba[:k].cpu(), bd=wo.bd[:k].cpu()); ring["ba2"] = wo.ba2[:k].cpu()
            dx, ins, M = Mi.net.dx, Mi.o.inside, Mi.M; L, R = side_sets(rows, info, seed)
            rec = dict(layout=lay, n=a.n, quarters=Q[:4], deaths_log=Q[4]["deaths_log"], earned=wo.earned(), arms={})
            with torch.no_grad():
                j = torch.randint(0, k, (4096,), device=DEV); rec["q_abs"] = float(wo.v(wo.bx[j]).abs().mean())
            rec["value_arm"] = a.value_arm; rec["pull_earned_end"] = None
            life_w = copy.deepcopy(wo); del Mi; torch.cuda.empty_cache()
            torch.save(ring, os.path.join(d, "life_ring.pt"))
            def _arb(Mi):
                Mi.arbitrate = a.arbitrate
            fo = fork(ck, life_w, rows, info, sd, L, R, prep=_arb); rec["arms"]["life"] = {q: fo[q] for q in ("value_pref_right", "first_match", "recover_share", "decided_by")}
            print(lay, "fork life", rec["arms"]["life"]["first_match"], rec["arms"]["life"]["recover_share"], "earned", round(rec["earned"], 2), "|Q|", round(rec["q_abs"], 2), round(time.time() - t0), flush=True)
            if a.far:                                               # V2: the far fork with the life's own value
                from tools.a1_relief import far_fork
                from tools.a1_sf import paths_of
                rec["arms"]["life"]["far"] = far_fork(ck, life_w, rows, info, sd, paths_of(rows, info), arbitrate=a.arbitrate); print(lay, "far life", rec["arms"]["life"]["far"], flush=True)
                rec["arms"]["life"]["stem"] = far_fork(ck, life_w, rows, info, sd, paths_of(rows, info), start="stem", arbitrate=a.arbitrate); print(lay, "stem life", rec["arms"]["life"]["stem"], flush=True)
            if a.latent:                                            # D21: relief-share arm on the many-body ring; junction fork and far fork
                from tools.a1_relief import relief_arm, shuffled, far_fork
                from tools.a1_sf import paths_of
                ring["bx2_true"], ring["bd_true"] = ring["bx2"], ring["bd"]; P = paths_of(rows, info)
                arms = {"life": life_w, "relief": relief_arm(ring, dx, ins, M, sd), "relief-shuf": relief_arm(shuffled(ring, ins, sd), dx, ins, M, sd)}
                for k_, arm in arms.items():
                    if k_ != "life":
                        fo = fork(ck, arm, rows, info, sd, L, R); rec["arms"][k_] = {q: fo[q] for q in ("value_pref_right", "first_match", "recover_share", "decided_by")}
                    rec["arms"][k_]["far"] = far_fork(ck, arm, rows, info, sd, P)
                    print(lay, k_, "junction", rec["arms"][k_]["value_pref_right"], rec["arms"][k_]["first_match"], rec["arms"][k_]["recover_share"], "| far", rec["arms"][k_]["far"], round(time.time() - t0), flush=True)
            res.append(rec); json.dump(dict(protocol=("atanor_os/A1_LATENT_PROTOCOL.md" if a.latent else "atanor_os/A1_MANY_PROTOCOL.md"), rows=res), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
