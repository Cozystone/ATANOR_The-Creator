"""D1 (atanor_os/A1_OPTION_DIAG.md, registered before running): does the value know what HOLDING each combo does at the A1 fork?
Caches the phase-A mind, its lived Worth ring and the A1 arms (1f, 2, 3) to probe-work/a1cache/<seed>_<mirror>/ for later diagnostics.
  PYTHONPATH=. python -m tools.a1_option OUT.json --seeds 9101-9103"""
import argparse, copy, json, os, time
import numpy as np, torch
from tools.need_fork import t_room, DEV, DEPTH
from tools.a1_channel import V2, junction, side_sets, trust_one, KEEP
from tools.a1_data import life
from tools.a1_hold import build3, zone
from atanor_os.emerge.room_body import RoomBody
from atanor_os.emerge.body_contract import Organs
from atanor_os.emerge.mind import Mind
from atanor_os.emerge.core import Worth
from atanor_os.emerge.channel_worth import ChannelWorth

CACHE = "a1cache"
RING = ("bx", "bx2", "br", "bd", "ba", "bp")


def _small(sd):
    return {k: v for k, v in sd.items() if k.split(".")[-1] not in RING}


def save_cache(d, Mi, arms):
    os.makedirs(d, exist_ok=True); Mi.save(os.path.join(d, "mind.pt")); wo = Mi.wo; k = int(wo.bn)
    torch.save(dict(bx=wo.bx[:k].cpu(), bx2=wo.bx2[:k].cpu(), ba=wo.ba[:k].cpu(), bd=wo.bd[:k].cpu(), br=wo.br[:k].cpu(), bi=int(wo.bi), bn=k), os.path.join(d, "life_ring.pt"))
    torch.save({a: _small(m.state_dict()) for a, m in arms.items()}, os.path.join(d, "arms.pt"))


def load_arms(d, dx, ins, M, width=128):
    sd = torch.load(os.path.join(d, "arms.pt"), map_location=DEV, weights_only=False); arms = {}
    arms["1f"] = Worth(dx, ins, lam=2.0, m=M, keep=KEEP, d=width).to(DEV); arms["1f"].load_state_dict(sd["1f"], strict=False)
    a3 = ChannelWorth(dx, ins, lam=2.0, m=M, keep=KEEP, d=width, mode="bedrock").to(DEV); a3.load_state_dict(sd["3"], strict=False); arms["3"] = a3
    a2 = ChannelWorth(dx, ins, lam=2.0, m=M, keep=KEEP, d=width, mode="aux", P=a3.P).to(DEV); a2.load_state_dict(sd["2"], strict=False); arms["2"] = a2
    return arms


def rank(v):
    r = np.empty(len(v)); r[np.argsort(v, kind="stable")] = np.arange(len(v)); return r


def spearman(a, b):
    a, b = rank(np.asarray(a, float)), rank(np.asarray(b, float))
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def truth_run(ck, arm, rows, info, seed, P=1, H=12, T=150):
    torch.manual_seed(seed + 7)
    M = len(RoomBody(1, rows, drain=0.0, device=DEV).combos); n = M * P * 2
    B = RoomBody(n, rows, drain=0.0, seed=seed, device=DEV, regrow=V2["regrow"]); O = Organs(B.c, n, dev=DEV, depth=DEPTH)
    Mi = Mind(O, B.c, n, dev=DEV, worth=0); Mi.load(ck); Mi.wo = copy.deepcopy(arm); Mi.wo.on = True     # worth=0: no 300k ring is made (GPU is shared)
    need = np.repeat([0, 1], M * P); combo = np.tile(np.arange(M), 2 * P)
    th = torch.tensor(need == 0, device=DEV)
    B.w.store[th, 0], B.w.store[th, 1] = 0.30, 0.95; B.w.store[~th, 0], B.w.store[~th, 1] = 0.95, 0.30
    junction(B, info); every = torch.ones(n, dtype=torch.bool, device=DEV); O.reset(every); Mi.begin(every); x = O(B.raw())
    with torch.no_grad():
        q = Mi.wo.v(x).cpu().numpy()
    Q = [q[need == g].mean(0) for g in (0, 1)]                       # all clones of one need share the same start state
    st0 = B.w.store.clone(); entry = np.zeros(n, int); act_forced = torch.tensor(combo, device=DEV)
    for t in range(H):                                              # the combo is held (the judge drives it; nothing is learned)
        B.step(act_forced); side, _ = zone(B, info); entry = np.where((entry == 0) & (side != 0), side, entry)
    Mi.begin(every); x = O(B.raw())
    with trust_one(Mi.wo):
        for t in range(H, T):
            act = Mi.choose(x); B.step(act); x2 = O(B.raw()); Mi.prev = (x, act); Mi.cut = torch.zeros(n, dtype=torch.bool, device=DEV); x = x2
            side, _ = zone(B, info); entry = np.where((entry == 0) & (side != 0), side, entry)
    rise = (B.w.store - st0).cpu().numpy(); got = rise[np.arange(n), need] > 0.05
    out = {}
    legs = np.array([B.combos[a][0] for a in range(M)])
    need_left = [info["water_left"], not info["water_left"]]
    for g in (0, 1):
        sel = need == g; rec = np.array([got[sel & (combo == a)].mean() for a in range(M)])
        ent = np.array([np.mean(entry[sel & (combo == a)] == (-1 if need_left[g] else 1)) for a in range(M)])
        grand = rec.mean(); ss = ((rec - grand) ** 2).sum()
        leg_eta = float(sum(((rec[legs == l].mean() - grand) ** 2) * (legs == l).sum() for l in np.unique(legs)) / ss) if ss > 0 else None
        top = np.argsort(-Q[g])[:5]; rng = np.random.default_rng(seed + g)
        out["thirsty" if g == 0 else "hungry"] = dict(
            spearman_Q_vs_recovery=spearman(Q[g], rec), spearman_Q_vs_right_arm=spearman(Q[g], ent),
            truth_spread=float(rec.max() - rec.min()), truth_mean=float(grand), truth_best=float(rec.max()), share_combos_recovering=float((rec > 0.5).mean()),
            leg_option_explains=leg_eta, value_top5_recovery=float(rec[top].mean()), random5_recovery=float(np.mean([rec[rng.choice(M, 5, replace=False)].mean() for _ in range(200)])),
            value_spread=float(Q[g].max() - Q[g].min()), value_top5_right_arm=float(ent[top].mean()), right_arm_mean=float(ent.mean()))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--seeds", default="9101-9103"); ap.add_argument("--updates", type=int, default=2000)
    a = ap.parse_args(); s0, s1 = map(int, a.seeds.split("-")); assert not (7701 <= s0 <= 7712 or 7701 <= s1 <= 7712), "sealed seeds stay untouched"
    V2["on"] = True; res = []; t0 = time.time()
    for seed in range(s0, s1 + 1):
        for mirror in (False, True):
            rows, info = t_room(seed, mirror); sd = seed * 10 + mirror; d = os.path.join(CACHE, f"{seed}_{int(mirror)}")
            tr, got, Mi = life(rows, info, sd, cap=V2["beats"], drain=V2["drain"], regrow=V2["regrow"], fixed=True)
            rec = dict(seed=seed, mirror=mirror, fills=[int(got[0]), int(got[1])], arms={})
            if (got > 0).all():
                arms = build3(Mi, sd, a.updates, 128); save_cache(d, Mi, arms)
                lw = Worth(Mi.net.dx, Mi.o.inside, lam=2.0, m=Mi.M, keep=KEEP).to(DEV); lw.load_state_dict(_small(Mi.wo.state_dict()), strict=False)
                arms["life"] = lw                                       # the being's own lived value (weights, trust stats; ring left out)
                del Mi; torch.cuda.empty_cache()
                ck = os.path.join(d, "mind.pt")
                for k in ("life", "1f", "2", "3"):
                    rec["arms"][k] = truth_run(ck, arms[k], rows, info, sd)
            res.append(rec)
            print(seed, mirror, rec["fills"], {k: {g: (None if v[g]["spearman_Q_vs_recovery"] is None else round(v[g]["spearman_Q_vs_recovery"], 2), round(v[g]["truth_spread"], 2),
                                                       round(v[g]["value_top5_recovery"], 2), round(v[g]["random5_recovery"], 2)) for g in v} for k, v in rec["arms"].items()}, round(time.time() - t0), flush=True)
            json.dump(dict(protocol="atanor_os/A1_OPTION_DIAG.md", rows=res), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
