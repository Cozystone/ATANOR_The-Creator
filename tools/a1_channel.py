"""A1 channel-outcome test (protocol atanor_os/A1_CHANNEL_PROTOCOL.md, committed before running).
  PYTHONPATH=. python -m tools.a1_channel fork OUT.json --seeds 9001-9012 [--updates 2000] [--width 128]
  PYTHONPATH=. python -m tools.a1_channel life OUT.json                     (regression life for adoption)
The judge builds the rooms, knows which arm holds which spring and which combos go left/right; the arms see only contract feelings."""
import argparse, copy, json, math, os, tempfile, time
import numpy as np, torch
from tools.need_fork import t_room, phase_a, DEV, DEPTH, K, SRC, BEING
from atanor_os.emerge.room_body import RoomBody
from atanor_os.emerge.body_contract import Organs
from atanor_os.emerge.mind import Mind
from atanor_os.emerge.core import Worth
from atanor_os.emerge.channel_worth import ChannelWorth, params

KEEP = 16384
ARMS = ["1L", "1f", "2", "3", "1f-shuf", "2-shuf", "3-shuf"]


V2 = dict(on=False, regrow=1 / 120.0, drain=1 / 180.0, beats=12000)


def steps_of(Mi, shuffle, seed):
    if V2["on"]:                                                     # v2: every lived step, from the value organ's own ring (cuts excluded)
        wo = Mi.wo; k = int(wo.bn)
        X, X2, A, died = wo.bx[:k].clone(), wo.bx2[:k].clone(), wo.ba[:k].clone(), wo.bd[:k] > 0.5
    else:
        X, A, X2, OK = [torch.cat([b[k] for b in Mi.buf]) for k in range(4)]
        died = OK < 0.5
    if shuffle:
        g = torch.Generator().manual_seed(seed); p = torch.randperm(len(X), generator=g).to(X.device)
        ins = Mi.o.inside; X2 = X2.clone(); X2[:, ins] = X2[p][:, ins]; died = died[p]
    return X, A, X2, died


def fill(wo, S):
    X, A, X2, died = S
    wo.bn.zero_(); wo.bi.zero_()
    while int(wo.bn) < 4096 + len(X):
        wo.see(X, X2, died, a=A)


def sleep(wo, S, updates):
    fill(wo, S); t = time.time()
    for _ in range(updates // 4):
        wo.learn(n=4)
    return time.time() - t


def build_arms(Mi, seed, updates, width):
    dx, ins, M = Mi.net.dx, Mi.o.inside, Mi.M
    arms, secs = {}, {}
    for shuf in (False, True):
        S = steps_of(Mi, shuf, seed); tag = "-shuf" if shuf else ""
        if not shuf:
            w = Worth(dx, ins, lam=2.0, m=M, keep=KEEP).to(DEV); w.load_state_dict({k: v for k, v in Mi.wo.state_dict().items() if not k.startswith("b")}, strict=False)
            secs["1L"] = sleep(w, S, updates); arms["1L"] = w
        torch.manual_seed(seed); w = Worth(dx, ins, lam=2.0, m=M, keep=KEEP, d=width).to(DEV); secs["1f" + tag] = sleep(w, S, updates); arms["1f" + tag] = w
        torch.manual_seed(seed); a3 = ChannelWorth(dx, ins, lam=2.0, m=M, keep=KEEP, d=width, mode="bedrock").to(DEV); secs["3" + tag] = sleep(a3, S, updates); arms["3" + tag] = a3
        torch.manual_seed(seed); a2 = ChannelWorth(dx, ins, lam=2.0, m=M, keep=KEEP, d=width, mode="aux", P=a3.P).to(DEV); secs["2" + tag] = sleep(a2, S, updates); arms["2" + tag] = a2
    return arms, secs


def junction(B, info, idx=None):
    w = B.w; i = torch.arange(B.n, device=DEV) if idx is None else idx
    w.pos[i, 0] = info["x0"]; w.pos[i, 1] = 2.0; w.head[i] = -math.pi / 2


def side_sets(rows, info, seed):
    B = RoomBody(len(RoomBody(1, rows, drain=0.0, device=DEV).combos), rows, drain=0.0, seed=seed, device=DEV)
    junction(B, info); act = torch.arange(B.n, device=DEV)
    for _ in range(6):
        B.step(act)
    dx = (B.w.pos[:, 0] - info["x0"]).cpu().numpy()
    return np.nonzero(dx < -0.5)[0], np.nonzero(dx > 0.5)[0]


class trust_one:
    def __init__(self, wo): self.wo = wo
    def __enter__(self): self.old = self.wo.earned; self.wo.earned = lambda: 1.0
    def __exit__(self, *a): self.wo.earned = self.old


def fork(ck, arm, rows, info, seed, L, R, T=400, prep=None):
    torch.manual_seed(seed + 1)
    n = 2 * K; B = RoomBody(n, rows, drain=0.0, seed=seed, device=DEV, regrow=V2["regrow"] if V2["on"] else None); O = Organs(B.c, n, dev=DEV, depth=DEPTH)
    Mi = Mind(O, B.c, n, dev=DEV); Mi.load(ck); Mi.wo = copy.deepcopy(arm); Mi.wo.on = True
    if prep is not None:
        prep(Mi)                                                     # judge-only probes (e.g. D18: silence the pull)
    B.w.store[:K, 0], B.w.store[:K, 1] = 0.30, 0.95                  # store order: water, food
    B.w.store[K:, 0], B.w.store[K:, 1] = 0.95, 0.30
    junction(B, info); every = torch.ones(n, dtype=torch.bool, device=DEV); O.reset(every); Mi.begin(every)
    x = O(B.raw()); st0 = B.w.store.clone()
    need_left = np.array([info["water_left"]] * K + [not info["water_left"]] * K)
    out = {}
    with torch.no_grad():                                            # 1-2: what the arm knows, before any new reward
        sc = Mi.wo.v(x)
        pref = (sc[:, L].mean(1) - sc[:, R].mean(1)).cpu().numpy()  # > 0: prefers left
        out["value_pref_right"] = [float(((pref[:K] > 0) == need_left[:K]).mean()), float(((pref[K:] > 0) == need_left[K:]).mean())]
        out["value_pref_margin"] = [float(np.mean(pref[:K])), float(np.mean(pref[K:]))]
        if isinstance(Mi.wo, ChannelWorth):
            ps = Mi.wo.psi(x)                                       # [n, M, ni]; inside order in x: [blank, food, water]
            kw, kf = 2, 1
            pw = (ps[:K, L, kw].mean(1) - ps[:K, R, kw].mean(1)).cpu().numpy(); pf = (ps[K:, L, kf].mean(1) - ps[K:, R, kf].mean(1)).cpu().numpy()
            out["channel_pred_right"] = [float(((pw > 0) == need_left[:K]).mean()), float(((pf > 0) == need_left[K:]).mean())]
            if Mi.wo.mode == "bedrock":
                ex = Mi.wo.exact; Mi.wo.exact = False; sg = Mi.wo.v(x); Mi.wo.exact = ex
                pg = (sg[:, L].mean(1) - sg[:, R].mean(1)).cpu().numpy()
                out["value_pref_right_grad"] = [float(((pg[:K] > 0) == need_left[:K]).mean()), float(((pg[K:] > 0) == need_left[K:]).mean())]
    choice = np.zeros(n, int); src = np.zeros(5, int); t_lat = []
    with trust_one(Mi.wo):
        for t in range(T):
            t0 = time.time(); act = Mi.choose(x); t_lat.append(time.time() - t0)
            live = choice == 0; s = Mi.src.cpu().numpy(); src += np.bincount(s[live], minlength=5)
            B.step(act); x2 = O(B.raw())
            Mi.prev = (x, act); Mi.cut = torch.zeros(n, dtype=torch.bool, device=DEV)
            px = B.w.pos[:, 0].cpu().numpy(); py = B.w.pos[:, 1].cpu().numpy(); upper = py < 3.0
            choice = np.where((choice == 0) & upper & (px < info["x0"] - 2.5), -1, choice)
            choice = np.where((choice == 0) & upper & (px > info["x0"] + 2.5), 1, choice)
            x = x2
    chose = choice != 0; match = (choice == -1) == need_left
    rise = (B.w.store - st0).cpu().numpy()                           # [n, 2] water, food
    need_c = np.array([0] * K + [1] * K); other_c = 1 - need_c
    got = rise[np.arange(n), need_c]; oth = rise[np.arange(n), other_c]
    out.update(first_match=[float(match[:K][chose[:K]].mean()) if chose[:K].any() else 0.5, float(match[K:][chose[K:]].mean()) if chose[K:].any() else 0.5],
               choice_rate=float(chose.mean()), left_share=float((choice[chose] == -1).mean()) if chose.any() else None,
               recover_share=[float((got[:K] > 0.05).mean()), float((got[K:] > 0.05).mean())],
               rise_needed=float(got.mean()), rise_other=float(oth.mean()), decided_by=dict(zip(SRC, src.tolist())),
               choose_ms=float(1000 * np.median(t_lat)))
    out["value_knew_habit_won"] = [bool(out["value_pref_right"][i] > 0.5 and out["first_match"][i] < 0.5) for i in range(2)]
    return out


def run_fork(a):
    s0, s1 = map(int, a.seeds.split("-")); rows_out = []; t0 = time.time()
    for seed in range(s0, s1 + 1):
        for mirror in (False, True):
            rows, info = t_room(seed, mirror)
            if V2["on"]:
                from tools.a1_data import life, trips
                tr, got, Mi = life(rows, info, seed * 10 + mirror, cap=V2["beats"], drain=V2["drain"], regrow=V2["regrow"], fixed=True)
                tp = trips(tr, info); wl = -1 if info["water_left"] else 1
                ok_w = sum(t["side"] == wl and t["first_water"] is not None for t in tp); ok_f = sum(t["side"] == -wl and t["first_food"] is not None for t in tp)
                pa = dict(beats=len(tr), fills_water=int(got[0]), fills_food=int(got[1]), ok=bool((got > 0).all()), trips_water=ok_w, trips_food=ok_f,
                          deaths=int(sum(q[4] for q in tr)), ring=int(Mi.wo.bn))
                both = bool(ok_w >= 5 and ok_f >= 5)
            else:
                Mi, pa = phase_a(rows, info, seed * 10 + mirror, drain=1 / 360.0, rebirth=True); both = bool(pa["ok"])
            rec = dict(seed=seed, mirror=mirror, info=info, phase_a=pa, both_lived=both, arms={})
            if pa["ok"]:                                              # v2: every layout that lived both is forked; the >= 5 trips flag selects the primary set
                arms, secs = build_arms(Mi, seed * 10 + mirror, a.updates, a.width)
                L, R = side_sets(rows, info, seed)
                rec["sides"] = dict(left=len(L), right=len(R)); rec["sleep_s"] = secs
                with tempfile.TemporaryDirectory() as d:
                    ck = os.path.join(d, "ck.pt"); Mi.save(ck)
                    for k in ARMS:
                        rec["arms"][k] = fork(ck, arms[k], rows, info, seed * 10 + mirror, L, R)
                if not rows_out or "params" not in rows_out[0]:
                    rec["params"] = {k: params(v) - (params(v.Pt) if isinstance(v, ChannelWorth) else 0) - (params(v.vt) if not isinstance(v, ChannelWorth) else 0)
                                     - (params(v.qt) if isinstance(v, ChannelWorth) and v.mode == "aux" else 0) - (params(v.P) if k.startswith("2") else 0) for k, v in arms.items()}
            rows_out.append(rec)
            print(seed, mirror, pa["ok"], pa["fills_water"], pa["fills_food"],
                  {k: (v["value_pref_right"], v["first_match"], v["recover_share"]) for k, v in rec["arms"].items() if k in ("1f", "2", "3", "3-shuf")}, round(time.time() - t0), flush=True)
    ok = [r for r in rows_out if r["both_lived"]]
    def m(key, arm, pop=False):
        src = rows_out if pop else ok; vals = []
        for r in src:
            if r["arms"]:
                v = r["arms"][arm].get(key)
                if v is not None:
                    vals += list(v) if isinstance(v, list) else [v]
            else:
                vals += [0.5, 0.5] if key in ("value_pref_right", "first_match", "channel_pred_right") else [0.0, 0.0]
        return float(np.mean(vals)) if vals else None
    keys = ("channel_pred_right", "value_pref_right", "value_pref_right_grad", "first_match", "recover_share", "rise_needed", "rise_other")
    summ = {arm: {k: m(k, arm) for k in keys} for arm in ARMS}
    pop = {arm: {k: m(k, arm, True) for k in ("value_pref_right", "first_match", "recover_share")} for arm in ARMS}
    habit = {arm: int(sum(sum(r["arms"][arm]["value_knew_habit_won"]) for r in ok)) for arm in ARMS}
    s3, s3s, s1f = summ["3"], summ["3-shuf"], summ["1f"]
    gate = dict(G1=bool(s3["value_pref_right"] >= 0.75 and s3["value_pref_right"] - s3s["value_pref_right"] >= 0.20),
                G2=bool(s3["first_match"] >= 0.65 and s3["first_match"] >= s3s["first_match"] + 0.10),
                G3=bool(s3["recover_share"] >= s1f["recover_share"] + 0.10)) if ok else {}
    if gate:
        gate["passed"] = all(gate.values())
    res = dict(protocol="atanor_os/A1_CHANNEL_PROTOCOL.md", seeds=a.seeds, updates=a.updates, width=a.width, layouts_both_lived=len(ok), layouts=len(rows_out),
               summary_both_lived=summ, summary_population=pop, value_knew_habit_won=habit, gate=gate, rows=rows_out)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(dict(layouts_both_lived=len(ok), summary=summ, habit=habit, gate=gate), indent=1))


def run_life(a):
    from atanor_os.emerge.room_vary import make_room
    out = {}
    for arm in ("worth", "bedrock"):
        res = []
        for seed in range(7601, 7605):
            torch.manual_seed(seed)
            B = RoomBody(16, make_room(seed), drain=1 / 360.0, seed=seed, device=DEV); O = Organs(B.c, 16, dev=DEV, depth=DEPTH)
            Mi = Mind(O, B.c, 16, dev=DEV, value_arm=arm); Mi.load(BEING)
            if Mi.wo is not None:
                Mi.wo.on = True
            every = torch.ones(16, dtype=torch.bool, device=DEV); O.reset(every); Mi.begin(every)
            x = O(B.raw()); fills = torch.zeros(16, device=DEV); deaths = 0; whole = []
            for t in range(a.beats):
                act = Mi.choose(x); alive0 = ~B.dead.clone(); f = B.step(act); fills += f.sum(1).float()
                died = alive0 & B.dead; x2 = O(B.raw()); Mi.after(x, act, x2, died, B.dead & ~died); x = x2
                whole.append(float(Mi.wo.whole(x)[~B.dead].mean()) if (~B.dead).any() else 0.0)
                deaths += int(died.sum())
                if B.dead.any():
                    dd = B.dead.clone(); B.reset(dd); O.reset(dd); Mi.begin(dd); x = O(B.raw())
            res.append(dict(seed=seed, whole=float(np.mean(whole)), fills=float(fills.mean()), deaths=deaths))
            print(arm, res[-1], flush=True)
        out[arm] = dict(rows=res, whole=float(np.mean([r["whole"] for r in res])), fills=float(np.mean([r["fills"] for r in res])), deaths=float(np.mean([r["deaths"] for r in res])))
    json.dump(out, open(a.out, "w"), indent=1); print(json.dumps({k: {q: v[q] for q in ("whole", "fills", "deaths")} for k, v in out.items()}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["fork", "life"]); ap.add_argument("out")
    ap.add_argument("--seeds", default="9001-9012"); ap.add_argument("--updates", type=int, default=2000); ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--beats", type=int, default=1200); ap.add_argument("--v2", action="store_true")
    a = ap.parse_args()
    V2["on"] = a.v2
    run_fork(a) if a.stage == "fork" else run_life(a)
