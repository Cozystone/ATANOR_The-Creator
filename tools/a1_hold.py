"""A1 hold diagnostic (atanor_os/A1_HOLD_DIAG.md, registered before running). Same phase A, arms and fork as A1 v2; at each clone's commit beat (M1)
the whole per-body state and the random state are copied, and three choices are compared from that one state: original, hold released for that beat,
value only (probe). Judge side only.
  PYTHONPATH=. python -m tools.a1_hold OUT.json --seeds 9101-9112"""
import argparse, copy, json, os, tempfile, time
import numpy as np, torch, torch.nn as nn
from tools.need_fork import t_room, DEV, DEPTH, K, SRC
from tools.a1_channel import V2, steps_of, sleep, junction, side_sets, trust_one, KEEP
from tools.a1_data import life
from atanor_os.emerge.room_body import RoomBody
from atanor_os.emerge.body_contract import Organs
from atanor_os.emerge.mind import Mind
from atanor_os.emerge.core import Worth
from atanor_os.emerge.channel_worth import ChannelWorth

ARMS = ["1f", "2", "3"]
RING = {"bx", "bx2", "br", "bd", "ba", "bp", "bi", "bn"}
T, AFTER = 400, 150


# ---- copying the whole per-body state ----
def _cl(v, n):
    if torch.is_tensor(v):
        return v.clone() if (v.dim() == 0 or v.shape[0] == n or v.numel() <= 4096) else None
    if isinstance(v, dict):
        return {k: _cl(q, n) for k, q in v.items()}
    if isinstance(v, (list, tuple)) and v and all(torch.is_tensor(q) for q in v):
        return type(v)(q.clone() for q in v)
    if isinstance(v, (int, float, bool, np.integer, np.floating)) or v is None:
        return v
    if isinstance(v, np.ndarray) or isinstance(v, np.random.Generator):
        return copy.deepcopy(v)
    return "__keep__"


def _objs(roots):
    out, seen, todo = [], set(), list(roots)
    while todo:
        o = todo.pop()
        if id(o) in seen:
            continue
        seen.add(id(o)); out.append(o)
        kids = list(o._modules.values()) if isinstance(o, nn.Module) else []
        kids += [v for v in vars(o).values() if (isinstance(v, nn.Module) or type(v).__module__.startswith("atanor_os")) and not isinstance(v, type)]
        if isinstance(o, nn.Module):
            kids = [k for k in kids if k is not None]
        todo += [k for k in kids if k is not None]
    return out


def snap(roots, n, x):
    st = []
    for o in _objs(roots):
        d = {}
        for k, v in vars(o).items():
            if k in ("_parameters", "_modules") or k in RING:
                continue
            if k == "_buffers":
                d[k] = {b: (t.clone() if t is not None and b not in RING and (t.dim() == 0 or t.numel() <= 4096 or t.shape[0] == n) else "__keep__") for b, t in v.items()}
                continue
            c = _cl(v, n)
            if c is not None and not (isinstance(c, str) and c == "__keep__"):
                d[k] = c
        st.append((o, d))
    return dict(objs=st, x=x.clone(), rng=torch.get_rng_state(), crng=torch.cuda.get_rng_state() if torch.cuda.is_available() else None)


def _rc(v):
    if torch.is_tensor(v):
        return v.clone()
    if isinstance(v, dict):
        return {k: _rc(q) for k, q in v.items()}
    if isinstance(v, (list, tuple)) and v and all(torch.is_tensor(q) for q in v):
        return type(v)(q.clone() for q in v)
    if isinstance(v, (np.ndarray, np.random.Generator)):
        return copy.deepcopy(v)
    return v


def restore(s):
    for o, d in s["objs"]:
        for k, v in d.items():
            if k == "_buffers":
                for b, t in v.items():
                    if not (isinstance(t, str) and t == "__keep__"):
                        o._buffers[b] = t.clone()
                continue
            if isinstance(v, dict) and isinstance(getattr(o, k, None), dict):
                cur = getattr(o, k)
                for kk, vv in v.items():
                    if not (isinstance(vv, str) and vv == "__keep__"):
                        cur[kk] = _rc(vv)
                continue
            setattr(o, k, _rc(v))
    torch.set_rng_state(s["rng"])
    if s["crng"] is not None:
        torch.cuda.set_rng_state(s["crng"])
    return s["x"].clone()


# ---- arms (A1 v2's 1f, 2, 3 only) ----
def build3(Mi, seed, updates, width):
    dx, ins, M = Mi.net.dx, Mi.o.inside, Mi.M; S = steps_of(Mi, False, seed); arms = {}
    torch.manual_seed(seed); arms["1f"] = Worth(dx, ins, lam=2.0, m=M, keep=KEEP, d=width).to(DEV); sleep(arms["1f"], S, updates)
    torch.manual_seed(seed); a3 = ChannelWorth(dx, ins, lam=2.0, m=M, keep=KEEP, d=width, mode="bedrock").to(DEV); sleep(a3, S, updates); arms["3"] = a3
    torch.manual_seed(seed); a2 = ChannelWorth(dx, ins, lam=2.0, m=M, keep=KEEP, d=width, mode="aux", P=a3.P).to(DEV); sleep(a2, S, updates); arms["2"] = a2
    return arms


# ---- the fork ----
def setup(ck, arm, rows, info, seed):
    torch.manual_seed(seed + 1)
    n = 2 * K; B = RoomBody(n, rows, drain=0.0, seed=seed, device=DEV, regrow=V2["regrow"]); O = Organs(B.c, n, dev=DEV, depth=DEPTH)
    Mi = Mind(O, B.c, n, dev=DEV); Mi.load(ck); Mi.wo = copy.deepcopy(arm); Mi.wo.on = True
    B.w.store[:K, 0], B.w.store[:K, 1] = 0.30, 0.95
    B.w.store[K:, 0], B.w.store[K:, 1] = 0.95, 0.30
    junction(B, info); every = torch.ones(n, dtype=torch.bool, device=DEV); O.reset(every); Mi.begin(every)
    return B, O, Mi, O(B.raw())


def beat(B, O, Mi, x, mask=None, mode=None):
    if mode == "release" and mask is not None:
        Mi.force = mask
    act = Mi.choose(x)
    if mode == "value" and mask is not None and Mi.lw_last is not None:
        v = Mi.lw_last.argmax(1); act = torch.where(mask, v, act); Mi.cur = torch.where(mask, v, Mi.cur)
    st0 = B.w.store.clone(); B.step(act); x2 = O(B.raw())
    Mi.prev = (x, act); Mi.cut = torch.zeros(B.n, dtype=torch.bool, device=DEV)
    return act, x2, (B.w.store - st0).cpu().numpy()


def zone(B, info):
    p = B.w.pos.cpu().numpy(); up = p[:, 1] < 3.0
    side = np.where(up & (p[:, 0] < info["x0"] - 2.5), -1, np.where(up & (p[:, 0] > info["x0"] + 2.5), 1, 0))
    return side, up & (np.abs(p[:, 0] - info["x0"]) < 1.5)


def margin(Mi, x, L, R, need_left):
    with torch.no_grad():
        sc = Mi.wo.v(x); pref = (sc[:, L].mean(1) - sc[:, R].mean(1)).cpu().numpy()
    return np.where(need_left, pref, -pref)                      # > 0: leans to the needed side


def fork_hold(ck, arm, rows, info, seed, L, R):
    n = 2 * K; need_left = np.array([info["water_left"]] * K + [not info["water_left"]] * K); need_c = np.array([0] * K + [1] * K)
    sideof = np.zeros(len(RoomBody(1, rows, drain=0.0, device=DEV).combos), int); sideof[L] = -1; sideof[R] = 1
    B, O, Mi, x = setup(ck, arm, rows, info, seed); roots = [B, O, Mi]
    snaps, entry, ent_side, zone_beats, rises, again_t, src_t, cur_t = [], np.full(n, -1), np.zeros(n, int), [], [], [], [], []
    m0 = margin(Mi, x, L, R, need_left)
    with trust_one(Mi.wo):
        for t in range(T):
            _, inz = zone(B, info); zone_beats.append(inz)
            snaps.append(snap(roots, n, x))
            act, x, rise = beat(B, O, Mi, x)
            again_t.append(Mi.again.cpu().numpy()); src_t.append(Mi.src.cpu().numpy()); cur_t.append(act.cpu().numpy()); rises.append(rise)
            side, _ = zone(B, info); new = (entry < 0) & (side != 0); entry[new] = t; ent_side[new] = side[new]
        rises = np.stack(rises); Z = np.stack(zone_beats); AG = np.stack(again_t); SR = np.stack(src_t); AC = np.stack(cur_t)
        clones = []
        for i in range(n):
            r = rises[:, i, need_c[i]]; any_r = rises[:, i].max(1) > 0.005; last = int(np.nonzero(any_r)[0][-1]) if any_r.any() else -1
            e = int(entry[i]); m1 = int(np.nonzero(Z[:e + 1, i])[0][-1]) if e >= 0 and Z[:e + 1, i].any() else (0 if e >= 0 else -1)
            rec_win = r[e:e + AFTER].sum() > 0.05 if e >= 0 else False
            clones.append(dict(need_left=bool(need_left[i]), m0_cmd=int(sideof[AC[0, i]]), m0_src=SRC[int(SR[0, i])], m0_again=bool(AG[0, i]), m0_margin=float(m0[i]),
                               entered=int(ent_side[i]) if e >= 0 else 0, entry_beat=e, right=bool(e >= 0 and (ent_side[i] == -1) == need_left[i]),
                               last_recovered=("water" if rises[last, i, 0] > rises[last, i, 1] else "food") if last >= 0 else None,
                               needed_recovered=bool(r.sum() > 0.05), needed_recovered_win=bool(rec_win),
                               m1=m1, m1_again=bool(AG[m1, i]) if m1 >= 0 else None, m1_src=SRC[int(SR[m1, i])] if m1 >= 0 else None,
                               m1_cmd=int(sideof[AC[m1, i]]) if m1 >= 0 else None))
        for i, c in enumerate(clones):                                  # value margin at M1 (from the copied state)
            if c["m1"] >= 0:
                c["m1_margin"] = float(margin(Mi, snaps[c["m1"]]["x"][i:i + 1].expand(n, -1), L, R, need_left)[i])
        buckets = {}
        for i, c in enumerate(clones):
            if c["m1"] >= 0:
                buckets.setdefault(c["m1"], []).append(i)
        replay = None
        for b, (t1, idx) in enumerate(sorted(buckets.items())):
            mask = torch.zeros(n, dtype=torch.bool, device=DEV); mask[idx] = True
            for mode in (["orig"] if b == 0 else []) + ["release", "value"]:
                x = restore(snaps[t1]); ent = {i: (0, -1) for i in idx}; rec = {i: 0.0 for i in idx}
                for t in range(t1, T):
                    _, x, rise = beat(B, O, Mi, x, mask if t == t1 else None, None if mode == "orig" else (mode if t == t1 else None))
                    side, _ = zone(B, info)
                    for i in idx:
                        if ent[i][1] < 0 and side[i] != 0:
                            ent[i] = (int(side[i]), t)
                        if ent[i][1] >= 0 and t < ent[i][1] + AFTER:
                            rec[i] += float(rise[i, need_c[i]])
                    if all(ent[i][1] >= 0 and (t >= ent[i][1] + AFTER or rec[i] > 0.05) for i in idx):
                        break
                for i in idx:
                    s, e = ent[i]
                    out = dict(entered=s, right=bool(e >= 0 and (s == -1) == need_left[i]), needed_recovered_win=bool(rec[i] > 0.05))
                    if mode == "orig":
                        clones[i]["replay"] = out
                    else:
                        clones[i][mode] = out
                if mode == "orig":
                    replay = float(np.mean([clones[i]["replay"]["entered"] == clones[i]["entered"] for i in idx]))
    return dict(clones=clones, replay_same_entry=replay, buckets=len(buckets))


def tally(clones):
    ent = [c for c in clones if c["entered"] != 0]
    out = dict(clones=len(clones), never_entered=len(clones) - len(ent), right=sum(c["right"] for c in ent),
               m0_again=sum(c["m0_again"] for c in clones), m1_hold=sum(c["m1_again"] is False for c in ent),
               m0_cmd_toward_need=sum((c["m0_cmd"] == (-1 if c["need_left"] else 1)) for c in clones),
               m0_margin_right=sum(c["m0_margin"] > 0 for c in clones))
    for mode in ("release", "value"):
        rows = [c for c in ent if mode in c]; d = {}
        for grp, sel in (("all", rows), ("m1_was_hold", [c for c in rows if c["m1_again"] is False]), ("m1_was_free", [c for c in rows if c["m1_again"]]),
                         ("margin_right", [c for c in rows if c.get("m1_margin", 0) > 0]), ("margin_wrong", [c for c in rows if c.get("m1_margin", 0) <= 0])):
            d[grp] = dict(n=len(sel), saved=sum((not c["right"]) and c[mode]["right"] for c in sel), spoiled=sum(c["right"] and not c[mode]["right"] for c in sel),
                          same_right=sum(c["right"] and c[mode]["right"] for c in sel), same_wrong=sum((not c["right"]) and not c[mode]["right"] for c in sel),
                          rec_saved=sum((not c["needed_recovered_win"]) and c[mode]["needed_recovered_win"] for c in sel),
                          rec_spoiled=sum(c["needed_recovered_win"] and not c[mode]["needed_recovered_win"] for c in sel))
        ms = sorted(abs(c.get("m1_margin", 0)) for c in rows)
        if ms:
            q = np.quantile(ms, [0.25, 0.5, 0.75]); bins = []
            for lo, hi in zip([-1, *q], [*q, 1e9]):
                sel = [c for c in rows if lo < abs(c.get("m1_margin", 0)) <= hi]
                bins.append(dict(upto=float(hi) if hi < 1e9 else None, n=len(sel), right=sum(c["right"] for c in sel), right_after=sum(c[mode]["right"] for c in sel)))
            d["by_margin_quartile"] = bins
        out[mode] = d
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--seeds", default="9101-9112"); ap.add_argument("--updates", type=int, default=2000)
    ap.add_argument("--width", type=int, default=128); a = ap.parse_args()
    s0, s1 = map(int, a.seeds.split("-")); assert not (7701 <= s0 <= 7712 or 7701 <= s1 <= 7712), "sealed seeds stay untouched"
    V2["on"] = True; rows_out = []; t0 = time.time()
    for seed in range(s0, s1 + 1):
        for mirror in (False, True):
            rows, info = t_room(seed, mirror); sd = seed * 10 + mirror
            tr, got, Mi = life(rows, info, sd, cap=V2["beats"], drain=V2["drain"], regrow=V2["regrow"], fixed=True)
            rec = dict(seed=seed, mirror=mirror, fills=[int(got[0]), int(got[1])], arms={})
            if (got > 0).all():
                arms = build3(Mi, sd, a.updates, a.width); L, R = side_sets(rows, info, seed)
                with tempfile.TemporaryDirectory() as d:
                    ck = os.path.join(d, "ck.pt"); Mi.save(ck)
                    for k in ARMS:
                        rec["arms"][k] = fork_hold(ck, arms[k], rows, info, sd, L, R)
            rows_out.append(rec)
            print(seed, mirror, rec["fills"], {k: (tally(v["clones"])["right"], v["replay_same_entry"], v["buckets"]) for k, v in rec["arms"].items()}, round(time.time() - t0), flush=True)
            summ = {k: tally([c for r in rows_out for c in r["arms"].get(k, {}).get("clones", [])]) for k in ARMS}
            json.dump(dict(protocol="atanor_os/A1_HOLD_DIAG.md", seeds=a.seeds, summary=summ, rows=rows_out), open(a.out, "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
