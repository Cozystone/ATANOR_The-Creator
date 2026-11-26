"""D39 (atanor_os/VALUE_CHOICE_PROTOCOL.md, registered before running): on what share of beats is the gesture done the one its value rates best,
and where is it lost - the gate (hold), the size of the value among the score's terms, the temperature, the value's earned share, or the value itself.
A readout only: the being (S3-end, one mind with 64 bodies) lives and learns as always; Mind.choose is not touched (record_parts only stores the terms).
  PYTHONPATH=. python -m tools.d39_choice OUT.json --lay 9101_0 --being s3_a [--beats 10000]"""
import argparse, json, time
from collections import defaultdict
import torch
from tools.need_fork import t_room, place, DEV
from tools.a1_channel import V2
from tools.l1_life import make
from tools.s4_kin import being, rebirth
from atanor_os.emerge.room_body import RoomBody
from atanor_os.emerge.mind import LOOSE

TERMS = ("pull", "harm", "futile", "value", "going")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--lay", default="9101_0"); ap.add_argument("--being", required=True)
    ap.add_argument("--beats", type=int, default=10000); ap.add_argument("--gate-usual", type=int, default=0); ap.add_argument("--d41", type=int, default=0); ap.add_argument("--d42", type=float, default=0.0); ap.add_argument("--k", type=int, default=20)   # D40 probe arm
    a = ap.parse_args(); V2["on"] = True; RoomBody.default_cues = True; t0 = time.time(); n = 64
    seed, mirror = map(int, a.lay.split("_")); assert not 7701 <= seed <= 7712; rows, info = t_room(seed, bool(mirror)); sd = seed * 10 + mirror + 39
    torch.manual_seed(sd)
    B = make(rows, n, sd, V2["drain"]); O, Mi = being(B, n, a.being); Mi.record_parts = True; Mi.gate_usual = bool(a.gate_usual); M = Mi.M
    every = torch.ones(n, dtype=torch.bool, device=DEV); B.w.store[:] = 0.75; place(B, info); O.reset(every); Mi.begin(every); Mi.prev = None
    x = O(B.raw()); halves = [defaultdict(float), defaultdict(float)]; deaths = 0
    EV = []; D41 = defaultdict(float); PEND, DONE = [], []                                    # D41: pending pick events (t, mask, category, lack third, whole at pick, alive)
    for t in range(a.beats):
        A = halves[t * 2 // a.beats]

        def add(k, v):
            A[k] += float(v)
        held = Mi.cur.clone(); act = Mi.choose(x)
        P, lw, again, src, cur = Mi.parts, Mi.lw_last, Mi.again, Mi.src, Mi.cur
        sc, strong = P["total"], P["strong"]; lack = (1 - x[:, O.inside]).clamp(min=0).max(1).values
        add("n", n); add("open", again.sum()); add("strong_all", strong.sum())
        for k in range(5):
            add(f"src{k}", (src == k).sum())
        cut = Mi.cut; pre = Mi.pre_last
        add("open_cut", (again & cut).sum()); add("open_pre", (again & ~cut & pre).sum()); add("open_novel", (again & ~cut & ~pre).sum())
        if lw is None:
            add("no_value", n)
        else:
            flat = lw.std(1) < 1e-9; ok = ~flat; add("flat", flat.sum()); add("ok", ok.sum())      # a value that rates every gesture the same has no best: counted apart
            vb = lw.argmax(1); hit = (act == vb) & ok; pick_hit = (cur == vb) & ok; hold = ~again & ok; so = again & strong & ok; wk = again & ~strong & ok
            add("open_ok", (again & ok).sum())
            for k in range(5):
                add(f"oksrc{k}", ((src == k) & ok).sum())
            add("hit", hit.sum()); add("pick_hit", pick_hit.sum()); add("hit_top3", ((lw.topk(3, 1).indices == act[:, None]).any(1) & ok).sum())
            for k in range(5):
                add(f"hit_src{k}", (hit & (src == k)).sum())
            for j, (lo, hi) in enumerate(((0, 0.3), (0.3, 0.6), (0.6, 2))):
                m = (lack >= lo) & (lack < hi) & ok; add(f"lack{j}_n", m.sum()); add(f"lack{j}_hit", (hit & m).sum()); add(f"lack{j}_pick_hit", (pick_hit & m).sum())
                add(f"lack{j}_open", (again & m).sum()); add(f"lack{j}_vspread", lw.std(1)[m].sum())
            rank = (lw > lw.gather(1, held[:, None])).sum(1)                       # where the held gesture stands in the value (0 = its best)
            add("hold_n", hold.sum()); add("hold_best", (hold & (held == vb)).sum()); add("hold_rank", rank[hold].float().sum())
            add("v_selfmass", torch.softmax(lw / LOOSE, 1).max(1).values[ok].sum()); add("v_spread", lw.std(1)[ok].sum()); add("earned", Mi.wo.earned() * n)
            for k in TERMS:
                if k in P:
                    add(f"all_spread_{k}", P[k].std(1).sum())
            if bool(so.any()):
                p = torch.softmax(sc / LOOSE, 1); t2 = sc.topk(2, 1).values; mg = (t2[:, 0] - t2[:, 1]) / LOOSE
                add("so_n", so.sum()); add("so_mass", p.gather(1, vb[:, None])[:, 0][so].sum()); add("so_agree", ((sc.argmax(1) == vb) & so).sum())
                add("so_decisive", ((sc.argmax(1) != (sc - lw).argmax(1)) & so).sum()); add("so_pick_hit", (pick_hit & so).sum())
                add("so_margin", mg[so].sum()); add("so_margin3", ((mg > 3) & so).sum()); add("so_spread_total", sc.std(1)[so].sum())
                for k in TERMS:
                    if k in P:
                        add(f"so_spread_{k}", P[k].std(1)[so].sum())
            if bool(wk.any()):
                add("wk_n", wk.sum()); add("wk_pick_hit", (pick_hit & wk).sum()); add("wk_not_tried", (wk & (src != 4)).sum()); add("wk_value_drawn", (wk & (src == 1)).sum())
        if a.d41 and lw is not None and "going" in P:
            gb = P["going"].argmax(1); vb2 = lw.argmax(1); pk = again & strong & (lw.std(1) > 1e-9) & (P["going"].std(1) > 1e-9)
            dis = pk & (gb != vb2); agr = pk & (gb == vb2)
            cat = torch.full((n,), -1, device=DEV, dtype=torch.long)
            if a.d42:                                                          # D42: the judge replaces the pick by the value's or the going's best, at random
                swap = dis & (torch.rand(n, device=DEV) < a.d42); coin = torch.rand(n, device=DEV) < 0.5; tgt = torch.where(coin, vb2, gb)
                tried = Mi.src == 4; swap = swap & ~tried
                act = torch.where(swap, tgt, act); Mi.cur = torch.where(swap, tgt, Mi.cur); cur = Mi.cur
                vbo, gbo = lw.argmax(1), P["going"].argmax(1); G_ = P["going"]
                mv = lw.gather(1, vbo[:, None])[:, 0] - lw.gather(1, gbo[:, None])[:, 0]; mg = G_.gather(1, gbo[:, None])[:, 0] - G_.gather(1, vbo[:, None])[:, 0]
                for i in torch.nonzero(swap)[:, 0].tolist():
                    PEND.append([t, i, int(coin[i]), float(mv[i]), float(mg[i]), float(lack[i]), float(Mi.wo.whole(x[i:i + 1])[0]), True])
                dis = swap; agr = torch.zeros_like(agr); vb2 = torch.where(coin, vb2, torch.full_like(vb2, -9)); gb = torch.where(coin, torch.full_like(gb, -9), gb)
            cat = torch.where(dis & (cur == vb2), torch.full_like(cat, 0), cat); cat = torch.where(dis & (cur == gb), torch.full_like(cat, 1), cat)
            cat = torch.where(dis & (cur != vb2) & (cur != gb), torch.full_like(cat, 2), cat); cat = torch.where(agr, torch.full_like(cat, 3), cat)
            third = torch.bucketize(lack, torch.tensor([0.3, 0.6], device=DEV))
            if bool((cat >= 0).any()):
                EV.append([t, cat >= 0, cat, third, Mi.wo.whole(x).clone(), torch.ones(n, dtype=torch.bool, device=DEV)])
            D41["dis"] += float(dis.sum()); D41["agr"] += float(agr.sum()); D41["picks"] += float(pk.sum())
        al = ~B.dead.clone(); s0 = B.w.store.clone(); f = B.step(act); x2 = O(B.raw()); died = al & B.dead; deaths += int(died.sum())
        add("deaths", died.sum()); add("need_fills", ((f[:, 0].bool() & (s0[:, 0] < s0[:, 1])) | (f[:, 1].bool() & (s0[:, 1] < s0[:, 0]))).sum())
        Mi.after(x, act, x2, died, B.dead & ~died); x = x2
        if PEND:
            dl = died.tolist(); wn = Mi.wo.whole(x2).tolist()
            for e in PEND:
                e[7] = e[7] and not dl[e[1]]
            while PEND and PEND[0][0] + a.k <= t:
                e = PEND.pop(0); DONE.append(dict(arm="value" if e[2] else "going", mv=round(e[3], 5), mg=round(e[4], 5), lack=round(e[5], 3), dw=round(wn[e[1]] - e[6], 5) if e[7] else None, died=not e[7]))
        if a.d41 and EV:
            for e in EV:
                e[5] &= ~died
            w_now = Mi.wo.whole(x2)
            while EV and EV[0][0] + a.k <= t:
                _, m, cat, th, w0, alive = EV.pop(0)
                for c in range(4):
                    for j in range(3):
                        mm = m & (cat == c) & (th == j); D41[f"n_{c}_{j}"] += float(mm.sum()); D41[f"die_{c}_{j}"] += float((mm & ~alive).sum())
                        ok = mm & alive; D41[f"ok_{c}_{j}"] += float(ok.sum()); D41[f"dw_{c}_{j}"] += float((w_now - w0)[ok].sum())
        if B.dead.any():
            rebirth(B, O, Mi, info); B.felt[:] = 0; x = O(B.raw())
        if (t + 1) % 1000 == 0:
            Mi.going.learn()
            with torch.enable_grad():
                Mi.wo.learn(n=max(1, 2500 // max(1, getattr(Mi.wo, "per_call", 1))))

    def read(A):
        g = lambda k: A.get(k, 0.0); d = lambda p, q: round(g(p) / g(q), 4) if g(q) else None; nv = g("ok")
        r = dict(deaths_per_1000=round(1000 * g("deaths") / g("n"), 3), need_fills_per_1000=round(1000 * g("need_fills") / g("n"), 2), gestures=M, chance=round(1 / M, 4), value_ready=round((g("n") - g("no_value")) / g("n"), 3), value_flat=d("flat", "n"), value_has_a_best=d("ok", "n"),
                 M1_done_is_value_best=round(g("hit") / nv, 4) if nv else None, M1_pick_is_value_best=round(g("pick_hit") / nv, 4) if nv else None,
                 done_in_value_top3=round(g("hit_top3") / nv, 4) if nv else None,
                 decided_by=dict(strong=d("src0", "n"), value=d("src1", "n"), random=d("src2", "n"), hold=d("src3", "n"), tried=d("src4", "n")),
                 value_best_done_when_decided_by=dict(strong=d("hit_src0", "oksrc0"), value=d("hit_src1", "oksrc1"), random=d("hit_src2", "oksrc2"), hold=d("hit_src3", "oksrc3"), tried=d("hit_src4", "oksrc4")),
                 by_lack=[dict(share=d(f"lack{j}_n", "n"), done_best=d(f"lack{j}_hit", f"lack{j}_n"), pick_best=d(f"lack{j}_pick_hit", f"lack{j}_n"), gate_open=d(f"lack{j}_open", f"lack{j}_n"),
                               value_spread=d(f"lack{j}_vspread", f"lack{j}_n")) for j in range(3)],
                 gate=dict(open=d("open", "n"), by_pre=d("open_pre", "n"), by_cut=d("open_cut", "n"), by_novel=d("open_novel", "n"), mean_hold_beats=round(g("n") / g("open"), 1) if g("open") else None,
                           strong_share_all_beats=d("strong_all", "n"), held_is_value_best=d("hold_best", "hold_n"), held_mean_rank_in_value=d("hold_rank", "hold_n")),
                 strong_open=dict(share=d("so_n", "n"), pick_is_value_best=d("so_pick_hit", "so_n"), softmax_mass_on_value_best=d("so_mass", "so_n"), score_best_is_value_best=d("so_agree", "so_n"),
                                  value_decisive=d("so_decisive", "so_n"), top_margin_in_temperatures=d("so_margin", "so_n"), margin_over_3=d("so_margin3", "so_n"),
                                  spread=dict(total=d("so_spread_total", "so_n"), **{k: d(f"so_spread_{k}", "so_n") for k in TERMS})),
                 weak_open=dict(share=d("wk_n", "n"), pick_is_value_best=d("wk_pick_hit", "wk_n"), value_drawn=d("wk_value_drawn", "wk_not_tried")),
                 value_itself=dict(own_softmax_mass_on_best=round(g("v_selfmass") / nv, 4) if nv else None, spread=round(g("v_spread") / nv, 5) if nv else None,
                                   earned_trust=round(g("earned") / nv, 3) if nv else None, spread_all_beats={k: d(f"all_spread_{k}", "n") for k in TERMS}))
        if nv:
            ph, ps, pw = g("hold_n") / nv, g("so_n") / nv, g("wk_n") / nv; qh, qs, qw = (g("hold_best") / g("hold_n") if g("hold_n") else 0), (g("so_pick_hit") / g("so_n") if g("so_n") else 0), (g("wk_pick_hit") / g("wk_n") if g("wk_n") else 0)
            r["arithmetic"] = dict(P_hold=round(ph, 4), best_given_hold=round(qh, 4), P_strong=round(ps, 4), best_given_strong=round(qs, 4), P_weak=round(pw, 4), best_given_weak=round(qw, 4),
                                   pick_total=round(ph * qh + ps * qs + pw * qw, 4), if_hold_perfect=round(ph + ps * qs + pw * qw, 4), if_strong_perfect=round(ph * qh + ps + pw * qw, 4),
                                   if_weak_perfect=round(ph * qh + ps * qs + pw, 4))
        return r

    tot = defaultdict(float)
    for A in halves:
        for k, v in A.items():
            tot[k] += v
    names = ("value_best", "going_best", "other", "agree"); d41 = None
    if a.d41:
        d41 = dict(picks=D41["picks"], disagree=D41["dis"], agree=D41["agr"], k=a.k, by_third=[{nm: dict(n=int(D41[f"n_{c}_{j}"]), died=round(D41[f"die_{c}_{j}"] / max(D41[f"n_{c}_{j}"], 1), 4),
                   dwhole=round(D41[f"dw_{c}_{j}"] / max(D41[f"ok_{c}_{j}"], 1), 5)) for c, nm in enumerate(names)} for j in range(3)])
    r = dict(d43_events=DONE if a.d42 else None, d41=d41, layout=a.lay, being=a.being, gate_usual=a.gate_usual, sd_usual=getattr(Mi, "sd_usual", None), beats=a.beats, bodies=n, temperature=LOOSE, deaths=deaths, all=read(tot), first_half=read(halves[0]), last_half=read(halves[1]), seconds=round(time.time() - t0))
    print(json.dumps(r), flush=True)
    try:
        old = json.load(open(a.out))["rows"]
    except (OSError, ValueError, KeyError):
        old = []
    json.dump(dict(protocol="atanor_os/VALUE_CHOICE_PROTOCOL.md (D39)", rows=old + [r]), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
