"""Judge-side walk diagnostics (WORLD_A_PROTOCOL.md C1-C7, founder 2026-11-27 "circling like a stereotypy").
Reads the body, mind and going organ after each beat; changes nothing the being does (Mind.record_parts only copies score parts).
Circling = over the last 15 beats the body moved > 3 cells yet ended < 1 cell from where it was. Every measure is split circling vs else.
Used by tools/a2_moving_food.py --circ 1."""
import math
import numpy as np
import torch


class WalkDiag:
    def __init__(self, n, M):
        self.n, self.M = n, M; self.ring = None
        self.C = dict(beats=0, circ=0, src=np.zeros((2, 5)), goal=np.zeros(2), hungry=np.zeros(2), acts=np.zeros((2, M)), dist=np.zeros((2, 3)),
                      atgoal=np.zeros(2), gbeats=np.zeros(2), th=[[], []], dd=[[], []], de=[[], []], pr=[], fe=[], top={}, topn=np.zeros(2),
                      spread={}, tch=np.zeros(2), tn_=np.zeros(2), dth=[[], []], blk=np.zeros(2), bn=np.zeros(2), fwd=[[], []],
                      bout_d0=np.full(n, np.nan), bouts=[])
        self.e_prev = None; self.tg_prev = None; self.prevc = np.zeros(n, bool)

    @torch.no_grad()
    def step(self, w, B, Mi, G, O, x2, act, died):
        C, n, dev = self.C, self.n, w.pos.device
        if self.ring is None:
            self.ring = w.pos[:, None].repeat(1, 16, 1).clone(); Mi.record_parts = True
        self.ring = torch.cat([self.ring[:, 1:], w.pos[:, None]], 1); seg = (self.ring[:, 1:] - self.ring[:, :-1]).norm(dim=2)
        circ = ((seg.sum(1) > 3.0) & ((self.ring[:, -1] - self.ring[:, 0]).norm(dim=1) < 1.0) & ~B.dead & (seg < 2).all(1)).cpu().numpy()
        alv = (~B.dead).cpu().numpy(); src = Mi.src.cpu().numpy()
        gp = getattr(Mi, "goal_prev", None)
        gl = (gp if gp is not None else torch.full((n,), -1, device=dev)).cpu().numpy() >= 0
        hu = (w.store.min(1).values < 0.7).cpu().numpy(); ac = act.cpu().numpy()
        split = lambda extra=True: [alv & (circ == bool(c)) & extra for c in (0, 1)]
        for c, m in enumerate(split()):                                                    # C1: who decides, goal, hunger, gestures
            C["src"][c] += np.bincount(src[m], minlength=5); C["goal"][c] += (gl & m).sum(); C["hungry"][c] += (hu & m).sum(); C["acts"][c] += np.bincount(ac[m], minlength=self.M)
        db = np.digitize((w.pos - (w.src[:, 1].float() + 0.5)).norm(dim=1).cpu().numpy(), [2.0, 5.0])
        for c, m in enumerate(split(hu)):                                                  # C1: hungry beats by distance to food
            C["dist"][c] += np.bincount(db[m], minlength=3)
        if gp is not None:                                                                 # C1b: at its own goal unit
            ug = (G.unit(G.place(x2)) == gp).cpu().numpy() & (gp >= 0).cpu().numpy()
            for c, m in enumerate(split((gp >= 0).cpu().numpy())):
                C["gbeats"][c] += m.sum(); C["atgoal"][c] += (m & ug).sum()
        e = ((O.path.h - w.head + math.pi) % (2 * math.pi) - math.pi).cpu().numpy()      # C2: heading drift per beat
        if self.e_prev is not None:
            de = np.abs((e - self.e_prev + np.pi) % (2 * np.pi) - np.pi)
            for c, m in enumerate(split()):
                C["de"][c] += de[m].tolist()
        self.e_prev = e
        th = getattr(G, "last_th", None)
        if th is not None and th.shape[0] == n:
            ok = G.last_ok.cpu().numpy(); thn = np.abs(th.cpu().numpy()); dn = G.last_d.cpu().numpy()
            for c, m in enumerate(split(ok)):                                              # C2: target angle and distance
                C["th"][c] += thn[m].tolist(); C["dd"][c] += dn[m].tolist()
            tg = G.last_tgt.cpu().numpy(); thv = th.cpu().numpy()
            if self.tg_prev is not None:                                                   # C5: target jumps
                for c, m in enumerate(split(ok & self.ok_prev)):
                    C["tn_"][c] += m.sum(); C["tch"][c] += (m & (tg != self.tg_prev)).sum()
                    C["dth"][c] += np.abs((thv - self.th_prev + np.pi) % (2 * np.pi) - np.pi)[m].tolist()
            self.tg_prev, self.th_prev, self.ok_prev = tg, thv, ok
            _, wd = w.cast((w.head + th)[:, None], step=0.05); wd = wd[:, 0].cpu().numpy()  # C6: line to the target blocked by a wall
            for c, m in enumerate(split(ok)):
                C["bn"][c] += m.sum(); C["blk"][c] += (m & (wd < dn)).sum()
            st = circ & ~self.prevc & ok; C["bout_d0"][st] = dn[st]                        # C7: own distance at bout start and end
            en = ~circ & self.prevc & ~np.isnan(C["bout_d0"])
            for i in np.nonzero(en)[0]:
                C["bouts"].append((float(C["bout_d0"][i]), float(dn[i])))
            C["bout_d0"][en] = np.nan; self.prevc = circ.copy()
            ph = G.last_ph.gather(1, act[:, None, None].expand(-1, 1, 3))[:, 0].cpu().numpy()
            for c, m in enumerate(split(ok)):                                              # C6: chosen gesture's foreseen forward motion
                C["fwd"][c] += ph[m, 0].tolist()
            m = alv & ok & ~died.cpu().numpy(); C["pr"] += ph[m].tolist(); C["fe"] += B.felt.cpu().numpy()[m].tolist()   # C3: foresight vs felt
            G.last_th = None
        pp, ag = getattr(Mi, "parts", None), getattr(Mi, "again", None)
        if pp is not None and ag is not None:                                              # C4: which score part chose
            ch = Mi.cur.cpu().numpy()
            for c, m in enumerate(split(ag.cpu().numpy())):
                C["topn"][c] += m.sum()
                for k, v in pp.items():
                    if k in ("total", "strong") or v.dim() != 2:
                        continue
                    sd = v.std(1).cpu().numpy(); tm = (v.argmax(1).cpu().numpy() == ch) & (sd > 1e-6)
                    C["top"].setdefault(k, np.zeros(2)); C["top"][k][c] += (tm & m).sum()
                    C["spread"].setdefault(k, np.zeros(2)); C["spread"][k][c] += sd[m].sum()
        C["beats"] += int(alv.sum()); C["circ"] += int((alv & circ).sum())

    def report(self, hops):
        C = self.C; r3 = lambda v: round(float(v), 3)
        per = lambda key, base: [r3(C[key][c] / max(C[base][c], 1)) for c in (1, 0)]
        med = lambda key, f=lambda v: v, k=1: [round(float(f(np.median(C[key][c]))), k) if C[key][c] else None for c in (1, 0)]
        P, F = np.array(C["pr"]), np.array(C["fe"]); b = np.array(C["bouts"]); srcn = C["src"].sum(1)
        fore = None
        if len(P):
            tt = np.abs(F[:, 2]) > 0.1
            fore = dict(corr=[r3(np.corrcoef(P[:, j], F[:, j])[0, 1]) for j in range(3)], turn_sign_agree=r3(np.mean(np.sign(P[tt, 2]) == np.sign(F[tt, 2]))),
                        sd_pred=np.round(P.std(0), 3).tolist(), sd_true=np.round(F.std(0), 3).tolist())
        thd = [np.degrees(np.array(C["th"][c])) for c in (1, 0)]
        return dict(share=r3(C["circ"] / max(C["beats"], 1)), order="circling, else", hops=hops,
                    src=[(C["src"][c] / max(srcn[c], 1)).round(3).tolist() for c in (1, 0)],
                    goal=[r3(C["goal"][c] / max(srcn[c], 1)) for c in (1, 0)], hungry=[r3(C["hungry"][c] / max(srcn[c], 1)) for c in (1, 0)],
                    hungry_food_dist=[(C["dist"][c] / max(C["dist"][c].sum(), 1)).round(3).tolist() for c in (1, 0)],
                    at_goal_unit=per("atgoal", "gbeats"),
                    acts=[(C["acts"][c] / max(C["acts"][c].sum(), 1)).round(3).tolist() for c in (1, 0)],
                    c2=dict(abs_th_deg=med("th", np.degrees), th_near90=[r3(np.mean((v > 60) & (v < 120))) if len(v) else None for v in thd],
                            d_median=med("dd", k=2), heading_err_step_deg=[r3(np.degrees(np.mean(C["de"][c]))) if C["de"][c] else None for c in (1, 0)], fore=fore),
                    c4=dict(top_match={k: [r3(v[c] / max(C["topn"][c], 1)) for c in (1, 0)] for k, v in C["top"].items()},
                            spread={k: [r3(v[c] / max(C["topn"][c], 1)) for c in (1, 0)] for k, v in C["spread"].items()}, choices=[float(C["topn"][1]), float(C["topn"][0])]),
                    c5=dict(target_change=per("tch", "tn_"), dth_deg_median=med("dth", np.degrees)),
                    c6=dict(line_blocked=per("blk", "bn"), chosen_foreseen_forward=[r3(np.mean(C["fwd"][c])) if C["fwd"][c] else None for c in (1, 0)]),
                    c7=(dict(bouts=len(b), d_start_median=round(float(np.median(b[:, 0])), 2), d_end_median=round(float(np.median(b[:, 1])), 2),
                             share_closer_by_1=r3(np.mean(b[:, 1] < b[:, 0] - 1)), share_same=r3(np.mean(np.abs(b[:, 1] - b[:, 0]) <= 1))) if len(b) else None))
