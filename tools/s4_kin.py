"""S4 (atanor_os/S3_S4_PROTOCOL.md, registered before running): newborns beside an elder of their kind.
Elder = the S3 being as it ended (its own twin room, same layout). Newborns = a fresh being, 1,000 protected babbling beats, then the wild.
Arms: alone (no elder drawn) | a (elder drawn in the newborn's eye) | b (elder drawn + resonance bedrock .3).
  PYTHONPATH=. python -m tools.s4_kin OUT.json --arm b --lay 9101_0 --elder s3_a"""
import argparse, os, json, math, time
import numpy as np, torch
from tools.need_fork import t_room, place, DEV, DEPTH, BEING
from tools.a1_channel import V2
from tools.a1_sf import paths_of
from tools.l1_life import make, forks
from atanor_os.emerge.room_body import RoomBody
from atanor_os.emerge.body_contract import Organs
from atanor_os.emerge.mind import Mind
from atanor_os.emerge.going import Going
from tools.d34b_organ_parts import code_of
from tools.n1_nurture import Carer, FAR
from collections import deque


class Guardian(Carer):
    """S4g (founder 2026-11-25: a hand-written parent for now, that keeps the child and leads it to learn). The world's, not a being: it goes to the
    spring of its own lower inside (N1 caregiver); while its child is more than 3 cells away it calls and walks back to fetch it (wall-free distance;
    waiting alone froze the pair in a smoke). Nothing in the child is told to follow."""

    def fetch_map(self, cell):
        g = self.F["w"]; H, W = g.shape; D = np.full((H, W), -1); D[cell[1], cell[0]] = 0; q = deque([cell])
        while q:
            x, y = q.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if 0 <= x + dx < W and 0 <= y + dy < H and D[y + dy, x + dx] < 0 and g[y + dy, x + dx] >= 0:
                    D[y + dy, x + dx] = D[y, x] + 1; q.append((x + dx, y + dy))
        return D

    def step(self, wait, fetch=None, child=None):
        if fetch is None or not fetch.any():
            return super().step(wait)
        B = self.B; pos = B.w.pos.cpu().numpy(); head = B.w.head.cpu().numpy(); act = np.full(self.n, self.STILL)
        st = B.w.store.cpu().numpy()
        for i in range(self.n):
            if wait[i] and not fetch[i]:
                continue
            F = self.fetch_map((int(child[i, 0]), int(child[i, 1]))) if fetch[i] else self.F["w" if st[i, 0] < st[i, 1] else "f"]
            cy, cx = int(pos[i, 1]), int(pos[i, 0])
            if F[cy, cx] <= 1:
                act[i] = self.STILL if fetch[i] else self.DO; continue
            best = min(((F[cy + dy, cx + dx], dy, dx) for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)) if F[cy + dy, cx + dx] >= 0), default=None)
            if best is None:
                continue
            want = math.atan2(best[1] + (cy + 0.5 - pos[i, 1]) * 0.5, best[2] + (cx + 0.5 - pos[i, 0]) * 0.5); d = (want - head[i] + math.pi) % (2 * math.pi) - math.pi
            act[i] = self.FWD if abs(d) < math.radians(20) else (self.RIGHT if d > 0 else self.LEFT)
        f = B.step(torch.as_tensor(act, device=DEV))
        if B.dead.any():
            dd = B.dead.clone(); idx = torch.nonzero(dd)[:, 0]; B.w.dead[idx] = False; B.w.store[idx] = 0.75
        return f


def being(B, n, path=None):
    flag_check()
    O = Organs(B.c, n, dev=DEV, depth=DEPTH); Mi = Mind(O, B.c, n, dev=DEV, value_arm="relief"); G = Going(O.dx, Mi.M, n, O.inside, DEV, W=1.0)
    if path is None:
        Mi.grow(BEING)
    else:
        Mi.load(f"{path}/mind.pt"); st = torch.load(f"{path}/going.pt", map_location=DEV, weights_only=False)
        G.skill.load_state_dict(st["skill"]); G.C, G.Cn, G.T, G.R, G.fitted = st["C"], st["Cn"], st["T"], st["R"], st["fitted"]
        G.mended = st.get("mended", True); G.Vn = st.get("Vn", G.Vn); G.born = st.get("born", True); G.used = st.get("used", G.KU); G.refresh()
    Mi.wo.on = True; Mi.going = G
    if os.environ.get("ATANOR_TURN_ONCE", "0") == "1" and hasattr(B, "no_turn"):   # T1: a held gesture does not keep turning
        Mi.turn_once = B.no_turn
    if os.environ.get("ATANOR_ONE_OP", "0") in ("1", "2") and hasattr(B, "one_op"):   # R1: the body offers one operation at a time; 2 = R1b, no side-steps
        Mi.allowed = B.vacuum if os.environ["ATANOR_ONE_OP"] == "2" else B.one_op
    if os.environ.get("ATANOR_INNATE_HOMING", "1") == "1":            # gate 1 adopted (WORLD_A_PROTOCOL.md): homing vector + own goal; =0 reproduces runs before 2026-11-26
        G.innate(O, Mi.net)
        if os.environ.get("ATANOR_LEARNED_TRUST", "1") == "1":       # LW3f adopted (LEARNED_WEIGHT_PROTOCOL.md): trust in the homing content = its kept promises; =0 = full weight (gate 1)
            from atanor_os.emerge.mind import LearnedWeight
            Mi.learned_home = LearnedWeight(w0=0.0, mode=os.environ.get("ATANOR_TRUST_MODE", "reli"))   # TC1: "cont" = contingency trust (registered)
            pf = os.environ.get("ATANOR_PLACE_FROM", "")                  # M1d (registered): where each place unit is, by its own path sense, learned in training
            if pf:
                sp = torch.load(pf, map_location=DEV, weights_only=False); G.Pu, G.Pn = sp["Pu"].to(DEV), sp["Pn"].to(DEV)
            tf = os.environ.get("ATANOR_TRUST_FROM", "")                  # M1a (registered): the trust learned in training carried into this life (default: none)
            if tf:
                import json as _j
                Mi.learned_home.carry(_j.load(open(tf))["home"])
    return O, Mi


def rebirth(B, O, Mi, info):
    dd = B.dead.clone(); idx = torch.nonzero(dd)[:, 0]; B.w.dead[idx] = False; B.w.store[idx] = 0.75; place(B, info, which=idx); O.reset(dd); Mi.begin(dd)
    if hasattr(B, "pitch"):
        B.pitch[idx] = 0.0                                              # HC1 audit: a reborn body kept the dead body's head tilt
    return dd


def reread(B, O, x, dd):
    """HC1 (2026-11-28 audit): after a rebirth only the reborn rows take the new reading. Before, every living body's x was replaced by a
    second organ call in the same beat, with its felt own-motion zeroed. ATANOR_OLD_REREAD=1 keeps that."""
    B.felt[:] = 0; xn = O(B.raw())
    return xn if os.environ.get("ATANOR_OLD_REREAD", "0") == "1" else torch.where(dd[:, None], xn, x)


def flag_check():
    """HC1 audit: flags that do nothing without another one - refuse to start instead of silently measuring the baseline (F2's seen-eye arm)."""
    on = lambda k, d="0": os.environ.get("ATANOR_" + k, d) not in ("0", "")
    need = [("SEEN", "IMAGINE_MAP"), ("RELY", "SEEN"), ("PARALLAX", "SEEN"), ("IMAGINE_GOAL", "IMAGINE_MAP"), ("OWN_LEAVE", "GUT"), ("PITCH_RETURN", "PITCH")]
    bad = [f"ATANOR_{a_} needs ATANOR_{b_}" for a_, b_ in need if on(a_) and not on(b_) and not (b_ == "IMAGINE_MAP" and os.environ.get("ATANOR_B0", "0") == "own")]
    homing = on("INNATE_HOMING", "1") and on("LEARNED_TRUST", "1")
    bad += [f"ATANOR_{a_} is ignored without innate homing and learned trust" for a_ in ("TRUST_FROM", "PLACE_FROM") if os.environ.get("ATANOR_" + a_, "") and not homing]
    assert not bad, "; ".join(bad)
    import sys
    print("recipe flags:", {k: os.environ.get("ATANOR_" + k, "-") for k in ("TURN_ONCE", "MISMATCH", "LEARNED_TRUST", "RENDER_FIX", "OLD_REREAD")}, file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--arm", default="b"); ap.add_argument("--lay", default="9101_0")
    ap.add_argument("--elder", required=True); ap.add_argument("--beats", type=int, default=10000); ap.add_argument("--babble", type=int, default=1000)
    ap.add_argument("--carry", type=int, default=0); ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--nurse", type=float, default=0.0)   # S4n: after babbling, childhood drains (floor .1) and a newborn within 1.5 cells of its parent is filled this much a beat (nursing)
    ap.add_argument("--imagine", type=float, default=0.0)
    ap.add_argument("--vdiag", type=int, default=0)   # S4j diagnostic (see protocol)
    ap.add_argument("--contact", type=int, default=0)   # S4i: the leading guardian sounds all through childhood (a contact call, louder when near), not only when the child is far
    ap.add_argument("--exit-store", type=float, default=0.0)   # S4g diagnostic: stores set to this when childhood ends (0 = as they are); tells a head start from what childhood taught
    ap.add_argument("--parent", type=int, default=0)   # 5: S4h leading guardian; 4: S4g fetching guardian; 1: through childhood the elder waits (stands) and calls while its newborn is more than 3 cells away; 2: waits, no call
    a = ap.parse_args(); V2["on"] = True; RoomBody.default_cues = True; t0 = time.time(); n = 64
    seed, mirror = map(int, a.lay.split("_")); assert not 7701 <= seed <= 7712; rows, info = t_room(seed, bool(mirror)); sd = seed * 10 + mirror + 77 + 1000 * a.rep; prot = max(a.babble, a.carry)
    torch.manual_seed(sd)
    Be = make(rows, n, sd + 1, V2["drain"]); Oe, Me = being(Be, n, a.elder)
    Bn = make(rows, n, sd, V2["drain"]); On, Mn = being(Bn, n)
    Mn.imagine = a.imagine                                             # S4l: choosing by its own imagination (rolled forecaster, scored by its own value); 0 = off
    every = torch.ones(n, dtype=torch.bool, device=DEV)
    for B, O, Mi in ((Be, Oe, Me), (Bn, On, Mn)):
        B.w.store[:] = 0.75; place(B, info); O.reset(every); Mi.begin(every); Mi.prev = None
    if a.arm != "alone":
        Bn.kin = Be.w.pos.clone()
    lg, nk, hd = Be.legs, Be.neck, Be.hand; STILL = int(torch.nonzero((lg == 0) & (nk == 0) & (hd == 0))[0, 0]); waited = 0; nursed = 0
    xe, xn = Oe(Be.raw()), On(Bn.raw())
    deaths = np.zeros(a.beats, int); need = np.zeros(a.beats, int); near = np.zeros(a.beats); seen = np.zeros(a.beats); felt = 0.0
    VD, VG, VB = [], [], []; exit_store = None; C = Guardian(rows, info, n, sd) if a.parent in (4, 5) else None; fetching = torch.zeros(n, dtype=torch.bool, device=DEV)
    for t in range(a.beats):
        if C is not None:                                              # S4g: the hand-written guardian through childhood, then it is gone
            if t < a.carry and a.parent == 5:                          # S4h: the leading guardian - never fetches nor waits; goes its own way between springs and calls while the child is more than 3 cells away
                gap = (Bn.w.pos - C.B.w.pos).norm(dim=1); calling = gap > 3.0; waited += int(calling.sum())
                fe = C.step(np.zeros(n, bool)); ppos = C.B.w.pos.clone(); Bn.call_pos = ppos; Bn.call_on = torch.ones_like(calling) if a.contact else calling
            elif t < a.carry:
                gap = (Bn.w.pos - C.B.w.pos).norm(dim=1); fetching = (gap > 3.0) | (fetching & (gap > 1.2)); wait = fetch = fetching.cpu().numpy(); waited += int(wait.sum())
                fe = C.step(wait, fetch, Bn.w.pos.cpu().numpy()); ppos = C.B.w.pos.clone()
                Bn.call_pos = ppos; Bn.call_on = torch.as_tensor(wait, device=DEV)
            else:
                fe = torch.zeros(n, 2, dtype=torch.bool, device=DEV); ppos = torch.full_like(Bn.w.pos, FAR); Bn.call_on = None
            de = torch.zeros(n, dtype=torch.bool, device=DEV)
        ae = Me.choose(xe) if C is None else None
        if a.parent and C is None and t < a.carry:
            wait = (Bn.w.pos - Be.w.pos).norm(dim=1) > 3.0; waited += int(wait.sum())
            if a.parent == 3:                                          # the elder goes back to fetch its newborn with its own going skill (goal = the newborn's place) and calls
                go = torch.multinomial(Me.going.toward(xe, code_of(Bn.w.pos.tolist(), info, DEV)).exp(), 1)[:, 0]; ae = torch.where(wait, go, ae)
            else:
                ae = torch.where(wait, torch.full_like(ae, STILL), ae)
            Bn.call_pos = Be.w.pos.clone(); Bn.call_on = wait if a.parent in (1, 3) else None
        elif a.parent and C is None:
            Bn.call_on = None
        if C is None:
            s0e = Be.w.store.clone(); al_e = ~Be.dead.clone(); fe = Be.step(ae); de = al_e & Be.dead; xe2 = Oe(Be.raw())
            Me.after(xe, ae, xe2, de, Be.dead & ~de); xe = xe2; ppos = Be.w.pos
        an = torch.randint(0, Mn.M, (n,), device=DEV) if t < a.babble else Mn.choose(xn)
        if a.vdiag and C is not None:
            gb = (Bn.w.pos - C.B.w.pos).norm(dim=1)
        if a.vdiag and prot - 1000 <= t < prot and C is not None:   # S4j: does its own value rate the guardian near above far (same lack), and do its steps close the gap
            with torch.no_grad():
                sc = Mn.wo.score(xn); g0 = (Bn.w.pos - C.B.w.pos).norm(dim=1)
            VD.append(torch.stack([sc.max(1).values, g0, 1.0 - Bn.w.store.min(1).values, sc.gather(1, an[:, None])[:, 0], (an == sc.argmax(1)).float(), sc.max(1).values - sc.mean(1)], 1).cpu().numpy())
        s0 = Bn.w.store.clone(); al = ~Bn.dead.clone(); f = Bn.step(an)
        if a.arm != "alone":
            Bn.kin = ppos.clone()
        if a.vdiag and a.babble - 500 <= t < a.babble and C is not None:   # random-step baseline for closing the gap
            VB.append(((Bn.w.pos - C.B.w.pos).norm(dim=1) < gb - 0.02).cpu().numpy())
        if a.vdiag and prot - 1000 <= t < prot and C is not None:
            VG.append(((Bn.w.pos - C.B.w.pos).norm(dim=1) < g0 - 0.02).cpu().numpy())
        dn = al & Bn.dead; xn2 = On(Bn.raw()); Mn.after(xn, an, xn2, dn, Bn.dead & ~dn)
        ks = getattr(Bn, "kin_seen", None) if a.arm != "alone" else None
        if a.arm == "b" and ks is not None:                            # the resonance bedrock: a seen rise of the elder's inside, felt where the newborn stands
            rose = fe.bool() & ks[:, None]; Mn.going.resonate(xn2, rose, 0.3); felt += float(rose.sum())
        xn = xn2
        if t < a.carry and not a.parent:                               # nurture: a newborn strayed beyond 3 cells is carried back to its elder; the carry is felt as motion
            far_ = (Bn.w.pos - Be.w.pos).norm(dim=1) > 3.0
            if bool(far_.any()):
                dp = torch.where(far_[:, None], Be.w.pos - Bn.w.pos, torch.zeros_like(Bn.w.pos)); c, s_ = torch.cos(Bn.w.head), torch.sin(Bn.w.head)
                Bn.w.pos = Bn.w.pos + dp; Bn.felt = torch.stack([dp[:, 0] * c + dp[:, 1] * s_, -dp[:, 0] * s_ + dp[:, 1] * c, torch.zeros_like(c)], 1)
                xn = On(Bn.raw())                                      # the re-read feels only the carry (the step's own motion was read already)
        if a.nurse and a.babble <= t < prot:                           # nursing: the child's inside drains as in life but is held above .1; close to its parent it is filled
            close = (Bn.w.pos - ppos).norm(dim=1) <= 1.5; nursed += int(close.sum())
            Bn.w.store = torch.where(close[:, None], (Bn.w.store + a.nurse).clamp(max=1.0), Bn.w.store).clamp(min=0.1); Bn.w.dead[:] = False
        elif t < prot:
            Bn.w.store[:] = 0.75; Bn.w.dead[:] = False
        if t == prot - 1:
            exit_store = round(float(Bn.w.store.mean()), 3)
            if a.exit_store:
                Bn.w.store[:] = a.exit_store
        deaths[t] = int(dn.sum()); need[t] = int(((f[:, 0].bool() & (s0[:, 0] < s0[:, 1])) | (f[:, 1].bool() & (s0[:, 1] < s0[:, 0]))).sum())
        near[t] = float(((Bn.w.pos - ppos).norm(dim=1) <= 3.0).float().mean()); seen[t] = float(ks.float().mean()) if ks is not None else 0.0
        if (t + 1) % 1000 == 0:
            Mn.going.learn()
            if C is None:
                Me.going.learn()
            with torch.enable_grad():
                if C is None:
                    Me.wo.learn(n=max(1, 2500 // max(1, getattr(Me.wo, 'per_call', 1))))
                if t >= a.babble:
                    Mn.wo.learn(n=max(1, 2500 // max(1, getattr(Mn.wo, 'per_call', 1))))
        for B, O, Mi in (((Be, Oe, Me),) if C is None else ()) + ((Bn, On, Mn),):
            if B.dead.any():
                rebirth(B, O, Mi, info)
        if bool(de.any()) or bool(dn.any()):                           # re-read after any rebirth; the re-read feels no motion (L1d drift fix)
            Be.felt[:] = 0; Bn.felt[:] = 0; xe = Oe(Be.raw()); xn = On(Bn.raw())
    per = lambda arr, lo, hi: round(1000 * float(arr[lo:hi].sum()) / max(n * (hi - lo), 1), 2)   # HC1 audit: the default --carry 0 divided by zero here
    fk = forks(Mn, On, rows, info, sd, paths_of(rows, info))
    r = dict(arm=a.arm, carry=a.carry, parent=a.parent, contact=a.contact, imagine=a.imagine, nurse=a.nurse, exit_store=exit_store, set_exit=a.exit_store, nursed=nursed, waited=waited, near_child=round(float(near[a.babble:max(a.carry, a.babble + 1)].mean()), 3), rep=a.rep, layout=a.lay, elder=a.elder, early_deaths=per(deaths, prot, prot + 3000), last_deaths=per(deaths, a.beats - 2000, a.beats),
             child_need=per(need, a.babble, prot), near_h1=round(float(near[a.babble:(a.babble + prot) // 2].mean()), 3), near_h2=round(float(near[(a.babble + prot) // 2:prot].mean()), 3), early_need=per(need, prot, prot + 3000), last_need=per(need, a.beats - 2000, a.beats), near_share=round(float(near[a.babble:].mean()), 3),
             seen_share=round(float(seen[a.babble:].mean()), 3), resonance_events=felt, forks={s: fk[s] for s in ("far", "stem")}, seconds=round(time.time() - t0))
    if VD:
        D = np.concatenate(VD); Gs = np.concatenate(VG); v, g, lk = D[:, 0], D[:, 1], D[:, 2]; q = np.quantile(lk, [1 / 3, 2 / 3]); rows_ = []
        for lo, hi in ((-1, q[0]), (q[0], q[1]), (q[1], 2)):
            m = (lk > lo) & (lk <= hi); nr = m & (g <= 1.5); fr = m & (g > 3.0)
            rows_.append(dict(lack=round(float(lk[m].mean()), 3), v_near=round(float(v[nr].mean()), 5) if nr.any() else None, v_far=round(float(v[fr].mean()), 5) if fr.any() else None, n_near=int(nr.sum()), n_far=int(fr.sum())))
        far_ = g > 1.5; ag = D[:, 4] > 0.5; hi_ = lk > q[1]; r["vdiag"] = dict(by_lack=rows_, closer_when_apart=round(float(Gs[far_].mean()), 3), closer_value_best=round(float(Gs[far_ & ag].mean()), 3), closer_not_best=round(float(Gs[far_ & ~ag].mean()), 3), share_value_best=round(float(ag[far_].mean()), 3), closer_value_best_hilack=round(float(Gs[far_ & ag & hi_].mean()), 3), spread_far=round(float(D[far_, 5].mean()), 5), spread_near=round(float(D[g <= 1.5, 5].mean()), 5), closer_random_babble=round(float(np.concatenate(VB).mean()), 3) if VB else None)
    print(json.dumps(r), flush=True)
    try:
        old = json.load(open(a.out))["rows"]
    except (OSError, ValueError, KeyError):
        old = []
    json.dump(dict(protocol="atanor_os/S3_S4_PROTOCOL.md (S4)", rows=old + [r]), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
