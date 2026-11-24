"""D17 (atanor_os/A1_RELIEF_DIAG.md, registered before running): per-channel futures as the share of a lack relieved (need-free, latent learning)
vs the absolute SARSA futures (D12), on cued-room lived rings. Value map along both paths, junction preference, the A1 v2 fork.
  PYTHONPATH=. python -m tools.a1_relief OUT.json"""
import argparse, copy, json, math, os, time
import numpy as np, torch
from tools.need_fork import t_room, DEV, DEPTH
from tools.a1_channel import V2, KEEP, side_sets, fork
from tools.a1_sf import arm3, paths_of
from atanor_os.emerge.room_body import RoomBody
from atanor_os.emerge.body_contract import Organs
from atanor_os.emerge.mind import Mind
from atanor_os.emerge.channel_worth import ChannelWorth, ChannelP

CACHE = "a1cache_cues"


class ReliefWorth(ChannelWorth):
    """psi = [relief share per channel (0..1), loss per channel (<= 0)], SARSA. S = whole(b + (1 - b) clip(psi+) + psi-) - whole(b)."""
    def __init__(self, dx, inside, m, d=128, keep=KEEP, gamma=0.97, sync=40):
        super().__init__(dx, inside, lam=2.0, gamma=gamma, m=m, keep=keep, d=d, mode="bedrock"); self.sync = sync
        self.P = ChannelP(dx, m, 2 * self.ni, d); self.Pt = copy.deepcopy(self.P)
        for p in self.Pt.parameters():
            p.requires_grad_(False)

    def _S(self, x, psi):
        b = x[..., self.inside]; pp = psi[..., :self.ni].clamp(0, 1); pm = psi[..., self.ni:]
        b2 = (b[..., None, :] + (1 - b)[..., None, :] * pp + pm).clamp(0, 1)
        return self._whole_b(b2) - self._whole_b(b)[..., None]

    def learn(self, n=4, batch=2048, lr=1e-3):
        top = int(self.bn)
        if self.popt is None:
            self.popt = torch.optim.Adam(self.P.parameters(), lr)
        ins = self.inside; ni2 = 2 * self.ni
        for _ in range(n):
            j = torch.randint(0, top, (batch,), device=self.bx.device); x, x2, a, d = self.bx[j], self.bx2[j], self.ba[j], self.bd[j]
            with torch.no_grad():
                _, p2 = self.Pt(x2); an = self.ba2[j]
                nxt = torch.where((an >= 0)[:, None], p2.gather(1, an.clamp(min=0)[:, None, None].expand(-1, 1, ni2))[:, 0], p2.mean(1))
                db = x2[:, ins] - x[:, ins]; dead = d[:, None] > 0
                fp = torch.where(dead, torch.zeros_like(db), (db.clamp(min=0) / (1 - x[:, ins] + 0.02)).clamp(max=1))
                fm = torch.where(dead, -x[:, ins], db.clamp(max=0))
                tg = torch.cat([fp, fm], 1) + self.gamma * nxt * (1 - d)[:, None]
            _, p = self.P(x); out = p.gather(1, a[:, None, None].expand(-1, 1, ni2))[:, 0]
            loss = ((out - tg) ** 2).mean(); self.popt.zero_grad(); loss.backward(); self.popt.step(); self.steps += 1
            if self.steps % self.sync == 0:
                self.Pt.load_state_dict(self.P.state_dict())
        return float(loss)


def relief_arm(ring, dx, ins, M, seed, updates=2000, gamma=0.97, sync=40):
    torch.manual_seed(seed); a = ReliefWorth(dx, ins, M, keep=max(KEEP, len(ring["ba"])), gamma=gamma, sync=sync).to(DEV)
    X, X2, A, D = [ring[q].to(DEV) for q in ("bx", "bx2", "ba", "bd")]; k = len(A); a.see(X, X2, D > 0.5, a=A)
    joined = torch.zeros(k, dtype=torch.bool, device=DEV); joined[:-1] = ((X[1:] - ring["bx2_true"].to(DEV)[:-1]).abs().max(1).values < 1e-5) & (ring["bd_true"].to(DEV)[:-1] < 0.5)
    nxt = torch.cat([A[1:], A[-1:]]); a.ba2[:k] = torch.where(joined, nxt, torch.full_like(nxt, -1))
    if "ba2" in ring:                                               # many-body ring: each body's own next action was recorded in the life
        a.ba2[:k] = ring["ba2"].to(DEV)
    for _ in range(updates // 4):
        a.learn(n=4)
    return a


def far_fork(ck, arm, rows, info, seed, P, K=16, T=400, start=None, arbitrate=False):
    """D21: thirsty clones start at the food spring, hungry clones at the water spring; the mind chooses as it is. Returns recovery of the needed channel."""
    import copy as _c
    from tools.a1_channel import trust_one
    torch.manual_seed(seed + 5); n = 2 * K
    B = RoomBody(n, rows, drain=0.0, seed=seed, device=DEV, regrow=V2["regrow"]); O = Organs(B.c, n, dev=DEV, depth=DEPTH)
    Mi = Mind(O, B.c, n, dev=DEV, arbitrate=arbitrate); Mi.load(ck); Mi.wo = _c.deepcopy(arm); Mi.wo.on = True
    at = {sp: cells[-1] for cells, sp in P.values()}
    if start == "stem":                                              # V4: where life starts - the bottom of the stem, facing up
        at = {"f": (info["x0"], info["ys"], -math.pi / 2), "w": (info["x0"], info["ys"], -math.pi / 2)}                # the cell before each spring, facing it
    for i in range(n):
        px, py, h = at["f"] if i < K else at["w"]; B.w.pos[i, 0], B.w.pos[i, 1], B.w.head[i] = px, py, h
    B.w.store[:K, 0], B.w.store[:K, 1] = 0.30, 0.95; B.w.store[K:, 0], B.w.store[K:, 1] = 0.95, 0.30
    every = torch.ones(n, dtype=torch.bool, device=DEV); O.reset(every); Mi.begin(every); x = O(B.raw()); st0 = B.w.store.clone(); first = np.full(n, -1)
    with trust_one(Mi.wo):
        for t in range(T):
            act = Mi.choose(x); B.step(act); x2 = O(B.raw()); Mi.prev = (x, act); Mi.cut = torch.zeros(n, dtype=torch.bool, device=DEV); x = x2
            r = (B.w.store - st0).cpu().numpy(); got = np.r_[r[:K, 0], r[K:, 1]] > 0.05; first = np.where((first < 0) & got, t, first)
    return dict(recover=[float((first[:K] >= 0).mean()), float((first[K:] >= 0).mean())], beats_to_recover=float(np.median(first[first >= 0])) if (first >= 0).any() else None)


def shuffled(ring, ins, seed):
    g = torch.Generator().manual_seed(seed); p = torch.randperm(len(ring["ba"]), generator=g); r = dict(ring)
    x2 = ring["bx2"].clone(); x2[:, ins] = ring["bx2"][p][:, ins]; r["bx2"] = x2; r["bd"] = ring["bd"][p]
    return r


def vmap(arm, rows, info, P, organs_state=None):
    gaps = {}
    for side, (cells, spring) in P.items():
        n = 2; B = RoomBody(n, rows, drain=0.0, device=DEV, regrow=V2["regrow"]); O = Organs(B.c, n, dev=DEV, depth=DEPTH); px, py, h = cells[-1]
        if organs_state is not None:
            O.load(organs_state)                                    # the being's own sense statistics (fresh ones normalise over the 2 clones: both paths looked identical)
        B.w.pos[:, 0], B.w.pos[:, 1], B.w.head[:] = px, py, h
        B.w.store[0, 0], B.w.store[0, 1] = 0.30, 0.95; B.w.store[1, 0], B.w.store[1, 1] = 0.95, 0.30
        O.reset(torch.ones(n, dtype=torch.bool, device=DEV)); x = O(B.raw())
        with torch.no_grad():
            s = arm.v(x).max(1).values.cpu().numpy()
        gaps[spring] = dict(thirsty=float(s[0]), hungry=float(s[1]))
    return dict(thirsty=gaps["w"]["thirsty"] - gaps["f"]["thirsty"], hungry=gaps["f"]["hungry"] - gaps["w"]["hungry"])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--nopull", action="store_true"); a_ = ap.parse_args(); V2["on"] = True; RoomBody.default_cues = True; res = []; t0 = time.time()
    for lay in sorted(os.listdir(CACHE)):
        if not os.path.exists(os.path.join(CACHE, lay, "life_ring.pt")):
            continue
        seed, mirror = map(int, lay.split("_")); d = os.path.join(CACHE, lay); rows, info = t_room(seed, bool(mirror)); sd = seed * 10 + mirror
        ring = torch.load(os.path.join(d, "life_ring.pt"), weights_only=False); ring["bx2_true"], ring["bd_true"] = ring["bx2"], ring["bd"]; ck = os.path.join(d, "mind.pt")
        c = RoomBody(1, rows, drain=0.0, device=DEV).c; tmp = Mind(Organs(c, 1, dev=DEV, depth=DEPTH), c, 1, dev=DEV, worth=0); dx, ins, M = tmp.net.dx, tmp.o.inside, tmp.M; del tmp
        L, R = side_sets(rows, info, seed); P = paths_of(rows, info); rec = dict(layout=lay, arms={}); ost = torch.load(ck, map_location=DEV, weights_only=False)["organs"]
        arms = {"absolute": arm3(ring, dx, ins, M, "sarsa", sd), "relief": relief_arm(ring, dx, ins, M, sd), "relief-shuf": relief_arm(shuffled(ring, ins, sd), dx, ins, M, sd)}
        def no_pull(Mi):                                             # D18 probe: the value's lean alone picks (pull and harm memory silenced)
            n_, M_ = Mi.n, Mi.M; Mi.harm = 0
            Mi.net.pull = lambda x: (torch.zeros(n_, M_, device=DEV), torch.zeros(n_, dtype=torch.bool, device=DEV))
        for k, arm in arms.items():
            if a_.nopull:
                fo = fork(ck, arm, rows, info, sd, L, R); fn = fork(ck, arm, rows, info, sd, L, R, prep=no_pull)
                rec["arms"][k] = dict(normal={q: fo[q] for q in ("first_match", "recover_share", "decided_by")}, value_only={q: fn[q] for q in ("first_match", "recover_share", "decided_by", "choice_rate")})
                print(lay, k, "normal", fo["first_match"], fo["recover_share"], "| value only", fn["first_match"], fn["recover_share"], fn["decided_by"], round(time.time() - t0), flush=True); continue
            fo = fork(ck, arm, rows, info, sd, L, R)
            rec["arms"][k] = dict(needed_gap=vmap(arm, rows, info, P, ost), **{q: fo[q] for q in ("value_pref_right", "first_match", "recover_share", "decided_by")})
            print(lay, k, {q: rec["arms"][k][q] for q in ("needed_gap", "value_pref_right", "first_match", "recover_share")}, round(time.time() - t0), flush=True)
        res.append(rec); json.dump(dict(protocol="atanor_os/A1_RELIEF_DIAG.md", rows=res), open(a_.out, "w"), indent=1)


if __name__ == "__main__":
    main()
