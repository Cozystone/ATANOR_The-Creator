"""The going (L1, atanor_os/P1_PATH_PROTOCOL.md): an organ of the mind that learns to take the body to a remembered place.
Reads the innate path organ's code (the last PathOrgan.N numbers of the sense). Everything here is learned from the being's own experience:
- place memory (fixed size): KU place units by online k-means of its place code; which unit follows which (decayed counts); for each inside, how often
  it rose in each unit (relief remembered with its place);
- the skill: sense + place code + goal place -> gesture, trained by hindsight on a bounded ring of its own recent beats (where it actually got to),
  never by reward (D33: Ghosh et al. 2019 GCSL; goal babbling, Rolf et al. 2010);
- the lean of the lack: for the lowest inside, the next unit on its own map toward where that inside was relieved (V = (I - g T)^-1 r, best
  neighbour); the skill's preference for going there, scaled by how low that inside is, is offered to the mind's choice score."""
import torch, torch.nn as nn, torch.nn.functional as F
from .body_contract import PathOrgan

NP = PathOrgan.N; PC = NP - 2                                          # place part of the code (the last two numbers are the heading)


class Skill(nn.Module):
    def __init__(self, dx, M):
        super().__init__(); self.register_buffer("mu", torch.zeros(dx)); self.register_buffer("sd", torch.ones(dx))
        self.f = nn.Sequential(nn.Linear(dx + PC, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, M))

    def forward(self, x, g):
        return self.f(torch.cat([((x - self.mu) / self.sd.clamp(min=0.05)).clamp(-5, 5), g], 1))


class Going:
    def __init__(self, dx, M, n, inside, dev, KU=64, ring=1024, far=32, gamma=0.99, W=1.0, decay=0.9995, mended=True, piece=64, born=True, hops=4,
                 ctx=0, margin=20.0, lam=0.99, fresh=0.998, ctx_by="both", dread=0.0, hold=0):
        self.dx, self.M, self.n, self.dev, self.KU, self.far, self.gamma, self.W, self.decay = dx, M, n, dev, KU, far, gamma, W, decay
        # L1b (D34b): mended = units settle (running mean), relief per fading visits, the skill's memory a reservoir of pieces over the whole life.
        # mended=False is L1 as it ran (units drift with a +.01 floor; visits never fade; ring of the latest beats).
        # L1c (D34b on L1b): born = units are born where a body is farther than one cell's worth of code from every unit (until KU are used);
        # born=False is L1b (all units start at the first places, i.e. at birth).
        self.hops = hops                                                # L1d (D34c): the lean aims this many hops ahead along its own chain (L1-L1c: 1)
        self.born = born; P0 = PathOrgan(2, dev); P0.p = torch.tensor([[0.0, 0.0], [1.0, 0.0]], device=dev); c0 = P0(torch.zeros(2, 3, device=dev))[:, :PC]
        self.rad = float((c0[0] - c0[1]).norm()); self.used = 0
        self.mended, self.piece = mended, piece; self.Vn = torch.zeros(KU, device=dev); self.slots = ring // piece - 1; self.pieces = 0
        self.ins = list(range(dx))[inside] if isinstance(inside, slice) else list(inside)     # (constant, food, water) in the room's order
        self.skill = Skill(dx, M).to(dev); self.opt = torch.optim.Adam(self.skill.f.parameters(), 3e-4); self.fitted = False
        self.C = None; self.Cn = torch.zeros(KU, device=dev); self.T = torch.zeros(KU, KU, device=dev); self.R = torch.zeros(KU, 2, device=dev)
        self.rX = torch.zeros(ring, n, dx, device=dev); self.rA = torch.zeros(ring, n, dtype=torch.long, device=dev); self.rL = torch.zeros(ring, n, dtype=torch.long, device=dev)
        self.ri, self.rn, self.life = 0, 0, torch.zeros(n, dtype=torch.long, device=dev); self.nxt = None; self.since = 0
        # S3b (atanor_os/S3_S4_PROTOCOL.md): contexts of the relief memory - latent-cause inference after Gershman & Niv. The place units and their
        # chain are one; what was relieved where is held per context. Each beat every held context, and a blank one that knows only how often
        # reliefs come at all, is scored by how well it foretold where the reliefs came (and did not); when another beats the one in use by
        # `margin` (log-likelihood, summed over bodies, fading with lam a beat) memory moves to it - a blank winning opens a new context.
        # Only the context in use learns and fades; the others are kept as they were left. ctx=0 is the one memory of L1d.
        self.ctx, self.margin, self.lam, self.beats = ctx, margin, lam, 0
        self.ctx_by = ctx_by
        # D37d (atanor_os/FEAR_PROTOCOL.md): place memory of harm, as of relief - where bodies died, per unit (fading as reliefs fade). A place's
        # worth for the map is its relief rate minus `dread` x its death rate (death counted against both insides). 0 = off (L1d as it was).
        self.dread = dread; self.Dd = torch.zeros(KU, device=dev)
        # S3d: suspend before learning (founder 2026-11-25: "do not believe the present experience at once"). A crossing the memory in use has never
        # seen is held out of it (the newcomer still learns it). If within `hold` beats the newcomer wins, it is another world (the held go with it);
        # if not, the memory in use was wrong and takes the held crossings in. hold=0: learn at once (S3c2 and before).
        self.hold = hold; self.Th = torch.zeros(KU, KU, device=dev); self.h_since = -1; self.merges = 0                                            # S3c: 'map' = a world is told by its layout only; relief expectations stay inside it (springs run dry)
        if ctx:
            self.Rs = torch.zeros(ctx, KU, 2, device=dev); self.Vns = torch.zeros(ctx, KU, device=dev); self.act, self.nctx = 0, 1
            self.R, self.Vn = self.Rs[0], self.Vns[0]; self.Ls = torch.zeros(ctx + 1, device=dev); self.last = [0] * ctx
            self.base_s = torch.zeros(2, device=dev); self.base_n = 0; self.switches = []
            # the newcomer: a memory grown only from recent beats (fading `fresh` a beat), the candidate for "another world" - a blank that knows
            # nothing loses to any old memory even in a changed world, because most places give no relief anywhere (dev diagnostic 2026-11-25)
            self.Rf = torch.zeros(KU, 2, device=dev); self.Vf = torch.zeros(KU, device=dev); self.fresh = fresh
            # the expectation scored is tied to its own act: relief when it reached out (drank / ate) at that place - sharp enough to be wrong
            # (per-beat relief at a place was too soft: .04 at best, the same score in its own world and a swapped one; dev 2026-11-25).
            self.Ds = torch.zeros(ctx, KU, 2, device=dev); self.Dn = torch.zeros(ctx, KU, device=dev)   # per context: reliefs on reaching beats, reaching beats
            self.do_mask = None
            # the map too is per context (hippocampal remapping read as hidden-state inference: Sanders, Wilson & Gershman 2020): which unit follows
            # which. Its expectations are tested every beat by moving, where a remembered spring may sit inside a new world's wall and never be tried.
            self.Ts = torch.zeros(ctx, KU, KU, device=dev); self.T = self.Ts[0]; self.Tf = torch.zeros(KU, KU, device=dev)                                         # [M] bool, set by whoever knows the body's gestures (the hand that reaches)

    def place(self, x):
        return x[:, -NP:-2]

    def unit(self, p):
        if self.born and self.used < self.KU:
            return torch.cdist(p, self.C[:self.used]).argmin(1)
        return torch.cdist(p, self.C).argmin(1)

    def _recruit(self, p):
        for _ in range(4):                                              # at most a few births a beat
            if self.used >= self.KU:
                return
            d = torch.cdist(p, self.C[:self.used]).min(1).values; i = int(d.argmax())
            if float(d[i]) <= self.rad:
                return
            self.C[self.used] = p[i]; self.used += 1

    def needs(self, x):
        """lack per body (1 - lowest inside) and which store is lowest (0 water, 1 food: the world's store order)."""
        food, water = x[:, self.ins[1]], x[:, self.ins[2]]
        return 1.0 - torch.minimum(food, water), (food < water).long()

    @torch.no_grad()
    def observe(self, x, a, x2, died):
        p, p2 = self.place(x), self.place(x2)
        if self.C is None:                                              # the first places it is in become its first units
            self.C = p[torch.randint(0, len(p), (self.KU,), device=self.dev)] + 0.01 * torch.randn(self.KU, PC, device=self.dev)
            if self.born:
                self.C[0] = p[0]; self.used = 1
        if self.born:
            self._recruit(p2)
        u, u2 = self.unit(p), self.unit(p2); live = ~died
        cnt = torch.bincount(u2, minlength=self.KU).float()            # online k-means, each unit moves toward the places it wins (one step for all units;
        sm = torch.zeros(self.KU, PC, device=self.dev).index_add_(0, u2, p2)   # same numbers as the old per-unit loop, checked 2026-11-25)
        self.Cn += cnt; hit = cnt > 0; rate = (cnt / self.Cn.clamp(min=1) + (0.0 if self.mended else 0.01)).clamp(max=1.0)
        self.C = torch.where(hit[:, None], self.C + (sm / cnt.clamp(min=1)[:, None] - self.C) * rate[:, None], self.C)
        if self.ctx and bool(live.any()):                               # S3b: the map's expectation is scored before this beat is learned
            self._moves(u[live], u2[live])
        self.T.mul_(self.decay); self.R.mul_(self.decay); self.Vn.mul_(self.decay)
        if self.dread:
            self.Dd.mul_(self.decay)
            if bool(died.any()):
                self.Dd.index_put_((u2[died],), torch.ones(int(died.sum()), device=self.dev), accumulate=True)
        self.Vn.index_put_((u2[live],), torch.ones(int(live.sum()), device=self.dev), accumulate=True)
        if self.ctx and self.hold and self.used >= self.KU:            # while place units are still being born (its first world young), it learns at once
            ul, vl = u[live], u2[live]; new = (ul != vl) & (self.T[ul, vl] < 0.05)
            self.Th.mul_(self.decay)
            if bool(new.any()):
                self.Th.index_put_((ul[new], vl[new]), torch.ones(int(new.sum()), device=self.dev), accumulate=True)
                if self.h_since < 0:
                    self.h_since = 0
            self.T.index_put_((ul[~new], vl[~new]), torch.ones(int((~new).sum()), device=self.dev), accumulate=True)
            if self.h_since >= 0:
                self.h_since += 1
                if self.h_since > self.hold:                            # no other world won in time: my map was wrong here - take them in
                    self.T += self.Th; self.Th.zero_(); self.h_since = -1; self.merges += 1
        else:
            self.T.index_put_((u[live], u2[live]), torch.ones(int(live.sum()), device=self.dev), accumulate=True)
        rose = torch.stack([x2[:, self.ins[2]] - x[:, self.ins[2]] > 0.01, x2[:, self.ins[1]] - x[:, self.ins[1]] > 0.01], 1) & live[:, None]
        self.beats += 1
        if self.ctx and bool(live.any()):
            sel = live if self.do_mask is None else live & self.do_mask[a]
            if bool(sel.any()):
                self._evidence(u2[sel], rose[sel])
        self.R.index_put_((u2[:, None].expand(-1, 2).reshape(-1), torch.arange(2, device=self.dev).repeat(len(u2))), rose.reshape(-1).float(), accumulate=True)
        if self.mended:                                                  # the piece being lived sits in the last 64 rows; a full piece may be kept in a slot
            j = self.slots * self.piece + self.ri; self.rX[j] = x; self.rA[j] = a; self.rL[j] = self.life; self.ri += 1
            if self.ri == self.piece:
                self.pieces += 1; s = self.pieces - 1 if self.pieces <= self.slots else int(torch.randint(0, self.pieces, (1,)))
                if s < self.slots:
                    sl, cur = slice(s * self.piece, (s + 1) * self.piece), slice(self.slots * self.piece, (self.slots + 1) * self.piece)
                    self.rX[sl] = self.rX[cur]; self.rA[sl] = self.rA[cur]; self.rL[sl] = self.rL[cur]
                self.ri = 0
        else:
            self.rX[self.ri] = x; self.rA[self.ri] = a; self.rL[self.ri] = self.life; self.ri = (self.ri + 1) % len(self.rX); self.rn = min(self.rn + 1, len(self.rX))
        self.life = self.life + died.long(); self.since += 1
        if self.since >= 100 or self.nxt is None:
            self.refresh(); self.since = 0

    @torch.no_grad()
    def _moves(self, u, u2):
        K = self.nctx; idx = list(range(K)) + [self.ctx]; m = u != u2
        if bool(m.any()):                                               # crossings only (staying in a unit says little)
            a, b = u[m], u2[m]; U = float(max(1, self.used))
            if self.ctx_by == "possible":                                # S3c2: only "never seen this crossing here" counts - how often is behaviour, not layout
                ll = torch.cat([(self.Ts[:K, a, b] < 0.05).float().sum(1), (self.Tf[a, b] < 0.05).float().sum()[None]]) * -4.6
            else:
                pT = (self.Ts[:K, a, b] + 1.0 / U) / (self.Ts[:K, a].sum(-1) - self.Ts[:K, a, a] + 1.0)
                pf = (self.Tf[a, b] + 1.0 / U) / (self.Tf[a].sum(-1) - self.Tf[a, a] + 1.0)
                ll = torch.cat([pT.clamp(1e-4, 1).log().sum(1), pf.clamp(1e-4, 1).log().sum()[None]])
            self.Ls[idx] += ll
        self.Tf.mul_(self.fresh); self.Tf.index_put_((u, u2), torch.ones(len(u), device=self.dev), accumulate=True)

    @torch.no_grad()
    def _evidence(self, u, rose):
        rf = rose.float(); self.base_s += rf.sum(0); self.base_n += len(u)
        base = ((self.base_s + 1) / (self.base_n + 100)).clamp(1e-3, 0.5); K = self.nctx
        p = ((self.Ds[:K, u] + 2 * base) / (self.Dn[:K, u, None] + 2)).clamp(1e-3, 1 - 1e-3)
        pf = ((self.Rf[u] + 2 * base) / (self.Vf[u, None] + 2)).clamp(1e-3, 1 - 1e-3)
        P = torch.cat([p, pf[None]], 0)
        ll = (rf * P.log() + (1 - rf) * (1 - P).log()).sum((1, 2))
        idx = list(range(K)) + [self.ctx]; self.Ls.mul_(self.lam)
        if self.ctx_by not in ("map", "possible", "possible+relief"):
            self.Ls[idx] += ll
        if self.ctx_by == "possible+relief":                            # S3e: where reliefs come is part of what a world is - weighed over a long
            if getattr(self, "Lr", None) is None:                       # window (fading .999, ~1,000 beats), since a spring running dry inside a
                self.Lr = torch.zeros_like(self.Ls)                     # world is brief and a moved spring in another world lasts
            self.Lr.mul_(0.999); self.Lr[idx] += ll
        c = self.act; self.Ds[c].mul_(self.decay); self.Dn[c].mul_(self.decay); self.Dn[c].index_put_((u,), torch.ones(len(u), device=self.dev), accumulate=True)
        self.Ds[c].index_put_((u[:, None].expand(-1, 2).reshape(-1), torch.arange(2, device=self.dev).repeat(len(u))), rf.reshape(-1), accumulate=True)
        self.Rf.mul_(self.fresh); self.Vf.mul_(self.fresh); self.Vf.index_put_((u,), torch.ones(len(u), device=self.dev), accumulate=True)
        self.Rf.index_put_((u[:, None].expand(-1, 2).reshape(-1), torch.arange(2, device=self.dev).repeat(len(u))), rf.reshape(-1), accumulate=True)
        tot = self.Ls + self.Lr if getattr(self, "Lr", None) is not None else self.Ls
        sc = tot[idx]; b = int(sc.argmax()); best = idx[b]; self.last[self.act] = self.beats
        if best != self.act and float(sc[b] - tot[self.act]) > self.margin:
            old = self.act
            if best == self.ctx:                                         # nothing held foretells this: a new context (or the longest unused one, emptied)
                if self.nctx < self.ctx:
                    best = self.nctx; self.nctx += 1
                else:
                    best = min(range(self.ctx), key=lambda c: self.last[c])
                self.Ds[best] = self.Rf; self.Dn[best] = self.Vf; self.Rs[best] = 0; self.Vns[best] = 0; self.Ts[best] = 0   # the newcomer's expectations become the new context's
            if self.Ts[best].sum() == 0:
                self.Ts[best] = self.Tf
            self.act = best; self.R, self.Vn, self.T = self.Rs[best], self.Vns[best], self.Ts[best]; self.Ls.zero_(); self.last[best] = self.beats
            if getattr(self, "Lr", None) is not None:
                self.Lr.zero_()
            self.Rf = torch.zeros_like(self.Rf); self.Vf = torch.zeros_like(self.Vf); self.Tf = torch.zeros_like(self.Tf)
            self.Th.zero_(); self.h_since = -1
            self.switches.append((self.beats, old, best)); self.refresh()

    @torch.no_grad()
    def resonate(self, x, rose, w=0.3):
        """S4 bedrock (founder 2026-11-25, S3_S4_PROTOCOL.md): another of its kind's seen inside rise [n, 2] (water, food; 0 where not seen) is felt
        as a small relief of the same inside where this body itself stands - remembered by its own place memory, weight w."""
        if self.C is None or not bool(rose.any()):
            return
        u = self.unit(self.place(x))
        self.R.index_put_((u[:, None].expand(-1, 2).reshape(-1), torch.arange(2, device=self.dev).repeat(len(u))), w * rose.float().reshape(-1), accumulate=True)

    @torch.no_grad()
    def refresh(self):
        if self.born and self.used < self.KU:                            # units not yet born are not on the map
            self.R[self.used:] = 0; self.T[self.used:] = 0; self.T[:, self.used:] = 0
        S = self.T + self.T.t(); Tm = S / S.sum(1, keepdim=True).clamp(min=1e-6); vis = self.Cn.clamp(min=1)
        r = self.R / (self.Vn.clamp(min=1)[:, None] if self.mended else vis[:, None])
        if self.dread:
            r = r - self.dread * (self.Dd / self.Vn.clamp(min=1))[:, None]
        V = torch.linalg.solve(torch.eye(self.KU, device=self.dev) - self.gamma * Tm, r)          # [KU, 2]
        nb = (S > 0.5) | torch.eye(self.KU, dtype=torch.bool, device=self.dev)
        self.nxt = torch.stack([torch.where(nb, V[None, :, c].expand(self.KU, -1), torch.full_like(S, -1e9)).argmax(1) for c in (0, 1)])   # [2, KU]
        self.V = V

    def learn(self, updates=1500, batch=1024):
        if self.mended:
            return self._learn_pieces(updates, batch)
        if self.rn < self.far + 2:
            return None
        L = self.rn; idx0 = (self.ri - L) % len(self.rX)
        if not self.fitted:
            Xf = self.rX[:L].reshape(-1, self.dx); self.skill.mu.copy_(Xf.mean(0)); self.skill.sd.copy_(Xf.std(0) + 1e-3); self.fitted = True
        le = 0.0
        with torch.enable_grad():
            for s in range(updates):
                o = torch.randint(0, L - 1, (batch,), device=self.dev); k = torch.randint(1, self.far + 1, (batch,), device=self.dev); o2 = (o + k).clamp(max=L - 1)
                t, t2 = (idx0 + o) % len(self.rX), (idx0 + o2) % len(self.rX); i = torch.randint(0, self.n, (batch,), device=self.dev)
                ok = self.rL[t, i] == self.rL[t2, i]; t, t2, i = t[ok], t2[ok], i[ok]
                x = self.rX[t, i].clone(); x[:, self.ins] = 0
                loss = F.cross_entropy(self.skill(x, self.place(self.rX[t2, i])), self.rA[t, i]); self.opt.zero_grad(); loss.backward(); self.opt.step()
                le += float(loss.detach()) / updates
        return le

    def _learn_pieces(self, updates, batch):
        held = min(self.pieces, self.slots)
        if held < 1:
            return None
        P = self.piece
        if not self.fitted:
            Xf = self.rX[:held * P].reshape(-1, self.dx); self.skill.mu.copy_(Xf.mean(0)); self.skill.sd.copy_(Xf.std(0) + 1e-3); self.fitted = True
        le = 0.0
        with torch.enable_grad():
            for s in range(updates):
                sl = torch.randint(0, held, (batch,), device=self.dev); k = torch.randint(1, self.far + 1, (batch,), device=self.dev)
                o = (torch.rand(batch, device=self.dev) * (P - k)).long(); t, t2 = sl * P + o, sl * P + o + k; i = torch.randint(0, self.n, (batch,), device=self.dev)
                ok = self.rL[t, i] == self.rL[t2, i]; t, t2, i = t[ok], t2[ok], i[ok]
                x = self.rX[t, i].clone(); x[:, self.ins] = 0
                loss = F.cross_entropy(self.skill(x, self.place(self.rX[t2, i])), self.rA[t, i]); self.opt.zero_grad(); loss.backward(); self.opt.step()
                le += float(loss.detach()) / updates
        return le

    @torch.no_grad()
    def toward(self, x, g):
        xx = x.clone(); xx[:, self.ins] = 0; return F.log_softmax(self.skill(xx, g), 1)

    @torch.no_grad()
    def lean(self, x):
        if not self.fitted or self.nxt is None:
            return None
        lack, c = self.needs(x); v = self.unit(self.place(x))
        for _ in range(self.hops):
            v = self.nxt[c, v]
        g = self.C[v]
        lp = self.toward(x, g); return self.W * lack[:, None] * (lp - lp.mean(1, keepdim=True))

    def state(self):
        d = dict(skill=self.skill.state_dict(), C=self.C, Cn=self.Cn, T=self.T, R=self.R, Vn=self.Vn, fitted=self.fitted, mended=self.mended, born=self.born, used=self.used)
        if self.ctx:
            d.update(ctx=self.ctx, Rs=self.Rs, Vns=self.Vns, Ds=self.Ds, Dn=self.Dn, Ts=self.Ts, act=self.act, nctx=self.nctx, switches=self.switches)
        return d
