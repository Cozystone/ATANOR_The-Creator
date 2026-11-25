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
    def __init__(self, dx, M, n, inside, dev, KU=64, ring=1024, far=32, gamma=0.99, W=1.0, decay=0.9995, mended=True, piece=64, born=True, hops=4):
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
        for k in range(self.KU):                                        # online k-means, each unit moves toward the places it wins
            m = u2 == k
            if m.any():
                self.Cn[k] += float(m.sum()); self.C[k] += (p2[m].mean(0) - self.C[k]) * min(1.0, float(m.sum()) / self.Cn[k].clamp(min=1).item() + (0.0 if self.mended else 0.01))
        self.T.mul_(self.decay); self.R.mul_(self.decay); self.Vn.mul_(self.decay)
        self.Vn.index_put_((u2[live],), torch.ones(int(live.sum()), device=self.dev), accumulate=True)
        self.T.index_put_((u[live], u2[live]), torch.ones(int(live.sum()), device=self.dev), accumulate=True)
        rose = torch.stack([x2[:, self.ins[2]] - x[:, self.ins[2]] > 0.01, x2[:, self.ins[1]] - x[:, self.ins[1]] > 0.01], 1) & live[:, None]
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
    def refresh(self):
        if self.born and self.used < self.KU:                            # units not yet born are not on the map
            self.R[self.used:] = 0; self.T[self.used:] = 0; self.T[:, self.used:] = 0
        S = self.T + self.T.t(); Tm = S / S.sum(1, keepdim=True).clamp(min=1e-6); vis = self.Cn.clamp(min=1)
        r = self.R / (self.Vn.clamp(min=1)[:, None] if self.mended else vis[:, None])
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
        return dict(skill=self.skill.state_dict(), C=self.C, Cn=self.Cn, T=self.T, R=self.R, Vn=self.Vn, fitted=self.fitted, mended=self.mended, born=self.born, used=self.used)
