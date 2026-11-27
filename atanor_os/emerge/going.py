"""The going (L1, atanor_os/P1_PATH_PROTOCOL.md): an organ of the mind that learns to take the body to a remembered place.
Reads the innate path organ's code (the last PathOrgan.N numbers of the sense). Everything here is learned from the being's own experience:
- place memory (fixed size): KU place units by online k-means of its place code; which unit follows which (decayed counts); for each inside, how often
  it rose in each unit (relief remembered with its place);
- the skill: sense + place code + goal place -> gesture, trained by hindsight on a bounded ring of its own recent beats (where it actually got to),
  never by reward (D33: Ghosh et al. 2019 GCSL; goal babbling, Rolf et al. 2010);
- the lean of the lack: for the lowest inside, the next unit on its own map toward where that inside was relieved (V = (I - g T)^-1 r, best
  neighbour); the skill's preference for going there, scaled by how low that inside is, is offered to the mind's choice score."""
import os, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from .body_contract import PathOrgan

NP = PathOrgan.N; PC = NP - 2                                          # place part of the code (the last two numbers are the heading)


class Skill(nn.Module):
    def __init__(self, dx, M):
        super().__init__(); self.register_buffer("mu", torch.zeros(dx)); self.register_buffer("sd", torch.ones(dx))
        self.f = nn.Sequential(nn.Linear(dx + PC, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, M))

    def forward(self, x, g):
        return self.f(torch.cat([((x - self.mu) / self.sd.clamp(min=0.05)).clamp(-5, 5), g], 1))


def _blend(*wv, wall=98.0):
    """B0: bilinear over a distance field, leaving out corners that are walls (no way) - a body against a wall is not 'far'."""
    num = 0.0; den = 0.0
    for w_, v_ in wv:
        ok = (v_ < wall).float(); num = num + w_ * ok * v_; den = den + w_ * ok
    return torch.where(den > 1e-6, num / den.clamp(min=1e-6), torch.full_like(num, 99.0))


class Going:
    def __init__(self, dx, M, n, inside, dev, KU=64, ring=1024, far=32, gamma=0.99, W=1.0, decay=0.9995, mended=True, piece=64, born=True, hops=4,
                 ctx=0, margin=20.0, lam=0.99, fresh=0.998, ctx_by="both", dread=0.0, hold=0, chan="room"):
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
        # G1 (K1_KNOWING_PROTOCOL.md): chan="contract" reads every bedrock channel the body contract declares, unnamed and in any order - one
        # relief memory per channel, the lowest channel is the lack. chan="room" (default) is the organ as it was: food = ins[1], water = ins[2].
        self.chan = chan; assert chan in ("room", "contract") and not (chan == "contract" and ctx), "contract channels: ctx not carried over yet"
        self.k = len(self.ins) if chan == "contract" else 2
        self.skill = Skill(dx, M).to(dev); self.opt = torch.optim.Adam(self.skill.f.parameters(), 3e-4); self.fitted = False
        self.C = None; self.Cn = torch.zeros(KU, device=dev); self.T = torch.zeros(KU, KU, device=dev); self.R = torch.zeros(KU, self.k, device=dev)
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

    @torch.no_grad()
    def _remap(self, p, live):
        """R1 (EMERGE_PLAN 2026-11-27, registered in WORLD_A_PROTOCOL.md): place cells remap. The number of units stays fixed (KU). When a body
        stands farther than one cell's worth of code (rad, the same rule units are born by) from every unit, the unit it has gone longest without
        visiting - if that is longer than its own map's memory (1 / (1 - decay) beats) - is cleared and placed here: its code, its relief memory,
        its crossings and its homing position start afresh. Nothing named: which units are used is its own walking. At the first beat of a life
        in this world every unit counts as last visited one memory span ago (it comes from elsewhere)."""
        span = 1.0 / (1.0 - self.decay)
        if getattr(self, "lastv", None) is None:
            self.lastv = torch.full((self.KU,), float(self.beats) - span, device=self.dev); self.remaps = 0
        for _ in range(4):                                              # at most a few a beat, as births
            d = torch.cdist(p, self.C[:self.used]).min(1).values; d = torch.where(live, d, torch.zeros_like(d)); i = int(d.argmax())
            if float(d[i]) <= self.rad:
                return
            k = int(self.lastv.argmin())
            if float(self.beats) - float(self.lastv[k]) <= span:
                return
            self.C[k] = p[i]; self.Cn[k] = 1.0; self.R[k] = 0; self.Vn[k] = 0; self.T[k, :] = 0; self.T[:, k] = 0; self.Dd[k] = 0; self.Th[k, :] = 0; self.Th[:, k] = 0
            if getattr(self, "V", None) is not None:
                self.V[k] = 0
            if getattr(self, "Pu", None) is not None and getattr(self, "organs", None) is not None:
                self.Pu[k] = self.organs.path.p[i]; self.Pn[k] = 1.0
            self.lastv[k] = float(self.beats); self.remaps += 1; self._fields = {}

    @torch.no_grad()
    def _r2_observe(self, x, a, x2, died):
        """R2 (EMERGE_PLAN 2026-11-27; Astra's prototype, LOKI c7c304c loki_memory/place_bundles.py, fetched with the founder's approval):
        64 place units shared by every world; what was relieved where, and which crossings were walked, kept per context (3 bundles, fixed
        size). Which context is now: its own reliefs and its own blocked moves where it stands, against each bundle's record; none fits ->
        read a blank one (no old world's relief pulls a goal). This organ then only reads the bundles' snapshot (C, R, Vn, T, Pu) - it no
        longer learns places itself. Consolidation ("sleep") every 100 beats with the map's refresh - an interim stand-in for real sleep."""
        if getattr(self, "r2", None) is None:
            import sys as _s
            _s.path.insert(0, "ext/LOKI-v13")
            from loki_memory.place_bundles import PlaceBundles
            self.r2 = PlaceBundles(channels=self.k, bodies=self.n, slots=self.KU, bundles=3).to(self.dev)
            if getattr(self, "Pu", None) is None:
                self.Pu = torch.zeros(self.KU, 2, device=self.dev); self.Pn = torch.zeros(self.KU, device=self.dev)
            self.r2.import_going(self)                                     # the grown being's map as context 0 (its earlier world)
            u_ = int(self.used); self.r2.Pu[:u_] = self.r2.odometry(self.r2.C[:u_])   # each unit's place, read from its own path code
            self.r2_read = -1; self.r2_reads = []; self.r2_wp = self.organs.path.p.clone(); self.r2_n = 0
        live = ~died; p2 = self.place(x2)
        if self.chan == "contract":
            rise = (x2[:, self.ins] - x[:, self.ins]).clamp(0, 1)
        else:
            rise = torch.stack([x2[:, self.ins[2]] - x[:, self.ins[2]], x2[:, self.ins[1]] - x[:, self.ins[1]]], 1).clamp(0, 1)
        rose = (rise > 0.01).float()
        fo = self._foreseen(x).gather(1, a[:, None, None].expand(-1, 1, 3))[:, 0]
        moved = (self.organs.path.p - self.r2_wp).norm(dim=1); self.r2_wp = self.organs.path.p.clone()
        err = ((fo[:, :2].norm(dim=1) > 0.15) & (moved < 0.03)).float()    # it pushed (its forecaster foresaw a step) and did not move
        b, _sup = self.r2.observe(p2, rose, err, live)
        self.r2_n += 1
        if self.r2_n % 100 == 0:
            self.r2.sleep(); b = int(self.r2.reading) if int(self.r2.reading) >= 0 else self.r2.identify()[0]
        self.r2_read = b
        if len(self.r2_reads) < 10000:
            self.r2_reads.append(b)
        sn = self.r2.going_snapshot(b)
        self.C = sn["C"].clone(); self.Cn = sn["Cn"].clone(); self.T = sn["T"].clone(); self.R = sn["R"].clone(); self.Vn = sn["Vn"].clone()
        self.Pu = sn["Pu"].clone(); self.Pn = sn["Pn"].clone(); self.used = max(1, int(sn["used"]))
        self._fields = {}
        if self.mended:                                                  # its going skill still learns from lived pieces, as before
            j = self.slots * self.piece + self.ri; self.rX[j] = x; self.rA[j] = a; self.rL[j] = self.life; self.ri += 1
            if self.ri == self.piece:
                self.pieces += 1; s_ = self.pieces - 1 if self.pieces <= self.slots else int(torch.randint(0, self.pieces, (1,)))
                if s_ < self.slots:
                    sl, cur = slice(s_ * self.piece, (s_ + 1) * self.piece), slice(self.slots * self.piece, (self.slots + 1) * self.piece)
                    self.rX[sl] = self.rX[cur]; self.rA[sl] = self.rA[cur]; self.rL[sl] = self.rL[cur]
                self.ri = 0
        self.life = self.life + died.long(); self.since += 1; self.beats += 1
        if self.since >= 100 or self.nxt is None:
            self.refresh(); self.since = 0

    def needs(self, x):
        """lack per body (1 - lowest inside) and which store is lowest (0 water, 1 food: the world's store order; contract: the channel's own place)."""
        if self.chan == "contract":
            b = x[:, self.ins]; m = b.min(1); return 1.0 - m.values, m.indices
        food, water = x[:, self.ins[1]], x[:, self.ins[2]]
        return 1.0 - torch.minimum(food, water), (food < water).long()

    @torch.no_grad()
    def observe(self, x, a, x2, died):
        if os.environ.get("ATANOR_SEEN", "0") == "1":                       # SM2: which inside rose this beat (channel order of needs)
            self._sm_rose = ((x2[:, self.ins] - x[:, self.ins] > 0.01) if self.chan == "contract" else torch.stack([x2[:, self.ins[2]] - x[:, self.ins[2]] > 0.01, x2[:, self.ins[1]] - x[:, self.ins[1]] > 0.01], 1)) & ~died[:, None]
        if (os.environ.get("ATANOR_IMAGINE_MAP", "0") == "1" or os.environ.get("ATANOR_B0", "0") == "own") and getattr(self, "organs", None) is not None and getattr(self, "net", None) is not None:
            self.wall_map_observe(x, a, died)                                   # IM1: its own wall map grows from where it walked and where it was stopped
        if os.environ.get("ATANOR_R2", "0") == "1" and self.C is not None and getattr(self, "organs", None) is not None:
            return self._r2_observe(x, a, x2, died)                     # R2 (Astra, LOKI c7c304c): the place memory is the bundles organ's
        if os.environ.get("ATANOR_B0", "0") != "0":                       # B0: its own felt drain a beat (insides falling when nothing rose)
            st_ = torch.stack([x[:, self.ins[2]], x[:, self.ins[1]]], 1); st2_ = torch.stack([x2[:, self.ins[2]], x2[:, self.ins[1]]], 1)
            fall = (st_ - st2_); quiet = (fall > -0.005).all(1) & ~died
            if bool(quiet.any()):
                self.b0_dr = 0.99 * float(getattr(self, "b0_dr", 0.001)) + 0.01 * float(fall[quiet].clamp(min=0).mean())
            if getattr(self, "organs", None) is not None:                 # its own free pace: how far a beat moves it when it moves freely
                pp_ = getattr(self, "_b0_pp", None); P_ = self.organs.path.p
                if pp_ is not None and pp_.shape == P_.shape:
                    mv = (P_ - pp_).norm(dim=1); fr = (mv > 0.1) & (mv < 2.0) & ~died
                    if bool(fr.any()):
                        self.b0_pace = 0.99 * float(getattr(self, "b0_pace", 0.3)) + 0.01 * float(mv[fr].quantile(0.9))
                self._b0_pp = P_.clone()
        p, p2 = self.place(x), self.place(x2)
        if self.C is None:                                              # the first places it is in become its first units
            self.C = p[torch.randint(0, len(p), (self.KU,), device=self.dev)] + 0.01 * torch.randn(self.KU, PC, device=self.dev)
            if self.born:
                self.C[0] = p[0]; self.used = 1
        if self.born:
            self._recruit(p2)
        if os.environ.get("ATANOR_REMAP", "0") == "1" and self.used >= self.KU:
            self._remap(p2, ~died)                                      # R1: a place no unit covers takes the unit it has gone longest without
        u, u2 = self.unit(p), self.unit(p2); live = ~died
        if getattr(self, "homing", False) and getattr(self, "organs", None) is not None and getattr(self.organs, "path", None) is not None:
            self.keep_where(u2, live)
        if getattr(self, "lastv", None) is not None:                     # R1: when each unit was last stood in
            self.lastv[torch.unique(u2[live])] = float(self.beats)
        cnt = torch.bincount(u2, minlength=self.KU).float()            # online k-means, each unit moves toward the places it wins (one step for all units;
        sm = torch.zeros(self.KU, PC, device=self.dev).index_add_(0, u2, p2)   # same numbers as the old per-unit loop, checked 2026-11-25)
        self.Cn += cnt; hit = cnt > 0; rate = (cnt / self.Cn.clamp(min=1) + (0.0 if self.mended else 0.01)).clamp(max=1.0)
        self.C = torch.where(hit[:, None], self.C + (sm / cnt.clamp(min=1)[:, None] - self.C) * rate[:, None], self.C)
        if self.ctx and bool(live.any()):                               # S3b: the map's expectation is scored before this beat is learned
            self._moves(u[live], u2[live])
        rd = getattr(self, "rdecay", None) or self.decay                 # V1b (WORLD_A_PROTOCOL.md; default = decay): where relief came (R, with its visits Vn) may fade faster than the map (T)
        self.T.mul_(self.decay); self.R.mul_(rd); self.Vn.mul_(rd)
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
        rose = ((x2[:, self.ins] - x[:, self.ins] > 0.01) if self.chan == "contract" else torch.stack([x2[:, self.ins[2]] - x[:, self.ins[2]] > 0.01, x2[:, self.ins[1]] - x[:, self.ins[1]] > 0.01], 1)) & live[:, None]
        self.beats += 1
        if self.ctx and bool(live.any()):
            sel = live if self.do_mask is None else live & self.do_mask[a]
            if bool(sel.any()):
                self._evidence(u2[sel], rose[sel])
        self.R.index_put_((u2[:, None].expand(-1, self.k).reshape(-1), torch.arange(self.k, device=self.dev).repeat(len(u2))), rose.reshape(-1).float(), accumulate=True)
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
        V = torch.linalg.solve(torch.eye(self.KU, device=self.dev) - self.gamma * Tm, r)          # [KU, k]
        nb = (S > 0.5) | torch.eye(self.KU, dtype=torch.bool, device=self.dev)
        self.nxt = torch.stack([torch.where(nb, V[None, :, c].expand(self.KU, -1), torch.full_like(S, -1e9)).argmax(1) for c in range(self.R.shape[1])])   # [k, KU]
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
        if getattr(self, "homing_lean", False) and getattr(self, "Pu", None) is not None and getattr(self, "organs", None) is not None and getattr(self, "net", None) is not None:
            return self._home(x, v, torch.ones_like(lack, dtype=torch.bool), lack)     # D47 (not met; off): its own lean walks by the homing vector to the unit its chain aims at
        g = self.C[v]
        lp = self.toward(x, g); return self.W * lack[:, None] * (lp - lp.mean(1, keepdim=True))

    @torch.no_grad()
    def lean_to(self, x, goal):
        """B1 (WORLD_A_PROTOCOL.md): the same lean, but toward a goal unit given from outside (goal [n] long, -1 = none) - e.g. where a memory
        organ (LOKI v12) believes the lacking channel will be relieved now. The way there follows this organ's own map: a shortest chain over
        crossings it has lived (T), `hops` units ahead; the walking is its own skill. Bodies without a goal or a lived way get 0."""
        if not self.fitted or self.nxt is None:
            return None
        if getattr(self, "oracle", None) is not None:                          # G2L ladder (judge-side diagnostic, never a pass): the judge gives the vector in the body's frame
            lack, c = self.needs(x); bx, by = self.oracle(c)
            return self._home(x, None, lack > 0.3, lack, vec=(bx, by))
        lack, _ = self.needs(x); v = self.unit(self.place(x)); n = len(v)
        S = ((self.T + self.T.t()) > 0.5).cpu().numpy(); KU = self.KU          # one copy to the host (speed, 2026-11-26; same answers as the per-body loop)
        nb = [np.nonzero(S[i])[0].tolist() for i in range(KU)]
        gl = goal.tolist(); vl = v.tolist(); tl = list(vl); okl = [False] * n; trees = {}
        for i in range(n):
            gi = gl[i]
            if gi < 0:
                continue
            if gi == vl[i]:
                tl[i] = gi; okl[i] = True; continue
            prev = trees.get(vl[i])
            if prev is None:                                               # breadth-first over its own lived crossings, once per starting unit
                prev = {vl[i]: None}; q = [vl[i]]; qi = 0
                while qi < len(q):
                    u = q[qi]; qi += 1
                    for w_ in nb[u]:
                        if w_ not in prev:
                            prev[w_] = u; q.append(w_)
                trees[vl[i]] = prev
            if gi in prev:
                path = [gi]
                while prev[path[-1]] is not None:
                    path.append(prev[path[-1]])
                path = path[::-1]; tl[i] = path[min(self.hops, len(path) - 1)]; okl[i] = True
        tgt = torch.tensor(tl, device=self.dev, dtype=torch.long); ok = torch.tensor(okl, device=self.dev, dtype=torch.bool)
        self._last_goal = goal.clone()
        has_ = goal >= 0
        self.n_goal = getattr(self, "n_goal", 0) + int(has_.sum()); self.n_noway = getattr(self, "n_noway", 0) + int((has_ & ~ok).sum())   # judge-side count
        if os.environ.get("ATANOR_IMAGINE_GOAL", "0") == "1" and getattr(self, "wmap", None) is not None and getattr(self, "Pu", None) is not None:
            nw_ = has_ & ~ok                                                # IM2: no lived way to its goal - imagine one on its own wall map to the goal itself
            if bool(nw_.any()):
                tgt = torch.where(nw_, goal.clamp(min=0), tgt); ok = ok | nw_; self._im_only = nw_
        if getattr(self, "homing", False) and getattr(self, "Pu", None) is not None and getattr(self, "organs", None) is not None and getattr(self, "net", None) is not None:
            return self._home(x, tgt, ok, lack)
        lp = self.toward(x, self.C[tgt]); out = self.W * lack[:, None] * (lp - lp.mean(1, keepdim=True))
        return torch.where(ok[:, None], out, torch.zeros_like(out))

    @torch.no_grad()
    def keep_where(self, u2, live):
        """D46 (innate homing vector, founder 2026-11-26): each place unit keeps the mean position the path organ held there (the body's own
        integrated position in its birth frame, not the judge's)."""
        P = self.organs.path.p
        if getattr(self, "Pu", None) is None:
            self.Pu = torch.zeros(self.KU, 2, device=self.dev); self.Pn = torch.zeros(self.KU, device=self.dev)
        if bool(live.any()):
            cnt = torch.bincount(u2[live], minlength=self.KU).float(); sm = torch.zeros(self.KU, 2, device=self.dev).index_add_(0, u2[live], P[live]); hit = cnt > 0
            self.Pn += cnt; r = (cnt / self.Pn.clamp(min=1)).clamp(min=0.01)
            self.Pu = torch.where(hit[:, None], self.Pu + (sm / cnt.clamp(min=1)[:, None] - self.Pu) * r[:, None], self.Pu)

    @torch.no_grad()
    def _foreseen(self, x):
        """Its own forecaster's felt motion per gesture [n, M, 3] (forward, side, turn) in the body's own units."""
        O = self.organs; nb = sum(int(np.prod(O.c.sensors[i].shape)) for i in O.bed); off = nb
        for i in O.vec:
            if i == O.mo:
                break
            off += int(np.prod(O.c.sensors[i].shape))
        st = O.stats[O.mo]; idx = torch.tensor(O.mo_idx, device=self.dev)
        fo = self.net.foresee(x)[:, :, off + idx]
        return fo * 4 * st.s[idx] + st.m[idx]

    @torch.no_grad()
    def b0_foresight(self, x, goal, H=300, hold=3):
        """B0 (Plan B, first step; EMERGE_PLAN 2026-11-27 "Yes, build B"): choice by imagination, scored by the bedrock only.
        For each gesture: its own forecaster says where holding it `hold` beats takes the body (a turn is one act, then its walking twin);
        from there the imagined way to the remembered relief place (its goal unit's own place) around the walls it knows (its own wall map;
        unknown cells passable) gives the beats until relief at its own pace. Its stores drain at `drain` a beat; the lacking one refills at
        arrival if that is within H. Score = imagined wholeness at H (1 - mean squared deficit). Returns (score [n, M], active [n]) - active
        where a goal is held and the imagined ways differ. mode "oracle" (judge-side diagnostic, never a pass): the judge's true field in
        the world frame (self.b0_true) instead of its own map."""
        n = x.shape[0]; M = self.M; dev = self.dev
        act_ = (goal >= 0)
        if not bool(act_.any()):
            return None, act_
        lack, c = self.needs(x)
        store = torch.stack([x[:, self.ins[2]], x[:, self.ins[1]]], 1) if self.chan == "room" else x[:, self.ins]   # its own insides (channel order of needs)
        drain = float(getattr(self, "b0_dr", 0.001))                          # its own felt drain a beat (learned in observe)
        fo = self._foreseen(x)                                              # [n, M, 3] forward, side, turn per beat (its own units)
        f, sd, tn = fo[..., 0], fo[..., 1], fo[..., 2]
        pace = torch.full((n,), float(getattr(self, "b0_pace", 0.3)), device=dev)   # its own free walking pace a beat (learned in observe)
        mag = (f * f + sd * sd).sqrt()
        moving = mag > 0.5 * mag.amax(1, keepdim=True).clamp(min=0.1)       # its forecaster foresees this gesture moving the body (vs turning in place)
        uf = torch.where(moving, f / mag.clamp(min=1e-6), torch.zeros_like(f)); us = torch.where(moving, sd / mag.clamp(min=1e-6), torch.zeros_like(sd))
        f, sd = uf * pace[:, None], us * pace[:, None]                      # imagined: the gesture's own direction at its free pace (walls come from the map)
        oracle = getattr(self, "b0_true", None)
        if oracle is not None:
            p, h, look = oracle["pos"](), oracle["head"](), (lambda xy, near=False: oracle["field"](c, xy, near))
        else:
            O = self.organs; p, h = O.path.p, O.path.h; Gs = 64; c0 = Gs // 2; wver = int((self.wmap > 1.0).sum())
            if getattr(self, "_fver", None) != wver:
                self._fields = {}; self._fver = wver
            gl = goal.clamp(min=0).tolist()
            for g in set(gl):
                if g not in self._fields:
                    D = self.imagine_route(self.Pu[g].tolist()); D[~np.isfinite(D)] = 99.0
                    self._fields[g] = torch.tensor(D, device=dev, dtype=torch.float32)
            Fd = torch.stack([self._fields[g] for g in gl])
            def look(xy, near=False):                                      # bilinear over its map's cells (cell centres at integers)
                if near:
                    i_ = (xy[..., 0] + c0).round().long().clamp(0, Gs - 1); j_ = (xy[..., 1] + c0).round().long().clamp(0, Gs - 1)
                    return Fd[torch.arange(n, device=dev).view(-1, *([1] * (xy.dim() - 2))), i_, j_]
                u_ = (xy[..., 0] + c0).clamp(0, Gs - 1.001); v_ = (xy[..., 1] + c0).clamp(0, Gs - 1.001)
                i0, j0 = u_.floor().long(), v_.floor().long(); a_, b_ = u_ - i0, v_ - j0; bi = torch.arange(n, device=dev).view(-1, *([1] * (xy.dim() - 2)))
                g_ = lambda i, j: Fd[bi, i.clamp(0, Gs - 1), j.clamp(0, Gs - 1)]
                return _blend(((1 - a_) * (1 - b_), g_(i0, j0)), (a_ * (1 - b_), g_(i0 + 1, j0)), ((1 - a_) * b_, g_(i0, j0 + 1)), (a_ * b_, g_(i0 + 1, j0 + 1)))
        h2 = h[:, None] + tn                                                # this gesture: its turn (one act) and one beat of its own motion
        cs, sn = torch.cos(h2), torch.sin(h2)
        p1 = torch.stack([p[:, 0:1] + f * cs - sd * sn, p[:, 1:2] + f * sn + sd * cs], -1)   # [n, M, 2]
        p1 = torch.where((look(p1, near=True) >= 98.0)[..., None], p[:, None, :].expand_as(p1), p1)   # into a wall it knows: it stays
        d_here = look(p[:, None, :])
        # then, in its head: further one-act turns (30 degrees each) to any of 12 headings, and walking `hold` beats that way at its pace
        th = torch.arange(12, device=dev, dtype=torch.float32) * (math.pi / 6)                         # [12]
        dturn = torch.remainder(th[None, None, :] - h2[..., None] + math.pi, 2 * math.pi) - math.pi    # [n, M, 12]
        acts = (dturn.abs() / (math.pi / 6)).round()
        step = hold * pace[:, None, None]
        p2 = torch.stack([p1[..., 0:1] + step * torch.cos(th), p1[..., 1:2] + step * torch.sin(th)], -1)  # [n, M, 12, 2]
        pm = 0.5 * (p1[..., None, :] + p2)                                  # walking into a wall it knows stops it there (half-way checked too)
        stop = (look(pm, near=True) >= 98.0) | (look(p2, near=True) >= 98.0)
        p2 = torch.where(stop[..., None], p1[..., None, :].expand_as(p2), p2)
        cost = 1 + acts + hold + look(p2) / pace[:, None, None]
        beats = cost.amin(-1)                                               # its best imagined way after this gesture
        s0 = store[:, None, :].expand(-1, M, -1).clone() - drain * H
        reach = beats <= H
        refill = torch.zeros_like(s0); refill.scatter_(2, c[:, None, None].expand(-1, M, 1), 1.0)
        s_end = torch.where(reach[..., None] & (refill > 0), torch.ones_like(s0) - drain * (H - beats[..., None]).clamp(min=0), s0)
        whole = 1.0 - ((1.0 - s_end.clamp(0, 1)) ** 2).mean(-1)
        spread = whole.amax(1) - whole.amin(1)
        return whole, act_ & (spread > 1e-4) & (d_here[:, 0] > 1.0)

    def wall_map_observe(self, x, a, died):
        with torch.no_grad():
            return self._wall_map_observe(x, a, died)

    def _wall_map_observe(self, x, a, died):
        """IM1 (registered): the being's own wall map in its own path frame (fixed 64 x 64, one cell = one of its own steps' units).
        Free where it has walked; a wall where it tried to move (its forecaster foresaw a step) and its own felt motion stayed ~0.
        Its own senses only (path sense, forecaster, felt motion); nothing from the judge."""
        O = self.organs; p = O.path.p; h = O.path.h; Gs = 64; c0 = Gs // 2
        if getattr(self, "wmap", None) is None:
            self.wmap = torch.zeros(Gs, Gs, device=self.dev); self.wp_prev = p.clone(); self.wfo = None
        live = ~died
        cell = (p + c0).round().long().clamp(0, Gs - 1)
        self.wmap[cell[live, 0], cell[live, 1]] = (self.wmap[cell[live, 0], cell[live, 1]] - 1.0).clamp(min=-4.0)       # walked: free
        if True:
            fo = self._foreseen(x).gather(1, a[:, None, None].expand(-1, 1, 3))[:, 0]                                            # what this gesture was foreseen to do
            moved = (p - self.wp_prev).norm(dim=1); tried = (fo[:, :2].norm(dim=1) > 0.15) & (moved < 0.03) & live
            if bool(tried.any()):
                ang = h[tried] + torch.atan2(fo[tried, 1], fo[tried, 0]); q = p[tried] + 0.7 * torch.stack([torch.cos(ang), torch.sin(ang)], 1)
                wc = (q + c0).round().long().clamp(0, Gs - 1)
                self.wmap[wc[:, 0], wc[:, 1]] = (self.wmap[wc[:, 0], wc[:, 1]] + 2.0).clamp(max=4.0)                     # pushed and did not move: wall
        if os.environ.get("ATANOR_SEEN", "0") == "1" and getattr(O, "view", None) is not None:
            with torch.enable_grad():
                self._seen(O.view, p, h, fo, tried, moved, live)
        self.wp_prev = p.clone()

    def _seen(self, view, p, h, fo, tried, moved, live):
        """SM1 (founder 2026-11-27: "does what it sees enter its map? a person takes in far structure by looking"): from its own walking it
        learns how far the way ahead is free in the picture it sees - the teacher is its own feet: each picture of the last 32 beats, taken
        facing the way it is now going, is paired with how far it then got (exactly, when it is stopped by a wall; at least that far, while
        it still walks). Then looking alone writes its map: free as far as it expects, a faint wall where it expects to be stopped (weaker
        than a wall it has touched; looking through a touched wall for a while clears it - an opened way). No geometry is given: a small
        net from a 12 x 20 picture to one number, grown from scratch. It writes only once its own error on new stops is under a cell."""
        n = p.shape[0]; dev = self.dev; L = 32
        if os.environ.get("ATANOR_PARALLAX", "0") == "1":                     # P1 (registered 2026-11-28): depth from its own motion - how the picture
            pv = getattr(self, "_par_v", None); pp = getattr(self, "_par_p", None); ph_ = getattr(self, "_par_h", None)   # changed with its own last step
            if pv is None or pv.shape != view.shape:
                pv, pp, ph_ = view.clone(), p.clone(), h.clone()
            dv_ = p - pp; c_, s_ = torch.cos(h), torch.sin(h)
            step_ = torch.stack([dv_[:, 0] * c_ + dv_[:, 1] * s_, -dv_[:, 0] * s_ + dv_[:, 1] * c_, torch.remainder(h - ph_ + math.pi, 2 * math.pi) - math.pi], 1)
            self._par_v, self._par_p, self._par_h = view.clone(), p.clone(), h.clone()
            view = torch.cat([view, view - pv, step_.clamp(-2, 2)], 1)
        if getattr(self, "sm_net", None) is None:
            self.sm_net = nn.Sequential(nn.Linear(view.shape[1], 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)).to(dev)
            self.sm_opt = torch.optim.Adam(self.sm_net.parameters(), 1e-3)
            self.sm_v = torch.zeros(L, n, view.shape[1], device=dev); self.sm_p = torch.zeros(L, n, 2, device=dev); self.sm_h = torch.zeros(L, n, device=dev)
            self.sm_ok = torch.zeros(L, n, dtype=torch.bool, device=dev); self.sm_i = 0
            self.sm_X = torch.zeros(4096, view.shape[1], device=dev); self.sm_Y = torch.zeros(4096, device=dev); self.sm_k = 0; self.sm_err = 9.0; self.sm_n = 0
            sf = os.environ.get("ATANOR_SEEN_FROM", "")
            if sf:
                self._seen_carry = torch.load(sf, map_location=dev, weights_only=False); c_ = self._seen_carry
                self.sm_net.load_state_dict(c_["sm"]); self.sm_err, self.sm_n = float(c_["sm_err"]), int(c_["sm_n"])
        fwd = torch.atan2(fo[:, 1], fo[:, 0]).abs() < 0.35                    # this gesture was (foreseen) a walk ahead
        u = torch.stack([torch.cos(self.sm_h), torch.sin(self.sm_h)], -1)     # [L, n, 2] where each kept picture was facing
        v = p[None] - self.sm_p; d = (v * u).sum(-1); lat = (v[..., 0] * u[..., 1] - v[..., 1] * u[..., 0]).abs()
        dh = (torch.remainder(self.sm_h - h[None] + math.pi, 2 * math.pi) - math.pi).abs()
        same = self.sm_ok & (dh < 0.2) & (lat < 0.6) & (d >= 0)
        stop = same & (tried & fwd & live)[None]; go_ = same & ((moved > 0.15) & fwd & live & ~tried)[None]
        loss = 0.0; k = 0
        if bool(stop.any()):                                                  # stopped: the pictures behind it on this line learn the exact way
            X = self.sm_v[stop]; Y = d[stop] + 0.5
            with torch.no_grad():
                e = float((F.softplus(self.sm_net(X))[:, 0] - Y).abs().mean()); self.sm_err = 0.98 * self.sm_err + 0.02 * e; self.sm_n += len(Y)
            idx = (torch.arange(len(Y), device=dev) + self.sm_k) % 4096; self.sm_X[idx] = X; self.sm_Y[idx] = Y; self.sm_k += len(Y)
            self.sm_ok = self.sm_ok & ~stop                                   # used once
        m = min(self.sm_k, 4096)
        if m >= 64:
            j = torch.randint(0, m, (256,), device=dev)
            loss = loss + F.smooth_l1_loss(F.softplus(self.sm_net(self.sm_X[j]))[:, 0], self.sm_Y[j]); k += 1
        if bool(go_.any()):                                                   # still walking: at least this far was free
            sel = torch.nonzero(go_.reshape(-1))[:, 0]; sel = sel[torch.randint(0, len(sel), (min(256, len(sel)),), device=dev)]
            loss = loss + F.relu(d.reshape(-1)[sel] + 0.3 - F.softplus(self.sm_net(self.sm_v.reshape(L * n, -1)[sel]))[:, 0]).mean(); k += 1
        if k:
            self.sm_opt.zero_grad(); loss.backward(); self.sm_opt.step()
        self._seen_relief(view, p, h, live, L)
        self.sm_v[self.sm_i] = view; self.sm_p[self.sm_i] = p; self.sm_h[self.sm_i] = h; self.sm_ok[self.sm_i] = live; self.sm_i = (self.sm_i + 1) % L
        with torch.no_grad():
            dhat = F.softplus(self.sm_net(view))[:, 0].clamp(max=20.0); self.sm_last = dhat
            if self.sm_n >= 300 and self.sm_err < 1.0:                        # it trusts its eye only once its eye has been right about new stops
                Gs = 64; c0 = Gs // 2; uh = torch.stack([torch.cos(h), torch.sin(h)], 1)
                for s_ in range(1, 15):
                    q = p + uh * float(s_); on = live & (dhat - 0.5 > s_)
                    if bool(on.any()):
                        c = (q[on] + c0).round().long().clamp(0, Gs - 1)
                        self.wmap[c[:, 0], c[:, 1]] = (self.wmap[c[:, 0], c[:, 1]] - 0.3).clamp(min=-4.0)       # seen free
                q = p + uh * (dhat[:, None] + 0.2); c = (q[live] + c0).round().long().clamp(0, Gs - 1)
                cur = self.wmap[c[:, 0], c[:, 1]]
                self.wmap[c[:, 0], c[:, 1]] = torch.where(cur < 2.0, (cur + 0.5).clamp(max=2.0), cur)                # seen wall (fainter than touched)
                self.sm_writes = getattr(self, "sm_writes", 0) + int(live.sum())

    def seen_state(self):
        """SM1 / SM2 as part of the being: what its eye has learned from its feet and its insides (carried from a 64-body life to one body)."""
        return dict(sm=self.sm_net.state_dict(), sm_err=self.sm_err, sm_n=self.sm_n, sr=self.sr_net.state_dict(), sr_hit=self.sr_hit, sr_n=self.sr_n)

    def _seen_relief(self, view, p, h, live, L):
        """SM2 (founder 2026-11-27 on the viewer: "the water is in view and it does not look at it"): what it sees enters its relief memory.
        When one of its insides rises, the pictures of the last 32 beats that were facing the place where it rose learn "relief of this
        kind lies ahead, this far" (teacher: its own insides and its own path sense); pictures kept long without relief learn "none". Then
        looking alone puts an expectation of relief at that place in its own place memory (fainter than lived relief) - so a spring it sees
        can become its goal before it has ever drunk there in this world. Nothing named: the kinds are its own inside channels."""
        n = p.shape[0]; dev = self.dev; k = self.k; rose = getattr(self, "_sm_rose", None)
        if getattr(self, "sr_net", None) is None:
            self.sr_net = nn.Sequential(nn.Linear(view.shape[1], 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 3 * k)).to(dev)   # per channel: is it ahead, how far, and where in the view (bearing)
            self.sr_opt = torch.optim.Adam(self.sr_net.parameters(), 1e-3)
            self.sr_X = torch.zeros(4096, view.shape[1], device=dev); self.sr_C = torch.zeros(4096, dtype=torch.long, device=dev); self.sr_D = torch.zeros(4096, device=dev); self.sr_B = torch.zeros(4096, device=dev)
            self.sr_k = 0; self.sr_hit = 0.0; self.sr_n = 0; self.sr_writes = 0; self.sr_NX = torch.zeros(4096, view.shape[1], device=dev); self.sr_nk = 0
            c_ = getattr(self, "_seen_carry", None)
            if c_ is not None:
                self.sr_net.load_state_dict(c_["sr"]); self.sr_hit, self.sr_n = float(c_["sr_hit"]), int(c_["sr_n"])
        if rose is not None and bool(rose.any()):
            lg = getattr(self, "_last_goal", None)
            for b, c in torch.nonzero(rose & live[:, None]).tolist():
                if lg is not None and lg.shape[0] == n and int(lg[b]) >= 0 and getattr(self, "Pu", None) is not None and float(self.Pn[int(lg[b])]) > 0:
                    e_ = float((self.Pu[int(lg[b])] - p[b]).norm())             # R5: how far its memory's place was from where relief came
                    self.mem_err = 0.95 * float(getattr(self, "mem_err", 2.0)) + 0.05 * e_; self.mem_errn = getattr(self, "mem_errn", 0) + 1
                v = p[b][None] - self.sm_p[:, b]; d = v.norm(dim=1); ang = torch.atan2(v[:, 1], v[:, 0])
                bear = torch.remainder(ang - self.sm_h[:, b] + math.pi, 2 * math.pi) - math.pi       # where it was in the view, by its own path sense
                face = bear.abs() < 0.65
                sel = self.sm_ok[:, b] & face & (d > 0.8) & (d < 14)
                if bool(sel.any()):
                    X = self.sm_v[sel, b]; m = len(X)
                    with torch.no_grad():
                        o = self.sr_net(X)
                        dq = F.softplus(o[:, k + c]).clamp(max=14.0); bq = o[:, 2 * k + c].clamp(-0.8, 0.8); hq = self.sm_h[sel, b] + bq
                        q_ = self.sm_p[sel, b] + dq[:, None] * torch.stack([torch.cos(hq), torch.sin(hq)], 1)
                        self.eye_err = 0.95 * float(getattr(self, "eye_err", 2.0)) + 0.05 * float((q_ - p[b][None]).norm(dim=1).mean())   # R5: the eye's own
                        self.sr_hit = 0.98 * self.sr_hit + 0.02 * float((torch.sigmoid(o[:, c]) > 0.5).float().mean()); self.sr_n += m
                    idx = (torch.arange(m, device=dev) + self.sr_k) % 4096; self.sr_X[idx] = X; self.sr_C[idx] = c; self.sr_D[idx] = d[sel]; self.sr_B[idx] = bear[sel]; self.sr_k += m
        old = (self.sm_i + 1) % L; keep = self.sm_ok[old]                    # the oldest kept pictures: no relief followed within 32 beats
        if bool(keep.any()):
            X = self.sm_v[old][keep][:8]; idx = (torch.arange(len(X), device=dev) + self.sr_nk) % 4096; self.sr_NX[idx] = X; self.sr_nk += len(X)
        mp, mn = min(self.sr_k, 4096), min(self.sr_nk, 4096)
        if mp >= 32 and mn >= 32:
            j = torch.randint(0, mp, (128,), device=dev); jn = torch.randint(0, mn, (128,), device=dev)
            o = self.sr_net(self.sr_X[j]); on = self.sr_net(self.sr_NX[jn]); c = self.sr_C[j]
            tgt = torch.zeros(128, k, device=dev); tgt[torch.arange(128, device=dev), c] = 1.0
            loss = F.binary_cross_entropy_with_logits(o[:, :k], tgt) + F.binary_cross_entropy_with_logits(on[:, :k], torch.zeros(128, k, device=dev))                 + F.smooth_l1_loss(F.softplus(o[:, k:2 * k]).gather(1, c[:, None])[:, 0], self.sr_D[j]) + F.smooth_l1_loss(o[:, 2 * k:].gather(1, c[:, None])[:, 0], self.sr_B[j])
            self.sr_opt.zero_grad(); loss.backward(); self.sr_opt.step()
        with torch.no_grad():
            o = self.sr_net(view); pr = torch.sigmoid(o[:, :k]); dd = F.softplus(o[:, k:2 * k]).clamp(max=14.0); bb = o[:, 2 * k:].clamp(-0.8, 0.8); self.sr_last = pr; self.sr_dd = dd; self.sr_bb = bb
            if self.sr_n >= 200 and self.sr_hit > 0.7 and getattr(self, "Pu", None) is not None and self.C is not None:
                if getattr(self, "_po", None) is None:
                    self._po = PathOrgan(1, dev)
                for b, c in torch.nonzero((pr > 0.5) & live[:, None]).tolist():       # more likely than not; how strongly it is written follows how sure it is
                    q = p[b] + float(dd[b, c]) * torch.stack([torch.cos(h[b] + bb[b, c]), torch.sin(h[b] + bb[b, c])])
                    self._po.p = q[None].clone(); self._po.h = torch.zeros(1, device=dev); code = self._po(torch.zeros(1, 6, device=dev))[:, :PC]
                    u_ = int(self.unit(code)[0])
                    self.R[u_, c] = torch.maximum(self.R[u_, c], 0.2 * float(pr[b, c]) * self.Vn[u_].clamp(min=1.0))   # an expected relief there (fainter than lived)
                    if float(self.Pn[u_]) < 1.0:
                        self.Pu[u_] = q; self.Pn[u_] = 1.0
                    self.sr_writes += 1

    @torch.no_grad()
    def imagine_route(self, goal_xy):
        """IM1: its imagined way to a place - walking its own wall map in its head (breadth-first over cells not known as wall, unknown
        cells allowed) -> distance field [64, 64] in steps. One field per distinct target, recomputed when the map has changed."""
        import collections
        Gs = 64; c0 = Gs // 2; wall = (self.wmap > 1.0).cpu().numpy()
        gx, gy = [int(min(Gs - 1, max(0, round(float(v) + c0)))) for v in goal_xy]
        D = np.full((Gs, Gs), np.inf); D[gx, gy] = 0; q = collections.deque([(gx, gy)])
        while q:
            i, j = q.popleft()
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                a_, b_ = i + di, j + dj
                if 0 <= a_ < Gs and 0 <= b_ < Gs and not wall[a_, b_] and D[a_, b_] == np.inf:
                    D[a_, b_] = D[i, j] + 1; q.append((a_, b_))
        return D

    @torch.no_grad()
    def _home(self, x, tgt, ok, lack, vec=None):
        """The homing vector to the next unit on its own chain, in the body's frame; each gesture scored by how its own forecaster says the
        body would then move: progress along the vector + alignment of the heading after the turn (equal weight, each standardised)."""
        O = self.organs; p, h = O.path.p, O.path.h
        if vec is None:
            v = self.Pu[tgt] - p; c, s = torch.cos(h), torch.sin(h)
            bx, by = v[:, 0] * c + v[:, 1] * s, -v[:, 0] * s + v[:, 1] * c
        else:
            bx, by = vec
        d = (bx * bx + by * by).sqrt().clamp(min=1e-3); th = torch.atan2(by, bx)
        nb = sum(int(np.prod(O.c.sensors[i].shape)) for i in O.bed); off = nb
        for i in O.vec:
            if i == O.mo:
                break
            off += int(np.prod(O.c.sensors[i].shape))
        st = O.stats[O.mo]; idx = torch.tensor(O.mo_idx, device=self.dev)
        fo = self.net.foresee(x)[:, :, off + idx]                          # [n, M, 3] its own foreseen felt motion per gesture (feeling scale)
        ph = fo * 4 * st.s[idx] + st.m[idx]                                # back to the body's own units
        f, sd, tn = ph[..., 0], ph[..., 1], ph[..., 2]
        prog = (bx[:, None] * f + by[:, None] * sd) / d[:, None]; algn = torch.cos(th[:, None] - tn)
        H_ = int(os.environ.get("ATANOR_HOME_HOLD", "1"))
        if H_ > 1:                                                         # D49: what each gesture does if held H beats, as gestures are held (its own one-beat foresight, integrated)
            X_ = torch.zeros_like(f); Y_ = torch.zeros_like(f); A_ = torch.zeros_like(f)
            for _ in range(H_):
                ca_, sa_ = torch.cos(A_), torch.sin(A_); X_ = X_ + f * ca_ - sd * sa_; Y_ = Y_ + f * sa_ + sd * ca_; A_ = A_ + tn
            prog = (bx[:, None] * X_ + by[:, None] * Y_) / d[:, None]; algn = torch.cos(th[:, None] - A_)
        self.last_th, self.last_d, self.last_ok, self.last_ph, self.last_tgt = th, d, ok, ph, tgt                   # judge-side read only (C2); changes nothing
        if getattr(self, "home_rem", False):                              # D46b: what is left of the way after the gesture (overshoot counts), not how far along it
            prog = -((bx[:, None] - f) ** 2 + (by[:, None] - sd) ** 2).sqrt()
        if vec is None and os.environ.get("ATANOR_IMAGINE_MAP", "0") == "1" and getattr(self, "wmap", None) is not None and bool(ok.any()):   # IM1: the way in its head, around its own walls
            Gs = 64; c0 = Gs // 2; wver = int((self.wmap > 1.0).sum())         # only walls change the imagined ways (unknown cells count as passable)
            if getattr(self, "_fver", None) != wver:
                self._fields = {}; self._fver = wver
            idx_ok = torch.nonzero(ok)[:, 0]; gl = tgt[idx_ok].tolist()
            for g in set(gl):
                if g not in self._fields:
                    D = self.imagine_route(self.Pu[g].tolist()); D[~np.isfinite(D)] = 99.0
                    self._fields[g] = torch.tensor(D, device=self.dev, dtype=torch.float32)
            Fd = torch.stack([self._fields[g] for g in gl])                       # [k, 64, 64]
            pk, hk = p[idx_ok], h[idx_ok]; cs, sn = torch.cos(hk)[:, None], torch.sin(hk)[:, None]
            look = lambda xy: Fd[torch.arange(len(gl), device=self.dev)[:, None], (xy[..., 0] + c0).round().long().clamp(0, Gs - 1), (xy[..., 1] + c0).round().long().clamp(0, Gs - 1)]
            fx, fy = f[idx_ok] * 3, sd[idx_ok] * 3                                 # each gesture held 3 beats, by its own foresight
            dest = torch.stack([pk[:, 0:1] + fx * cs - fy * sn, pk[:, 1:2] + fx * sn + fy * cs], -1)
            d_now = look(pk[:, None, :])                                           # [k, 1]
            prog = prog.clone(); prog[idx_ok] = d_now - look(dest)
            nbr = torch.tensor([[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]], device=self.dev, dtype=torch.float32)
            dn = look(pk[:, None, :] + nbr[None]); best = dn.argmin(1)               # which way the imagined route goes from here
            way = torch.atan2(nbr[best, 1], nbr[best, 0]) - torch.atan2(torch.sin(hk), torch.cos(hk))
            algn = algn.clone(); algn[idx_ok] = torch.cos(way[:, None] - tn[idx_ok])
        if vec is None and os.environ.get("ATANOR_SEEN", "0") == "1" and getattr(self, "sr_last", None) is not None and getattr(self, "sr_dd", None) is not None                 and self.sr_last.shape[0] == lack.shape[0] and self.sr_n >= 200 and self.sr_hit > 0.7:
            # SM3 (registered 2026-11-28; founder on the viewer: "it sees the water and walks past"): what it sees now is fresher than what it
            # remembers. While it lacks an inside and its eye says relief of that kind is in view (more likely than not), the homing content
            # points where its eye says it is (its own learned bearing and distance) instead of where its memory places it.
            _, c_ = self.needs(x); ar = torch.arange(len(c_), device=self.dev)
            m = (self.sr_last[ar, c_] > 0.5) & (lack > 0.3)
            if os.environ.get("ATANOR_RELY", "0") == "1":                      # R5 (registered 2026-11-28; Ernst & Banks 2002): each source counts by its own
                we = self.sr_last[ar, c_] / (float(getattr(self, "eye_err", 2.0)) ** 2 + 0.05)      # record - how far off it has been about where relief came.
                wm = ok.float() / (float(getattr(self, "mem_err", 2.0)) ** 2 + 0.05)                # No "the eye wins" rule: the surer one leads this beat.
                m = (lack > 0.3) & (we > wm)
                self.rely_eye = getattr(self, "rely_eye", 0) + int(m.sum()); self.rely_mem = getattr(self, "rely_mem", 0) + int(((lack > 0.3) & ok & ~m).sum())
            if bool(m.any()):
                d2 = self.sr_dd[ar, c_].clamp(min=0.3); b2 = self.sr_bb[ar, c_]; bx2, by2 = d2 * torch.cos(b2), d2 * torch.sin(b2)
                pm = (bx2[:, None] * f + by2[:, None] * sd) / d2[:, None]; am = torch.cos(b2[:, None] - tn)
                prog = torch.where(m[:, None], pm, prog); algn = torch.where(m[:, None], am, algn); ok = ok | m
                self.sm3_n = getattr(self, "sm3_n", 0) + int(m.sum())
        z = lambda a: (a - a.mean(1, keepdim=True)) / (a.std(1, keepdim=True) + 1e-6)
        out = self.W * lack[:, None] * (z(prog) + z(algn))
        return torch.where(ok[:, None], out, torch.zeros_like(out))

    def innate(self, organs, net, rdecay=0.99):
        """Gate 1 adopted (founder 2026-11-26, WORLD_A_PROTOCOL.md): the innate homing vector (D46) and seeking its own believed relief (D48)
        become the main line. rdecay: where relief came fades faster than the map (V1b)."""
        self.homing, self.organs, self.net, self.seek, self.rdecay = True, organs, net, True, rdecay; return self

    @torch.no_grad()
    def own_goal(self, x, at=0.3):
        """D48: for a body whose lack is over `at`, the place unit its own relief memory values most for the lacking channel; -1 otherwise.
        No judge: the goal is its own belief, walked by lean_to (its lived crossings + the homing vector)."""
        if getattr(self, "V", None) is None:
            return None
        lack, c = self.needs(x); Vc = self.V.clone()
        ok = torch.ones_like(Vc, dtype=torch.bool)                             # gate 2 run 1 fix 2: only at a place it has been in this life (its homing
        if getattr(self, "Pn", None) is not None and os.environ.get("ATANOR_GOAL_FIX", "0") == "1":   # position set); faint relief memories still count
            ok &= (self.Pn > 0)[:, None]                                       # ATANOR_GOAL_FIX=0: D48 as it ran (goals may be units with no homing position)
        if self.used < self.KU:
            ok[self.used:] = False
        Vc[~ok] = -1e9; best = Vc.argmax(0); has = ok.any(0)
        g = best[c]; return torch.where((lack > at) & has[c], g, torch.full((len(lack),), -1, device=self.dev, dtype=torch.long))

    def state(self):
        d = dict(skill=self.skill.state_dict(), C=self.C, Cn=self.Cn, T=self.T, R=self.R, Vn=self.Vn, fitted=self.fitted, mended=self.mended, born=self.born, used=self.used)
        if self.ctx:
            d.update(ctx=self.ctx, Rs=self.Rs, Vns=self.Vns, Ds=self.Ds, Dn=self.Dn, Ts=self.Ts, act=self.act, nctx=self.nctx, switches=self.switches)
        return d
