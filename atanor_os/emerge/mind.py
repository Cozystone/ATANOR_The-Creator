"""하나의 마음 (공통 접면): 계약을 읽은 기관 (body_contract.Organs)이 만든 느낌 x만 받고 몸짓 번호를 낸다. 방이든 마크든 같은 고리, 같은 존재 파일.
고르기 (손 규칙 없음): 알맹이의 끌림 (풀림의 기억) - 헛됨 (넓힘 + 탈습관화) + 다침의 기억 (좁힘) + 배우는 심장 (몸짓마다의 값, 바탕만 보상, 겪어서 얻은 믿음만큼).
하던 몸짓은 스스로 붙든다: 놀라거나, 더 끌리는 것이 나타나거나, 헛됨이 쌓여 덜 끌릴 때만 새로 고른다 (상한 없음). 끌림이 없으면 얻은 믿음만큼 값을, 나머지는 아무거나.
배움: 내다보기 (겪은 걸음의 고정 크기 고리에서), 풀림·다침의 기억 (바탕의 오름 / 온전함의 갑작스런 줄어듦), 값 (되감기).
  m = Mind(organs, contract); act = m.choose(x); m.after(x, act, x2, died, cut)   # cut: 손이 옮겼거나 다시 태어남 (사슬 끊기)
  m.save(path) / Mind.load(path, organs, contract)
"""
import math, os
import numpy as np
import torch, torch.nn.functional as F
from .core import Core, Worth
from .common_senses import Z, ZF

LEAK, TIRE, LOOSE = 0.95, 4.0, 0.05


class LearnedWeight:
    """LW1 (LEARNED_WEIGHT_PROTOCOL.md): how much one organ's content counts in choice, learned from the bedrock only. The content f [n, M]
    (per gesture, e.g. what the homing organ says each gesture does to the way left) enters the score as w * f. w is learned by a linear
    policy gradient on the being's own wholeness change per beat (a death = losing the wholeness it had), with an eligibility trace (credit
    to the choices of the last ~1/(1-lam) beats) and a running baseline; both signals standardised by their running spread, so lr is
    scale-free. No other reward; w starts where it is told (0 = the content is not trusted at birth)."""

    def __init__(self, w0=0.0, lr=1e-3, lam=0.98, mode="beat", reach=2.0):
        self.w, self.lr, self.lam = float(w0), lr, lam; self.e = None; self.base = 0.0; self.rv = 1e-6; self.gv = 1e-6; self.hist = []
        # LW2 (mode="retro"; LEARNED_WEIGHT_PROTOCOL.md): learning is triggered by a meaningful event and looks back (Jeong et al. 2022,
        # ANCCR). An event = a beat whose wholeness change is far outside its own usual (a fill, a death) - nothing is named. At an event
        # the body's trace of how much its choices followed the content, minus what that trace usually is, moves the weight by the
        # event's size and sign. The trace reaches back `reach` x its own mean interval between events (a long wait = a long memory).
        self.mode, self.reach = mode, reach; self.T = 100.0; self.since = None; self.M = 0.0; self.Ev = 1e-6; self.am = 0.0; self.av = 1e-6; self.rs = 1e-6; self.n_ev = 0

    @torch.no_grad()
    def _contingent(self, r, alive):
        """TC1 (WORLD_A_PROTOCOL.md, registered 2026-11-27): the trust = how much more relief comes after following the content than after
        not following it (contingency; Rescorla 1968, ANCCR - Jeong et al. 2022), judged on every held beat, not at arrival. Each body keeps two
        traces fading over ~30 beats - of beats it followed and of beats it did not, while the content was held. A relief event (a wholeness rise
        far outside its own usual, as LW3) credits both traces; each trace's credit is divided by how much of that trace was lived (its own
        opportunity). w = p(relief | followed) / (p(followed) + p(not followed)): .5 = following makes no difference, 1 = only following brings
        relief. Counts fade x .9995 a beat. Nothing named; its only teacher is its own wholeness."""
        n = len(r); dev = r.device
        if getattr(self, "ef", None) is None or self.ef.shape[0] != n:
            self.ef = torch.zeros(n, device=dev); self.en = torch.zeros(n, device=dev); self.Kf, self.Kn, self.Sf, self.Sn = 1.0, 1.0, 1.0, 1.0; self.n_rel = 0
        fol = getattr(self, "_fol", None); held = getattr(self, "_held", None)
        if fol is None or fol.shape[0] != n:
            fol = torch.zeros(n, dtype=torch.bool, device=dev); held = fol.clone()
        self._fol = self._held = None
        k = 1.0 - 1.0 / 30.0
        self.ef = torch.where(alive, k * self.ef + fol.float(), torch.zeros_like(self.ef)); self.en = torch.where(alive, k * self.en + (held & ~fol).float(), torch.zeros_like(self.en))
        a_ = r.abs()
        if bool(alive.any()):
            self.am = 0.999 * self.am + 0.001 * float(a_[alive].mean()); self.av = 0.999 * self.av + 0.001 * float(((a_[alive] - self.am) ** 2).mean())
        rel = alive & (r > self.am + 3.0 * self.av ** 0.5)
        d = 0.9995
        self.Kf *= d; self.Kn *= d; self.Sf *= d; self.Sn *= d
        self.Sf += float(self.ef.sum()); self.Sn += float(self.en.sum())
        if bool(rel.any()):
            self.Kf += float(self.ef[rel].sum()); self.Kn += float(self.en[rel].sum()); self.n_rel += int(rel.sum())
        pf, pn = self.Kf / self.Sf, self.Kn / self.Sn; self.w = pf / max(pf + pn, 1e-12)
        if len(self.hist) < 10000:
            self.hist.append(self.w)

    @torch.no_grad()
    def arrive(self, at_goal, goal, wrise, alive):
        """LW3: at_goal [n] the body is at its goal unit now; goal [n] its goal (-1 none); wrise [n] wholeness change this beat."""
        n = len(at_goal)
        if getattr(self, "win", None) is None or self.win.shape[0] != n:
            self.win = torch.full((n,), -1, dtype=torch.long, device=at_goal.device); self.got = torch.zeros(n, dtype=torch.bool, device=at_goal.device)
            if not getattr(self, "carried", False):                              # M1a: a trust carried from the being file keeps its record (else the prior 1 kept, 1 broken)
                self.kept, self.broken = 1.0, 1.0
            self.w = self.kept / (self.kept + self.broken); self.n_arr = 0
        start = at_goal & (self.win < 0) & alive
        self.win = torch.where(start, torch.zeros_like(self.win), self.win); self.got = torch.where(start, torch.zeros_like(self.got), self.got)
        on = (self.win >= 0) & alive
        a_ = wrise.abs(); self.am = 0.999 * self.am + 0.001 * float(a_[alive].mean()) if bool(alive.any()) else self.am
        self.av = 0.999 * self.av + 0.001 * float(((a_[alive] - self.am) ** 2).mean()) if bool(alive.any()) else self.av
        relief = wrise > self.am + 3.0 * self.av ** 0.5                         # a rise far outside its own usual (LW2's event, rises only) - no fixed size
        self.got = self.got | (on & relief); self.win = torch.where(on, self.win + 1, self.win)
        done = (self.win > 30) | (self.win >= 0) & (self.got | ~alive)
        if bool(done.any()):
            k = int((done & self.got).sum()); b = int((done & ~self.got).sum())
            for _ in range(k + b):
                self.kept *= 0.995; self.broken *= 0.995
            self.kept += k; self.broken += b; self.n_arr += k + b; self.w = self.kept / (self.kept + self.broken)
            self.win = torch.where(done, torch.full_like(self.win, -1), self.win)
        if len(self.hist) < 10000:
            self.hist.append(self.w)

    @torch.no_grad()
    def stall(self, goal, d, ok, alive, W=30, need=1.0):
        """S1 (founder 2026-11-27, "if going round gets nothing it should stop"): the homing content promises that following it brings the
        body closer to its goal. While a goal is held, every W beats the body's own homing distance d (its path organ to its goal's place) is
        compared with W beats before: not closer by `need` cells -> one broken promise (same record as arrival; nothing counted as kept)."""
        n = len(goal)
        if getattr(self, "sg", None) is None or self.sg.shape[0] != n:
            self.sg = torch.full((n,), -1, dtype=torch.long, device=goal.device); self.sd = torch.zeros(n, device=goal.device); self.st = torch.zeros(n, dtype=torch.long, device=goal.device)
            self.n_stall = 0
            if not hasattr(self, "kept"):
                self.kept, self.broken = 1.0, 1.0
        on = (goal >= 0) & ok & alive
        fresh = on & (goal != self.sg)
        self.sd = torch.where(fresh, d, self.sd); self.st = torch.where(fresh, torch.zeros_like(self.st), self.st); self.sg = torch.where(on, goal, torch.full_like(self.sg, -1))
        self.st = torch.where(on, self.st + 1, torch.zeros_like(self.st))
        due = on & (self.st >= W)
        if bool(due.any()):
            b = int((due & (d > self.sd - need)).sum())
            for _ in range(b):
                self.kept *= 0.995; self.broken *= 0.995
            self.broken += b; self.n_stall += b; self.w = self.kept / (self.kept + self.broken)
            self.sd = torch.where(due, d, self.sd); self.st = torch.where(due, torch.zeros_like(self.st), self.st)

    def state(self):
        """M1a: the trust's own record, kept with the being (what it has learned of this organ's promises)."""
        return dict(kept=float(getattr(self, "kept", 1.0)), broken=float(getattr(self, "broken", 1.0)), am=float(self.am), av=float(self.av))

    def carry(self, st):
        self.kept, self.broken, self.am, self.av = st["kept"], st["broken"], st["am"], st["av"]; self.w = self.kept / (self.kept + self.broken); self.carried = True

    @torch.no_grad()
    def credit(self, f, pi, act, made):
        """f [n, M] content, pi [n, M] the choice's own probabilities, act [n] what the body did, made [n] whether a choice was made this beat."""
        g = f.gather(1, act[:, None])[:, 0] - (pi * f).sum(1); g = torch.where(made, g, torch.zeros_like(g))
        self.gv = 0.999 * self.gv + 0.001 * float((g * g).mean())
        if self.mode == "cont":                                                   # TC1: did this beat's gesture go the content's way (above its usual gesture)?
            self._held = f.abs().sum(1) > 0; self._fol = (f.gather(1, act[:, None])[:, 0] > 0) & self._held
            return
        if self.mode == "retro":
            dec = float(torch.exp(torch.tensor(-1.0 / max(self.reach * self.T, 1.0)))); gz = g / (self.gv ** 0.5 + 1e-8)
            self.e = gz if self.e is None or self.e.shape != g.shape else dec * self.e + gz
            return
        self.e = g / (self.gv ** 0.5 + 1e-8) if self.e is None or self.e.shape != g.shape else self.lam * self.e + g / (self.gv ** 0.5 + 1e-8)

    @torch.no_grad()
    def learn(self, r, alive):
        if self.mode == "cont":
            return self._contingent(r, alive)
        if self.e is None:
            return
        if self.mode == "reli":                                                    # LW3: the weight = how often this organ's promise was kept on arrival
            return
        if self.mode == "retro":
            n = len(r); a = r.abs()
            if self.since is None or self.since.shape != r.shape:
                self.since = torch.zeros_like(r)
            self.since += 1
            ev = alive & (a > self.am + 3.0 * self.av ** 0.5) & (self.n_ev > 200 or True)
            self.am = 0.999 * self.am + 0.001 * float(a[alive].mean()) if bool(alive.any()) else self.am
            self.av = 0.999 * self.av + 0.001 * float(((a[alive] - self.am) ** 2).mean()) if bool(alive.any()) else self.av
            self.M = 0.999 * self.M + 0.001 * float(self.e.mean()); self.Ev = 0.999 * self.Ev + 0.001 * float(((self.e - self.M) ** 2).mean())
            if bool(ev.any()):
                k = int(ev.sum()); self.rs = 0.99 * self.rs + 0.01 * float((r[ev] ** 2).mean())
                zr = r[ev] / (self.rs ** 0.5 + 1e-8); ze = (self.e[ev] - self.M) / (self.Ev ** 0.5 + 1e-8)
                per = self.lr * zr * ze / n; self.w += float(per.sum()); self.n_ev += k
                self.T = 0.98 * self.T + 0.02 * float(self.since[ev].mean()); self.since[ev] = 0
                if getattr(self, "kinds", None) is not None:
                    k_ = self.kinds[ev]; self.split = getattr(self, "split", {"death": 0.0, "fill": 0.0, "other": 0.0})
                    for nm, m_ in (("death", k_ == 2), ("fill", k_ == 1), ("other", k_ == 0)):
                        self.split[nm] += float(per[m_].sum())
            self.kinds = None
            if len(self.hist) < 10000:
                self.hist.append(self.w)
            return
        m = float(r[alive].mean()) if bool(alive.any()) else 0.0; self.base = 0.999 * self.base + 0.001 * m
        d = r - self.base; self.rv = 0.999 * self.rv + 0.001 * float((d * d).mean())
        per = self.lr * (d / (self.rv ** 0.5 + 1e-8) * self.e) / len(d)               # each body's share of this beat's step
        self.w += float(per.sum())
        if getattr(self, "kinds", None) is not None:                             # LW diagnostic (judge side): the step split by what happened this beat
            k_ = self.kinds; self.split = getattr(self, "split", {"death": 0.0, "fill": 0.0, "other": 0.0})
            for nm, m_ in (("death", k_ == 2), ("fill", k_ == 1), ("other", k_ == 0)):
                self.split[nm] += float(per[m_].sum())
            self.kinds = None
        if len(self.hist) == 0 or len(self.hist) < 10000:
            self.hist.append(self.w)


class Earned:
    """D37e: a controller's earned trust - correlation between how hard it pushed (e [n]) and the wholeness change over the next K beats (a death
    = losing all wholeness), discounted, kept as running moments (same form as Mind._track_pull)."""

    def __init__(self, K=60, gamma=0.97):
        self.K, self.g, self.tr, self.oc = K, gamma, None, None

    @torch.no_grad()
    def track(self, e, whole, alive):
        n, K = len(e), self.K
        if self.tr is None or self.tr["v"].shape[1] != n:
            z = lambda: torch.zeros(K, n, device=e.device)
            self.tr = dict(v=z(), w=z(), a=z(), t=0); self.oc = torch.zeros(5, device=e.device)
        tr = self.tr; i = tr["t"] % K
        if tr["t"] >= K:
            idx = [(i + k) % K for k in range(K)]; a_ = tr["a"][idx]; alive_k = torch.cumprod(a_, 0); w_ = tr["w"][idx] * alive_k
            w_end = torch.cat([w_[1:], (torch.where(alive, whole, torch.zeros_like(whole)) * alive_k[-1])[None]])
            disc = self.g ** torch.arange(K, device=e.device, dtype=torch.float32)[:, None]; gsum = (disc * (w_end - w_)).sum(0)
            ok = a_[0] > 0
            if ok.sum() >= 1:
                v = tr["v"][i][ok]; gg = gsum[ok]; m = torch.stack([v.mean(), gg.mean(), (v * v).mean(), (gg * gg).mean(), (v * gg).mean()])
                self.oc.copy_(m if float(self.oc[2]) == 0 else 0.99 * self.oc + 0.01 * m)
        tr["v"][i] = e; tr["w"][i] = whole; tr["a"][i] = alive.float(); tr["t"] += 1

    def weight(self):
        if self.oc is None or self.tr["t"] < 2 * self.K or float(self.oc[2]) == 0:
            return 1.0
        ev, eg, ev2, eg2, evg = [float(z) for z in self.oc]
        c = (evg - ev * eg) / max(((ev2 - ev * ev) * (eg2 - eg * eg)) ** 0.5, 1e-9)
        return max(0.1, min(1.0, c))


class Mind:
    def __init__(self, organs, contract, n, dev="cuda", far=0.3, worth=2.0, try_p=0.1, harm=1.0, learn_every=50, keep=4000, lp=0.0, lp_every=1000, inner=0, imagine=0.0, horizon=6, rolls=2, value_arm="worth", value_target="max", arbitrate=False):
        self.o, self.c, self.n, self.dev = organs, contract, n, dev
        combos, C = contract.codes(); self.M = len(combos)
        # R1 이어진 내면 상태 (inner > 0): h = GRU([느낌, 방금 한 몸짓의 부호], h). 앞으로 올 느낌 (1-4걸음 앞)을 내다보며 스스로 배운다 -
        # 어둠 뒤 불이 켜질 때 보일 것을 맞히려면 들고 있어야 하므로 따로 정답 없이 기억이 자란다. 알맹이와 값은 [느낌, h]를 느낀다.
        self.inner = inner; self.Cc = torch.tensor(C, device=dev); self.inner_every = 10
        dxe = organs.dx + inner
        if inner:
            import torch.nn as nn
            self.gru = nn.GRUCell(organs.dx + C.shape[1], inner).to(dev)
            self.HOR = (1, 4, 16, 32)                                                    # 내다보는 거리: 어둠 (20걸음)을 건너야 맞힐 수 있게 (1-4걸음만 보면 기억할 까닭이 없었다, 2026-11-22 진단)
            # 기억을 바탕에 묶는다 (2026-11-22, 해마의 '어디서 먹었나'): 목표는 앞날의 느낌 전체가 아니라 '앞으로 몸 안이 얼마나 채워질까' (바탕 칸마다, 거리마다).
            # 몸을 이렇게 하면 몇 걸음 뒤 마시는지 맞히려면 샘의 자리를 들고 있어야 하므로 쓸모 있는 기억만 자란다 (느낌 전체를 맞히면 그것이 묻혔다).
            self.inner_target = "relief"; self.ni = organs.inside.stop - organs.inside.start
            self.ahead = nn.Sequential(nn.Linear(inner + organs.dx + C.shape[1], 256), nn.ELU(), nn.Linear(256, len(self.HOR) * (self.ni if self.inner_target == 'relief' else organs.dx))).to(dev)
            self.iopt = torch.optim.Adam(list(self.gru.parameters()) + list(self.ahead.parameters()), lr=1e-3)
            self.h = torch.zeros(n, inner, device=dev); self.seq = []
        self.net = Core(dxe, self.M, organs.inside, slice(0, 0), codes=C).to(dev); self.net.far = far; self.net.harm_sharp = True
        self.opt = torch.optim.AdamW(self.net.parameters(), lr=1e-3)
        # value_arm (A1 시험, 2026-11-23): 'worth' = 기존 값 (기본, 주 줄기), 'bedrock' / 'aux' = 채널별 앞날 (channel_worth.py, 기본 꺼짐)
        if worth and value_arm == "relief":                                   # 잠재 학습 값 (D17-D23): 채널별 '풀린 몫' 앞날 x 지금의 결핍 (기본 꺼짐, V2 시험)
            from .channel_worth import ReliefWorth
            self.wo = ReliefWorth(dxe, organs.inside, m=self.M, lam=worth).to(dev)
        elif worth and value_arm != "worth":
            from .channel_worth import ChannelWorth
            self.wo = ChannelWorth(dxe, organs.inside, lam=worth, m=self.M, mode=value_arm).to(dev)
        else:
            self.wo = Worth(dxe, organs.inside, lam=worth, m=self.M).to(dev) if worth else None
        if self.wo is not None and hasattr(self.wo, "target"):
            self.wo.target = value_target                                       # V1: 'sarsa' = 값이 실제로 한 다음 몸짓에서 배움
        self.try_p, self.harm, self.learn_every, self.keep = try_p, harm, learn_every, keep
        # V4 (2026-11-24): who decides between the habit-like relief pull and the value, by each one's own track record (Daw, Niv & Dayan 2005:
        # the more reliable controller takes over). The pull's record = how well its expectation (its strongest score) foretold the wholeness change
        # over the next K beats, tracked like the value's own earned trust. No hand-set weight. Default off.
        self.arbitrate = arbitrate; self.ptrace = None; self.poc = torch.zeros(5, device=dev); self.pK = 60
        # D37b (atanor_os/FEAR_PROTOCOL.md, founder 2026-11-25): conflict monitoring (Botvinick et al. 2001) - when the learned value leans against the
        # gesture being held by more than `conflict` (in the lean's own units), this beat is a new choice (the usual choosing makes it). It weakens no
        # habit and names no place: it only lets a strong value-act conflict call for re-choosing. 0 = off (as before).
        self.conflict = 0.0; self.conflict_n = 0
        self.buf = []; self.steps = 0
        # 값으로 매기는 상상 (2026-11-22, 창립자 검토: 겉모습의 닮음이 아니라 여러 걸음 뒤 제 온전함으로): 알맹이의 내다보기를 (기억 상태까지 품고)
        # 몸짓마다 horizon걸음 굴려, 끝을 '온전함의 바뀜 + 끝 장면의 값 (몸짓마다의 값 중 가장 나은 것)'으로 매긴다. 고름에 드는 몫 = imagine x 값이 얻은 믿음.
        self.imagine, self.horizon, self.rolls = imagine, horizon, rolls
        # imagine_mode: 'roll' = 알맹이 내다보기를 굴림 (실패: 한 걸음부터 몸 안 바뀜을 못 맞힘), 'ahead' = 기억 기관의 '몸짓에 따른 앞날의 채움' 머리 (빛 속에서 몸짓을 가림)
        self.imagine_mode = "roll"
        # 둘째 바탕 (창립자 확정, E20 'ALIVE_ODD'의 다음 꼴): 배움의 진척 - 예전의 나 (lp_every걸음 전의 내다보기)보다 지금의 내가 이 걸음을 덜 틀리게 내다본 만큼.
        # 배울 수 없는 소음은 진척이 0, 다 배운 것도 0 (지루함) -> 배움이 느는 곳에만 끌림. 그 몫 lp (장부).
        self.lp, self.lp_every, self.f_old, self.lp_scale = lp, lp_every, None, None
        self.begin(torch.ones(n, dtype=torch.bool, device=dev))

    def begin(self, which):
        if not hasattr(self, "fut"):
            n, d = self.n, self.dev
            self.fut = torch.zeros(n, self.M, device=d); self.s_m = torch.zeros(n, device=d); self.s_v = torch.full((n,), 1e-4, device=d)
            self.cur = torch.zeros(n, dtype=torch.long, device=d); self.prev = None; self.cut = torch.ones(n, dtype=torch.bool, device=d)
        self.fut[which] = 0; self.cut |= which
        if getattr(self, "wo", None) is not None and hasattr(self.wo, "forget"):
            self.wo.forget(torch.nonzero(which)[:, 0])
        if getattr(self, "inner", 0):
            self.h[which] = 0

    @torch.no_grad()
    def feel(self, x):
        """느낌에 이어진 내면 상태를 붙인다 (inner일 때). 몸 쪽 고리는 choose/after 앞에 이것을 부른다."""
        if not self.inner:
            return x
        return torch.cat([x, self.h], 1)

    @torch.no_grad()
    def carry(self, x_raw, act, cut):
        """한 걸음 뒤: h를 새 느낌과 방금 한 몸짓으로 갱신 (끊긴 몸은 0에서)."""
        if not self.inner:
            return
        self.h = self.gru(torch.cat([x_raw, self.Cc[act]], 1), self.h * (~cut).float()[:, None])
        self.seq.append((x_raw, act, cut)); self.seq = self.seq[-400:]

    def learn_inner(self, L=64, B=64):
        """기억 기관의 배움: 고리에서 L걸음 창을 잘라 h를 굴리고, 1-4걸음 뒤 느낌을 내다본 틀림으로 배운다 (끊긴 데서 h를 0으로)."""
        H = max(self.HOR)
        if not self.inner or len(self.seq) < L + H + 2:
            return None
        X = torch.stack([s[0] for s in self.seq]); A = torch.stack([s[1] for s in self.seq]); Cu = torch.stack([s[2] for s in self.seq])
        t0 = int(torch.randint(0, len(self.seq) - L - H - 1, (1,))); b = torch.randint(0, self.n, (min(B, self.n),), device=self.dev)
        h = torch.zeros(len(b), self.inner, device=self.dev); loss = 0
        for k in range(L):
            t = t0 + k
            h = self.gru(torch.cat([X[t, b], self.Cc[A[t, b]]], 1), h * (~Cu[t, b]).float()[:, None])
            if k >= L // 2:
                pred = self.ahead(torch.cat([h, X[t + 1, b], self.Cc[A[t + 1, b]]], 1)).view(len(b), len(self.HOR), -1)
                if self.inner_target == "relief":                          # 지금부터 j걸음 안에 몸 안이 오른 만큼 (가장 높이 찬 때 - 지금), 바탕 칸마다
                    ins = self.o.inside
                    tgt = torch.stack([(X[t + 1:t + 2 + j, b][:, :, ins].max(0).values - X[t + 1, b][:, ins]).clamp(min=0) * 10 for j in self.HOR], 1)
                else:
                    tgt = torch.stack([X[t + 1 + j, b] for j in self.HOR], 1)
                ok = torch.stack([~Cu[t + 1:t + 2 + j, b].any(0) for j in self.HOR], 1).float()       # 사이에 끊김 (죽음, 옮김)이 없을 때만
                loss = loss + (((pred - tgt) ** 2).mean(2) * ok).sum() / ok.sum().clamp(min=1)
        if not torch.is_tensor(loss):
            return None
        self.iopt.zero_grad(); loss.backward(); self.iopt.step()
        return float(loss)

    @torch.no_grad()
    def imagined(self, x):
        """몸짓마다 [B, M]: 그 몸짓으로 시작해 아무 몸짓으로 이어 굴린 끝의 값 (rolls개 가운데 가장 나은 것)."""
        B, M, R, H = x.shape[0], self.M, self.rolls, self.horizon
        z = x[:, None, None].expand(B, M, R, x.shape[1]).reshape(-1, x.shape[1])
        a = torch.arange(M, device=self.dev)[None, :, None].expand(B, M, R).reshape(-1)
        w0 = self.wo.whole(z)
        for k in range(H):
            z = z + self.net.f(torch.cat([z, self.net.codes[a]], 1))
            a = torch.randint(0, M, (len(z),), device=self.dev)
        g = self.wo.gamma ** H
        v = (self.wo.whole(z) - w0) + g * self.wo.v(z).max(1).values
        return v.view(B, M, R).max(2).values

    @torch.no_grad()
    def unfamiliar(self, x, H=4, k=4):
        """E1 (atanor_os/E1_ROAM_PROTOCOL.md; default off). Per gesture [B, M]: how little the being's own memory of recently sensed states knows the state it
        foresees after holding that gesture H beats (distance to the k-th nearest remembered state, outside senses only), standardised per body."""
        B, M, dx = x.shape[0], self.M, x.shape[1]; a = torch.arange(M, device=self.dev)[None].expand(B, M).reshape(-1)
        z = x[:, None].expand(B, M, dx).reshape(-1, dx)
        for _ in range(H):
            z = z + self.net.f(torch.cat([z, self.net.codes[a]], 1))
        mem = self.pm[:self.pm_n]; sd = mem.std(0) + 1e-6; q = (z[:, self.pm_out] / sd); m = mem / sd
        dk = torch.cat([torch.cdist(q[j:j + 2048], m).topk(k, dim=1, largest=False).values[:, -1] for j in range(0, len(q), 2048)]).view(B, M)
        return (dk - dk.mean(1, keepdim=True)) / (dk.std(1, keepdim=True) + 1e-6)

    @torch.no_grad()
    def remember_place(self, x, live):
        """E1: one sensed state per beat (a random living body's) into a bounded ring of 4,096; the oldest is replaced."""
        if not hasattr(self, "pm"):
            ins = self.net.inside; out = [i for i in range(x.shape[1]) if not (ins.start <= i < ins.stop)]
            self.pm_out = torch.tensor(out, device=self.dev); self.pm = torch.zeros(4096, len(out), device=self.dev); self.pm_n = 0; self.pm_i = 0
        idx = torch.nonzero(live)[:, 0]
        if len(idx):
            j = idx[int(torch.randint(0, len(idx), (1,)))]; self.pm[self.pm_i] = x[j, self.pm_out]; self.pm_i = (self.pm_i + 1) % 4096; self.pm_n = min(4096, self.pm_n + 1)

    @torch.no_grad()
    def fill_ahead(self, x):
        """몸짓마다 [B, M]: 기억 기관이 내다본 '앞으로 몸 안이 얼마나 채워질까' (거리 16과 32의 합, 바탕 칸 합)."""
        B, M, dx = x.shape[0], self.M, x.shape[1] - self.inner
        xr, h = x[:, :dx], x[:, dx:]
        z = torch.cat([h[:, None].expand(B, M, self.inner), xr[:, None].expand(B, M, dx), self.Cc[None].expand(B, M, self.Cc.shape[1])], 2).reshape(B * M, -1)
        p = self.ahead(z).view(B, M, len(self.HOR), -1).sum(3)
        return p[:, :, [self.HOR.index(16), self.HOR.index(32)]].sum(2)

    @torch.no_grad()
    def choose(self, x):
        n, d = self.n, self.dev
        if self.prev is not None:
            px, pa = self.prev
            fo = self.net.foresee(px).gather(1, pa[:, None, None].expand(-1, 1, x.shape[1]))[:, 0]
            ext = getattr(self, "ext_fore", None)                              # T2 (LOKI_TRANSPLANT_PROTOCOL.md; default None): the body's own forecaster (LOKI) for its first columns
            if ext is not None:
                fo = fo.clone(); fo[:, :ext.shape[1]] = ext; self.ext_fore = None
            more, eased = self.net.unmet(px, pa, x, fo=fo)
            self.fut = self.fut * LEAK; self.fut[torch.arange(n, device=d), pa] += more * (~self.cut); self.fut[eased] = 0
            odd = ((fo - x) ** 2).mean(1); novel = (odd > self.s_m + 2 * self.s_v.sqrt()) & (self.steps > 20) & ~self.cut
            self.fut[novel] = 0; self.s_m = self.s_m + 0.02 * (odd - self.s_m); self.s_v = self.s_v + 0.02 * ((odd - self.s_m) ** 2 - self.s_v)
        else:
            novel = torch.ones(n, dtype=torch.bool, device=d)
        sc, strong = self.net.pull(x); rec = getattr(self, "record_parts", False); parts = {"pull": sc.clone()} if rec else None
        if getattr(self.net, "trail", False):
            ts, tst = self.net.trail_pull(x); sc = sc + ts; strong = strong | tst
        if self.harm and int(self.net.harm_n) > 0:
            av = self.net.avoid(x); sc = sc + self.harm * av; strong = strong | (av.abs().max(1).values > 0.02)
            if rec:
                parts["harm"] = self.harm * av
        sc = sc - TIRE * self.fut
        if rec:
            parts["futile"] = -TIRE * self.fut
        lw = None
        if self.wo is not None:
            self.wo.track(x, ~self.cut)
            lw = self.wo.lean(self.net, x)
            if lw is not None and self.imagine and self.wo.on:
                im = self.fill_ahead(x) if self.imagine_mode == "ahead" else self.imagined(x); c = im - im.mean(1, keepdim=True)
                s = c.std(1).mean(); self.im_scale = s if getattr(self, 'im_scale', None) is None else 0.999 * self.im_scale + 0.001 * s
                lw = lw + self.imagine * self.wo.lam * self.wo.earned() * c / (self.im_scale + 1e-6)
            if lw is not None:
                sc = sc + lw
                if rec:
                    parts["value"] = lw
        go = getattr(self, "going", None)
        if go is not None:                                                     # L1: the lean of the lack toward its next place (atanor_os/emerge/going.py)
            gl = go.lean(x)
            if gl is not None and getattr(self, "going_earned", None) is not None and self.wo is not None:   # D37e: the lean weighed by its own record
                self.going_earned.track(gl.max(1).values - gl.mean(1), self.wo.whole(x), ~self.cut)
                gl = gl * self.going_earned.weight()
            if gl is not None and os.environ.get("ATANOR_TRUST_LEAN", "0") == "1" and getattr(self, "learned_home", None) is not None and getattr(go, "homing", False):
                gl = gl * self.learned_home.w                                    # S1b: the going lean reads the same homing vector, so the same learned trust weighs it
            if gl is not None and os.environ.get("ATANOR_QUIET", "0") == "1":     # Q1 (registered 2026-11-28): an organ counts by how often what it foretold came
                gl = gl * self._quiet("lean"); self._q_gl = gl                 # true - here: "going this way brings me nearer that place". Nothing to say or
            if gl is not None:                                                 # nothing borne out -> it goes quiet by itself (never below a whisper, so it can return)
                sc = sc + gl
                if rec:
                    parts["going"] = gl
            if getattr(go, "seek", False) and getattr(self, "plan_goal", None) is None and getattr(self, "beat_n", 0) >= 50:   # gate 1 adopted: its own goal at full weight
                self.plan_goal = go.own_goal(x); self.plan_full = True
                if os.environ.get("ATANOR_GUT", "0") == "1" and self.plan_goal is not None:
                    self.plan_goal = self._gut_goal(x, self.plan_goal)
                if self.plan_goal is not None:                                 # judge-side count only (Copilot's question): how often its own goal changes while held
                    gp = getattr(self, "goal_prev", None)
                    if gp is not None:
                        both = (self.plan_goal >= 0) & (gp >= 0); self.goal_held = getattr(self, "goal_held", 0) + int(both.sum())
                        self.goal_switch = getattr(self, "goal_switch", 0) + int((both & (self.plan_goal != gp)).sum())
                    self.goal_prev = self.plan_goal.clone()
                    Pn = getattr(go, "Pn", None)                                # judge-side: goals at a unit with no homing position (walked toward the birth point)
                    if Pn is not None:
                        hd_ = self.plan_goal >= 0; self.goal_set_n = getattr(self, "goal_set_n", 0) + int(hd_.sum())
                        self.goal_unset = getattr(self, "goal_unset", 0) + int((hd_ & (Pn[self.plan_goal.clamp(min=0)] == 0)).sum())
            self.beat_n = getattr(self, "beat_n", 0) + 1
            pg = getattr(self, "plan_goal", None)                              # B1 (WORLD_A_PROTOCOL.md; default None): a goal place from a memory organ (LOKI v12),
            if pg is not None and self.wo is not None:                         # walked by the going organ's own map and skill, weighed by its own earned record
                pl = go.lean_to(x, pg)
                if pl is not None:
                    if getattr(self, "plan_earned", None) is None:
                        self.plan_earned = Earned()
                    self.plan_earned.track(pl.max(1).values - pl.mean(1), self.wo.whole(x), ~self.cut)
                    lwh = getattr(self, "learned_home", None)                   # LW1: the weight on the homing content learned from the bedrock (default None = as before)
                    if lwh is not None and getattr(self, "plan_full", False):
                        self._lw_f = pl.clone(); pl = pl * lwh.w
                    else:
                        pl = pl * (1.0 if getattr(self, "plan_full", False) else self.plan_earned.weight())
                    if os.environ.get("ATANOR_QUIET", "0") == "1":
                        pl = pl * self._quiet("plan"); self._q_pl = pl
                    sc = sc + pl
                    if rec:
                        parts["plan"] = pl
                self.plan_goal = None; self.plan_full = False
        if rec:
            parts["total"] = sc.clone(); parts["strong"] = strong.clone(); self.parts = parts
        roam_on = getattr(self, "roam", 0) and getattr(self, "pm_n", 0) >= 256
        self._roam_swing = None
        if roam_on and getattr(self, "roam_mode", "slot") == "score":            # E1b: the unfamiliarity competes in the score, as the value and harm do
            u_ = self.unfamiliar(x); lwr = getattr(self, "learned_roam", None)   # G2 run 2: its weight = its own kept promises (LW3 frame), else the fixed roam
            wr_ = lwr.w if lwr is not None else self.roam
            self._roam_swing = (u_.argmax(1), sc.argmax(1) != u_.argmax(1))      # judge-free: which way the unfamiliarity points, and whether it differs from the rest of the score
            sc = sc + wr_ * u_
        alw = getattr(self, "allowed", None)                                  # R1 (founder 2026-11-27, "one operation at a time, like a robot vacuum"): the body
        if alw is not None:                                                    # offers only the gestures it has; the rest are not choosable
            sc = sc.masked_fill(~alw[None], -1e9)
            if lw is not None:
                lw = lw.masked_fill(~alw[None], -1e9)
        pick = torch.multinomial(torch.softmax(sc / LOOSE, 1), 1)[:, 0]
        rnd = torch.randint(0, self.M, (n,), device=d) if alw is None else torch.nonzero(alw)[:, 0][torch.randint(0, int(alw.sum()), (n,), device=d)]
        usedv = torch.zeros(n, dtype=torch.bool, device=d)
        if roam_on and getattr(self, "roam_mode", "slot") == "slot":             # E1: with no strong pull, drawn to the gesture whose foreseen state its place memory knows least
            rnd = torch.multinomial(torch.softmax(self.roam * self.unfamiliar(x), 1), 1)[:, 0]
        if lw is not None:
            byv = torch.multinomial(torch.softmax(lw / LOOSE, 1), 1)[:, 0]
            usedv = torch.rand(n, device=d) < self.wo.earned()
            rnd = torch.where(usedv, byv, rnd)
        pick = torch.where(strong, pick, rnd)
        if self.arbitrate and lw is not None:
            self._track_pull(x, torch.where(strong, sc.max(1).values, torch.zeros(n, device=d)), ~self.cut)
            ev, ep = self.wo.earned(), self.pull_earned(); pv = ev / max(ev + ep, 1e-6) if (ev + ep) > 0 else 0.0
            take = strong & (torch.rand(n, device=d) < pv)           # on a strong beat the value decides with the share its record earns
            pick = torch.where(take, byv, pick); usedv = usedv | take; strong = strong & ~take
        gap = sc.max(1).values - sc.gather(1, self.cur[:, None])[:, 0]; sd = sc.std(1).clamp(min=1e-6)
        if getattr(self, "gate_usual", False):                                # D40 (VALUE_CHOICE_PROTOCOL.md; default off): the bar is 2x the being's own usual score spread (slow mean over its lived beats), not this beat's
            m = float(sd.mean()); self.sd_usual = m if getattr(self, "sd_usual", None) is None else 0.999 * self.sd_usual + 0.001 * m
            sd = torch.full_like(sd, self.sd_usual)
        pre = strong & (gap > 2.0 * sd); self.pre_last = pre                 # pre_last: judge-side record only
        again = novel | pre | self.cut
        if self.conflict and lw is not None:
            clash = (lw.mean(1) - lw.gather(1, self.cur[:, None])[:, 0]) > self.conflict
            self.clash = clash; self.conflict_n += int((clash & ~again).sum()); again = again | clash
        force = getattr(self, "force", None)                                  # 판정 쪽 탐침 (A1 유지 진단): 이 박자만 새로 고르게. 기본 없음
        if force is not None:
            again = again | force; self.force = None
        self.again, self.lw_last = again, lw                                # 재는 쪽 기록만
        motor_candidate = getattr(self, "motor_candidate", None)
        if motor_candidate is not None:
            pick = motor_candidate(x, pick, again, alw)
        self.cur = torch.where(again, pick, self.cur)
        lwh = getattr(self, "learned_home", None)
        if lwh is not None:                                                    # LW1: credit for the homing content in the choices made this beat
            f_ = getattr(self, "_lw_f", None)
            if f_ is None:
                f_ = torch.zeros(n, self.M, device=d)
            lwh.credit(f_, torch.softmax(sc / LOOSE, 1), self.cur, again); self._lw_f = None
        tried = torch.rand(n, device=d) < self.try_p
        act = torch.where(tried, torch.randint(0, self.M, (n,), device=d) if alw is None else torch.nonzero(alw)[:, 0][torch.randint(0, int(alw.sum()), (n,), device=d)], self.cur)
        nt = getattr(self, "turn_once", None)                                  # T1 (registered): a turn is one act (30 degrees, like a head turn or a
        if nt is not None:                                                     # saccade); a held gesture keeps its walking and hand, not its turn
            act = torch.where(again | tried, act, nt[act])
        if os.environ.get("ATANOR_W0", "0") == "1" and go is not None and getattr(go, "organs", None) is not None and getattr(go, "V", None) is not None and self.wo is not None:
            # W0 (registered 2026-11-27: the smallest world-model test). Prophet: its own learned forecaster (core.foresee: the next senses per
            # gesture), rolled H beats for a few futures made of its own gestures (walk on; turn 1, 2, 3 or 6 times either way, then walk).
            # Judge: its own wholeness in the imagined end (+ what its place memory holds there for what it lacks). Executor: the first
            # `K` beats of the best future. Called only while it lacks (a goal is held) and its last plan has run out. Habit otherwise.
            H_, K_ = 20, 4
            if getattr(self, "w0_plan", None) is None or self.w0_plan.shape[0] != n:
                self.w0_plan = torch.zeros(n, K_, dtype=torch.long, device=d); self.w0_i = torch.full((n,), K_, dtype=torch.long, device=d); self.w0_n = 0; self.w0_spread = 0.0
            gp_ = getattr(self, "goal_prev", None)
            need = (gp_ >= 0) if gp_ is not None else torch.zeros(n, dtype=torch.bool, device=d)
            due = need & (self.w0_i >= K_)
            if bool(due.any()):
                fo_ = go._foreseen(x); fw, tn_ = fo_[..., 0], fo_[..., 2]
                tmax = tn_.abs().amax(1, keepdim=True).clamp(min=1e-6)
                b_ = torch.where(tn_.abs() < 0.2 * tmax, fw, torch.full_like(fw, -1e9)).argmax(1)       # its own best walking gesture
                L_, R_ = tn_.argmax(1), tn_.argmin(1)
                seqs = [b_[:, None].expand(-1, H_)]
                for k in (1, 2, 3, 6):
                    for t_ in (L_, R_):
                        seqs.append(torch.cat([t_[:, None].expand(-1, k), b_[:, None].expand(-1, H_ - k)], 1))
                S_ = torch.stack(seqs, 1); C_ = S_.shape[1]                                             # [n, 9, H]
                xi = x[:, None, :].expand(-1, C_, -1).reshape(n * C_, -1)
                for t in range(H_):
                    g_ = S_[:, :, t].reshape(-1)
                    xi = self.net.foresee(xi).gather(1, g_[:, None, None].expand(-1, 1, xi.shape[-1]))[:, 0]
                wh = self.wo.whole(xi).view(n, C_)
                lk_, ch_ = go.needs(x); u_ = go.unit(go.place(xi)).view(n, C_)
                sc_ = wh + lk_[:, None] * go.V[u_, ch_[:, None].expand(-1, C_)]
                best = sc_.argmax(1); self.w0_spread = 0.99 * self.w0_spread + 0.01 * float((sc_.amax(1) - sc_.amin(1))[due].mean())
                self.w0_plan = torch.where(due[:, None], S_[torch.arange(n, device=d), best, :K_], self.w0_plan); self.w0_i = torch.where(due, torch.zeros_like(self.w0_i), self.w0_i)
                self.w0_pick = torch.bincount(best[due], minlength=C_) + (getattr(self, "w0_pick", None) if getattr(self, "w0_pick", None) is not None else 0)
            run = need & (self.w0_i < K_)
            if bool(run.any()):
                pa = self.w0_plan.gather(1, self.w0_i.clamp(max=K_ - 1)[:, None])[:, 0]
                act = torch.where(run, pa, act); self.cur = torch.where(run, pa, self.cur); again = again | run; tried = tried & ~run
                self.w0_i = torch.where(run, self.w0_i + 1, self.w0_i); self.w0_n += int(run.sum())
            self.w0_i = torch.where(need, self.w0_i, torch.full_like(self.w0_i, K_))
        if os.environ.get("ATANOR_E1", "0") == "1" and go is not None and getattr(go, "organs", None) is not None:
            # E1 control arm (Astra's one-step prototype, LOKI 27f33dd loki_memory/embodied_choice.py, fetched with the founder's approval):
            # each gesture's learned effect on its own path position (given its heading) + the remembered place of relief per channel; the
            # present deficits join the two - the gesture whose foreseen next place is nearest to where it was relieved. One beat ahead only.
            # Same calling rule as W0: while it lacks (a goal is held) and it remembers a relief place; habit otherwise.
            ec = self._e1()
            P_ = go.organs.path.p; h_ = go.organs.path.h; sn_ = torch.stack([torch.cos(h_), torch.sin(h_)], 1)
            df_ = 1.0 - torch.stack([x[:, go.ins[2]], x[:, go.ins[1]]], 1).clamp(0, 1)
            gp_ = getattr(self, "goal_prev", None)
            need = ((gp_ >= 0) if gp_ is not None else torch.zeros(n, dtype=torch.bool, device=d)) & bool((ec.mass > 0).any())
            if bool(need.any()):
                pa = act.clone()
                for i in torch.nonzero(need)[:, 0].tolist():
                    pa[i] = ec.choose(sn_[i], P_[i], df_[i])
                act = torch.where(need, pa, act); self.cur = torch.where(need, pa, self.cur); again = again | need; tried = tried & ~need
                self.e1_n = getattr(self, "e1_n", 0) + int(need.sum())
            self._e1_prev = (sn_.clone(), P_.clone(), df_.clone())
        if os.environ.get("ATANOR_QUIET", "0") == "1" and go is not None and getattr(go, "organs", None) is not None:
            P_ = go.organs.path.p; pend = {}
            fo_ = go._foreseen(x); mg_ = (fo_[..., 0] ** 2 + fo_[..., 1] ** 2).sqrt()       # Q1b: "nearer" is foretold only by a gesture its own forecaster
            walks = mg_.gather(1, act[:, None])[:, 0] > 0.5 * mg_.amax(1).clamp(min=0.1)   # sees as a move; a turn asked for is not scored as a failed approach
            for nm, cont, tg in (("lean", getattr(self, "_q_gl", None), getattr(go, "_t_lean", None)), ("plan", getattr(self, "_q_pl", None), getattr(go, "_t_plan", None))):
                if cont is None or tg is None or cont.shape[0] != n or tg[0].shape[0] != n:
                    continue
                said = cont.abs().sum(1) > 1e-6                                 # the organ had something to say this beat
                led = said & tg[1] & (cont.argmax(1) == act) & walks             # and what the body does is what it asked for, and is a move
                self.q_turn = getattr(self, "q_turn", 0) + int((said & tg[1] & (cont.argmax(1) == act) & ~walks).sum())
                pend[nm] = (led, tg[0].clone(), (tg[0] - P_).norm(dim=1))
            self._q_pend = (pend, P_.clone()); self._q_gl = None; self._q_pl = None
        b0m = os.environ.get("ATANOR_B0", "0")
        if b0m in ("own", "oracle") and go is not None and getattr(self, "goal_prev", None) is not None and getattr(go, "organs", None) is not None:
            wsc, on_ = go.b0_foresight(x, self.goal_prev)   # B0 (Plan B first step): the imagined outcome chooses
            if wsc is not None and bool(on_.any()):
                if alw is not None:
                    wsc = torch.where(alw[None, :], wsc, torch.full_like(wsc, -1e9))
                best = wsc.argmax(1); act = torch.where(on_, best, act); self.cur = torch.where(on_, best, self.cur)
                again = again | on_; tried = tried & ~on_; self.b0_n = getattr(self, "b0_n", 0) + int(on_.sum())
        # 재는 쪽 기록 (고름을 바꾸지 않음): 이 박자의 몸짓을 누가 정했나 - 0 강한 끌림 (기억·다침·헛됨·값을 더한 점수), 1 값 (끌림이 약할 때 믿음만큼),
        # 2 고른 무작위 (끌림이 약할 때), 3 하던 것 유지, 4 시도
        src = torch.where(strong, 0, torch.where(usedv, 1, 2)); src = torch.where(again, src, torch.full_like(src, 3)); src = torch.where(tried, torch.full_like(src, 4), src)
        self.src = src
        if getattr(self, "src_count", None) is None or self.src_count.shape[0] != n:
            self.src_count = torch.zeros(n, 5, dtype=torch.long, device=d)
        self.src_count[torch.arange(n, device=d), src] += 1
        self.steps += 1
        return act

    @torch.no_grad()
    def _gut_goal(self, x, default):
        """G1 (registered 2026-11-28; Astra's somatic readout, LOKI de50f9f): the gut feeling over journeys. A journey = setting out for a
        remembered place until it ends (relief, giving up, death). The gut remembers, for "where I was and what I lacked" and "which place I
        set out for", how its insides changed by the journey's end (the drain on the way included; a death = everything lost). Choosing a
        goal asks the gut: among its best-remembered places for each inside, the one whose past journeys leave it most whole. No record for
        any of them -> the relief memory's own choice (as before). A journey is kept until it ends; it gives up after three times as long
        as its own paying journeys have usually taken."""
        go = self.going; n = len(default); d = default.device; k = go.k
        if getattr(self, "gut", None) is None:
            import sys as _s
            _s.path.insert(0, "ext/LOKI-v13")
            from loki_memory.somatic_readout import SomaticReadout
            self.gut = SomaticReadout(cue_width=2 + k, channels=k, slots=64, events=256).to(d)
            self.gj = torch.full((n,), -1, dtype=torch.long, device=d); self.gcue = torch.zeros(n, 2 + k, device=d); self.gbefore = torch.zeros(n, k, device=d)
            self.gt = torch.zeros(n, dtype=torch.long, device=d); self.g_since = torch.full((n,), 999, dtype=torch.long, device=d); self.g_far = torch.zeros(n, dtype=torch.bool, device=d); self.g_len = 100.0; self.g_asked = 0; self.g_known = 0; self.g_diff = 0; self.g_ends = 0
        df = 1.0 - (torch.stack([x[:, go.ins[2]], x[:, go.ins[1]]], 1) if go.chan == "room" else x[:, go.ins]).clamp(0, 1)
        P_ = go.organs.path.p if getattr(go, "organs", None) is not None and getattr(go.organs, "path", None) is not None else torch.zeros(n, 2, device=d)
        cue = torch.cat([P_ / 8.0, df], 1)
        self.gt = torch.where(self.gj >= 0, self.gt + 1, self.gt)
        rose = (self.gbefore - df).max(1).values > 0.05                         # an inside rose since it set out: the journey paid
        self.g_since = torch.where(rose & (self.gj >= 0), torch.zeros_like(self.g_since), self.g_since + 1)   # beats since its last relief
        late = self.gt > (self.g_len if os.environ.get("ATANOR_OWN_LEAVE", "0") == "1" else 3 * self.g_len)   # L7 (registered): past its own usual journey it
        end = (self.gj >= 0) & (rose | (default < 0) | late | self.cut)                                          # books what the journey cost so far and asks again
        for i in torch.nonzero(end)[:, 0].tolist():
            after = torch.ones(k, device=d) if bool(self.cut[i]) and not bool(rose[i]) else df[i]
            self.gut.remember(self.gcue[i], int(self.gj[i]), self.gbefore[i], after); self.g_ends += 1
            if bool(rose[i]) and bool(self.g_far[i]):                            # L7b: only journeys that began away from a meal count (a sip right after a sip is
                                                                                 # the same meal, not a journey). Its usual: the mean of paying journeys; with L7 the longest
                self.g_len = max(0.995 * self.g_len, float(self.gt[i])) if os.environ.get("ATANOR_OWN_LEAVE", "0") == "1" else 0.95 * self.g_len + 0.05 * float(self.gt[i])   # of late (fading), so sips at a spring do not shrink it
        self.gj = torch.where(end, torch.full_like(self.gj, -1), self.gj)
        out = default.clone(); start = (default >= 0) & (self.gj < 0)
        if bool(start.any()) and getattr(go, "V", None) is not None:
            V = go.V.clone()
            if go.used < go.KU:
                V[go.used:] = -1e9
            top = torch.cat([V[:, c].topk(2).indices for c in range(V.shape[1])])   # its two best-remembered places for each inside
            for i in torch.nonzero(start)[:, 0].tolist():
                acts = torch.unique(torch.cat([top, default[i:i + 1]]))
                known = (self.gut.fast_action[None, :] == acts[:, None]).any(1)
                pick = int(default[i]); self.g_asked += 1
                if bool(known.any()):
                    pick, conf, called = self.gut.choose(cue[i], acts, df[i]); self.g_known += 1; self.g_diff += int(pick != int(default[i]))
                self.gj[i] = pick; self.gcue[i] = cue[i]; self.gbefore[i] = df[i]; self.gt[i] = 0; self.g_far[i] = bool(self.g_since[i] > 8)
        held = self.gj >= 0
        return torch.where(held, self.gj, out)

    def _quiet(self, name):
        """Q1: the organ's own record - of the beats the body did what it asked, the share on which the body came nearer the place it named
        (by its own path sense). Starts at 1 (as it was); a fading mean; never under .05."""
        q = getattr(self, "q_rel", None)
        if q is None:
            self.q_rel = q = dict(lean=1.0, plan=1.0); self.q_n = dict(lean=0, plan=0)
        return max(0.05, q[name])

    def _quiet_check(self, died, cut):
        pe = getattr(self, "_q_pend", None); go = self.going
        if pe is None or getattr(go, "organs", None) is None:
            return
        pend, _ = pe; P_ = go.organs.path.p; self._quiet("lean")
        for nm, (led, tgt, d0) in pend.items():
            m = led & ~died & ~cut & (d0 > 0.7)
            if bool(m.any()):
                hit = ((d0 - (tgt - P_).norm(dim=1)) > 0.05)[m].float().mean()
                k = min(0.05, 0.002 * int(m.sum()))                              # more bodies checked this beat, more is learned (one mind)
                if os.environ.get("ATANOR_OWN_RATE", "0") == "1":                # SF3: and faster when the reading departs from its usual
                    dv = self.q_dev.get(nm, 0.2) if hasattr(self, "q_dev") else 0.2
                    if not hasattr(self, "q_dev"):
                        self.q_dev = {}
                    sur = abs(float(hit) - self.q_rel[nm]) / (dv + 1e-6); self.q_dev[nm] = 0.98 * dv + 0.02 * abs(float(hit) - self.q_rel[nm]); k = min(0.3, k * (1.0 + sur))
                self.q_rel[nm] = (1 - k) * self.q_rel[nm] + k * float(hit); self.q_n[nm] += int(m.sum())
        self._q_pend = None

    def _e1(self):
        if getattr(self, "e1", None) is None:
            import sys as _s
            _s.path.insert(0, "ext/LOKI-v13")
            from loki_memory.embodied_choice import EmbodiedChoice
            self.e1 = EmbodiedChoice(senses=2, channels=2, actions=self.M, path_width=2).to(self.going.dev)
        return self.e1

    def after(self, x, act, x2, died, cut):
        """한 걸음의 겪음. died [n] bool: 이 걸음에 죽음. cut [n] bool: 다음 걸음과 잇지 않음 (손이 옮김, 다시 태어남)."""
        ins = self.net.inside
        if getattr(self, "going", None) is not None:                           # L1: the going organ keeps its own place memory and ring
            self.going.observe(x, act, x2, died | cut)
            if os.environ.get("ATANOR_QUIET", "0") == "1":
                self._quiet_check(died, cut)
        if os.environ.get("ATANOR_E1", "0") == "1" and getattr(self, "_e1_prev", None) is not None and getattr(self.going, "organs", None) is not None:
            go_ = self.going; ec = self._e1(); sn_, P0, d0 = self._e1_prev; P1 = go_.organs.path.p
            d1 = 1.0 - torch.stack([x2[:, go_.ins[2]], x2[:, go_.ins[1]]], 1).clamp(0, 1)
            if sn_.shape[0] == P1.shape[0]:
                for i in torch.nonzero(~died & ~cut)[:, 0].tolist():           # one aligned transition per living body (one mind, one wiring)
                    ec.experience(sn_[i], P0[i], d0[i], int(act[i]), P1[i], d1[i])
        with torch.no_grad():
            rise = (x2[:, ins] - x[:, ins]).clamp(min=0)
            ok = ~died & ~cut & (rise > 0.01).any(1)
            if ok.any():
                self.net.keep(rise[ok], x[ok], act[ok])
                if getattr(self.net, "trail", False):
                    self.keep_trail(ok, rise)
            lwh = getattr(self, "learned_home", None)
            if lwh is not None and self.wo is not None:                          # LW1: the bedrock's own change this beat (a death = all its wholeness lost)
                w0_, w1_ = self.wo.whole(x), self.wo.whole(x2)
                lwh.learn(torch.where(died, -w0_, w1_ - w0_), ~cut | died)
            if self.wo is not None and getattr(self, "going", None) is not None and getattr(self, "goal_prev", None) is not None:
                # J1 and S1 no longer sit inside the learned trust (2026-11-27: with ATANOR_LEARNED_TRUST=0 they were silently off - found while
                # building TC1; gate 2 run 4's RW and the R1 full-weight diagnostics ran without J1)
                w0_, w1_ = self.wo.whole(x), self.wo.whole(x2)
                go_ = self.going; u2_ = go_.unit(go_.place(x2)); gp_ = self.goal_prev
                if lwh is not None and lwh.mode == "reli":
                    lwh.arrive((gp_ >= 0) & (u2_ == gp_), gp_, torch.where(died, -w0_, w1_ - w0_), ~died & ~cut)
                if os.environ.get("ATANOR_MISMATCH", "0") == "1" and getattr(go_, "V", None) is not None:   # J1 (registered): expectation broken -> judge again at once
                    lk_, ch_ = go_.needs(x2); ins_ = go_.ins
                    rose_ = torch.stack([x2[:, ins_[2]] - x[:, ins_[2]] > 0.01, x2[:, ins_[1]] - x[:, ins_[1]] > 0.01], 1).gather(1, ch_[:, None])[:, 0]
                    at_ = (gp_ >= 0) & (u2_ == gp_) & ~died & ~cut & (lk_ > 0.3)
                    mm_ = getattr(self, "_mm", None)
                    mm_ = torch.zeros(len(gp_), dtype=torch.long, device=gp_.device) if mm_ is None or mm_.shape[0] != len(gp_) else mm_
                    mm_ = torch.where(at_ & ~rose_, mm_ + 1, torch.zeros_like(mm_)); fire_ = mm_ >= 5      # 5 beats at its believed place, lacking, nothing comes
                    if bool(fire_.any()):
                        go_.R[gp_[fire_], ch_[fire_]] = 0.0                   # the place's relief memory for that need is the present: empty now
                        go_.refresh(); self.force = fire_ if getattr(self, "force", None) is None else (self.force | fire_)
                        self.mm_n = getattr(self, "mm_n", 0) + int(fire_.sum()); mm_ = torch.where(fire_, torch.zeros_like(mm_), mm_)
                    self._mm = mm_
                if lwh is not None and os.environ.get("ATANOR_STALL", "0") == "1" and getattr(go_, "Pu", None) is not None and getattr(go_, "organs", None) is not None:   # S1 (registered)
                    dg_ = (go_.Pu[gp_.clamp(min=0)] - go_.organs.path.p).norm(dim=1)   # its own distance to the goal's own place (not the hop target, which moves ahead)
                    lwh.stall(gp_, dg_, go_.Pn[gp_.clamp(min=0)] > 0, ~died & ~cut)
            lwr = getattr(self, "learned_roam", None)
            if lwr is not None and self.wo is not None:                          # G2 run 2: a promise = a choice the unfamiliarity swung; kept = a relief event within 30 beats
                sw_ = getattr(self, "_roam_swing", None); r0_, r1_ = self.wo.whole(x), self.wo.whole(x2)
                fol_ = (sw_[1] & (act == sw_[0])) if sw_ is not None else torch.zeros_like(died)
                lwr.arrive(fol_, None, torch.where(died, -r0_, r1_ - r0_), ~died & ~cut)
            if self.wo is not None:
                wdrop = (self.wo.whole(x) - self.wo.whole(x2))
                drop = torch.where(died, torch.ones_like(wdrop), wdrop.clamp(min=0))
                hi = (drop > 0.01) & ~cut
                if hi.any() and self.harm:
                    self.net.keep_harm(drop[hi], x[hi], act[hi])
                live = ~cut | died
                extra = None
                if self.lp and self.f_old is not None:
                    def ferr(f):
                        xe = torch.cat([x, self.net.codes[act]], 1)
                        return (((x + f(xe)) - x2) ** 2).mean(1)
                    prog = (ferr(self.f_old) - ferr(self.net.f)).clamp(min=0)
                    s = prog.mean() + 1e-6
                    self.lp_scale = s if self.lp_scale is None else 0.99 * self.lp_scale + 0.01 * s
                    extra = (self.lp * 0.01 * prog / self.lp_scale)[live]           # 온전함의 한 걸음 바뀜 (약 0.01)과 같은 눈금으로
                self.wo.see(x[live], x2[live], died[live], a=act[live], extra=extra, who=torch.nonzero(live)[:, 0])
                if self.steps % 20 == 0:
                    with torch.enable_grad():
                        self.wo.learn()
        if getattr(self, "roam", 0):
            self.remember_place(x2, ~died & ~cut)
        self.buf.append((x, act, x2, (~died).float() * (~cut).float())); self.buf = self.buf[-self.keep:]
        if self.lp and self.steps % self.lp_every == 0:
            import copy
            self.f_old = copy.deepcopy(self.net.f).eval()
            for p in self.f_old.parameters():
                p.requires_grad_(False)
        if self.inner and self.steps % self.inner_every == 0:
            with torch.enable_grad():
                self.learn_inner()
        if self.steps % self.learn_every == 0 and len(self.buf) > 100:
            if os.environ.get("ATANOR_REST_LEARN", "0") == "1":                # RL1 (registered 2026-11-28; Astra: "take the one call that stops the waking loop out of it"):
                self.owed = getattr(self, "owed", 0) + 1                       # awake it only keeps what it lived (its ring, as before) and reads the weights it has;
            else:                                                              # the forecaster's big learning waits for a rest (Mind.rest)
                self._learn_burst(20)
        self.prev = (x, act); self.cut = cut.clone()

    def _learn_burst(self, steps):
        if len(self.buf) <= 100:
            return
        X, A, X2, OK = [torch.cat([b[k] for b in self.buf]) for k in range(4)]
        self.net.train()
        with torch.enable_grad():
            for _ in range(steps):
                i = torch.randint(0, len(X), (2048,), device=self.dev)
                loss = self.net.loss(X[i], A[i], X2[i], OK[i]); self.opt.zero_grad(); loss.backward(); self.opt.step()
        self.net.eval()

    def rest(self, cap=400):
        """RL1: a rest. The forecaster's learning owed from the waking beats is done now (20 steps per waking burst skipped, at most `cap`
        steps a rest; what is over the cap is counted as lost, not hidden). Returns (steps done, bursts lost)."""
        owed = getattr(self, "owed", 0); steps = min(cap, 20 * owed); lost = max(0, owed - cap // 20)
        self._learn_burst(steps); self.owed = 0; self.rest_steps = getattr(self, "rest_steps", 0) + steps; self.rest_lost = getattr(self, "rest_lost", 0) + lost
        return steps, lost

    @torch.no_grad()
    def _track_pull(self, x, e, alive):
        n, K = len(x), self.pK
        if self.ptrace is None or self.ptrace["v"].shape[1] != n:
            self.ptrace = dict(v=torch.zeros(K, n, device=x.device), w=torch.zeros(K, n, device=x.device), a=torch.zeros(K, n, device=x.device), t=0)
        tr = self.ptrace; i = tr["t"] % K; g_ = self.wo.gamma
        if tr["t"] >= K:
            idx = [(i + k) % K for k in range(K)]; a_ = tr["a"][idx]; alive_k = torch.cumprod(a_, 0); w_ = tr["w"][idx] * alive_k
            w_end = torch.cat([w_[1:], (torch.where(alive, self.wo.whole(x), torch.zeros(n, device=x.device)) * alive_k[-1])[None]])
            disc = g_ ** torch.arange(K, device=x.device, dtype=torch.float32)[:, None]; gsum = (disc * (w_end - w_) * alive_k).sum(0)
            ok = (a_[0] > 0) & (tr["v"][i] != 0)
            if ok.sum() >= 1:
                v = tr["v"][i][ok]; gg = gsum[ok]; m = torch.stack([v.mean(), gg.mean(), (v * v).mean(), (gg * gg).mean(), (v * gg).mean()])
                self.poc.copy_(m if float(self.poc[2]) == 0 else 0.99 * self.poc + 0.01 * m)
        tr["v"][i] = e; tr["w"][i] = self.wo.whole(x); tr["a"][i] = alive.float(); tr["t"] += 1

    def pull_earned(self):
        ev, eg, ev2, eg2, evg = [float(z) for z in self.poc]
        c = (evg - ev * eg) / max(((ev2 - ev * ev) * (eg2 - eg * eg)) ** 0.5, 1e-9)
        return max(0.0, min(1.0, c))

    def use_trail(self, L=160, S=4, k=512):
        """길 기억을 켠다 (2026-11-23): 몸 안이 오른 순간 지난 L박자의 길에서 S박자마다 한 점씩 (느낀 것, 한 몸짓)을 든다. 한 몸이 L/2박자 안에 다시 담지 않는다 (샘에서 여러 번 마셔도 길은 하나)."""
        self.net.enable_trail(k); self.trail_L, self.trail_S = L, S
        self.trail_last = torch.full((self.n,), -10 ** 9, dtype=torch.long)

    @torch.no_grad()
    def keep_trail(self, ok, rise):
        L, S = self.trail_L, self.trail_S
        for j in torch.nonzero(ok)[:, 0].tolist():
            if self.steps - int(self.trail_last[j]) < L // 2:
                continue
            xs, aa = [], []
            for k in range(1, min(L, len(self.buf)) + 1):
                bx, ba, _, bok = self.buf[-k]
                if float(bok[j]) < 0.5:                                   # 끊김 (죽음·옮김) 너머는 이 길이 아니다
                    break
                if k % S == 0:
                    xs.append(bx[j]); aa.append(ba[j])
            if xs:
                self.net.keep_trail(rise[j], torch.stack(xs), torch.stack(aa)); self.trail_last[j] = self.steps

    def save(self, path):
        ring = ("bx", "bx2", "br", "bd", "ba", "ba2", "bp", "bi", "bn")          # 값의 되감기 고리 (해마의 단기 겪음)는 존재 파일에 넣지 않는다: 배운 것은 무게에 굳어 있다 (기억은 크기 고정, 가볍게)
        wst = {k: v for k, v in self.wo.state_dict().items() if k not in ring} if self.wo is not None else None
        torch.save(dict(signature=self.c.signature(), core=self.net.state_dict(), worth=wst,
                        worth_oc=(self.wo.oc.cpu() if self.wo is not None else None), organs=self.o.state(), steps=self.steps,
                        inner=(dict(n=self.inner, gru=self.gru.state_dict(), ahead=self.ahead.state_dict()) if self.inner else None)), path)

    def wear(self, path):
        """다른 몸의 존재 파일을 이 몸에 입힌다 (R2 몸 다시 짝짓기의 시작, 2026-11-22). 존재 안에 사는 눈 기관의 느낌 칸 (그림마다 z, zf, 응시, 벌레, 깊이)은
        두 몸에서 뜻이 같으므로 그 칸의 무게를 옮기고, 알맹이·값의 가운데 층 (모양이 같음)도 옮긴다. 몸마다 다른 칸 (바탕, 날 장치)과 몸짓 갈래의 무게는
        새로 시작 - 존재가 겪으며 스스로 짝짓는다. 겪음 기억 (rel/fnd/harm/slow)과 값의 얻은 믿음은 옮기지 않는다 (옛 몸의 것).
        돌려줌: 옮긴 것의 요약 (재는 쪽 기록)."""
        st = torch.load(path, map_location=self.dev, weights_only=False)
        osig, nsig = st["signature"], self.c.signature()
        if osig == nsig and st["core"]["f.4.weight"].shape[0] == self.o.dx:      # same body AND same sense width (G1: the path organ widens it; the signature does not say so)
            return dict(same_body=True, steps=self.load(path))
        def eye_map(sig, depth):
            off = sum(int(np.prod(s)) for k, s, b in sig[0] if b) + sum(int(np.prod(s)) for k, s, b in sig[0] if not b and k in ("vector", "audio"))
            per = Z + ZF + 2 + 1 + (5 if depth else 0)
            return [off + j * per + q for j in range(sum(1 for k, s, b in sig[0] if k == "image")) for q in range(per)], off + per * sum(1 for k, s, b in sig[0] if k == "image")
        ndx = self.o.dx; depth = self.o.DE is not None
        oe, odx = eye_map(osig, depth); ne, ndx2 = eye_map(nsig, depth)
        from .body_contract import PathOrgan
        pn = PathOrgan.N if getattr(self.o, "path", None) is not None else 0      # G1: the path code sits after the eye; new on this body, starts fresh
        po = PathOrgan.N if st["core"]["f.4.weight"].shape[0] == odx + PathOrgan.N else 0
        assert ndx2 + pn == ndx and odx + po == st["core"]["f.4.weight"].shape[0], "눈 칸 배치를 맞추지 못함"
        odx = odx + po
        k = min(len(oe), len(ne)); oe, ne = torch.tensor(oe[:k]), torch.tensor(ne[:k])
        oc = st["core"]["f.0.weight"].shape[1] - odx; nc = self.net.f[0].weight.shape[1] - ndx
        skip = ("rel_", "fnd_", "harm_", "slow_", "trl_", "e2", "t2", "ascale", "oc")
        moved = {}

        def graft(new_sd, old_sd, tag):
            out = {}
            for name, nv in new_sd.items():
                ov = old_sd.get(name)
                if ov is None or name.startswith(skip) or name.split(".")[-1] in skip:
                    continue
                ov = ov.to(nv.device, nv.dtype)
                if ov.shape == nv.shape and (nv.dim() < 1 or nv.shape[-1] not in (ndx, ndx + nc) or ndx == odx):
                    out[name] = ov.clone(); moved[f"{tag}.{name}"] = "whole"; continue
                v = nv.clone(); done = False
                if nv.dim() == 2 and (ov.shape[1], nv.shape[1]) in ((odx, ndx), (odx + oc, ndx + nc)):        # 들어오는 칸: 눈 칸 열만
                    if ov.shape[0] == nv.shape[0]:
                        v[:, ne] = ov[:, oe]; done = True
                    elif (ov.shape[0], nv.shape[0]) == (odx, ndx):
                        v[ne[:, None], ne[None]] = ov[oe[:, None], oe[None]]; done = True
                elif nv.dim() == 2 and (ov.shape[0], nv.shape[0]) == (odx, ndx) and ov.shape[1] == nv.shape[1]:   # 나가는 칸 (내다본 느낌): 눈 칸 행만
                    v[ne] = ov[oe]; done = True
                elif nv.dim() == 1 and (ov.shape[0], nv.shape[0]) == (odx, ndx):
                    v[ne] = ov[oe]; done = True
                if done:
                    out[name] = v; moved[f"{tag}.{name}"] = "eye part"
            return out

        self.net.load_state_dict(graft(self.net.state_dict(), st["core"], "core"), strict=False)
        if self.wo is not None and st.get("worth") is not None:
            self.wo.load_state_dict(graft(self.wo.state_dict(), st["worth"], "worth"), strict=False)
        oi = [i for i, (kk, s, b) in enumerate(osig[0]) if kk == "image"]; ni = self.o.img
        for a, b in zip(oi, ni):                                            # 눈의 셈 (z, zf 고르기)도 옮김
            self.o.zs[b].load(st["organs"]["z"].get(a), self.dev); self.o.fs[b].load(st["organs"]["f"].get(a), self.dev)
        self.steps = st.get("steps", 0)                                     # 나이는 잇는다 (하나의 이어지는 존재)
        return dict(same_body=False, steps=self.steps, eye_dims=int(k), old_dx=odx, new_dx=ndx, moved=moved)

    def grow(self, path):
        """P1 (atanor_os/P1_PATH_PROTOCOL.md): load a being file made before its body had the innate path organ. The organ's code sits at the end of the
        organs' sense (columns odx .. ndx); every weight that reads the sense gets zero columns there (so the being starts as it was and learns to use its
        new sense), every weight that predicts the sense keeps its own fresh rows there. Everything else is copied. Returns what was grown."""
        st = torch.load(path, map_location=self.dev, weights_only=False)
        assert st["signature"] == self.c.signature(), "다른 몸 계약"
        ndx = self.o.dx; odx = st["core"]["f.4.weight"].shape[0]; add = ndx - odx
        if add == 0:
            return dict(grown=0, steps=self.load(path))
        ring = ("bx", "bx2", "br", "bd", "ba", "ba2", "bp", "bi", "bn")
        grown = {}

        def fit(new_sd, old_sd, tag):
            out = {}
            for name, nv in new_sd.items():
                ov = old_sd.get(name)
                if ov is None or name.split(".")[-1] in ring:
                    continue
                ov = ov.to(nv.device, nv.dtype)
                if ov.shape == nv.shape:
                    out[name] = ov.clone(); continue
                v = nv.clone(); sl = [slice(None)] * nv.dim(); ok = True
                for d in range(nv.dim()):
                    if nv.shape[d] - ov.shape[d] == add and ov.shape[d] >= odx:       # this dimension runs over the sense: old part, new part, the rest
                        continue
                    if nv.shape[d] != ov.shape[d]:
                        ok = False
                if not ok:
                    continue
                idx_new = [torch.arange(s, device=nv.device) for s in nv.shape]; idx_old = []
                for d in range(nv.dim()):
                    if nv.shape[d] - ov.shape[d] == add and ov.shape[d] >= odx:
                        keep = torch.cat([torch.arange(odx, device=nv.device), torch.arange(odx + add, nv.shape[d], device=nv.device)]); idx_new[d] = keep
                        if nv.dim() == 2 and d == 1 and name.split(".")[-1] in ("rel_x", "fnd_x", "harm_x", "slow_x"):
                            # moments remembered before the organ: where they happened is not known. A code of a random place for each keeps the
                            # remembered place widely spread, so it carries little weight in the pull (a zero code would read as 'always at birth').
                            from .body_contract import PathOrgan
                            po = PathOrgan(nv.shape[0], nv.device); po.p = (torch.rand(nv.shape[0], 2, device=nv.device) - 0.5) * 24
                            po.h = (torch.rand(nv.shape[0], device=nv.device) - 0.5) * 2 * math.pi
                            v[:, odx:odx + add] = po(torch.zeros(nv.shape[0], 3, device=nv.device)).to(nv.dtype)
                        elif nv.dim() == 2 and d == 1:
                            v[:, odx:odx + add] = 0                                       # reading the new sense: starts at zero
                    idx_old.append(torch.arange(ov.shape[d], device=nv.device))
                v[torch.meshgrid(*idx_new, indexing="ij")] = ov[torch.meshgrid(*idx_old, indexing="ij")]
                out[name] = v; grown[f"{tag}.{name}"] = tuple(ov.shape)
            return out
        self.net.load_state_dict(fit(self.net.state_dict(), st["core"], "core"), strict=False)
        if self.wo is not None and st.get("worth") is not None:
            self.wo.load_state_dict(fit(self.wo.state_dict(), st["worth"], "worth"), strict=False)
        self.o.load(st["organs"]); self.steps = st.get("steps", 0)
        return dict(grown=add, old_dx=odx, new_dx=ndx, tensors=grown)

    def load(self, path):
        st = torch.load(path, map_location=self.dev, weights_only=False)
        assert st["signature"] == self.c.signature(), "다른 몸 계약: 갈래가 같지 않다 (새 갈래를 붙이려면 알맹이를 키워야 함)"
        self.net.load_state_dict(st["core"], strict=False)
        if self.wo is not None and st["worth"] is not None:
            self.wo.load_state_dict(st["worth"], strict=False)
        self.o.load(st["organs"])
        if self.inner and st.get("inner"):
            self.gru.load_state_dict(st["inner"]["gru"]); self.ahead.load_state_dict(st["inner"]["ahead"])
        self.steps = st.get("steps", 0)                                    # 존재의 나이 (걸음)를 잇는다 (전에는 불러도 0부터 다시 세어, 저장하면 나이가 줄었다)
        return self.steps
