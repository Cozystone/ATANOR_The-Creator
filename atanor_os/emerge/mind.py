"""하나의 마음 (공통 접면): 계약을 읽은 기관 (body_contract.Organs)이 만든 느낌 x만 받고 몸짓 번호를 낸다. 방이든 마크든 같은 고리, 같은 존재 파일.
고르기 (손 규칙 없음): 알맹이의 끌림 (풀림의 기억) - 헛됨 (넓힘 + 탈습관화) + 다침의 기억 (좁힘) + 배우는 심장 (몸짓마다의 값, 바탕만 보상, 겪어서 얻은 믿음만큼).
하던 몸짓은 스스로 붙든다: 놀라거나, 더 끌리는 것이 나타나거나, 헛됨이 쌓여 덜 끌릴 때만 새로 고른다 (상한 없음). 끌림이 없으면 얻은 믿음만큼 값을, 나머지는 아무거나.
배움: 내다보기 (겪은 걸음의 고정 크기 고리에서), 풀림·다침의 기억 (바탕의 오름 / 온전함의 갑작스런 줄어듦), 값 (되감기).
  m = Mind(organs, contract); act = m.choose(x); m.after(x, act, x2, died, cut)   # cut: 손이 옮겼거나 다시 태어남 (사슬 끊기)
  m.save(path) / Mind.load(path, organs, contract)
"""
import torch, torch.nn.functional as F
from .core import Core, Worth

LEAK, TIRE, LOOSE = 0.95, 4.0, 0.05


class Mind:
    def __init__(self, organs, contract, n, dev="cuda", far=0.3, worth=2.0, try_p=0.1, harm=1.0, learn_every=50, keep=4000, lp=0.0, lp_every=1000, inner=0):
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
            self.ahead = nn.Sequential(nn.Linear(inner + organs.dx + C.shape[1], 256), nn.ELU(), nn.Linear(256, len(self.HOR) * organs.dx)).to(dev)
            self.iopt = torch.optim.Adam(list(self.gru.parameters()) + list(self.ahead.parameters()), lr=1e-3)
            self.h = torch.zeros(n, inner, device=dev); self.seq = []
        self.net = Core(dxe, self.M, organs.inside, slice(0, 0), codes=C).to(dev); self.net.far = far; self.net.harm_sharp = True
        self.opt = torch.optim.AdamW(self.net.parameters(), lr=1e-3)
        self.wo = Worth(dxe, organs.inside, lam=worth, m=self.M).to(dev) if worth else None
        self.try_p, self.harm, self.learn_every, self.keep = try_p, harm, learn_every, keep
        self.buf = []; self.steps = 0
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
                tgt = torch.stack([X[t + 1 + j, b] for j in self.HOR], 1)
                ok = torch.stack([~Cu[t + 1:t + 2 + j, b].any(0) for j in self.HOR], 1).float()       # 사이에 끊김 (죽음, 옮김)이 없을 때만
                loss = loss + (((pred - tgt) ** 2).mean(2) * ok).sum() / ok.sum().clamp(min=1)
        if not torch.is_tensor(loss):
            return None
        self.iopt.zero_grad(); loss.backward(); self.iopt.step()
        return float(loss)

    @torch.no_grad()
    def choose(self, x):
        n, d = self.n, self.dev
        if self.prev is not None:
            px, pa = self.prev
            more, eased = self.net.unmet(px, pa, x)
            self.fut = self.fut * LEAK; self.fut[torch.arange(n, device=d), pa] += more * (~self.cut); self.fut[eased] = 0
            fo = self.net.foresee(px).gather(1, pa[:, None, None].expand(-1, 1, x.shape[1]))[:, 0]
            odd = ((fo - x) ** 2).mean(1); novel = (odd > self.s_m + 2 * self.s_v.sqrt()) & (self.steps > 20) & ~self.cut
            self.fut[novel] = 0; self.s_m = self.s_m + 0.02 * (odd - self.s_m); self.s_v = self.s_v + 0.02 * ((odd - self.s_m) ** 2 - self.s_v)
        else:
            novel = torch.ones(n, dtype=torch.bool, device=d)
        sc, strong = self.net.pull(x)
        if self.harm and int(self.net.harm_n) > 0:
            av = self.net.avoid(x); sc = sc + self.harm * av; strong = strong | (av.abs().max(1).values > 0.02)
        sc = sc - TIRE * self.fut
        lw = None
        if self.wo is not None:
            self.wo.track(x, ~self.cut)
            lw = self.wo.lean(self.net, x)
            if lw is not None:
                sc = sc + lw
        pick = torch.multinomial(torch.softmax(sc / LOOSE, 1), 1)[:, 0]
        rnd = torch.randint(0, self.M, (n,), device=d)
        if lw is not None:
            byv = torch.multinomial(torch.softmax(lw / LOOSE, 1), 1)[:, 0]
            rnd = torch.where(torch.rand(n, device=d) < self.wo.earned(), byv, rnd)
        pick = torch.where(strong, pick, rnd)
        gap = sc.max(1).values - sc.gather(1, self.cur[:, None])[:, 0]
        pre = strong & (gap > 2.0 * sc.std(1).clamp(min=1e-6))
        again = novel | pre | self.cut
        self.cur = torch.where(again, pick, self.cur)
        act = torch.where(torch.rand(n, device=d) < self.try_p, torch.randint(0, self.M, (n,), device=d), self.cur)
        self.steps += 1
        return act

    def after(self, x, act, x2, died, cut):
        """한 걸음의 겪음. died [n] bool: 이 걸음에 죽음. cut [n] bool: 다음 걸음과 잇지 않음 (손이 옮김, 다시 태어남)."""
        ins = self.net.inside
        with torch.no_grad():
            rise = (x2[:, ins] - x[:, ins]).clamp(min=0)
            ok = ~died & ~cut & (rise > 0.01).any(1)
            if ok.any():
                self.net.keep(rise[ok], x[ok], act[ok])
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
                self.wo.see(x[live], x2[live], died[live], a=act[live], extra=extra)
                if self.steps % 20 == 0:
                    with torch.enable_grad():
                        self.wo.learn()
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
            X, A, X2, OK = [torch.cat([b[k] for b in self.buf]) for k in range(4)]
            self.net.train()
            for _ in range(20):
                i = torch.randint(0, len(X), (2048,), device=self.dev)
                loss = self.net.loss(X[i], A[i], X2[i], OK[i]); self.opt.zero_grad(); loss.backward(); self.opt.step()
            self.net.eval()
        self.prev = (x, act); self.cut = cut.clone()

    def save(self, path):
        ring = ("bx", "bx2", "br", "bd", "ba", "bp", "bi", "bn")          # 값의 되감기 고리 (해마의 단기 겪음)는 존재 파일에 넣지 않는다: 배운 것은 무게에 굳어 있다 (기억은 크기 고정, 가볍게)
        wst = {k: v for k, v in self.wo.state_dict().items() if k not in ring} if self.wo is not None else None
        torch.save(dict(signature=self.c.signature(), core=self.net.state_dict(), worth=wst,
                        worth_oc=(self.wo.oc.cpu() if self.wo is not None else None), organs=self.o.state(), steps=self.steps,
                        inner=(dict(n=self.inner, gru=self.gru.state_dict(), ahead=self.ahead.state_dict()) if self.inner else None)), path)

    def load(self, path):
        st = torch.load(path, map_location=self.dev, weights_only=False)
        assert st["signature"] == self.c.signature(), "다른 몸 계약: 갈래가 같지 않다 (새 갈래를 붙이려면 알맹이를 키워야 함)"
        self.net.load_state_dict(st["core"], strict=False)
        if self.wo is not None and st["worth"] is not None:
            self.wo.load_state_dict(st["worth"], strict=False)
        self.o.load(st["organs"])
        if self.inner and st.get("inner"):
            self.gru.load_state_dict(st["inner"]["gru"]); self.ahead.load_state_dict(st["inner"]["ahead"])
        return st.get("steps", 0)
