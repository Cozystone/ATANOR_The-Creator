"""마음의 알맹이 (세계의 이름이 한 줄도 없다. 어느 몸의 세계에든 그대로 꽂는다).

받는 것은 숫자뿐이다: 느끼는 것 x [dx], 그 가운데 몸 안의 저장고 자리(inside), 자리 감각의 자리(place: 내용마다 [어느 쪽, 어느 쪽, 아는가] 셋씩), 할 수 있는 것의 수 m.
하는 일은 셋이다 (초단순 세계 tiny3.py에서 가려낸 그대로):
  내다보기   - 이 행동을 하면 다음에 무엇을 느낄까 (제 삶에서 걸음마다 배운다. 보답이 없어도)
  풀림의 기억 - 몸 안의 모자람이 풀린 순간마다, 그때 느낀 것과 한 것을 든다 (크기 고정. 가득 차면 가장 많이 든 종류의 가장 오래된 것이 자리를 내준다)
  끌림       - 가장 급한 모자람 하나가 그것이 풀렸던 순간들을 떠올린다. 내다본 다음 느낌이 거기에 가까워지는 행동에 끌리고, 그때 한 것은 여기서 그것이 모자람을 풀어 주리라 내다보는 만큼 끌린다.
               풀리던 순간 몸에 닿아 있던 것이 무겁고, 겪을수록 "매번 같았던 것"으로 추려진다. 모자람이 없으면 끌림도 없다.
값을 길게 배우지 않는다. 말도 이름표도 밖에서 준 목표도 계획기도 없다.
"""
import torch, torch.nn as nn, torch.nn.functional as F


class Core(nn.Module):
    def __init__(self, dx, m, inside, place, k=32, d=128, beta=30.0, inner=100.0):
        super().__init__()
        self.dx, self.m, self.inside, self.place, self.k, self.beta = dx, m, inside, place, k, beta
        ni = inside.stop - inside.start
        self.f = nn.Sequential(nn.Linear(dx + m, d), nn.GELU(), nn.Linear(d, d), nn.GELU(), nn.Linear(d, dx))
        self.g = nn.Sequential(nn.Linear(dx, d), nn.GELU(), nn.Linear(d, m))      # 이 행동을 하면 내 내다보기가 얼마나 틀릴까 (제 틀림에서 배운다)
        self.register_buffer("rel_de", torch.zeros(k, ni)); self.register_buffer("rel_x", torch.zeros(k, dx))
        self.register_buffer("rel_a", torch.zeros(k, dtype=torch.long)); self.register_buffer("rel_n", torch.zeros((), dtype=torch.long))
        self.register_buffer("rel_s", torch.zeros(k, dtype=torch.long)); self.register_buffer("rel_c", torch.zeros((), dtype=torch.long))
        self.register_buffer("fnd_x", torch.zeros(k, dx)); self.register_buffer("fnd_a", torch.zeros(k, dtype=torch.long)); self.register_buffer("fnd_n", torch.zeros((), dtype=torch.long))
        pl = torch.zeros(dx); pl[place] = 1
        self.register_buffer("pl", pl)
        self.also = None
        self.far, self._last = 0.0, None                              # far: 가까워지지 않은 것도 헛됨으로 치는 세기 (0이면 풀림만)
        self.like = True                                             # 그때 한 것은, 지금이 그때와 같게 느껴지는 만큼만 끌린다
        hv = torch.ones(dx); hv[inside] = inner
        self.register_buffer("heavy", hv)

    # ---- 내다보기 ----
    def foresee(self, x):
        eye = torch.eye(self.m, device=x.device).expand(*x.shape[:-1], self.m, self.m)
        xe = x[..., None, :].expand(*x.shape[:-1], self.m, -1)
        return xe + self.f(torch.cat([xe, eye], -1))

    def loss(self, X, A, X2, ok):
        fo = self.foresee(X).gather(-2, A[..., None, None].expand(*A.shape, 1, self.dx))[..., 0, :]
        e = (fo - X2) ** 2 * self.heavy                              # 제 몸 안을 내다본 것이 틀리면 무겁게 배운다 (몸 안이 차는 일은 드물어서, 고르게 배우면 바깥 감각의 자잘한 틀림에 묻혀 끝내 배우지 못한다. 재어 둔 것: EMERGE_PLAN 2026-11-21)
        l = (e.mean(-1) * ok).sum() / ok.sum().clamp(min=1)
        odd = ((fo - X2) ** 2).detach().mean(-1).sqrt()                # 방금 얼마나 틀렸는가 (몸 안을 무겁게 치지 않은 그대로)
        return l + (((self.g(X).gather(-1, A[..., None])[..., 0] - odd) ** 2) * ok).sum() / ok.sum().clamp(min=1)

    @torch.no_grad()
    def wonder(self, x):
        """둘째 바탕: 제 내다보기가 틀릴 곳에 끌린다. 행동마다, 틀리리라 보는 크기가 제 평균보다 얼마나 큰가 [B, m]."""
        g = self.g(x).clamp(min=0)
        return g / g.mean(1, keepdim=True).clamp(min=1e-4) - 1

    # ---- 풀림의 기억 ----
    @torch.no_grad()
    def keep(self, de, x, a):
        """de [n, ni] 몸 안에서 오른 것, x [n, dx] 그때 느낀 것, a [n] 그때 한 것."""
        for i in range(len(a)):
            if int(self.rel_n) < self.k:
                j = int(self.rel_n); self.rel_n += 1
            else:
                dn = F.normalize(self.rel_de, dim=-1); many = ((dn @ dn.t()) > 0.9).float().sum(1)
                j = int(torch.where(many >= many.max(), self.rel_s, torch.full_like(self.rel_s, 2 ** 62)).argmin())
            self.rel_de[j], self.rel_x[j], self.rel_a[j] = de[i].clamp(min=0), x[i], a[i]
            self.rel_c += 1; self.rel_s[j] = self.rel_c

    # ---- 찾아냄의 기억 (풀림의 기억과 같은 얼개를 한 단 앞에: 끌릴 데가 없다가 생긴 순간) ----
    @torch.no_grad()
    def keep_found(self, x, a):
        """모자란데 끌릴 데가 없다가, 이 걸음 뒤에 끌릴 데가 생겼다. 그 직전에 느낀 것 x [n, dx]와 한 것 a [n]을 든다 (크기 고정, 오래된 것부터 내준다)."""
        for i in range(len(a)):
            j = int(self.fnd_n) % self.k
            self.fnd_x[j], self.fnd_a[j] = x[i], a[i]; self.fnd_n += 1

    @torch.no_grad()
    def sees(self, x):
        """가장 급한 모자람이 풀렸던 순간에 닿아 있던 것이, 지금 어디 있는지 아는가 [B]."""
        n = int(self.rel_n)
        if n == 0:
            return torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        need = 1 - x[:, self.inside]
        need = need * F.one_hot(need.argmax(1), need.shape[1])
        dn = F.normalize(self.rel_de[:n], dim=-1)
        many = ((dn @ dn.t()) > 0.9).float().sum(1).clamp(min=1)
        w = torch.softmax(self.beta * (need @ dn.t()) - torch.log(many)[None], 1)
        g = self.rel_x[:n, self.place].view(n, -1, 3)
        touch = w @ (torch.exp(-4.0 * g[..., :2].norm(dim=-1)) * g[..., 2])
        return (touch * x[:, self.place].view(x.shape[0], -1, 3)[..., 2]).sum(1) > 0.5 * touch.sum(1)

    @torch.no_grad()
    def seek(self, x):
        """끌릴 데가 없을 때: 찾아냈던 순간들의 직전처럼 느껴지게 되는 행동, 그리고 그때 하던 행동에 끌린다 [B, m]. 든 것이 없으면 None."""
        n = min(int(self.fnd_n), self.k)
        if n < 4:
            return None
        mx = self.fnd_x[:n]; mu = mx.mean(0); var = mx.var(0)
        wt = 1 / (var + 0.02); wt[self.inside] = 0; wt = wt / wt.sum()
        d = (wt * (self.foresee(x) - mu) ** 2).sum(-1)
        return -10.0 * d + F.one_hot(self.fnd_a[:n], self.m).float().mean(0)[None]

    # ---- 헛됨 (창립자 2026-11-21 채택. 마음의 고르기에 든 고정된 성질 - 물고기와 선충에도 있는 바탕) ----
    @torch.no_grad()
    def unmet(self, x, a, x2):
        """방금 한 행동이 가장 급한 모자람을 풀어 주리라 내다본 만큼에서, 실제로 풀린 만큼을 뺀 것 [B] (0 이상), 그리고 실제로 풀렸는가 [B].
        이것을 행동마다 쌓아 두고 (새어 나가게 하며) 쌓인 만큼 그 행동의 끌림을 던다: 헛되면 놓는다. 쌓는 것은 몸을 사는 쪽이 한다 (존재마다 따로)."""
        ins = self.inside
        lack = F.one_hot((1 - x[:, ins]).argmax(1), ins.stop - ins.start).float()
        fo = self.foresee(x).gather(1, a[:, None, None].expand(-1, 1, self.dx))[:, 0]
        hoped = (((fo - x)[:, ins]) * lack).sum(1).clamp(min=0); came = (((x2 - x)[:, ins]) * lack).sum(1).clamp(min=0)
        more = (hoped - came).clamp(min=0)
        if self.far and self._last is not None and self._last[0].shape[0] == x.shape[0]:
            # 풀림만이 아니라, 떠올린 순간에 가까워지리라 내다본 만큼 가까워지지 않은 것도 헛됨이다 (벽을 밀고 있을 때처럼). 끌림과 같은 단위로 센다
            mu, wt, sc = self._last
            d0, dh, d2 = (wt * (x - mu) ** 2).sum(-1), (wt * (fo - mu) ** 2).sum(-1), (wt * (x2 - mu) ** 2).sum(-1)
            more = more + self.far * sc * ((d0 - dh).clamp(min=0) - (d0 - d2).clamp(min=0)).clamp(min=0)
        return more, came > 0.05

    # ---- 끌림 ----
    @torch.no_grad()
    def pull(self, x):
        """x [B, dx] -> 행동마다의 끌림 [B, m] (끌림이 없는 줄은 0), 끌림이 있는가 [B]"""
        n = int(self.rel_n)
        B = x.shape[0]
        if n == 0:
            return torch.zeros(B, self.m, device=x.device), torch.zeros(B, dtype=torch.bool, device=x.device)
        need = 1 - x[:, self.inside]
        need = need * F.one_hot(need.argmax(1), need.shape[1])
        dn = F.normalize(self.rel_de[:n], dim=-1)
        rel = need @ dn.t()
        many = ((dn @ dn.t()) > 0.9).float().sum(1).clamp(min=1)
        w = torch.softmax(self.beta * rel - torch.log(many)[None], 1)
        strong = (w * rel).sum(1) > 0.02
        mx = self.rel_x[:n]
        mu = w @ mx
        var = (w[:, :, None] * (mx[None] - mu[:, None]) ** 2).sum(1)
        g = mx[:, self.place].view(n, -1, 3)
        touch = w @ (torch.exp(-4.0 * g[..., :2].norm(dim=-1)) * g[..., 2])
        wt = torch.zeros_like(var)
        wt[:, self.place] = touch.repeat_interleave(3, 1) / (var[:, self.place] + 0.02)
        if self.also is not None:                                    # 자리 감각 밖의 느낌 가운데 풀린 순간마다 같았던 것도 끌림에 든다 (눈의 초점에 차 있던 빛깔 같은 것)
            cap = wt[:, self.place].sum(1, keepdim=True) / (wt[:, self.place] > 0).sum(1, keepdim=True).clamp(min=1)      # 눈의 자리 감각이 받는 무게의 평균
            wt[:, self.also] = torch.minimum(1 / (var[:, self.also] + 0.02), cap)      # 다른 느낌(소리 같은 것)은 그보다 무겁지 않다 (흩어짐이 작다고 끌림을 덮어쓰지 않게)
        fo = self.foresee(x)
        d = (wt[:, None] * (fo - mu[:, None]) ** 2).sum(-1)
        here = torch.exp(-(wt * (x - mu) ** 2).sum(-1))
        act = w @ F.one_hot(self.rel_a[:n], self.m).float()
        rise = ((fo[:, :, self.inside] - x[:, None, self.inside]) * (need > 0).float()[:, None]).sum(-1).clamp(min=0)     # 이 행동이 지금 급한 그 모자람을 얼마나 풀 것이라 내다보는가
        act = act * (rise / need.max(1, keepdim=True).values.clamp(min=0.05)).clamp(max=1) ** 2 * (here[:, None] if self.like else 1.0)
        #    # 그때 한 것은, 여기서 그것이 풀어 주리라 내다보는 만큼만 끌린다 (멀리서 헛되이 하지 않는다)
        self._last = (mu, wt, need.max(1).values ** 2 * strong)       # 방금 무엇에 끌렸는가 (헛됨을 셈할 때 쓴다)
        return (need.max(1).values ** 2 * strong)[:, None] * (-d + 10.0 * act), strong
