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
    def __init__(self, dx, m, inside, place, k=32, d=128, beta=30.0, inner=100.0, codes=None):
        super().__init__()
        # 몸짓의 부호 [m, c]: 없으면 몸짓마다 제 칸 하나 (단위 행렬, 처음부터의 꼴). 갈래 몸이면 다리·목·손 칸을 함께 켠다 -> 내다보기가 갈래의 공통을 나눠 배우고, 기억 속 '그때 한 것'은 부호의 닮음으로 견준다
        codes = torch.eye(m) if codes is None else torch.as_tensor(codes, dtype=torch.float32)
        cn = codes / codes.norm(dim=1, keepdim=True).clamp(min=1e-6)
        self.register_buffer("codes", codes, persistent=False); self.register_buffer("same", cn @ cn.t(), persistent=False)
        self.dx, self.m, self.inside, self.place, self.k, self.beta = dx, m, inside, place, k, beta
        ni = inside.stop - inside.start
        self.f = nn.Sequential(nn.Linear(dx + codes.shape[1], d), nn.GELU(), nn.Linear(d, d), nn.GELU(), nn.Linear(d, dx))
        self.g = nn.Sequential(nn.Linear(dx, d), nn.GELU(), nn.Linear(d, m))      # 이 행동을 하면 내 내다보기가 얼마나 틀릴까 (제 틀림에서 배운다)
        self.register_buffer("rel_de", torch.zeros(k, ni)); self.register_buffer("rel_x", torch.zeros(k, dx))
        self.register_buffer("rel_a", torch.zeros(k, dtype=torch.long)); self.register_buffer("rel_n", torch.zeros((), dtype=torch.long))
        self.register_buffer("rel_s", torch.zeros(k, dtype=torch.long)); self.register_buffer("rel_c", torch.zeros((), dtype=torch.long))
        self.register_buffer("rel_inn", torch.zeros(k, dtype=torch.bool))       # 타고난 본능의 씨앗인가 (저장에도 남게)
        self.register_buffer("fnd_x", torch.zeros(k, dx)); self.register_buffer("fnd_a", torch.zeros(k, dtype=torch.long)); self.register_buffer("fnd_n", torch.zeros((), dtype=torch.long))
        # 다침의 기억 (풀림의 기억과 짝, 크기 고정): 몸 안이 깎인 순간 (얼마나, 그때 느낀 것, 그때 한 것). 닮은 때에 그 장면에서 멀어지는 쪽으로 밀리고 그때 한 것을 꺼린다
        self.register_buffer("harm_de", torch.zeros(k)); self.register_buffer("harm_x", torch.zeros(k, dx)); self.register_buffer("harm_a", torch.zeros(k, dtype=torch.long))
        self.register_buffer("harm_n", torch.zeros((), dtype=torch.long)); self.register_buffer("harm_c", torch.zeros((), dtype=torch.long))
        # 느린 층 (다침을 굳혀 둠, 크기 고정): 빠른 기억에서 밀려나는 순간을 버리지 않고 가장 닮은 '다침의 꼴'에 섞는다 (온라인 k-평균). 행동은 섞인 순간들의 비율로
        self.register_buffer("slow_x", torch.zeros(k, dx)); self.register_buffer("slow_de", torch.zeros(k)); self.register_buffer("slow_a", torch.zeros(k, m))
        self.register_buffer("slow_cnt", torch.zeros(k)); self.register_buffer("slow_n", torch.zeros((), dtype=torch.long))
        self.slow = False
        pl = torch.zeros(dx); pl[place] = 1
        self.register_buffer("pl", pl)
        self.also = None
        self.far, self._last = 0.0, None                              # far: 가까워지지 않은 것도 헛됨으로 치는 세기 (0이면 풀림만)
        self.like = True                                             # 그때 한 것은, 지금이 그때와 같게 느껴지는 만큼만 끌린다
        hv = torch.ones(dx); hv[inside] = inner
        self.register_buffer("heavy", hv)

    # ---- 내다보기 ----
    def foresee(self, x):
        eye = self.codes.expand(*x.shape[:-1], self.m, self.codes.shape[1])
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
    def keep(self, de, x, a, innate=False):
        """de [n, ni] 몸 안에서 오른 것, x [n, dx] 그때 느낀 것, a [n] 그때 한 것.
        innate: 타고난 본능의 씨앗 (손으로 준 것, 장부). 그 행동은 '풀어 주리라 내다봄' 없이도 끌린다 (본능은 알아챔과 몸짓이 한 묶음: 개구리는 먹어 본 적 없이도 문다)."""
        for i in range(len(a)):
            if int(self.rel_n) < self.k:
                j = int(self.rel_n); self.rel_n += 1
            else:
                dn = F.normalize(self.rel_de, dim=-1); many = ((dn @ dn.t()) > 0.9).float().sum(1)
                j = int(torch.where(many >= many.max(), self.rel_s, torch.full_like(self.rel_s, 2 ** 62)).argmin())
            self.rel_de[j], self.rel_x[j], self.rel_a[j] = de[i].clamp(min=0), x[i], a[i]
            self.rel_inn[j] = bool(innate)
            self.rel_c += 1; self.rel_s[j] = self.rel_c

    @torch.no_grad()
    def keep_harm(self, drop, x, a):
        """drop [n] 몸 안이 깎인 크기, x [n, dx] 그때(깎이기 직전) 느낀 것, a [n] 그때 한 것. 크기 고정: 가득 차면 가장 오래된 것부터."""
        for i in range(len(a)):
            j = int(self.harm_c) % self.k
            if self.slow and int(self.harm_n) >= self.k:                        # 밀려날 순간을 느린 층에 굳힌다
                self.consolidate(self.harm_de[j], self.harm_x[j], self.harm_a[j])
            self.harm_de[j], self.harm_x[j], self.harm_a[j] = float(drop[i]), x[i], a[i]
            self.harm_c += 1; self.harm_n = torch.clamp(self.harm_n + 1, max=self.k)

    @torch.no_grad()
    def consolidate(self, de, x, a):
        """빠른 기억에서 밀려나는 다친 순간 하나를 느린 층에: 가득 차기 전에는 새 칸, 가득 차면 가장 닮은 꼴에 겪은 수로 고르게 섞는다 (문턱 없음)."""
        n = int(self.slow_n); oh = self.same[a.long()]
        if n < self.k:
            j = n; self.slow_n += 1
            self.slow_x[j], self.slow_de[j], self.slow_a[j], self.slow_cnt[j] = x, de, oh, 1.0
            return
        other = torch.ones(self.dx, dtype=torch.bool, device=x.device); other[self.inside] = False
        q = 1.0 / (self.slow_x[:, other].var(0, unbiased=False) + 0.02)
        dn = (((self.slow_x[:, other] - x[other]) ** 2) * q).sum(-1)
        sp = self.slow_x[:, other] * q.sqrt(); pd = torch.cdist(sp, sp) ** 2; pd.fill_diagonal_(float("inf"))
        if bool(getattr(self, "slow_pair", True)) and float(dn.min()) > float(pd.min()):
            # 새 순간이 어느 꼴과도 꼴끼리보다 멀다 = 새로운 종류의 다침: 가장 닮은 두 꼴을 하나로 합치고 빈 칸에 새 순간을 둔다 (한 덩어리가 다 먹는 것을 막음, 문턱 없음)
            f = int(pd.argmin()); i, j = f // self.k, f % self.k
            ci, cj = self.slow_cnt[i], self.slow_cnt[j]; t = ci + cj
            self.slow_x[i] = (ci * self.slow_x[i] + cj * self.slow_x[j]) / t; self.slow_de[i] = (ci * self.slow_de[i] + cj * self.slow_de[j]) / t
            self.slow_a[i] = (ci * self.slow_a[i] + cj * self.slow_a[j]) / t; self.slow_cnt[i] = t
            self.slow_x[j], self.slow_de[j], self.slow_a[j], self.slow_cnt[j] = x, de, oh, 1.0
            return
        j = int(dn.argmin())
        c = self.slow_cnt[j] + 1; r = 1.0 / c
        self.slow_x[j] += r * (x - self.slow_x[j]); self.slow_de[j] += r * (de - self.slow_de[j]); self.slow_a[j] += r * (oh - self.slow_a[j]); self.slow_cnt[j] = c

    @torch.no_grad()
    def avoid(self, x):
        """x [B, dx] -> 행동마다 밀림 [B, m] (다침의 기억이 없으면 0). 끌림(pull)과 같은 얼개를 거꾸로:
        지금과 닮은 다친 순간들을 떠올려 (닮음은 그 순간들 사이에서 늘 같았던 느낌으로 잰다), 내다본 다음 느낌이 그 장면에서 멀어질수록 좋고, 그때 한 것은 꺼린다.
        모자람과 상관없이 늘 켜진다 (아픔은 배가 불러도 아프다)."""
        n = int(self.harm_n)
        B = x.shape[0]
        if n == 0:
            return torch.zeros(B, self.m, device=x.device)
        hx, hde, hoh = self.harm_x[:n], self.harm_de[:n], self.same[self.harm_a[:n]]
        ns = int(self.slow_n) if self.slow else 0
        if ns:                                                                      # 빠른 것과 느린 것을 한데 놓고 같은 닮음으로 떠올린다
            hx = torch.cat([hx, self.slow_x[:ns]]); hde = torch.cat([hde, self.slow_de[:ns]]); hoh = torch.cat([hoh, self.slow_a[:ns]])
        n = len(hx)
        other = torch.ones(self.dx, dtype=torch.bool, device=x.device); other[self.inside] = False
        var = hx[:, other].var(0, unbiased=False) if n > 1 else torch.ones(int(other.sum()), device=x.device)
        q = 1.0 / (var + 0.02); wt = torch.zeros(self.dx, device=x.device); wt[other] = q / q.mean()
        dist = ((x[:, None] - hx[None]) ** 2 * wt).sum(-1) / other.sum()          # [B, n] 지금과 그 순간들의 거리 (느낌 한 칸당)
        w = torch.softmax(-dist * 4.0, 1) * hde[None]              # 닮은 순간일수록, 크게 다친 순간일수록
        dmin = dist.min(1).values
        if getattr(self, "harm_sharp", False):                                      # 좁힘: 평소에 다친 장면과 떨어져 있는 거리보다 뚜렷이 가까울 때만 (위험한 데서만 피하고 나머지는 자유롭게)
            if not hasattr(self, "harm_dbar"):
                self.harm_dbar = dmin.mean().detach()
            self.harm_dbar = 0.99 * self.harm_dbar + 0.01 * dmin.mean().detach()
            sim = torch.sigmoid((self.harm_dbar - dmin) / (0.15 * self.harm_dbar + 1e-6))
        else:
            sim = torch.exp(-dmin * 2.0)                                            # 지금이 다친 순간과 얼마나 닮았나 (0~1)
        mu = (w / w.sum(1, keepdim=True).clamp(min=1e-6)) @ hx
        fo = self.foresee(x)
        away = ((fo - mu[:, None]) ** 2 * wt).sum(-1) / other.sum() - ((x - mu) ** 2 * wt).sum(-1, keepdim=True) / other.sum()   # 이 행동으로 그 장면에서 멀어지는 만큼
        did = w @ hoh / w.sum(1, keepdim=True).clamp(min=1e-6)
        return sim[:, None] * (away - did)

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
        return -10.0 * d + self.same[self.fnd_a[:n]].mean(0)[None]

    # ---- 헛됨 (창립자 2026-11-21 채택. 마음의 고르기에 든 고정된 성질 - 물고기와 선충에도 있는 바탕) ----
    @torch.no_grad()
    def unmet(self, x, a, x2):
        """방금 한 행동이 가장 급한 모자람을 풀어 주리라 내다본 만큼에서, 실제로 풀린 만큼을 뺀 것 [B] (0 이상), 그리고 실제로 풀렸는가 [B].
        이것을 행동마다 쌓아 두고 (새어 나가게 하며) 쌓인 만큼 그 행동의 끌림을 던다: 헛되면 놓는다. 쌓는 것은 몸을 사는 쪽이 한다 (존재마다 따로)."""
        ins = self.inside
        if getattr(self, "whole", False):                            # 온전함의 잣대: 모자람마다 결핍^2의 몫으로 (가장 급한 하나만이 아니라)
            dfc = (1 - x[:, ins]).clamp(min=0) ** 2
            lack = dfc / dfc.sum(1, keepdim=True).clamp(min=1e-6)
        else:
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
        """x [B, dx] -> 행동마다의 끌림 [B, m] (끌림이 없는 줄은 0), 끌림이 있는가 [B]
        whole이 켜지면: 모자람마다 따로 떠올려 끌림을 내고, 온전함(1 - 평균 제곱 결핍)이 그 모자람으로 깎이는 만큼(결핍^2)의 무게로 더한다.
        꺼져 있으면: 가장 급한 모자람 하나만 (처음부터의 꼴)."""
        if not getattr(self, "whole", False):
            full = 1 - x[:, self.inside]
            return self._pull(x, full * F.one_hot(full.argmax(1), full.shape[1]))
        full = (1 - x[:, self.inside]).clamp(min=0)
        tot, any_, keep, top = 0.0, None, None, None
        for k in range(full.shape[1]):                               # 모자람 k의 끌림은 _pull 안에서 결핍_k^2 의 무게를 받는다
            sc, st = self._pull(x, full * F.one_hot(torch.full((x.shape[0],), k, device=x.device), full.shape[1]))
            tot = tot + sc; any_ = st if any_ is None else (any_ | st)
            if keep is None:
                keep, top = self._last, full[:, k]
            else:                                                    # 헛됨이 쓸 '방금 끌린 것'은 가장 급한 모자람의 것으로
                use = full[:, k] > top
                keep = tuple(torch.where(use[:, None] if v.dim() == 2 else use, v, o) for v, o in zip(self._last, keep)); top = torch.maximum(full[:, k], top)
        self._last = keep
        return tot, any_

    @torch.no_grad()
    def _pull(self, x, need):
        """need [B, ni]: 이번에 떠올릴 모자람 (그 줄만 0이 아님)."""
        n = int(self.rel_n)
        B = x.shape[0]
        if n == 0:
            return torch.zeros(B, self.m, device=x.device), torch.zeros(B, dtype=torch.bool, device=x.device)
        dn = F.normalize(self.rel_de[:n], dim=-1)
        rel = need @ dn.t()
        many = ((dn @ dn.t()) > 0.9).float().sum(1).clamp(min=1)
        w = torch.softmax(self.beta * rel - torch.log(many)[None], 1)
        strong = (w * rel).sum(1) > 0.02
        mx = self.rel_x[:n]
        mu = w @ mx
        var = (w[:, :, None] * (mx[None] - mu[:, None]) ** 2).sum(1)
        wt = torch.zeros_like(var)
        if self.place.stop == self.place.start:                      # 자리 감각을 따로 묶지 않은 몸 (마크처럼): 몸 안 밖의 모든 느낌 가운데 풀린 순간마다 같았던 것이 무겁다
            other = torch.ones(self.dx, dtype=torch.bool, device=x.device); other[self.inside] = False
            q = 1.0 / (var[:, other] + 0.02)
            wt[:, other] = q / q.mean(1, keepdim=True)
        else:
            g = mx[:, self.place].view(n, -1, 3)
            touch = w @ (torch.exp(-4.0 * g[..., :2].norm(dim=-1)) * g[..., 2])
            wt[:, self.place] = touch.repeat_interleave(3, 1) / (var[:, self.place] + 0.02)
        if self.also is not None and self.place.stop != self.place.start:                                    # 자리 감각 밖의 느낌 가운데 풀린 순간마다 같았던 것도 끌림에 든다 (눈의 초점에 차 있던 빛깔 같은 것)
            cap = wt[:, self.place].sum(1, keepdim=True) / (wt[:, self.place] > 0).sum(1, keepdim=True).clamp(min=1)      # 눈의 자리 감각이 받는 무게의 평균
            wt[:, self.also] = torch.minimum(1 / (var[:, self.also] + 0.02), cap)      # 다른 느낌(소리 같은 것)은 그보다 무겁지 않다 (흩어짐이 작다고 끌림을 덮어쓰지 않게)
        fo = self.foresee(x)
        d = (wt[:, None] * (fo - mu[:, None]) ** 2).sum(-1)
        here = torch.exp(-(wt * (x - mu) ** 2).sum(-1))
        inn = getattr(self, "rel_inn", None)
        inn = inn[:n].float() if inn is not None else torch.zeros(n, device=x.device)
        oh = self.same[self.rel_a[:n]]
        act = (w * (1 - inn)[None]) @ oh
        rise = ((fo[:, :, self.inside] - x[:, None, self.inside]) * (need > 0).float()[:, None]).sum(-1).clamp(min=0)     # 이 행동이 지금 급한 그 모자람을 얼마나 풀 것이라 내다보는가
        act = act * (rise / need.max(1, keepdim=True).values.clamp(min=0.05)).clamp(max=1) ** 2 * (here[:, None] if self.like else 1.0)
        act = act + ((w * inn[None]) @ oh) * (here[:, None] if self.like else 1.0)      # 본능의 씨앗: 내다봄 없이, 지금이 그 장면과 닮은 만큼만 끌린다
        #    # 그때 한 것은, 여기서 그것이 풀어 주리라 내다보는 만큼만 끌린다 (멀리서 헛되이 하지 않는다)
        self._last = (mu, wt, need.max(1).values ** 2 * strong)       # 방금 무엇에 끌렸는가 (헛됨을 셈할 때 쓴다)
        return (need.max(1).values ** 2 * strong)[:, None] * (-d + 10.0 * act), strong


class Worth(nn.Module):
    """배우는 심장 (창립자 결정 2026-11-22: 바탕만을 보상으로 하는 값 배움 허락). 기저핵 자리: '이 느낌 뒤에 내 온전함이 앞으로 어떻게 바뀌나'를 배운다.
    보상은 오직 제 온전함 (1 - 몸 안 결핍 제곱의 평균, 고정된 첫 바탕)의 바뀜, 죽음 = 온전함이 다 사라짐. 다른 보상 없음, 모자람 사이의 무게를 짜지 않음.
    겪은 걸음은 크기 고정 고리에 들고 (기억이 자라지 않음), learn()이 되감아 고친다 (쉼의 자리에서 부르면 해마의 되감기).
    lean(core, x): 몸짓마다 '내다본 다음 느낌의 값 - 지금 값'을 몸짓들 사이의 흩어짐으로 나눈 것 x lam. 장부: lam, gamma (참을성), 고리 크기, 그물 꼴."""
    def __init__(self, dx, inside, lam=2.0, gamma=0.97, keep=300000, d=128, m=None):
        super().__init__()
        self.inside, self.lam, self.gamma, self.keep = inside, lam, gamma, keep
        # m이 주어지면 몸짓마다의 값 (선조체의 행동 값, Samejima 등 2005): '이 장면에서 이 몸짓을 하면 앞으로 내 온전함이' - 내다보기를 거치지 않음
        self.m = m
        mk = lambda: nn.Sequential(nn.Linear(dx, d), nn.ELU(), nn.Linear(d, d), nn.ELU(), nn.Linear(d, m or 1))
        self.v, self.vt = mk(), mk(); self.vt.load_state_dict(self.v.state_dict())
        for p in self.vt.parameters():
            p.requires_grad_(False)
        self.register_buffer("bx", torch.zeros(keep, dx)); self.register_buffer("bx2", torch.zeros(keep, dx))
        self.register_buffer("br", torch.zeros(keep)); self.register_buffer("bd", torch.zeros(keep)); self.register_buffer("ba", torch.zeros(keep, dtype=torch.long))
        self.register_buffer("bi", torch.zeros((), dtype=torch.long)); self.register_buffer("bn", torch.zeros((), dtype=torch.long))
        self.register_buffer("bp", torch.ones(keep))                # 되감기의 몫: 놀라웠던 (값이 크게 어긋났던) 걸음을 더 자주 (해마가 놀랍고 중요한 일을 더 되감듯, Mattar & Daw 2018)
        self.prio = False
        self.trust_on = False                                     # 믿음: 값이 제 겪음을 얼마나 맞히는가 (설명된 흩어짐)만큼만 고름에 든다 - 덜 익은 값이 고름을 흔들지 않게 (습관이 차츰 굳듯)
        self.register_buffer("e2", torch.ones(())); self.register_buffer("t2", torch.ones(()))
        self.register_buffer("ascale", torch.zeros(()))            # 몸짓 사이 값 차이의 평소 크기 (제 겪음에 걸친 이동 평균): 이것으로 나눈다 (지금 장면의 흩어짐으로 나누면 값이 익어 차이가 작아질 때 소음이 커진다)
        self.scale_mode = "typical"
        # 겪어서 얻은 믿음 (기본): '내가 기대한 것 (V)'과 'K걸음 동안 실제로 온 온전함의 바뀜'의 상관을 이동 평균으로. 기댐은 그만큼만 고름에 든다.
        # (값이 제 목표 (스스로 만든 것)를 맞히는 몫은 믿음이 아니다 - 재 보니 맞힘이 0일 때도 0.9였다)
        self.trust_mode, self.K = "outcome", 60
        self.trace = None; self.register_buffer("oc", torch.zeros(5))   # 상관을 위한 이동 합: [E v, E g, E v^2, E g^2, E vg]
        self.opt, self.steps = None, 0
        self.on = True                                              # 고름에 드는가 (배우기만 할 때는 False)

    def whole(self, x):
        return 1 - ((1 - x[..., self.inside]).clamp(min=0) ** 2).mean(-1)

    @torch.no_grad()
    def see(self, x, x2, died, a=None, extra=None):
        """겪은 걸음들 x [k, dx] -> x2, died [k] (이 걸음에 죽었나), a [k] 그때 한 몸짓 (몸짓 값일 때). 보상은 온전함의 바뀜 (첫 바탕)
        + extra [k]: 둘째 바탕 (창립자 확정: 제 내다보기에 끌림 - 여기서는 배움의 진척, 몸 쪽에서 셈해 넘김)."""
        r = torch.where(died, -self.whole(x), self.whole(x2) - self.whole(x))
        if extra is not None:
            r = r + extra
        k = len(x)
        if k == 0:
            return
        idx = (torch.arange(k, device=x.device) + self.bi) % self.keep
        self.bx[idx], self.bx2[idx], self.br[idx], self.bd[idx] = x, x2, r, died.float()
        if a is not None:
            self.ba[idx] = a.long()
        self.bp[idx] = self.bp[:max(int(self.bn), 1)].max()          # 새 걸음은 한 번은 꼭 되감기게
        self.bi.copy_((self.bi + k) % self.keep); self.bn.copy_(torch.clamp(self.bn + k, max=self.keep))

    def learn(self, n=4, batch=2048, lr=1e-3):
        """겪은 것을 되감아 값을 고친다 (n번). 든 것이 적으면 하지 않는다."""
        top = int(self.bn)
        if top < 4096:
            return None
        if self.opt is None:
            self.opt = torch.optim.Adam(self.v.parameters(), lr)
        for _ in range(n):
            if self.prio:
                j = torch.multinomial(self.bp[:top] ** 0.6, batch, replacement=True)
            else:
                j = torch.randint(0, top, (batch,), device=self.bx.device)
            with torch.no_grad():
                nxt = self.vt(self.bx2[j])
                nxt = nxt.max(1).values if self.m else nxt[:, 0]          # 몸짓 값: 다음 장면에서 가장 나은 몸짓의 값으로
                tg = self.br[j] + self.gamma * nxt * (1 - self.bd[j])
            out = self.v(self.bx[j])
            err = (out.gather(1, self.ba[j][:, None])[:, 0] if self.m else out[:, 0]) - tg
            loss = (err ** 2).mean(); self.opt.zero_grad(); loss.backward(); self.opt.step()
            with torch.no_grad():
                self.e2.mul_(0.98).add_(0.02 * (err.detach() ** 2).mean()); self.t2.mul_(0.98).add_(0.02 * tg.var())
            if self.prio:
                self.bp[j] = err.detach().abs() + 1e-3
            self.steps += 1
            if self.steps % 40 == 0:
                self.vt.load_state_dict(self.v.state_dict())
        return float(loss)

    @torch.no_grad()
    def track(self, x, alive):
        """몸 모두 (n, 같은 차례)의 지금 느낌과 살아 있나를 걸음마다. K걸음 뒤 실제로 온 (할인된) 온전함의 바뀜과 그때의 V를 견준다."""
        n = len(x)
        if self.trace is None or self.trace["v"].shape[1] != n:
            self.trace = dict(v=torch.zeros(self.K, n, device=x.device), w=torch.zeros(self.K, n, device=x.device), a=torch.zeros(self.K, n, device=x.device), t=0)
        tr = self.trace; i = tr["t"] % self.K
        if tr["t"] >= self.K:                                   # i 자리에 든 K걸음 전의 것: 그 뒤 K걸음의 할인된 바뀜 (죽으면 그때까지 + 온전함이 다 사라짐)
            idx = [(i + k) % self.K for k in range(self.K)]
            a_ = tr["a"][idx]; alive_k = torch.cumprod(a_, 0); w_ = tr["w"][idx] * alive_k          # 창 안에서 죽으면 그 뒤 온전함은 0 (see()의 '죽음 = 다 사라짐'과 같게)
            w_end = torch.cat([w_[1:], (torch.where(alive, self.whole(x), torch.zeros(n, device=x.device)) * alive_k[-1])[None]])
            disc = self.gamma ** torch.arange(self.K, device=x.device, dtype=torch.float32)[:, None]
            g = (disc * (w_end - w_) * alive_k).sum(0)
            ok = a_[0] > 0
            if ok.sum() >= 1:                                    # 몸 하나여도 (시간에 걸쳐 모음)
                v = tr["v"][i][ok]; g = g[ok]
                m = torch.stack([v.mean(), g.mean(), (v * v).mean(), (g * g).mean(), (v * g).mean()])
                self.oc.copy_(m if float(self.oc[2]) == 0 else 0.99 * self.oc + 0.01 * m)
        vx = self.v(x)
        tr["v"][i] = vx.max(1).values if self.m else vx[:, 0]; tr["w"][i] = self.whole(x); tr["a"][i] = alive.float(); tr["t"] += 1

    def earned(self):
        ev, eg, ev2, eg2, evg = [float(z) for z in self.oc]
        c = (evg - ev * eg) / max(((ev2 - ev * ev) * (eg2 - eg * eg)) ** 0.5, 1e-9)
        return max(0.0, min(1.0, c))

    @torch.no_grad()
    def lean(self, core, x):
        if not self.on or int(self.bn) < 4096:
            return None
        adv = self.v(x) if self.m else self.v(core.foresee(x))[..., 0] - self.v(x)
        trust = (1 - self.e2 / self.t2.clamp(min=1e-8)).clamp(0, 1) if self.trust_on else 1.0
        if self.trust_mode == "outcome":
            trust = trust * self.earned()
        c = adv - adv.mean(1, keepdim=True)
        if self.scale_mode == "scene":
            return self.lam * trust * c / (c.std(1, keepdim=True) + 1e-4)
        s = c.std(1).mean()
        self.ascale.copy_(s if float(self.ascale) == 0 else 0.999 * self.ascale + 0.001 * s)
        return self.lam * trust * c / (self.ascale + 1e-6)

    def trust(self):
        return float((1 - self.e2 / self.t2.clamp(min=1e-8)).clamp(0, 1))
