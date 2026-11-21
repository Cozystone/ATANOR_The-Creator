"""속마음 (로드맵 3번 1단계). 세계의 이름이 없다. 마음의 알맹이(core.Core)와 함께 쓴다.

순간과 순간 사이에 이어지는 속 상태 h 하나. 지금 느낀 것과 한 것을 받아 h를 바꾸고, h에서 다음에 느낄 것을 내다본다.
눈을 감은 채로도 (느낀 것 없이 한 것만으로) 몇 걸음 앞을 내다볼 수 있게 배운다 - 그래야 h가 안 보이게 된 것을 들고 있을 이유가 생긴다.
쓰는 길: 행동 묶음 (같은 행동을 몇 틱 이어 하기) 마다 h에서 눈을 감고 몇 걸음 앞을 내다보고, 그 가운데 풀렸던 순간에 가장 가까워지는 묶음에 끌린다.
지은 지도도, 적어 둔 자리 목록도, 가장 짧은 길 셈도 없다.
"""
import torch, torch.nn as nn, torch.nn.functional as F


class Inner(nn.Module):
    def __init__(self, dx, m, h=128, mo=0):
        super().__init__()
        self.dx, self.m, self.hd, self.mo = dx, m, h, mo                    # mo: 몸이 느낀 제 움직임의 수 (앞으로, 옆으로, 돈 각). 눈을 감아도 몸은 제 걸음을 느낀다
        self.g = nn.GRUCell(dx + m + 1 + mo, h)
        self.o = nn.Sequential(nn.Linear(h, 128), nn.GELU(), nn.Linear(128, dx))

    def zero(self, n):
        return torch.zeros(n, self.hd)

    def step(self, h, x, a, shut=None, motion=None):
        """x [B, dx] 지금 느낀 것 (shut이면 0으로 받는다), a [B] 한 것. 새 h와, 그 h가 내다본 다음 느낌."""
        shut = torch.zeros(x.shape[0]) if shut is None else shut
        parts = [x * (1 - shut[:, None]), F.one_hot(a, self.m).float(), shut[:, None]]
        if self.mo:
            parts.append(torch.zeros(x.shape[0], self.mo) if motion is None else motion)      # 속으로 내다볼 때는 아직 겪지 않은 움직임이라 0
        h = self.g(torch.cat(parts, -1), h)
        return h, self.o(h)

    def run(self, X, A, shut, MO=None, dream=None):
        """X [B, T, dx], A [B, T], shut [B, T] -> 내다본 것 [B, T, dx] (t번째가 X[:, t+1]을 내다본다)."""
        h = self.zero(X.shape[0]); out = []
        for t in range(A.shape[1]):
            mo = None if MO is None else MO[:, t] * (1 - (dream[:, t:t + 1] if dream is not None else 0))
            h, y = self.step(h, X[:, t], A[:, t], shut[:, t], mo); out.append(y)
        return torch.stack(out, 1)

    @torch.no_grad()
    def chunks(self, h, x, k=6):
        """행동마다: 지금 느낀 것을 받고 그 행동을 k틱 이어 할 때, 눈을 감은 채 내다본 느낌들 [B, m, k, dx]."""
        B = x.shape[0]
        hh = h[:, None].expand(B, self.m, self.hd).reshape(B * self.m, self.hd)
        xx = x[:, None].expand(B, self.m, self.dx).reshape(B * self.m, self.dx)
        aa = torch.arange(self.m).repeat(B)
        ys = []
        hh, y = self.step(hh, xx, aa); ys.append(y)
        one = torch.ones(B * self.m)
        for _ in range(k - 1):
            hh, y = self.step(hh, torch.zeros_like(xx), aa, one); ys.append(y)
        return torch.stack(ys, 1).view(B, self.m, k, self.dx)



@torch.no_grad()
def chunk_pull(inner, h, x, mu, wt, scale, k=6, keep=()):
    """행동 묶음마다, 눈을 감고 내다본 k틱 동안 떠올린 순간(mu, 무게 wt)에 가장 가까이 가는 정도. 가장 나은 묶음보다 얼마나 못한가만 돌려준다 [B, m] (0 이하).
    keep: 이 행동들은 묶음으로 따지지 않는다 (예: '하기' - 이어 해도 자리 감각이 바뀌지 않으니 묶음으로는 늘 나빠 보인다. 한 걸음 끌림에 맡긴다)."""
    fo = inner.chunks(h, x, k)
    d = (wt[:, None, None] * (fo - mu[:, None, None]) ** 2).sum(-1).min(-1).values
    d = d - d.min(1, keepdim=True).values
    for j in keep:
        d[:, j] = 0
    return -scale[:, None] * d


def train_inner(inner, X, A, iters=4000, shut_p=0.35, window=64, batch=32, lr=1e-3, seed=0, log=None, shut_len=(4, 17), place=None, rare=1.0, MO=None):
    """X [N, T+1, dx], A [N, T]. 눈을 가린 구간(4~16틱)을 섞어, 다음 느낌을 내다보게 배운다."""
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(inner.parameters(), lr=lr)
    for it in range(iters):
        r = torch.randint(0, X.shape[0], (batch,), generator=gen); s = int(torch.randint(0, X.shape[1] - 1 - window, (1,), generator=gen))
        x, a, y = X[r, s:s + window], A[r, s:s + window], X[r, s + 1:s + window + 1]
        mo = None if MO is None else MO[r, s:s + window]
        shut = torch.zeros(batch, window)
        n = int(batch * window * shut_p / (0.5 * (shut_len[0] + shut_len[1])))
        st = torch.randint(0, window, (n,), generator=gen); rr = torch.randint(0, batch, (n,), generator=gen); L = torch.randint(shut_len[0], shut_len[1], (n,), generator=gen)      # 길게 눈을 감으면, 그동안 몸이 한 것만으로 여러 걸음 앞을 내다봐야 한다 (숨은 것을 들고 있어야 맞힌다)
        for i in range(n):
            shut[rr[i], st[i]:st[i] + L[i]] = 1
        dream = None
        if mo is not None:                                    # 꿈처럼 속으로만 내다보는 구간: 눈도 감고 몸의 느낌도 없이 (속으로 걸어 볼 때와 같은 꼴을 배운다)
            dream = torch.zeros(batch, window)
            for i in range(max(1, n // 2)):
                b_ = int(torch.randint(0, batch, (1,), generator=gen)); s_ = int(torch.randint(0, window, (1,), generator=gen))
                dream[b_, s_:s_ + 8] = 1
            shut = torch.maximum(shut, dream)
        e = (inner.run(x, a, shut, mo, dream) - y) ** 2
        if place is not None and rare != 1.0:                 # 무언가가 눈에 들어와 있는 순간의 자리 감각은 드물다: 무겁게 배운다 (몸 안이 차는 일을 무겁게 배운 것과 같은 까닭)
            sl = e[..., place]; seen = (y[..., place].abs() > 1e-6).float()
            e = torch.cat([e[..., :place.start], sl * (1 + (rare - 1) * seen), e[..., place.stop:]], -1)
        loss = e.mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if log and it % 1000 == 999:
            log(f"inner {it + 1}: {loss.item():.4f}")
    return inner
