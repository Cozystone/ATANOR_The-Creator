"""살면서 스스로를 고치는 힘 (온라인 가소성). 세계의 이름이 없다. 마음의 알맹이(core.Core) 어디에든 붙는다.

라운드가 끝나고 모아서 고치는 것이 아니라, 고를 때마다 방금 겪은 것으로 조금씩 고친다. 그러면서 앞서 배운 것을 망가뜨리지 않게 둘을 함께 든다:
  되새김   - 겪은 걸음들 가운데 크기가 고정된 한 줌을 고르게 남겨 둔다 (저수지 뽑기: 오래 산 만큼 옛 것도 고르게 남는다). 고칠 때마다 방금 것과 옛 것을 섞어 배운다.
  느린 나  - 마음의 연결을 천천히 따라오는 두 번째 사본을 든다. 고칠 때마다 빠른 연결이 느린 사본에서 너무 멀어지지 않게 당긴다 (빠르게 배우되 뼈대는 천천히 바뀐다).
            느린 사본은 빠른 것을 천천히 따라간다 (굳히기). 잠도, 라운드도 없이.
크기는 고정이다: 되새김 칸 수와 사본 하나.
"""
import copy
import torch


class Plastic:
    def __init__(self, net, lr=3e-4, recent=8, keep=4096, mix=56, pull=1.0, settle=0.999, seed=0):
        self.net, self.recent, self.keep, self.mix, self.pull, self.settle = net, recent, keep, mix, pull, settle
        self.opt = torch.optim.Adam(net.parameters(), lr=lr)
        self.slow = [p.detach().clone() for p in net.parameters()]
        self.buf, self.seen = None, 0
        self.gen = torch.Generator().manual_seed(seed)
        self.steps, self.last = 0, 0.0

    @torch.no_grad()
    def _keep(self, x, a, x2):
        """저수지 뽑기로 겪은 걸음을 남긴다 (크기 고정)."""
        n = x.shape[0]
        if self.buf is None:
            self.buf = [torch.zeros(self.keep, x.shape[1]), torch.zeros(self.keep, dtype=torch.long), torch.zeros(self.keep, x2.shape[1])]
        for i in range(n):
            self.seen += 1
            if self.seen <= self.keep:
                j = self.seen - 1
            else:
                j = int(torch.randint(0, self.seen, (1,), generator=self.gen))
                if j >= self.keep:
                    continue
            self.buf[0][j], self.buf[1][j], self.buf[2][j] = x[i], a[i], x2[i]

    def step(self, X, A, X2, OK):
        """X [B, T, dx] 방금 겪은 몇 걸음 (A, X2, OK 같은 꼴). 한 번 고친다."""
        live = OK[:, -1] > 0
        self._keep(X[live, -1], A[live, -1], X2[live, -1])
        loss = self.net.loss(X, A, X2, OK)
        filled = min(self.seen, self.keep)
        if filled > self.mix:
            i = torch.randint(0, filled, (self.mix,), generator=self.gen)
            loss = loss + self.net.loss(self.buf[0][i], self.buf[1][i], self.buf[2][i], torch.ones(self.mix))
        if self.pull:
            loss = loss + self.pull * sum(((p - s) ** 2).sum() for p, s in zip(self.net.parameters(), self.slow)) * 1e-3
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        with torch.no_grad():
            for p, s in zip(self.net.parameters(), self.slow):
                s.mul_(self.settle).add_(p.detach(), alpha=1 - self.settle)
        self.steps += 1; self.last = float(loss)
        return self.last
