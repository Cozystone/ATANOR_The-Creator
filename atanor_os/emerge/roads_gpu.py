"""길의 기억 (routes.Routes)을 몸 수천 개에 대해 GPU에서 한꺼번에. 규칙은 같다:
걸음마다 한 번 적고 (어디, 그다음 어디, 거기서 얼마 뒤에 어느 모자람이 풀렸나), 모자랄 때 가까이를 지나간 길 가운데 가장 빨리 풀렸던 길을 몇 걸음 앞까지 따라간다.
풀리던 자리에 와서 한참 있어도 풀리지 않으면 그리로 가는 길은 한동안 떠올리지 않는다 (얼마 동안인지는 겪은 것에서: 비었던 자리가 다시 풀리기까지 걸린 시간들의 가운데값).
신경망을 고치지 않는다. 크기 고정 (몸마다 L칸, 가득 차면 가장 오래된 것부터 덮어쓴다).
자리는 모두 제 걸음을 더해 간 것 (태어난 자리 0). 남의 길도 같은 꼴로 적을 수 있다 (watch).
"""
import math
import torch

INF = float("inf")


class RoadsGPU:
    def __init__(self, n, device, L=4000, kinds=2, stride=0.5, E=8):
        self.n, self.dev, self.L, self.K, self.stride, self.E = n, device, L, kinds, stride, E
        z = lambda *s, **k: torch.zeros(*s, device=device, **k)
        self.pos = z(n, L, 2); self.t = z(n, L); self.ser = torch.full((n, L), -1, device=device, dtype=torch.long)
        self.nxt = torch.full((n, L), -1, device=device, dtype=torch.long)
        self.ttr = torch.full((n, L, kinds), INF, device=device); self.goal = z(n, L, kinds, 2); self.won = z(n, L, kinds)
        self.count = z(n, dtype=torch.long); self.last = torch.full((n,), -1, device=device, dtype=torch.long); self.lastpos = torch.full((n, 2), 1e9, device=device)
        self.lifeser = z(n, dtype=torch.long); self.relser = torch.full((n, kinds), -1, device=device, dtype=torch.long)
        self.emp = torch.full((n, kinds, E, 2), 1e9, device=device); self.empt = torch.full((n, kinds, E), -INF, device=device); self.empk = z(n, kinds, dtype=torch.long)
        self.back = torch.full((n, 32), float("nan"), device=device); self.backk = z(n, dtype=torch.long)
        self.there = z(n, kinds); self.used = z(n)
        self.ar = torch.arange(n, device=device)

    def newlife(self):
        self.last.fill_(-1); self.lastpos.fill_(1e9); self.lifeser = self.count.clone(); self.relser.fill_(-1); self.there.zero_()

    # ---- 적기 ----
    def walk(self, pos, t, who):
        w = who & ((pos - self.lastpos).norm(dim=1) >= self.stride)
        if not w.any():
            return
        b = w.nonzero()[:, 0]; i = self.count[b] % self.L
        self.pos[b, i] = pos[b]; self.t[b, i] = t[b]; self.ser[b, i] = self.count[b]; self.nxt[b, i] = -1; self.ttr[b, i] = INF
        hasl = self.last[b] >= 0
        self.nxt[b[hasl], self.last[b[hasl]]] = i[hasl]
        self.last[b] = i; self.lastpos[b] = pos[b]; self.count[b] += 1

    def relief(self, j, who, pos, t):
        if not who.any():
            return
        b = who.nonzero()[:, 0]
        since = torch.maximum(self.lifeser[b], self.relser[b, j] + 1)[:, None]
        op = (self.ser[b] >= since) & (self.ser[b] >= 0)                          # 지난번 풀림 뒤로 이 삶에서 적은 칸들
        tt = (t[b, None] - self.t[b]).masked_fill(~op, INF)
        self.ttr[b, :, j] = torch.where(op, tt, self.ttr[b, :, j])
        self.goal[b, :, j] = torch.where(op[..., None], pos[b, None].expand(-1, self.L, 2), self.goal[b, :, j])
        self.won[b, :, j] = torch.where(op, t[b, None].expand(-1, self.L), self.won[b, :, j])
        self.relser[b, j] = self.count[b] - 1; self.there[b, j] = 0
        near = (self.emp[b, j] - pos[b, None]).norm(dim=-1) < 1.5                  # 비었다고 여겼던 자리에서 다시 풀렸다: 그 사이를 든다
        gap = (t[b, None] - self.empt[b, j]).masked_fill(~near | torch.isinf(self.empt[b, j]), float("nan"))
        g = torch.nanmean(gap, 1) if gap.numel() else gap
        okg = ~torch.isnan(g)
        if okg.any():
            bb = b[okg]; k = self.backk[bb] % 32
            self.back[bb, k] = g[okg]; self.backk[bb] += 1
        self.emp[b, j] = torch.where(near[..., None], torch.full_like(self.emp[b, j], 1e9), self.emp[b, j])
        self.empt[b, j] = torch.where(near, torch.full_like(self.empt[b, j], -INF), self.empt[b, j])

    def watch(self, who, path_pos, path_t, j, at, t):
        """남이 걸어 간 길 (제 걸음의 자리로 옮긴 것) path_pos [n, P, 2], 그 끝 at에서 모자람 j가 풀리는 것을 보았다: 그 길을 제 길처럼 적는다."""
        P = path_pos.shape[1]
        for p in range(P):
            ok = who & ~torch.isnan(path_pos[:, p, 0])
            if not ok.any():
                continue
            b = ok.nonzero()[:, 0]; i = self.count[b] % self.L
            self.pos[b, i] = path_pos[b, p]; self.t[b, i] = path_t[b, p]; self.ser[b, i] = self.count[b]
            self.nxt[b, i] = -1; self.ttr[b, i] = INF
            self.ttr[b, i, j] = t[b] - path_t[b, p]; self.goal[b, i, j] = at[b]; self.won[b, i, j] = t[b]
            prev = (self.count[b] - 1) % self.L
            chain = (p > 0) & (self.ser[b, prev] == self.count[b] - 1)
            self.nxt[b[chain], prev[chain]] = i[chain]
            self.count[b] += 1
        self.last[who] = -1; self.lastpos[who] = 1e9                               # 제 길과 잇지 않는다

    # ---- 떠올리기 ----
    def recall(self, pos, j, t, want, ahead=5, near=1.5, linger=25):
        """want [n]인 몸들에게 갈 자리 [n, 2] (없으면 nan)."""
        out = torch.full((self.n, 2), float("nan"), device=self.dev)
        if not want.any():
            return out
        b = want.nonzero()[:, 0]
        ok = torch.isfinite(self.ttr[b, :, j]) & (self.ser[b] >= 0)
        wait = torch.nanmedian(self.back[b], 1).values
        wait = torch.where((self.backk[b] >= 3) & ~torch.isnan(wait), wait, torch.full_like(wait, INF))
        for e in range(self.E):
            ep, et = self.emp[b, j, e], self.empt[b, j, e]
            active = torch.isfinite(et) & ((t[b] - et) < wait)
            blk = ((self.goal[b, :, j] - ep[:, None]).norm(dim=-1) < 1.5) & (self.won[b, :, j] <= et[:, None]) & active[:, None]
            ok &= ~blk
        d = (self.pos[b] - pos[b, None]).norm(dim=-1)
        cand = ok & (d < near)
        has = cand.any(1)
        score = (self.ttr[b, :, j] + 4.0 * d).masked_fill(~cand, INF)
        i = score.argmin(1)
        g = self.goal[b, i, j]
        at = has & ((g - pos[b]).norm(dim=1) < 1.2)
        self.there[b, j] = torch.where(at, self.there[b, j] + 1, torch.zeros_like(self.there[b, j]))
        gone = at & (self.there[b, j] > linger)                                    # 한참 있어도 풀리지 않는다
        if gone.any():
            bb = b[gone]; k = self.empk[bb, j] % self.E
            self.emp[bb, j, k] = g[gone]; self.empt[bb, j, k] = t[bb]; self.empk[bb, j] += 1; self.there[bb, j] = 0
        cur = i.clone(); ttr0 = self.ttr[b, i, j]
        for _ in range(ahead):                                                     # 그 길을 몇 걸음 앞까지 따라간다
            k = self.nxt[b, cur]
            good = (k >= 0) & has & ~at
            kk = k.clamp(min=0)
            good &= (self.ser[b, kk] == self.ser[b, cur] + 1) & torch.isfinite(self.ttr[b, kk, j]) & (self.ttr[b, kk, j] <= ttr0)
            cur = torch.where(good, kk, cur)
        wp = torch.where(at[:, None], g, self.pos[b, cur])
        wp = torch.where(((wp - pos[b]).norm(dim=1) < 0.3)[:, None], g, wp)
        res = torch.where((has & ~gone)[:, None], wp, torch.full_like(wp, float("nan")))
        out[b] = res; self.used[b] += (has & ~gone).float()
        return out
