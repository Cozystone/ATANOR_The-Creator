"""남을 보고 배우기 (로드맵 4를 앞당김). 두 몸이 한 세계를 나눈다 (짝: i 와 i ^ 1). 짝수 몸은 존재, 홀수 몸은 어른.
어른: 손으로 쓴 몸이다 (ATANOR가 아니다, 재는 데 쓰는 본보기). 세계의 참 지도를 알고 모자란 것의 샘으로 가장 짧은 길로 가서 먹는다.
존재의 눈에 어른은 풀빛 기둥으로 보인다 (같은 종. 빛깔의 맞섬 가운데 풀빛 칸에 잡힌다).
본 것을 적기 (watch): 어른이 먹는 순간 존재의 눈에 어른이 보이고 있었다면, 어른이 걸어온 길 (최근 40틱)을 제 걸음의 자리로 옮겨 제 길의 기억에 적는다 -
"저 자리들을 지나 저기에 가면 그 모자람이 풀린다". 무엇을 배울지는 정해 주지 않는다: 존재가 그 길을 떠올려 쓸지는 제 모자람과 제 자리가 정한다.
"""
import math
from collections import deque
import torch
from . import room3d_gpu as G
from .life_gpu import Life


def next_hops(base):
    """벽은 그대로인 세계에서, 칸마다 각 목표 칸으로 가는 가장 짧은 길의 다음 칸 (어른만 쓴다)."""
    S = base.shape[0]; free = (base == G.FLOOR).cpu()
    INF = 10 ** 6
    nh = torch.full((S * S, S * S), -1, dtype=torch.long); dist = torch.full((S * S, S * S), INF, dtype=torch.long)
    for g in range(S * S):
        gx, gy = divmod(g, S)
        dq = deque([(gx, gy)]); dist[g, g] = 0
        while dq:
            x, y = dq.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                u, v = x + dx, y + dy
                if 0 <= u < S and 0 <= v < S and free[u, v] and dist[g, u * S + v] == INF:
                    dist[g, u * S + v] = dist[g, x * S + y] + 1; nh[g, u * S + v] = x * S + y; dq.append((u, v))
    return nh, dist


class PairWorld(G.World):
    def reset(self, which):
        super().reset(which)
        ev = torch.arange(0, self.n, 2, device=self.dev)
        self.map[ev + 1] = self.map[ev]; self.src[ev + 1] = self.src[ev]                 # 짝은 같은 세계

    def _move(self, hit, j, kind):
        super()._move(hit, j, kind)
        for i in hit.nonzero()[:, 0].tolist():                                           # 옮겨 간 샘은 짝의 세계에서도 옮겨 간다
            self.map[i ^ 1] = self.map[i]; self.src[i ^ 1] = self.src[i]

    def cast(self, ang, step=0.1, far=None, chunk=64):
        kind, dist = super().cast(ang, step, far, chunk)
        ar = torch.arange(self.n, device=self.dev)
        mate = self.pos[ar ^ 1]
        hide = getattr(self, "hide", None)
        if hide is not None:
            mate = torch.where(hide[ar ^ 1][:, None], torch.full_like(mate, -99.0), mate)
        d = (mate - self.pos)[:, None]                                                   # [n, 1, 2]
        dirs = torch.stack([torch.cos(ang), torch.sin(ang)], -1)                         # [n, K, 2]
        t = (d * dirs).sum(-1); perp = (d[..., 0] * dirs[..., 1] - d[..., 1] * dirs[..., 0]).abs()
        hit = (t > 0.2) & (perp < 0.3) & (t < dist)
        kind = torch.where(hit, torch.full_like(kind, G.MATE), kind); dist = torch.where(hit, t, dist)
        return kind, dist


class Social(Life):
    def __init__(self, net, n, rows, device="cuda", seed=0, watch_mask=None, **kw):
        super().__init__(net, n, rows, device, seed, **kw)
        self.world = PairWorld(n, rows, device, seed=seed, drain=kw.get("drain", 1 / 240.0))
        self.world.hide = torch.zeros(n, dtype=torch.bool, device=device)
        self.elder = (torch.arange(n, device=device) % 2) == 1
        self.watching = watch_mask if watch_mask is not None else torch.zeros(n, dtype=torch.bool, device=device)
        self.nh, self.dist = [x.to(device) for x in next_hops(self.world.base)]
        self.path = torch.full((n, 40, 2), float("nan"), device=device); self.path_t = torch.full((n, 40), float("nan"), device=device); self.pk = 0
        self.watched = torch.zeros(n, device=device)
        self.override = self._elder_acts; self.after = self._after
        self.alone = False

    def begin(self):
        super().begin()
        self.path.fill_(float("nan")); self.path_t.fill_(float("nan"))
        self.world.hide.fill_(False)
        if self.alone:                                                                   # 혼자 사는 시험: 어른은 없다
            self.world.hide |= self.elder; self.world.dead |= self.elder

    def _elder_acts(self, a):
        w = self.world; S = w.S
        cx = w.pos[:, 0].long().clamp(0, S - 1); cy = w.pos[:, 1].long().clamp(0, S - 1); c = cx * S + cy
        k = w.store.argmin(1)
        tgt = w.src[torch.arange(self.n, device=self.dev), k]; g = tgt[:, 0] * S + tgt[:, 1]
        dd = self.dist[g, c]; nxt = self.nh[g, c].clamp(min=0)
        aim = torch.where((dd <= 1)[:, None], tgt.float() + 0.5, torch.stack([nxt // S, nxt % S], 1).float() + 0.5)
        want = torch.atan2(aim[:, 1] - w.pos[:, 1], aim[:, 0] - w.pos[:, 0])
        diff = torch.remainder(want - w.head + math.pi, 2 * math.pi) - math.pi
        act = torch.where(diff.abs() > 0.3, torch.where(diff > 0, 3, 2), torch.where(dd <= 1, 4, 1))
        act = torch.where(w.dead, torch.zeros_like(act), act)
        return torch.where(self.elder, act, a)

    def _after(self, filled):
        w = self.world
        k = self.pk % 40; self.path[:, k] = w.pos; self.path_t[:, k] = self.t; self.pk += 1
        if self.alone or not (filled & self.elder[:, None]).any():
            return
        learner = torch.arange(self.n, device=self.dev) ^ 1
        saw = self.seen_now[:, 1]                                                        # 풀빛 칸 (다른 몸)이 눈에 들어와 있었나
        start = w.start
        order = (torch.arange(40, device=self.dev) + self.pk) % 40
        for j in range(2):
            ate = filled[:, j] & self.elder & ~w.dead[learner]
            if not ate.any():
                continue
            e = ate.nonzero()[:, 0]; l = learner[e]
            ok = saw[l] & self.watching[l]
            if not ok.any():
                continue
            e, l = e[ok], l[ok]
            who = torch.zeros(self.n, dtype=torch.bool, device=self.dev); who[l] = True
            pp = torch.full((self.n, 40, 2), float("nan"), device=self.dev); pt = torch.full((self.n, 40), float("nan"), device=self.dev)
            pp[l] = self.path[e][:, order] - start; pt[l] = self.path_t[e][:, order]
            at = torch.zeros(self.n, 2, device=self.dev); at[l] = w.pos[e] - start
            self.roads.watch(who, pp, pt, j, at, self.t)
            self.watched[l] += 1
