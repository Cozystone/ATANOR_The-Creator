"""몸 수천 개가 GPU에서 한꺼번에 사는 고리. 마음의 알맹이(core.Core, 얼린 채) + 헛되면 놓기 + (고를 수 있게) 길의 기억 + (고를 수 있게) 남을 보기.
세계: room3d_gpu.World (그려 둔 익숙한 세계, 샘은 정해진 자리들 사이를 옮겨 간다). 눈: 지금 보이는 것만 (지은 자리 목록도 지도도 없다).
길의 기억이 있는 몸은, 모자란데 그 빛깔이 안 보이면 떠올린 길의 몇 걸음 앞을 그 빛깔의 자리 감각으로 느낀다 (있으리라 짐작할 뿐이므로 가까이 와도 닿았다고 느끼지 않게 크기를 0.3 이하로).
어느 빛깔의 칸에 길을 얹을지는 겪은 것에서 안다 (그 모자람이 풀릴 때 눈앞에 가장 크게 있던 빛깔).
"""
import math, time
import torch
from . import room3d_gpu as G
from .roads_gpu import RoadsGPU

LEAK, TIRE, LOOSE = 0.95, 4.0, 0.05


class Life:
    def __init__(self, net, n, rows, device="cuda", seed=0, roads=None, pool="max", drain=1 / 240.0, watch=None, explore=None):
        """roads: [n] 길의 기억을 쓰는 몸. watch: [n] 남(같은 세계의 짝)을 보고 그 길을 적는 몸 - 짝은 i ^ 1 (두 몸이 한 세계를 나눈다)."""
        self.net, self.n, self.dev, self.rows, self.seed = net, n, device, rows, seed
        self.world = G.World(n, rows, device, seed=seed, drain=drain)
        self.eye = G.Eye(n, device, pool=pool)
        self.roads = RoadsGPU(n, device)
        self.use_roads = roads if roads is not None else torch.zeros(n, dtype=torch.bool, device=device)
        self.watch = watch
        self.explore = explore if explore is not None else torch.zeros(n, dtype=torch.bool, device=device)
        self.attach = torch.zeros(n, dtype=torch.bool, device=device)      # 곁에 붙어 있기: 급하지 않을 때 보이는 같은 종 곁으로 간다 (새끼의 애착. 손으로 넣은 타고난 버릇)
        self.follow = torch.zeros(n, dtype=torch.bool, device=device)      # 갈 데를 모를 때 눈에 보이는 같은 종(풀빛)을 따라간다 (새끼가 어미를 따르는 타고난 버릇)
        self.link = torch.zeros(n, 2, G.CH, device=device)
        self.t = torch.zeros(n, device=device); self.gen = torch.Generator(device=device).manual_seed(seed + 7)
        self.lives = 0

    def _pi_reset(self):
        self.pi = torch.zeros(self.n, 2, device=self.dev); self.ph = torch.zeros(self.n, device=self.dev)

    def _pi_step(self):
        f = self.world.felt
        self.ph = self.ph + f[:, 1]
        c, s = torch.cos(self.ph), torch.sin(self.ph)
        self.pi = self.pi + f[:, 0:1] * torch.stack([c, s], 1) + f[:, 5:6] * torch.stack([-s, c], 1)

    def _fresh(self, who, K=7, reach=3.0):
        """눈에 트여 보이는 쪽 K갈래 가운데 제 길의 기억에 가장 적게 적힌 곳 [n, 2] (제 걸음의 자리). 다 막혀 보이면 등 뒤로."""
        offs = torch.linspace(-0.5, 0.5, K, device=self.dev) * G.FOV
        _, free = self.world.cast(self.world.head[:, None] + offs[None], step=0.25, far=reach + 1)
        d = (free - 0.6).clamp(min=0, max=reach)
        ang = self.ph[:, None] + offs[None]
        pts = self.pi[:, None] + d[..., None] * torch.stack([torch.cos(ang), torch.sin(ang)], -1)          # [n, K, 2]
        b = who.nonzero()[:, 0]
        cnt = ((self.roads.pos[b][:, None] - pts[b][:, :, None]).norm(dim=-1) < 1.5) & (self.roads.ser[b][:, None] >= 0)
        score = cnt.float().sum(-1) + 1000.0 * (d[b] < 1.0) + torch.rand(len(b), K, device=self.dev, generator=self.gen)
        k = score.argmin(1)
        out = torch.full((self.n, 2), float("nan"), device=self.dev)
        best = pts[b, k]
        blocked = (d[b] < 1.0).all(1)
        behind = self.pi[b] - 2.0 * torch.stack([torch.cos(self.ph[b]), torch.sin(self.ph[b])], 1)
        out[b] = torch.where(blocked[:, None], behind, best)
        return out

    def _with_roads(self, x, where, seen, size=None):
        """모자란데 그 빛깔이 안 보이면, 떠올린 길의 몇 걸음 앞을 그 빛깔의 자리 감각으로."""
        NS = G.CH
        for k in range(2):
            want_k = self.use_roads & (self.world.store[:, k] < 0.75) & (self.link[:, k].sum(1) > 0) & ~self.world.dead
            if not want_k.any():
                continue
            c = self.link[:, k].argmax(1)
            notseen = ~seen.gather(1, c[:, None])[:, 0]
            want = want_k & notseen
            wp = self.roads.recall(self.pi, k, self.t, want)
            fol = want & torch.isnan(wp[:, 0]) & self.follow & seen[:, 1]
            if fol.any():                                        # 갈 데를 모르는데 같은 종이 보인다: 그쪽으로 (보이는 크기에서 어림한 거리의 조금 앞까지)
                ang = self.ph + torch.nan_to_num(where[:, 1]); dd = (1.0 / size[:, 1].clamp(min=0.05) - 0.8).clamp(min=0.3, max=6.0)
                tw = self.pi + dd[:, None] * torch.stack([torch.cos(ang), torch.sin(ang)], 1)
                wp = torch.where(fol[:, None], tw, wp)
            lost = want & torch.isnan(wp[:, 0]) & self.explore
            if lost.any():                                       # 떠올릴 길이 없다: 눈에 트여 보이는 쪽 가운데 덜 걸어 본 쪽으로 (방금 뒤진 데로 돌아가지 않는다)
                wp = torch.where(lost[:, None], self._fresh(lost), wp)
            ok = want & ~torch.isnan(wp[:, 0])
            if hasattr(self, "wp"):
                self.wp = torch.where(ok[:, None], wp, self.wp)
            if not ok.any():
                continue
            d = wp - self.pi; dist = d.norm(dim=1).clamp(min=0.3)
            off = torch.remainder(torch.atan2(d[:, 1], d[:, 0]) - self.ph + math.pi, 2 * math.pi) - math.pi
            szw = (1.0 / dist).clamp(max=0.3)
            for jj in range(NS):
                m = ok & (c == jj)
                if not m.any():
                    continue
                x[m, 3 + 3 * jj] = 1 - szw[m]; x[m, 4 + 3 * jj] = off[m].abs() / math.pi; x[m, 5 + 3 * jj] = 1.0
                x[m, 3 + 3 * NS + 2 * jj] = szw[m] / 2; x[m, 4 + 3 * NS + 2 * jj] = (off[m] / (G.FOV / 2)).clamp(-1, 1)
                x[m, 3 + 5 * NS + jj] = torch.sin(off[m])
        return x

    @torch.no_grad()
    def begin(self):
        w = self.world
        w.reset(torch.ones(self.n, dtype=torch.bool, device=self.dev)); self.eye = G.Eye(self.n, self.dev, pool=self.eye.pool)
        self.roads.newlife(); self._pi_reset()
        self.fut = torch.zeros(self.n, self.net.m, device=self.dev); self.fills = torch.zeros(self.n, device=self.dev); self.prev = None

    @torch.no_grad()
    def tick(self):
        """한 틱. 돌려줌: (고른 행동 [n], 끌림 [n, m], 끌림이 있는가 [n], 길로 짐작한 자리 [n, 2] 또는 nan)."""
        w = self.world
        x = G.senses(w, self.eye)
        where, size, seen = self.eye.last
        self.wp = torch.full((self.n, 2), float("nan"), device=self.dev)
        x = self._with_roads(x, where, seen, size)
        if self.prev is not None:
            more, eased = self.net.unmet(self.prev[0], self.prev[1], x)
            self.fut = self.fut * LEAK; self.fut[torch.arange(self.n, device=self.dev), self.prev[1]] += more; self.fut[eased] = 0
        sc, strong = self.net.pull(x); sc = sc - TIRE * self.fut
        a = torch.multinomial(torch.softmax(sc / LOOSE, 1), 1, generator=self.gen)[:, 0]
        a = torch.where(strong, a, torch.randint(0, self.net.m, (self.n,), device=self.dev, generator=self.gen))
        if self.attach.any():
            calm = self.attach & seen[:, 1] & (w.store.min(1).values > 0.4) & ~w.dead
            if calm.any():
                b = torch.nan_to_num(where[:, 1]); far = torch.nan_to_num(size[:, 1]) < 0.45
                go = torch.where(b.abs() > 0.3, torch.where(b > 0, 3, 2), torch.where(far, 1, 0))
                a = torch.where(calm, go, a)
        a = torch.where(w.dead, torch.zeros_like(a), a)
        if getattr(self, "override", None) is not None:
            a = self.override(a)
        self.seen_now = seen
        filled = w.step(a)
        if getattr(self, "after", None) is not None:
            self.after(filled)
        self._pi_step(); self.t += 1
        alive = ~w.dead
        self.roads.walk(self.pi, self.t, alive & self.use_roads)
        for j in range(2):
            fj = filled[:, j]
            if fj.any():
                self.roads.relief(j, fj & self.use_roads, self.pi, self.t)
                sz = torch.where(seen, size, torch.full_like(size, -1.0))
                big = sz.argmax(1); has = seen.any(1) & fj
                self.link[has, j, big[has]] += 1
        self.fills += filled.any(1).float()
        self.prev = (x, a)
        return a, sc, strong, filled

    @torch.no_grad()
    def live(self, T, log=None):
        """한 삶 (T틱). 돌려줌: 끝까지 온전 [n], 채움 [n]."""
        self.begin()
        for tick in range(T):
            self.tick()
            if log and tick % 500 == 0:
                log(f"  tick {tick}: alive {float((~self.world.dead).float().mean()):.2f}")
            if self.world.dead.all():
                break
        self.lives += 1
        return (~self.world.dead).float(), self.fills
