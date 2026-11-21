"""몸 수천 개가 GPU에서 한꺼번에 사는 고리. 마음의 알맹이(core.Core, 얼린 채) + 헛되면 놓기. 그 밖에 손으로 넣은 행동 규칙은 없다.
세계: room3d_gpu.World (그려 둔 익숙한 세계, 샘은 정해진 자리들 사이를 옮겨 간다). 눈: 지금 보이는 것만 (지은 자리 목록도 지도도 없다).

2026-11-21 창립자 지시 ("손으로 넣은거 다 제거해")로 걷어 낸 것 (git 기록에만 남김): 길의 기억과 그 목적지 끼워 넣기, 덜 걸어 본 쪽,
곁에 붙기/따라가기, 믿을 것 고르기(톰슨), 한 바퀴 둘러보면 걸어가기, 둘러본 쪽 세기, 급한 것만 떠올리기.
남은 것: 겪어서 배운 알맹이 (내다보기, 풀림의 기억, 헛됨)와 눈 기관 (눈의 반사는 창립자가 정한 마지막 타고난 것).
"""
import torch
from . import room3d_gpu as G

LEAK, TIRE, LOOSE = 0.95, 4.0, 0.05


class Life:
    def __init__(self, net, n, rows, device="cuda", seed=0, pool="max", drain=1 / 240.0):
        self.net, self.n, self.dev, self.rows, self.seed = net, n, device, rows, seed
        self.world = G.World(n, rows, device, seed=seed, drain=drain)
        self.eye = G.Eye(n, device, pool=pool)
        self.t = torch.zeros(n, device=device); self.gen = torch.Generator(device=device).manual_seed(seed + 7)
        self.lives = 0
        self.dishab = False
        self.extra, self.record, self.try_p, self.learn = None, None, 0.0, False
        self.begin()

    @torch.no_grad()
    def begin(self):
        w = self.world
        w.reset(torch.ones(self.n, dtype=torch.bool, device=self.dev)); self.eye = G.Eye(self.n, self.dev, pool=self.eye.pool)
        self.t.zero_()
        self.fut = torch.zeros(self.n, self.net.m, device=self.dev); self.fills = torch.zeros(self.n, device=self.dev); self.prev = None
        self.s_m = torch.zeros(self.n, device=self.dev); self.s_v = torch.ones(self.n, device=self.dev) * 1e-4; self.novel = torch.zeros(self.n, dtype=torch.bool, device=self.dev)

    @torch.no_grad()
    def tick(self):
        """한 틱. 돌려줌: (고른 행동 [n], 끌림 [n, m], 끌림이 있는가 [n], 채워졌는가 [n, 2])."""
        w = self.world
        x = G.senses(w, self.eye)
        if self.extra is not None:                               # 몸에 더한 감각 (닿음, 깊이 같은 것): 느끼는 것 끝에 붙는다
            x = torch.cat([x, self.extra(w)], 1)
        where, size, seen = self.eye.last
        if self.record is not None and self.prev is not None:   # 기를 때: 겪은 걸음 (그때 느낀 것, 한 것, 다음에 느낀 것, 살아 있었나)
            self.record.append((self.prev[0], self.prev[1], x, self.prev[2]))
        if self.prev is not None:                                # 헛되면 놓는다: 내다본 것이 오지 않은 행동은 덜 끌린다
            more, eased = self.net.unmet(self.prev[0], self.prev[1], x)
            self.fut = self.fut * LEAK; self.fut[torch.arange(self.n, device=self.dev), self.prev[1]] += more; self.fut[eased] = 0
            if self.dishab:                                      # 탈습관화: 내다본 것과 제 평소보다 크게 다른 것이 오면 쌓인 헛됨이 풀린다 (새 자극이 시든 반응을 되살린다)
                fo = self.net.foresee(self.prev[0]).gather(1, self.prev[1][:, None, None].expand(-1, 1, x.shape[1]))[:, 0]
                odd = ((fo - x) ** 2).mean(1)
                novel = (odd > self.s_m + 2.0 * self.s_v.sqrt()) & (self.t > 20)
                self.fut[novel] = 0; self.novel = novel
                self.s_m = self.s_m + 0.02 * (odd - self.s_m); self.s_v = self.s_v + 0.02 * ((odd - self.s_m) ** 2 - self.s_v)
        sc, strong = self.net.pull(x); sc = sc - TIRE * self.fut
        a = torch.multinomial(torch.softmax(sc / LOOSE, 1), 1, generator=self.gen)[:, 0]
        a = torch.where(strong, a, torch.randint(0, self.net.m, (self.n,), device=self.dev, generator=self.gen))
        if self.try_p:                                           # 기를 때: 얼마쯤은 아무렇게나 해 본다 (원래 기르던 꼴)
            a = torch.where(torch.rand(self.n, device=self.dev, generator=self.gen) < self.try_p, torch.randint(0, self.net.m, (self.n,), device=self.dev, generator=self.gen), a)
        if getattr(self, "override", None) is not None:          # 이음 자리: 다른 기관(상상으로 고르기 같은)이 고른 행동으로 바꿀 수 있다
            a = self.override(a)
        a = torch.where(w.dead, torch.zeros_like(a), a)
        self.seen_now = seen
        e0 = w.store.clone(); live0 = ~w.dead
        filled = w.step(a)
        if self.learn:                                           # 기를 때: 몸 안이 오른 순간을 풀림의 기억에 든다
            rose = ((w.store - e0) > 0.05).any(1) & live0
            if rose.any():
                self.net.keep((w.store - e0)[rose].float(), x[rose], a[rose])
        self.t += 1
        self.fills += filled.any(1).float()
        self.prev = (x, a, live0.float())
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
