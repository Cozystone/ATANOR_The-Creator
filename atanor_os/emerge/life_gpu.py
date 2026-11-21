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
        self.begin()

    @torch.no_grad()
    def begin(self):
        w = self.world
        w.reset(torch.ones(self.n, dtype=torch.bool, device=self.dev)); self.eye = G.Eye(self.n, self.dev, pool=self.eye.pool)
        self.t.zero_()
        self.fut = torch.zeros(self.n, self.net.m, device=self.dev); self.fills = torch.zeros(self.n, device=self.dev); self.prev = None

    @torch.no_grad()
    def tick(self):
        """한 틱. 돌려줌: (고른 행동 [n], 끌림 [n, m], 끌림이 있는가 [n], 채워졌는가 [n, 2])."""
        w = self.world
        x = G.senses(w, self.eye)
        where, size, seen = self.eye.last
        if self.prev is not None:                                # 헛되면 놓는다: 내다본 것이 오지 않은 행동은 덜 끌린다
            more, eased = self.net.unmet(self.prev[0], self.prev[1], x)
            self.fut = self.fut * LEAK; self.fut[torch.arange(self.n, device=self.dev), self.prev[1]] += more; self.fut[eased] = 0
        sc, strong = self.net.pull(x); sc = sc - TIRE * self.fut
        a = torch.multinomial(torch.softmax(sc / LOOSE, 1), 1, generator=self.gen)[:, 0]
        a = torch.where(strong, a, torch.randint(0, self.net.m, (self.n,), device=self.dev, generator=self.gen))
        a = torch.where(w.dead, torch.zeros_like(a), a)
        self.seen_now = seen
        filled = w.step(a)
        self.t += 1
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
