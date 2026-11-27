"""1인칭 3D 세계를 GPU에서 한꺼번에 (몸 수천 개). room3d.Room + room3d_live.pictures + Retina(지금 보이는 것만)를 텐서로 옮긴 것.
세계의 규칙은 같다: 시계로 흐르고 (고를 때마다 TICK초), 저장고가 줄고, 벽에 축마다 막히고, '하기'는 정면 REACH 안의 샘을 채우고, 샘은 정해진 자리들 사이를 옮겨 간다.
눈: 64x40 그림을 그리고, 초점(8칸)은 원래 곱기로 / 고리 셋(16, 32, 64칸)은 8x8로 (고르게 뭉개거나 가장 센 빛깔). 빛깔의 맞섬 셋, 쪽은 좌우로 다 들어오는 가장 고운 겹에서,
크기는 위아래가 다 들어오는 가장 고운 겹에서. 눈의 반사 (보이는 빛깔 가운데 가장 크게 보이는 것으로 초점이 간다).
지은 기억(자리 목록, 지도, 길)은 여기 없다 - 그것들은 몸마다의 작은 기관이라 따로 붙인다.
"""
import math
import torch
import torch.nn.functional as F

FLOOR, WALL, WATER, FOOD = 0, 1, 2, 3
import os as _os
WORLD_FIX = _os.environ.get("ATANOR_WORLD_FIX", "1") == "1"   # used by room_body.RoomBody.step (the room's movement); set default 1 once gate 1's pair is rerun
COLOR = torch.tensor([[0.0, 0.0, 0.0], [0.62, 0.60, 0.56], [0.20, 0.45, 0.95], [0.90, 0.35, 0.25], [0.25, 0.75, 0.30], [0.07, 0.05, 0.05]])      # 4: 다른 몸 (같은 종), 5: 구덩이
MATE = 4
PIT = 5                                                                # D37 (atanor_os/FEAR_PROTOCOL.md): a drop - seen as a dark low rim, walked into, it kills
SKY, GROUND = torch.tensor([0.82, 0.84, 0.86]), torch.tensor([0.38, 0.40, 0.34])
FOV = math.radians(80)
SPEED, TURN, REACH, TICK = 2.0, math.radians(120), 1.2, 0.2
PW, PH, CH = 64, 40, 3
MOTOR = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])      # 앞으로, 돌기, 하기


def parse(rows):
    S = len(rows)
    m = torch.zeros(S, S, dtype=torch.int8); spots = {WATER: [], FOOD: []}; start = (1, 1)
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch == "#":
                m[x, y] = WALL
            elif ch == "p":
                m[x, y] = PIT
            elif ch in "wf":
                spots[WATER if ch == "w" else FOOD].append((x, y))
            elif ch == "s":
                start = (x, y)
    return m, spots, start


class World:
    def __init__(self, n, rows, device="cuda", seed=0, drain=1 / 240.0, regrow=None):
        self.n, self.dev, self.drain = n, device, drain
        # 샘의 고갈 (2026-11-23, 세계의 성질 - 행동 규칙 아님): regrow가 있으면 샘마다 남은 양 (0-1)이 있어, 마신 만큼 줄고 초당 regrow만큼 다시 찬다.
        # 없으면 예전 그대로 (쓰면 늘 가득).
        self.regrow = regrow
        base, spots, start = parse(rows)
        self.S = base.shape[0]
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.base = base.to(device)
        self.spots = {k: torch.tensor(v, device=device) for k, v in spots.items()}
        self.start = torch.tensor(start, device=device, dtype=torch.float32) + 0.5
        self.reset(torch.ones(n, dtype=torch.bool, device=device))

    def reset(self, which):
        n = int(which.sum())
        if not hasattr(self, "map"):
            self.map = self.base[None].repeat(self.n, 1, 1)
            self.src = torch.zeros(self.n, 2, 2, dtype=torch.long, device=self.dev)       # [몸, 마실 것 / 먹을 것, x y]
            self.pos = torch.zeros(self.n, 2, device=self.dev); self.head = torch.zeros(self.n, device=self.dev)
            self.store = torch.zeros(self.n, 2, device=self.dev); self.dead = torch.zeros(self.n, dtype=torch.bool, device=self.dev)
            self.clock = torch.zeros(self.n, device=self.dev); self.felt = torch.zeros(self.n, 6, device=self.dev)
            self.stock = torch.ones(self.n, 2, device=self.dev)
        idx = which.nonzero()[:, 0]
        self.map[idx] = self.base
        for k, kind in enumerate((WATER, FOOD)):
            sp = self.spots[kind]; c = sp[torch.randint(0, len(sp), (n,), device=self.dev, generator=self.gen)]
            self.src[idx, k] = c; self.map[idx, c[:, 0], c[:, 1]] = kind
        self.pos[idx] = self.start; self.head[idx] = 0.0
        self.store[idx] = 0.5 + 0.5 * torch.rand(n, 2, device=self.dev, generator=self.gen)
        self.dead[idx] = False; self.clock[idx] = 0; self.felt[idx] = 0; self.stock[idx] = 1.0

    def cell(self, p):
        x = p[:, 0].long().clamp(0, self.S - 1); y = p[:, 1].long().clamp(0, self.S - 1)
        return self.map[torch.arange(self.n, device=self.dev), x, y]

    def step(self, act):
        """act [n] (0 가만히, 1 앞으로, 2 왼쪽, 3 오른쪽, 4 하기). 세계의 시계가 TICK초 흐른다. 돌려줌: 채워졌는가 [n, 2]."""
        mo = MOTOR.to(self.dev)[act]; fwd, turn, do = mo[:, 0], mo[:, 1], mo[:, 2] > 0
        alive = ~self.dead
        was = self.pos.clone(); bumped = torch.zeros(self.n, dtype=torch.bool, device=self.dev)
        h, t = 1 / 60.0, 0.0
        while t < TICK - 1e-9:
            d = min(h, TICK - t); t += d
            self.head = torch.where(alive, (self.head + turn * TURN * d) % (2 * math.pi), self.head)
            stp = fwd * SPEED * d * alive
            nx = self.pos + stp[:, None] * torch.stack([torch.cos(self.head), torch.sin(self.head)], 1)
            for ax in (0, 1):
                trial = self.pos.clone(); trial[:, ax] = nx[:, ax]
                ok = self.cell(trial) == FLOOR
                if WORLD_FIX:                                           # world fix (WORLD_A_PROTOCOL.md, 2026-11-27): a body standing in a cell that became a
                    here = self.pos.long(); there = trial.long()         # block (food moved onto it) may move within it and step out to floor
                    ok = ok | ((here == there).all(1) & (self.cell(self.pos) != FLOOR))
                self.pos = torch.where(ok[:, None], trial, self.pos)
                bumped |= (~ok) & (stp.abs() > 0)
        self.store = self.store - self.drain * TICK * alive[:, None]
        filled = torch.zeros(self.n, 2, dtype=torch.bool, device=self.dev)
        if do.any():
            k, dist = self.cast(self.head[:, None], step=0.02, far=3.0)
            k, dist = k[:, 0], dist[:, 0]
            for j, kind in enumerate((WATER, FOOD)):
                hit = do & alive & (k == kind) & (dist <= REACH)
                if hit.any():
                    if self.regrow is None:
                        filled[:, j] = hit
                        self.store[hit, j] = 1.0
                    else:
                        give = torch.minimum(self.stock[:, j], 1.0 - self.store[:, j]) * hit
                        self.store[:, j] += give; self.stock[:, j] -= give
                        filled[:, j] = hit & (give > 0.01)
                    self._move(hit, j, kind)
        if self.regrow is not None:
            self.stock = (self.stock + self.regrow * TICK).clamp(max=1.0)
        self.clock += TICK * alive
        self.dead |= (self.store <= 0).any(1)
        self.store.clamp_(min=0)
        dd = self.pos - was; c, s = torch.cos(self.head), torch.sin(self.head)
        self.felt = torch.stack([dd[:, 0] * c + dd[:, 1] * s, turn * TURN * TICK * alive, bumped.float(), filled.any(1).float(), torch.full_like(c, TICK), -dd[:, 0] * s + dd[:, 1] * c], 1)
        return filled

    def _move(self, hit, j, kind):
        """채워 준 샘은 정해진 자리들 가운데 다른 곳으로 옮겨 간다 (몸 바로 곁은 아닌 곳)."""
        idx = hit.nonzero()[:, 0]; sp = self.spots[kind]
        for i in idx.tolist():
            here = self.src[i, j]
            cand = [c for c in sp.tolist() if (c[0], c[1]) != (int(here[0]), int(here[1])) and int(self.map[i, c[0], c[1]]) == FLOOR
                    and max(abs(c[0] + 0.5 - float(self.pos[i, 0])), abs(c[1] + 0.5 - float(self.pos[i, 1]))) > 1.0]
            if cand:
                c = cand[int(torch.randint(0, len(cand), (1,), device=self.dev, generator=self.gen))]
                self.map[i, int(here[0]), int(here[1])] = FLOOR; self.map[i, c[0], c[1]] = kind; self.src[i, j] = torch.tensor(c, device=self.dev)

    # ---- 눈 ----
    def cast(self, ang, step=0.1, far=None, chunk=64, through=()):
        """ang [n, K] -> (종류 [n, K], 거리 [n, K]). 시선의 걸음들을 한 번에 (덩이씩) 내다본다 - 파이썬 고리 대신 GPU가 한꺼번에."""
        n, K = ang.shape; far = far or 1.1 * self.S
        dx, dy = torch.cos(ang)[..., None], torch.sin(ang)[..., None]
        kind = torch.zeros(n, K, dtype=torch.int8, device=self.dev); dist = torch.full((n, K), 99.0, device=self.dev)
        live = torch.ones(n, K, dtype=torch.bool, device=self.dev)
        flat = self.map.view(n, -1)
        rs = torch.arange(step, far, step, device=self.dev)
        for c0 in range(0, len(rs), chunk):
            r = rs[c0:c0 + chunk][None, None]                                     # [1, 1, C]
            cx = (self.pos[:, 0, None, None] + dx * r).long().clamp(0, self.S - 1); cy = (self.pos[:, 1, None, None] + dy * r).long().clamp(0, self.S - 1)
            m = torch.gather(flat, 1, (cx * self.S + cy).view(n, -1)).view(n, K, -1)
            hit = m != FLOOR
            for k_ in through:                                                    # render fix: see past low things to the wall behind
                hit = hit & (m != k_)
            any_ = hit.any(2)
            first = hit.float().argmax(2)
            new = live & any_
            kind = torch.where(new, torch.gather(m, 2, first[..., None])[..., 0], kind)
            dist = torch.where(new, r[0, 0][first], dist)
            live &= ~any_
            if not live.any():
                break
        return kind, dist

    def picture(self, PW=PW, PH=PH, step=0.1):
        """눈앞 그림 [n, PH, PW, 3]."""
        offs = (torch.arange(PW, device=self.dev) / (PW - 1) - 0.5) * FOV
        kind, dist = self.cast(self.head[:, None] + offs[None], step=step)
        dist = dist * torch.cos(offs)[None]
        half = (0.5 * PH / dist.clamp(min=0.05)).clamp(0, PH)
        rows = torch.arange(PH, device=self.dev)[None, :, None]
        img = torch.where((rows < PH / 2)[..., None], SKY.to(self.dev), GROUND.to(self.dev)).expand(self.n, PH, PW, 3)
        wall = (rows - PH / 2 + 0.5).abs() <= half[:, None, :]
        shade = (1.0 - dist / 12.0).clamp(0.25, 1.0)
        col = COLOR.to(self.dev)[kind.long()] * shade[..., None]
        return torch.where(wall[..., None], col[:, None], img)


def opponent(px):
    """[..., 3] -> [..., 3] 한 빛이 나머지 둘의 평균보다 센 만큼."""
    return torch.stack([(px[..., k] - 0.5 * (px[..., (k + 1) % 3] + px[..., (k + 2) % 3])).clamp(min=0) for k in range(3)], -1)


class Eye:
    """지금 보이는 것만. 초점은 눈의 반사로 간다."""
    def __init__(self, n, device, pool="max", thr=0.06):
        self.n, self.dev, self.pool, self.thr = n, device, pool, thr
        self.gx = torch.full((n,), 0.5, device=device); self.lock = torch.full((n,), -1, device=device, dtype=torch.long)
        self.dwell = torch.zeros(n, device=device)

    def take(self, img):
        """img [n, PH, PW, 3] -> 맞섬마다 (쪽 [n,3], 크기 [n,3], 보이는가 [n,3]), 초점의 빛깔 [n,3]."""
        n = self.n
        cx = (self.gx * PW).round().long()
        where = torch.full((n, CH), float("nan"), device=self.dev); size = torch.full((n, CH), float("nan"), device=self.dev)
        lastw = torch.full((n, CH), float("nan"), device=self.dev); last = torch.zeros(n, CH, device=self.dev)
        fov_col = None
        opp_full = opponent(img)                                                        # [n, PH, PW, 3]
        for li, ext in enumerate((8, 16, 32, 64)):
            w_ = min(ext, PW); h_ = min(ext, PH)
            x0 = (cx - w_ // 2).clamp(0, PW - w_); y0 = max(0, (PH - h_) // 2)
            cols = x0[:, None] + torch.arange(w_, device=self.dev)[None]                # [n, w_]
            patch = torch.gather(opp_full[:, y0:y0 + h_], 2, cols[:, None, :, None].expand(n, h_, w_, CH))      # [n, h_, w_, 3]
            if li == 0:
                g = patch.permute(0, 3, 1, 2)                                           # 초점은 원래 곱기
                fov_col = g.mean((2, 3))
            else:
                p = patch.permute(0, 3, 1, 2)
                g = F.adaptive_max_pool2d(p, (8, 8)) if self.pool == "max" else F.adaptive_avg_pool2d(p, (8, 8))
            r_, c_ = g.shape[2], g.shape[3]
            cellh, cellw = h_ / r_, w_ / c_
            mask = g > self.thr                                                          # [n, 3, r, c]
            tall = mask.sum(2).float() * cellh / PH                                      # [n, 3, c]
            if self.pool == "max" and li > 0:
                tall = torch.where(tall > 0, (tall - 0.5 * cellh / PH).clamp(min=0.5 * cellh / PH), tall)
            present = tall.max(2).values > 0
            colsum = (tall > 0).float(); cnt = colsum.sum(2).clamp(min=1)
            xc = (colsum * torch.arange(c_, device=self.dev)[None, None]).sum(2) / cnt
            ang = ((x0[:, None].float() + (xc + 0.5) * cellw) / PW - 0.5) * FOV
            side_cut = (w_ < PW) & ((tall[..., 0] > 0) | (tall[..., -1] > 0))
            cut = (h_ < PH) & (mask[:, :, 0].any(2) | mask[:, :, -1].any(2))
            ok_w = present & ~side_cut & torch.isnan(where)
            where = torch.where(ok_w, ang, where)
            lastw = torch.where(present & torch.isnan(where) & torch.isnan(lastw), ang, lastw)
            ok_s = present & ~cut & torch.isnan(size)
            size = torch.where(ok_s, tall.max(2).values, size)
            last = torch.where(present, tall.max(2).values, last)
        where = torch.where(torch.isnan(where), lastw, where)
        seen = ~torch.isnan(where)
        size = torch.where(torch.isnan(size), last, size)
        # 눈의 반사: 보이는 것 가운데 가장 크게 보이는 빛깔로 (붙든 것은 잠시 계속 본다)
        sz = torch.where(seen, size, torch.full_like(size, -1.0))
        best = sz.argmax(1); any_ = seen.any(1)
        keep = any_ & (self.lock >= 0) & seen.gather(1, self.lock.clamp(min=0)[:, None])[:, 0] & (self.dwell < 10)
        self.lock = torch.where(keep, self.lock, torch.where(any_, best, torch.full_like(best, -1)))
        self.dwell = torch.where(keep, self.dwell + 1, torch.zeros_like(self.dwell))
        tgt = where.gather(1, self.lock.clamp(min=0)[:, None])[:, 0]
        self.gx = torch.where(self.lock >= 0, (tgt / FOV + 0.5).clamp(0, 1), torch.full_like(self.gx, 0.5))
        self.last = (where, size, seen)
        return where, size, seen, fov_col


def senses(world, eye, NS=3):
    """지금 보이는 것만으로 된 느낌 [n, DX] (room3d_live.Retina.sense에서 기억에서 온 것을 뺀 것과 같은 꼴). DX = 3 + 6*NS + 1 + 3."""
    where, size, seen, fov = eye.take(world.picture())
    n = world.n
    out = torch.zeros(n, 3 + 6 * NS + 1 + CH, device=world.dev)
    out[:, 0:2] = world.store; out[:, 2] = world.felt[:, 2]
    off = torch.nan_to_num(where); sz = torch.nan_to_num(size)
    s = seen.float()
    for j in range(NS):
        out[:, 3 + 3 * j] = s[:, j] * (1 - sz[:, j].clamp(max=1)); out[:, 4 + 3 * j] = s[:, j] * off[:, j].abs() / math.pi; out[:, 5 + 3 * j] = s[:, j]
        out[:, 3 + 3 * NS + 2 * j] = s[:, j] * sz[:, j].clamp(max=2) / 2; out[:, 4 + 3 * NS + 2 * j] = s[:, j] * (off[:, j] / (FOV / 2)).clamp(-1, 1)
        out[:, 3 + 5 * NS + j] = s[:, j] * torch.sin(off[:, j])
    out[:, 3 + 6 * NS] = (eye.gx - 0.5) * 2
    out[:, 4 + 6 * NS:4 + 6 * NS + CH] = (fov * 2).clamp(max=1)
    return out
