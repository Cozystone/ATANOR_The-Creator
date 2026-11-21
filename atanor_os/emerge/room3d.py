"""아주 가벼운 1인칭 3D 몸의 세계 (Crafter 다음 무대의 뼈대. 아직 마음은 연결하지 않았다).

원칙 (창립자 2026-11-20):
  - 눈이 보는 정면이 곧 몸의 앞이다. 하늘에서 내려다본 지도도, 칸마다의 이름표도 없다. 무엇이 무엇인지는 눈이 알아내야 한다.
  - 세계는 걸음이 아니라 흐르는 시간 위에 있다. 세계는 제 시계로 돌고, 마음이 고르는 동안에도 기다려 주지 않는다 (배고픔이 쌓인다).
  - 몸은 운동 명령(앞으로 가는 세기, 도는 세기, 하기)을 붙들고 있고, 시간이 흐르는 만큼 그것이 일어난다.
  - 빠르게 돌리는 실행은 연습이다 (advance에 넘기는 시간으로 구분한다. 세계의 규칙은 같다).

감각:
  eye()   - 눈앞의 그림 [H, W, 3] (0~1). 벽과 샘들이 원근으로 보인다. 그림일 뿐 이름은 없다.
  body()  - 몸이 느끼는 것: 저장고 둘, 방금 흐른 시간 동안 실제로 나아간 거리와 돈 각 (제 걸음의 감각), 막혔는가, 방금 한 '하기'가 무언가를 채웠는가.

  python -m atanor_os.emerge.room3d out.png      (그림 몇 장을 이어 붙여 저장한다. 세계가 그려지는지 눈으로 보려고)
"""
import math, sys
import numpy as np

FLOOR, WALL, WATER, FOOD = 0, 1, 2, 3
COLOR = np.array([[0.0, 0.0, 0.0], [0.62, 0.60, 0.56], [0.20, 0.45, 0.95], [0.90, 0.35, 0.25]])
FOV = math.radians(80)
SKY, GROUND = [0.82, 0.84, 0.86], [0.38, 0.40, 0.34]      # 하늘과 땅은 빛깔이 옅다 (샘의 빛깔과 섞이지 않게)
SPEED, TURN = 2.0, math.radians(120)              # 1초에 나아가는 칸, 1초에 도는 각
DRAIN = 1 / 60.0                                  # 저장고가 1초에 주는 양 (가득에서 60초)
REACH = 1.2                                       # '하기'가 닿는 거리 (칸)


class Room:
    def __init__(self, seed=0, size=16, walls=5, wall_len=5, roam=False, spots=0, layout=None):
        r = np.random.default_rng(seed)
        if layout is not None:
            self._layout(layout, r, roam); return
        self._r, self.roam = r, roam                # roam: 샘은 한 번 채워 주면 다른 곳으로 옮겨 간다 (계속 찾아다녀야 한다)
        self.size = size
        m = np.zeros((size, size), dtype=np.int8)
        m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = WALL
        for _ in range(walls):
            x, y = r.integers(2, size - 2, 2)
            for k in range(wall_len):
                cx, cy = (x + k, y) if r.random() < 0.5 else (x, y + k)
                if 1 <= cx < size - 1 and 1 <= cy < size - 1:
                    m[cx, cy] = WALL
        free = np.argwhere(m == FLOOR)
        pick = free[r.choice(len(free), 3, replace=False)]
        m[tuple(pick[0])] = WATER; m[tuple(pick[1])] = FOOD
        self.map = m
        self.spots = None                          # spots: 샘마다 머무는 자리가 몇 곳으로 정해져 있다 (옮겨 갈 때 그 가운데 다른 곳으로 간다. 익히면 어디로 갔을지 짐작할 수 있다)
        if spots:
            rest = free[r.choice(len(free), 2 * spots + 3, replace=False)]
            rest = [c for c in rest if m[tuple(c)] == FLOOR and tuple(c) != tuple(pick[2])][:2 * (spots - 1)]
            self.spots = {WATER: [tuple(pick[0])] + [tuple(c) for c in rest[:spots - 1]], FOOD: [tuple(pick[1])] + [tuple(c) for c in rest[spots - 1:]]}
        self.pos = pick[2].astype(np.float64) + 0.5
        self.head = float(r.random() * 2 * math.pi)
        self.store = np.array([1.0, 1.0])          # 마실 것, 먹을 것
        self.clock = 0.0                           # 세계의 시계 (초)
        self.motor = (0.0, 0.0, False)             # 앞으로 가는 세기 [-1, 1], 도는 세기 [-1, 1], 하기
        self.dead = False
        self._felt = np.zeros(6)

    def _layout(self, rows, r, roam):
        """그려 둔 세계: # 벽, w / f 마실 것과 먹을 것이 머무는 자리들, s 태어나는 자리 (오른쪽을 보고 태어난다)."""
        self._r, self.roam = r, roam
        self.size = len(rows)
        m = np.zeros((self.size, self.size), dtype=np.int8)
        sp = {WATER: [], FOOD: []}; start = (1, 1)
        for y, row in enumerate(rows):
            for x, ch in enumerate(row):
                if ch == "#":
                    m[x, y] = WALL
                elif ch in "wf":
                    sp[WATER if ch == "w" else FOOD].append((x, y))
                elif ch == "s":
                    start = (x, y)
        for kind in (WATER, FOOD):
            m[sp[kind][int(r.integers(len(sp[kind])))]] = kind
        self.map, self.spots = m, sp
        self.pos = np.array(start, dtype=np.float64) + 0.5
        self.head, self.store, self.clock, self.motor, self.dead = 0.0, np.array([1.0, 1.0]), 0.0, (0.0, 0.0, False), False
        self._felt = np.zeros(6)

    # ---- 몸 ----
    def set_motor(self, forward, turn, do=False):
        self.motor = (float(np.clip(forward, -1, 1)), float(np.clip(turn, -1, 1)), bool(do))

    def advance(self, dt):
        """세계의 시계가 dt초 흐른다. 마음이 무엇을 하고 있든 흐른다."""
        if self.dead:
            return
        fwd, turn, do = self.motor
        moved, bumped, t, h = 0.0, False, 0.0, 1 / 60.0
        was, turned0 = self.pos.copy(), self.head
        while t < dt - 1e-9:                       # 작은 틱으로 쌓는다 (얇은 벽을 건너뛰지 않게)
            d = min(h, dt - t); t += d
            self.head = (self.head + turn * getattr(self, 'turn_speed', TURN) * d) % (2 * math.pi)
            step = fwd * SPEED * d
            nx = self.pos + step * np.array([math.cos(self.head), math.sin(self.head)])
            for ax in (0, 1):                      # 축마다 따로 막힌다 (벽을 따라 미끄러진다)
                trial = self.pos.copy(); trial[ax] = nx[ax]
                if self.map[int(trial[0]), int(trial[1])] == FLOOR:
                    moved += abs(trial[ax] - self.pos[ax]); self.pos = trial
                elif abs(step) > 0:
                    bumped = True
        self.store = self.store - getattr(self, "drain", DRAIN) * dt      # drain: 이 세계에서 저장고가 주는 빠르기 (따로 주지 않으면 DRAIN)
        filled = False
        if do:
            kind, dist = self._ahead()
            if kind in (WATER, FOOD) and dist <= REACH:
                self.store[kind - WATER] = 1.0; filled = True
                if self.roam and self.spots:
                    here = tuple(np.argwhere(self.map == kind)[0])
                    others = [c for c in self.spots[kind] if c != here and self.map[c] == FLOOR and max(abs(c[0] + 0.5 - self.pos[0]), abs(c[1] + 0.5 - self.pos[1])) > 1.0]
                    if others:
                        self.map[here] = FLOOR; self.map[others[self._r.integers(len(others))]] = kind
                elif self.roam:
                    free = np.argwhere(self.map == FLOOR)
                    free = free[(np.abs(free + 0.5 - self.pos).max(1) > 1.0)]
                    self.map[self.map == kind] = FLOOR
                    self.map[tuple(free[self._r.integers(len(free))])] = kind
        self.clock += dt
        if (self.store <= 0).any():
            self.store = np.clip(self.store, 0, None); self.dead = True
        d = self.pos - was; c, s_ = math.cos(self.head), math.sin(self.head)
        # 몸이 느끼는 제 걸음: 이 시간 동안 실제로 옮겨 간 만큼을, 지금 보는 쪽을 앞으로 삼아 앞으로 / 옆으로 (벽에 미끄러지면 옆으로 간 것도 느낀다)
        self._felt = np.array([d[0] * c + d[1] * s_, turn * getattr(self, 'turn_speed', TURN) * dt, float(bumped), float(filled), dt, -d[0] * s_ + d[1] * c])

    def body(self):
        """[마실 것, 먹을 것, 앞으로 간 거리, 돈 각, 막혔는가, 채워졌는가, 흐른 시간, 옆으로 간 거리]"""
        return np.concatenate([self.store, self._felt])

    # ---- 눈 ----
    def _cast(self, ang):
        """각 ang들로 쏜 시선이 처음 닿는 것: (종류, 거리)."""
        dx, dy = np.cos(ang), np.sin(ang)
        kind = np.zeros(len(ang), dtype=np.int8); dist = np.full(len(ang), 99.0)
        live = np.ones(len(ang), dtype=bool)
        for s in np.arange(0.02, self.size * 1.5, 0.02):
            cx = (self.pos[0] + dx * s).astype(int).clip(0, self.size - 1); cy = (self.pos[1] + dy * s).astype(int).clip(0, self.size - 1)
            hit = live & (self.map[cx, cy] != FLOOR)
            kind[hit] = self.map[cx, cy][hit]; dist[hit] = s; live &= ~hit
            if not live.any():
                break
        return kind, dist

    def _ahead(self):
        k, d = self._cast(np.array([self.head]))
        return int(k[0]), float(d[0])

    def eye(self, W=64, H=40):
        ang = self.head + (np.arange(W) / (W - 1) - 0.5) * FOV
        kind, dist = self._cast(ang)
        dist = dist * np.cos(ang - self.head)                       # 어안 보정
        half = np.clip(0.5 * H / np.maximum(dist, 0.05), 0, H)      # 가까울수록 높게 보인다
        img = np.zeros((H, W, 3))
        rows = np.arange(H)[:, None]
        sky, ground = rows < H / 2, rows >= H / 2
        img[:] = np.where(sky[..., None], SKY, GROUND)
        img[ground[:, 0]] *= (0.55 + 0.45 * (rows[ground[:, 0]] - H / 2) / (H / 2))[:, :, None]      # 먼 바닥은 어둡다
        wall = np.abs(rows - H / 2 + 0.5) <= half[None]
        shade = np.clip(1.0 - dist / 12.0, 0.25, 1.0)
        img = np.where(wall[..., None], (COLOR[kind] * shade[:, None])[None], img)
        return img.astype(np.float32)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "room3d.png"
    from PIL import Image
    w = Room(seed=3)
    frames = []
    plan = [(0, 1, 0.5), (1, 0, 1.0), (0, -1, 0.6), (1, 0, 1.5), (1, 0.6, 1.0), (1, 0, 1.0), (0, 1, 0.8), (1, 0, 1.2)]      # (앞으로, 돌기, 몇 초)
    for fwd, turn, sec in plan:
        w.set_motor(fwd, turn); w.advance(sec); frames.append(w.eye(160, 100))
    strip = np.concatenate([np.concatenate(frames[:4], 1), np.concatenate(frames[4:], 1)], 0)
    Image.fromarray((strip * 255).astype(np.uint8)).resize((strip.shape[1] * 2, strip.shape[0] * 2), Image.NEAREST).save(out)
    print("saved", out, "| clock", round(w.clock, 2), "s | stores", w.store.round(2), "| body", w.body().round(2))
