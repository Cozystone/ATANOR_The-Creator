"""이월 검사의 첫 걸음: 마음의 알맹이(core.py)를 한 줄도 고치지 않고 1인칭 3D 세계(room3d.py)의 몸에 꽂는다 (재는 것. 아직 눈은 쓰지 않는다).

이 세계는 초단순 세계와 다르다: 몸에 향하는 쪽이 있고 (돌기 / 앞으로 가기), 세계는 걸음이 아니라 시계로 흐르고 (고를 때마다 TICK초가 지나간다), 샘은 그 위에 서는 것이 아니라 마주 보고 닿아야 찬다.
느끼는 것: 저장고 둘, 막혔는가, 그리고 자리 감각 - 본 적 있는 샘이 얼마나 멀고 어느 쪽인가 (몸 기준. 멀리까지 거칠게, 그리고 몸 가까이 두 칸은 곱게). 무엇이 샘인지는 아직 세계가 알려 준다 (눈이 알아내는 것은 다음 일이다).

  python -m atanor_os.emerge.room3d_live --rounds 30
"""
import argparse, math, time
import numpy as np, torch
from .room3d import Room, WATER, FOOD, REACH, FOV, COLOR, SKY, GROUND
from .core import Core
from .room3d_map import Map
from .routes import Routes

HOME_SMALL = ["################",    # 처음 그린 작은 세계 (문이 한 칸. 창립자: 문이 좁다 -> 아래 큰 세계)
        "#w...#.....#..f#",
        "#....#.....#...#",
        "#..........#...#",
        "#....#.........#",
        "##.###..####.###",
        "#......s.......#",
        "#..............#",
        "###.####.###.###",
        "#....#.....#...#",
        "#f...#..w..#...#",
        "#....#.....#...#",
        "#..........#..w#",
        "#....#.........#",
        "#...f#.....#...#",
        "################"]
HOME = ["########################",
        "#w.......#.......#....f#",
        "#........#.......#.....#",
        "#........#.......#.....#",
        "#................#.....#",
        "#................#.....#",
        "#........#.............#",
        "#........#.............#",
        "###..#####..######..####",
        "#......................#",
        "#..........s...........#",
        "#......................#",
        "#......................#",
        "####..######..#####..###",
        "#.......#........#.....#",
        "#f......#........#.....#",
        "#.......#...w....#.....#",
        "#................#.....#",
        "#................#....w#",
        "#.......#..............#",
        "#.......#..............#",
        "#...f...#........#.....#",
        "#.......#........#.....#",
        "########################"]      # 그려 둔 익숙한 세계 (24 x 24: 방 여섯과 가운데 큰 마당, 문은 두 칸). 샘은 w / f 자리들 사이를 옮겨 다닌다

DEV = "cpu"
TICK = 0.2                                # 한 번 고르는 동안 세계에서 흐르는 시간 (초). 연습에서는 이만큼씩 건너뛴다
MOTOR = [(0, 0, False), (1, 0, False), (0, -1, False), (0, 1, False), (0, 0, True)]      # 가만히, 앞으로, 왼쪽으로 돌기, 오른쪽으로 돌기, 하기
M = len(MOTOR)
LEAK = 0.95                               # 헛됨의 통이 틱마다 남기는 몫 (손으로 준 수)
GAZE = 1.5                                # 눈동자가 1초에 옮겨 가는 거리 (그림 너비에 대한 비율)
PW, PH = 64, 40                           # 눈앞 그림의 크기 (존재의 눈이 받는 것)
CHROMA = 3                                # 빛깔의 맞섬 셋: 붉음(붉음-풀빛), 풀빛(풀빛-붉음), 파랑(파랑-누름). 눈이 타고난 것이고 이름이 아니다


BODY = list(MOTOR)                        # 몸의 운동 (set_motors가 뒤로 걷기를 더할 수 있다)
NAMES = ["가만히", "앞으로", "왼쪽으로 돌기", "오른쪽으로 돌기", "하기"]
PATTERN = False                           # 벽에 세로 줄무늬가 있고, 벽이 향한 쪽마다 밝기가 다른가
FACE = np.array([0.50, 0.66, 0.82, 0.98])  # 벽이 향한 네 쪽의 밝기
FLOW = 16                                 # 무늬가 옆으로 밀린 것을 재는 거친 띠의 칸 수


def set_motors(gaze, back=False, both=False):
    """gaze = 눈동자를 왼쪽 / 오른쪽으로 옮기는 운동 둘을 더하는가 (몸은 그대로 둔다). back = 뒤로 걷기를 더하는가.
    both = 걸으면서 도는 것을 한꺼번에 할 수 있는가 (걷기와 돌기가 서로를 막지 않는다. 휘어 걷는다)."""
    global M, MOTOR, NAMES
    MOTOR = BODY + ([(-1, 0, False)] if back else []) + ([(1, -1, False), (1, 1, False)] if both else [])
    NAMES = (["가만히", "앞으로", "왼쪽으로 돌기", "오른쪽으로 돌기", "하기"] + (["뒤로 걷기"] if back else []) + (["걸으며 왼쪽으로", "걸으며 오른쪽으로"] if both else [])
             + (["눈동자 왼쪽으로", "눈동자 오른쪽으로"] if gaze else []))
    M = len(MOTOR) + (2 if gaze else 0)


def set_world(pattern):
    global PATTERN
    PATTERN = pattern
NS = 2                                    # 자리 감각의 칸 수 (칸마다: 얼마나 먼가, 정면에서 얼마나 벗어났는가, 아는가 / 몸 가까이 앞-옆 / 어느 옆)
DX = 3 + 6 * NS
INSIDE, PLACE = slice(0, 2), slice(3, 3 + 3 * NS)
RAYS, TINTS = 9, 4                        # 눈: 시야에 고르게 편 시선의 수, 빛깔을 가르는 칸의 수


def set_slots(ns, depth=0):
    """depth = 시선마다의 거리도 느끼는가 (그 수. 자리 감각 뒤에 붙는다)."""
    global NS, DX, PLACE
    NS, DX, PLACE = ns, 3 + 6 * ns + depth, slice(3, 3 + 3 * ns)


def put(out, j, a, b):
    """자리 감각의 j째 칸에, 몸 기준 앞으로 a 옆으로 b에 있는 것을 적는다."""
    th = math.atan2(b, a)
    out[3 + 3 * j:6 + 3 * j] = [math.log1p(math.hypot(a, b)) / LG, abs(th) / math.pi, 1.0]
    out[3 + 3 * NS + 2 * j:5 + 3 * NS + 2 * j] = np.clip([a / 2.0, b / 2.0], -1, 1)
    out[3 + 5 * NS + j] = math.sin(th)


def tint(rgb):
    """빛깔을 거칠게 가른다: 빛깔 없음(잿빛) 0, 그 밖은 가장 센 빛 (1~3). 이름이 아니다."""
    return 0 if rgb.max() - rgb.min() < 0.15 else 1 + int(rgb.argmax())


class Eye:
    """눈으로만 안다: 시야에 편 시선마다 닿은 것의 빛깔과 거리를 받아 (그림의 가운데 줄이 주는 것 그대로: 빛깔, 그리고 가까울수록 높게 보이는 것),
    빛깔 칸마다 본 자리 몇 곳을 제 걸음과 돈 것을 더해 간 제 자리에서 적어 둔다 (크기 고정: 칸마다 KEEP곳, 가득 차면 가장 오래된 것이 자리를 내준다).
    세계는 무엇이 샘인지 알려 주지 않는다. 어느 빛깔이 모자람을 풀어 주는지는 풀림의 기억이 가린다."""
    KEEP = 6
    forget = True

    def __init__(self, w):
        self.w, self.pos, self.head = w, np.zeros(2), 0.0
        self.at = [[] for _ in range(TINTS)]; self.now = [None] * TINTS

    def step(self):
        moved, turned, side = self.w._felt[0], self.w._felt[1], self.w._felt[5]
        self.head += turned
        self.pos = self.pos + moved * np.array([math.cos(self.head), math.sin(self.head)]) + side * np.array([-math.sin(self.head), math.cos(self.head)])

    def take(self, kinds, dists, offs):
        self.depth = np.log1p(np.asarray(dists)) / LG             # 시선마다 얼마나 멀리까지 트여 있는가
        self.lines = [(-1 if k == 0 else tint(COLOR[k]), float(r)) for k, r in zip(kinds, dists)]      # 시선마다 닿은 빛깔 칸과 거리 (뷰어가 보여 주려고)
        if self.forget:                                           # 적어 둔 자리를 다시 보았는데 그 빛깔이 거기 없으면 지운다 (시선이 그 자리를 지나 더 멀리 닿았다)
            offs_ = np.asarray(offs); dd = np.asarray(dists)
            for j, pts in enumerate(self.at):
                for q in list(pts):
                    d = q - self.pos; r = float(np.hypot(*d))
                    off = (math.atan2(d[1], d[0]) - self.head + math.pi) % (2 * math.pi) - math.pi
                    i = int(np.abs(offs_ - off).argmin())
                    if abs(offs_[i] - off) < 0.09 and dd[i] > r + 0.6:
                        pts[:] = [o for o in pts if o is not q]
        self.now = [None] * TINTS                                 # 지금 눈에 들어와 있는 것: 빛깔 칸마다 정면에 가장 가까운 시선이 닿은 곳
        for k, r, off in zip(kinds, dists, offs):
            if k == 0:
                continue
            j = tint(COLOR[k])
            if self.now[j] is None or abs(off) < abs(self.now[j][1]):
                self.now[j] = (r, off)
            q = self.pos + r * np.array([math.cos(self.head + off), math.sin(self.head + off)])      # 시선이 닿은 겉의 자리
            pts = self.at[j]
            for i, o in enumerate(pts):
                if np.abs(o - q).max() < 0.5:
                    pts.pop(i); break
            pts.append(q)
            if len(pts) > self.KEEP:
                pts.pop(0)

    def sense(self):
        w = self.w
        out = np.zeros(DX, dtype=np.float32)
        out[0:2] = w.store; out[2] = w._felt[2]
        c, s = math.cos(self.head), math.sin(self.head)
        for j, pts in enumerate(self.at):
            if self.now[j] is not None:                           # 보이는 것은 본 대로
                r, off = self.now[j]
                put(out, j, r * math.cos(off), r * math.sin(off))
            elif pts:                                             # 안 보이면 적어 둔 것 가운데 가장 가까운 곳
                d = np.array(pts) - self.pos
                dx, dy = d[np.hypot(d[:, 0], d[:, 1]).argmin()]
                put(out, j, dx * c + dy * s, -dx * s + dy * c)
        if DX > 3 + 6 * NS:
            out[3 + 6 * NS:] = self.depth
        return out


_FOVEA = None


def canon_eye():
    """창립자의 눈 기관 (packages/eye/fovea.py). 고치지 않고 그대로 불러 쓴다. 없으면 없다고 하고 멈춘다."""
    global _FOVEA
    if _FOVEA is None:
        import importlib.util, os, sys
        path = os.environ.get("ATANOR_EYE_PATH", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eye", "fovea.py"))
        spec = importlib.util.spec_from_file_location("atanor_canon_fovea", path)
        _FOVEA = importlib.util.module_from_spec(spec); sys.modules["atanor_canon_fovea"] = _FOVEA; spec.loader.exec_module(_FOVEA)
    return _FOVEA


def pictures(ws, step=0.08, PW=None, PH=None):
    """모든 몸의 눈앞 그림 [n, PH, PW, 3] (0~1). room3d.eye와 같은 그림을 한꺼번에 그린다 (PATTERN이면 벽에 줄무늬와 쪽마다의 밝기). 거리는 그림 밖으로 내보내지 않는다."""
    PW, PH = PW or globals()["PW"], PH or globals()["PH"]
    n = len(ws)
    maps = np.stack([w.map for w in ws]); pos = np.stack([w.pos for w in ws]); head = np.array([w.head for w in ws])
    offs = (np.arange(PW) / (PW - 1) - 0.5) * FOV
    ang = head[:, None] + offs[None]
    dx, dy = np.cos(ang), np.sin(ang)
    kind = np.zeros((n, PW), dtype=np.int8); dist = np.full((n, PW), 99.0); live = np.ones((n, PW), dtype=bool)
    face = np.zeros((n, PW), dtype=np.int8); along = np.zeros((n, PW))
    idx = np.arange(n)[:, None]
    pcx = pos[:, 0:1].astype(int) * np.ones((1, PW), dtype=int)
    top = maps.shape[1] - 1
    for r in np.arange(step, 1.5 * top, step):
        hx, hy = pos[:, 0:1] + dx * r, pos[:, 1:2] + dy * r
        cx = hx.astype(int).clip(0, top); cy = hy.astype(int).clip(0, top)
        m = maps[idx, cx, cy]
        hit = live & (m != 0)
        kind[hit] = m[hit]; dist[hit] = r; live &= ~hit
        xf = cx != pcx                                             # 앞 걸음과 견주어 어느 쪽 면으로 들어왔는가
        face[hit] = np.where(xf, np.where(dx > 0, 0, 1), np.where(dy > 0, 2, 3))[hit]
        along[hit] = np.where(xf, hy % 1.0, hx % 1.0)[hit]
        pcx = cx
        if not live.any():
            break
    dist = dist * np.cos(offs)[None]
    half = np.clip(0.5 * PH / np.maximum(dist, 0.05), 0, PH)
    rows = np.arange(PH)[None, :, None]
    img = np.where((rows < PH / 2)[..., None], np.array(SKY), np.array(GROUND)) * np.ones((n, PH, PW, 1))
    wall = np.abs(rows - PH / 2 + 0.5) <= half[:, None, :]
    shade = np.clip(1.0 - dist / 12.0, 0.25, 1.0)
    col = COLOR[kind] * shade[..., None]
    if PATTERN:                                                    # 벽에만: 향한 쪽의 밝기 x 세로 줄무늬 (한 칸에 세 줄)
        lum = FACE[face] * np.where((along * 3.0) % 1.0 < 0.5, 0.62, 1.0) * shade
        col = np.where((kind == 1)[..., None], lum[..., None] * np.array([1.0, 0.98, 0.94]), col)
    return np.where(wall[..., None], col[:, None], img).astype(np.float32)


class Retina:
    """그림으로만 안다 (거리도, 무엇이 무엇인지도 받지 않는다). 창립자의 눈 기관으로 그림을 받는다: 초점 둘레는 원래 곱기로, 바깥 고리들은 거칠게.
    눈이 타고난 것은 빛깔의 맞섬 셋뿐이다. 맞섬마다: 가장 크게(높게) 보이는 덩이가 정면에서 얼마나 벗어나 있고 얼마나 크게 보이는가 (가까우면 크다).
    거친 고리에서는 작은 것이 묻히고 자리도 크기도 뭉개진다 - 또렷이 알려면 초점이 거기 가 있어야 한다.
    안 보일 때는 마지막으로 본 쪽(제가 돈 것을 더해 간 방향)과 그때의 크기를 든다 (맞섬마다 하나. 크기 고정). 초점으로 그쪽을 보았는데 없으면 지운다."""
    THR = 0.06
    EAR = False                                                   # 귀를 드는가 (hear)
    reflex = False                                                # 눈의 반사 (눈 기관이 하는 일. 마음이 고르지 않는다): 주변에 빛깔이 잡히면 초점이 그리로 뛰고, 잡은 것은 몸이 돌아도 붙든다
    ROUTES = False                                                # 길의 기억을 드는가 (routes.Routes: 걸어 본 길과, 거기서 얼마 뒤에 풀렸는지)
    MAP = False                                                   # 머릿속 지도를 드는가 (room3d_map.Map)
    IMAGINE = False                                               # 떠올린 자리로 갈 때 지도 위에서 속으로 걸어 보고, 갈 데를 모르면 아직 모르는 쪽으로 걸어 보는가
    AGAIN = None                                                  # 가 보니 없던 자리를 다시 가 볼 만하다고 여기기까지의 틱. None = 제가 겪은 것에서 안다 (없던 자리가 다시 차 있는 것을 보기까지 걸린 시간들의 가운데값. 겪기 전에는 되돌아가지 않고 모르는 쪽을 찾는다). 수를 주면 그 수
    SPOTS = 1                                                     # 빛깔마다 적어 두는 자리의 수 (크기 고정. 가득 차면 가장 오래 안 본 자리가 나간다)
    DWELL = 10                                                    # 한 곳을 이만큼(틱) 보고 나면 다른 빛깔이 있을 때 그리로 옮긴다 (본 곳으로 곧바로 되돌아가지 않는다)
    POOL = "mean"                                                 # 바깥 고리가 빛깔을 줄여 담는 법: "mean" = 고르게 뭉갬 (눈 기관 그대로) / "max" = 칸 안에서 가장 센 빛깔을 남김 (좁게 낀 빛깔이 벽에 묻히지 않는다. 밝기는 그대로 뭉갠다)
    OPPONENT = "rgb"                                              # 빛깔의 맞섬을 내는 법: "rgb" = 한 빛이 나머지 둘의 평균보다 센 만큼 / "old" = 붉음-풀빛, 파랑-누름 (앞선 갈래들)

    def __init__(self, w):
        self.w, self.head, self.gx = w, 0.0, 0.5
        self.pos = np.zeros(2); self.band = None; self.flow = np.zeros(5)
        self.lock, self.dwell, self.fine = None, 0, [False] * CHROMA
        self.map = Map() if self.MAP else None; self.way = [None] * CHROMA; self.thought = 0; self.routes = [None] * CHROMA
        self.back = []
        self.road = Routes() if self.ROUTES else None; self.link = np.zeros((2, CHROMA)); self.store0 = w.store.copy(); self.byroad = [False] * CHROMA
        self.spots, self.t = [[] for _ in range(CHROMA)], 0       # 자리마다 [자리, 마지막으로 본 때, 가 보니 없던 때, 곱게 본 것인가]
        self.kept = [None] * CHROMA                               # 마지막으로 본 자리 (제 걸음을 더해 간 자리. 거리는 보인 크기에서: 크게 보일수록 가깝다)
        self.now = [None] * CHROMA                                # (벗어남, 크기, 곱게 본 것인가)
        self.fov = np.zeros(CHROMA)
        self.box = (0, 0, 0, 0)
        self.spec = canon_eye().RetinaSpec(fovea_px=8, rings=3, ring_px=8, falloff=2.0)
        self.spec_peak = canon_eye().RetinaSpec(fovea_px=8, rings=3, ring_px=8, falloff=2.0, ring_peak=True)      # 눈 기관 정본의 ring_peak (2026-11-21 창립자 승인으로 정본에 넣음)

    def step(self, gaze=0.0):
        self.head += self.w._felt[1]
        f_, s_ = self.w._felt[0], self.w._felt[5]
        self.pos = self.pos + f_ * np.array([math.cos(self.head), math.sin(self.head)]) + s_ * np.array([-math.sin(self.head), math.cos(self.head)])
        self.gx = float(np.clip(self.gx + gaze * GAZE * TICK, 0.0, 1.0))

    def take(self, img):
        o = canon_eye().sample(img, self.gx, 0.5, self.spec_peak if self.POOL == "max" else self.spec)
        self.box = o["fovea_box"]
        levels = [(o["fovea"], o["fovea_box"])] + [(lv["px"], lv["box"]) for lv in o["levels"]]
        peaks = [None] + [lv.get("peak") for lv in o["levels"]]
        where, size, last, sharp, lastw = [None] * CHROMA, [None] * CHROMA, [0.0] * CHROMA, [False] * CHROMA, [None] * CHROMA
        for li, (px, (x0, y0, x1, y1)) in enumerate(levels):
            px = np.asarray(px, np.float32)
            if self.OPPONENT == "old":
                rg = px[..., 0] - px[..., 1]; by = px[..., 2] - 0.5 * (px[..., 0] + px[..., 1])
                ch = np.stack([np.maximum(rg, 0), np.maximum(-rg, 0), np.maximum(by, 0)])      # [3, rows, cols]
            else:
                ch = np.maximum(np.stack([px[..., k] - 0.5 * (px[..., (k + 1) % 3] + px[..., (k + 2) % 3]) for k in range(3)]), 0)
                if self.POOL == "max" and li > 0 and peaks[li] is not None:      # 바깥 고리는 칸마다 가장 센 빛깔을 남긴 것을 정본 눈 기관에서 받는다
                    ch = np.asarray(peaks[li], np.float32).transpose(2, 0, 1)
            rows, cols = ch.shape[1:]
            cellh, cellw = (y1 - y0) / rows, (x1 - x0) / cols
            tall = (ch > self.THR).sum(1) * cellh / PH                                          # [3, cols] 그 빛깔이 세로로 얼마나 차 있는가
            if self.POOL == "max" and li > 0:                      # 가장 센 것을 남기면 걸친 칸도 찬 것으로 세어진다: 반 칸을 던다
                tall = np.where(tall > 0, np.maximum(tall - 0.5 * cellh / PH, 0.5 * cellh / PH), 0)
            if li == 0:
                self.fov = ch.reshape(CHROMA, -1).mean(1)
            mask = ch > self.THR
            for j in range(CHROMA):
                c = int(tall[j].argmax())
                if tall[j, c] <= 0:
                    continue
                cut = (y1 - y0) < PH and bool(mask[j, 0].any() or mask[j, -1].any())           # 이 겹의 위아래 끝에 걸려 다 들어오지 않았다
                xs = np.flatnonzero(tall[j] > 0)
                side_cut = (x1 - x0) < PW and bool(tall[j, 0] > 0 or tall[j, -1] > 0)          # 이 겹의 좌우 끝에 걸려 다 들어오지 않았다 (가까이 있어 크게 보이면 초점 상자에 다 안 들어온다)
                if where[j] is None and not side_cut:              # 어느 쪽인가: 그것이 좌우로 다 들어오는 가장 고운 겹에서 (그 빛깔이 있는 칸들의 가운데)
                    where[j] = ((x0 + (xs.mean() + 0.5) * cellw) / PW - 0.5) * FOV; sharp[j] = li == 0
                if where[j] is None:
                    lastw[j] = ((x0 + (xs.mean() + 0.5) * cellw) / PW - 0.5) * FOV
                if size[j] is None and not cut:                    # 얼마나 큰가: 위아래가 다 들어오는 가장 고운 겹에서
                    size[j] = float(tall[j, c])
                last[j] = float(tall[j, c])
        where = [where[j] if where[j] is not None else lastw[j] for j in range(CHROMA)]
        best = [None if where[j] is None else (where[j], size[j] if size[j] is not None else last[j], sharp[j]) for j in range(CHROMA)]
        self.t += 1
        for j in range(CHROMA):
            hit = None
            if best[j] is not None:
                far = 1.0 / max(best[j][1], 0.04)
                q = self.pos + far * np.array([math.cos(self.head + best[j][0]), math.sin(self.head + best[j][0])])
                for sp in self.spots[j]:
                    d = sp[0] - self.pos
                    off = (math.atan2(d[1], d[0]) - self.head + math.pi) % (2 * math.pi) - math.pi
                    if abs(off - best[j][0]) < 0.15 and float(np.hypot(*(sp[0] - q))) < max(1.5, 0.35 * float(np.hypot(*(q - self.pos)))):      # 같은 쪽이고, 거친 눈의 거리 어림이 틀릴 만한 만큼 안에 있을 때만 그것이다 (같은 쪽 더 먼 / 가까운 곳의 다른 것을 옛 자리로 덮어쓰지 않는다)
                        hit = sp; break
                if hit is None:
                    hit = [q, self.t, -1, best[j][2]]; self.spots[j].append(hit)
                    if len(self.spots[j]) > self.SPOTS:
                        old = min((sp for sp in self.spots[j] if sp is not hit), key=lambda sp: sp[1])
                        self.spots[j] = [sp for sp in self.spots[j] if sp is not old]
                elif self.reflex and not best[j][2] and hit[3]:
                    d = hit[0] - self.pos                          # 곱게 보아 둔 자리가 지금의 거친 모습과 어긋나지 않으면 곱게 본 쪽을 믿는다
                    best[j] = ((math.atan2(d[1], d[0]) - self.head + math.pi) % (2 * math.pi) - math.pi, 1.0 / max(float(np.hypot(*d)), 0.3), False)
                else:
                    hit[0], hit[3] = q, best[j][2]
                if hit[2] > hit[1]:                                # 없던 자리가 다시 차 있다: 그 사이가 얼마였는지를 든다 (크기 고정)
                    self.back = (self.back + [self.t - hit[2]])[-32:]
                hit[1] = self.t
                self.spots[j] = [sp for sp in self.spots[j] if sp is hit or np.abs(sp[0] - hit[0]).max() > 2.0]      # 한 가지 것은 한 자리에: 지금 본 것 바로 곁에 적어 둔 같은 빛깔의 자리들은 그것이었다
            for sp in self.spots[j]:                               # 적어 둔 자리가 가까이 눈앞에 있는데 그 빛깔이 거기 없다: 가 보니 없던 것으로 적는다
                if sp is hit:
                    continue
                d = sp[0] - self.pos
                off = (math.atan2(d[1], d[0]) - self.head + math.pi) % (2 * math.pi) - math.pi
                if (float(np.hypot(*d)) < 3.0 and abs(off) < 0.35 * FOV) or float(np.hypot(*d)) < 1.0:      # 눈앞 가까이인데 없다, 또는 그 자리에 와 서 있는데 없다
                    sp[2] = self.t
            alive = [sp for sp in self.spots[j] if sp[1] > sp[2]]
            if alive:                                              # 있으리라 여기는 자리 가운데 가장 가까운 곳
                sp = min(alive, key=lambda sp: float(np.hypot(*(sp[0] - self.pos))))
            else:                                                  # 다 없었다면, 없던 것을 본 지 가장 오래된 곳 (돌아와 있을지 모른다). 자리가 하나뿐이면 얼마 동안은 모르는 것으로 둔다
                wait = self.AGAIN if self.AGAIN is not None else (float(np.median(self.back)) if len(self.back) >= 3 else 1e9)
                old = [sp for sp in self.spots[j] if self.t - sp[2] > wait]      # 방금 가 보고 없던 곳으로는 곧바로 되돌아가지 않는다 (그사이에는 모르는 쪽을 찾는다)
                sp = min(old, key=lambda sp: sp[2]) if old else None
            self.kept[j], self.fine[j] = (None, False) if sp is None else (sp[0], sp[3])
        self.now = best
        if self.map is not None:
            self.map.stand(self.pos)
            if self.w._felt[2]:
                self.map.bump(self.pos, self.head)
            K = 16; bw = PW // K                                   # 시선 열여섯 줄: 줄마다 눈높이에 걸린 것이 서 있는 높이 -> 크게 보일수록 가깝다
            up = img[:PH // 2]                                 # 눈높이 줄에 걸린 것이 위로 어디까지 이어지는가 (눈높이 줄과 빛이 같은 칸의 수)
            tall = (np.abs(up - up[-1:]).sum(-1) < 0.05).sum(0).reshape(K, bw).mean(1) * 2.0 / PH
            offs_ = ((np.arange(K) + 0.5) * bw / PW - 0.5) * FOV
            self.map.see(self.pos, self.head, offs_, 1.0 / np.maximum(tall, 0.07) / np.cos(offs_), tall > 0.07)
            if self.IMAGINE:                                       # 속으로 걸어 보고 고른 길은 붙들고 따라간다. 다시 걸어 보는 때: 막혔을 때, 길 끝에 왔을 때, 가려는 데가 바뀌었을 때, 오래되었을 때
                for j in range(CHROMA):
                    if j == 1 and self.kept[j] is None:
                        self.way[j] = None; continue
                    key = None if self.kept[j] is None else tuple(np.rint(self.kept[j]).astype(int))
                    r = self.routes[j]
                    if r is not None and r[0] == key and not self.w._felt[2] and self.t - r[2] < 40:
                        k = int(np.abs(r[1] - self.pos).sum(1).argmin())
                        if k < len(r[1]) - 1:
                            self.way[j] = r[1][min(k + 3, len(r[1]) - 1)]; continue
                    self.thought += 1
                    self.way[j] = self.map.way(self.pos, self.kept[j])[0]
                    self.routes[j] = None if self.way[j] is None or self.map.route is None else (key, self.map.route, self.t)
        if self.road is not None:                                  # 길의 기억: 걸음을 적고, 풀리면 그 길에 적고, 모자란 것마다 풀렸던 길을 떠올린다
            self.road.walk(self.pos, self.t)
            rose = (self.w.store - self.store0) > 0.05; self.store0 = self.w.store.copy()
            for k in np.flatnonzero(rose):
                self.road.relief(int(k), self.pos, self.t)
                near = [(best[c][1], c) for c in range(CHROMA) if best[c] is not None]
                if near:                                           # 그 모자람이 풀릴 때 눈앞에 가장 크게 있던 빛깔을 센다 (어느 빛깔의 칸에 길을 얹을지를 겪어서 안다)
                    self.link[k, max(near)[1]] += 1
            self.byroad = [False] * CHROMA
            for k in range(2):
                if self.link[k].sum() == 0 or self.w.store[k] > 0.75:
                    continue
                c = int(self.link[k].argmax())
                if best[c] is None:
                    wp = self.road.recall(self.pos, k, self.t)
                    if wp is not None:
                        self.way[c] = wp; self.byroad[c] = True
        if self.reflex:
            seen = [j for j in range(CHROMA) if best[j] is not None and abs(best[j][0]) < FOV / 2]
            if self.lock in seen and (self.dwell < self.DWELL or len(seen) == 1):
                self.dwell += 1                                    # 붙든 것을 계속 본다 (몸이 돌면 눈이 반대로 돈다)
            elif seen:
                rest = [j for j in seen if j != self.lock] or seen
                self.lock, self.dwell = max(rest, key=lambda j: best[j][1]), 0      # 가장 크게 잡힌 빛깔로 뛴다
            else:
                self.lock, self.dwell = None, 0
            self.gx = 0.5 if self.lock is None else float(np.clip(best[self.lock][0] / FOV + 0.5, 0.0, 1.0))
        # 밝고 어두움의 무늬: 가운데 띠를 거칠게 받아, 앞의 것과 견주어 옆으로 얼마나 밀렸는지를 잰다 (제 눈이 재는 것. 세계가 알려 주지 않는다)
        lum = img[PH // 2 - 4:PH // 2 + 4].mean((0, 2)).reshape(FLOW, -1).mean(1)
        fl = 0.0
        if self.band is not None:
            errs = [np.abs(lum[max(k, 0):FLOW + min(k, 0)] - self.band[max(-k, 0):FLOW + min(-k, 0)]).mean() for k in range(-5, 6)]
            if max(errs) - min(errs) > 0.01:
                fl = (int(np.argmin(errs)) - 5) / 5.0
        self.band = lum
        self.flow = np.array([fl, min(float(lum.std()) * 4, 1.0), lum[:FLOW // 3].mean(), lum[FLOW // 3:2 * FLOW // 3].mean(), lum[2 * FLOW // 3:].mean()])

    def recall(self, j):
        d = self.kept[j] - self.pos
        return (math.atan2(d[1], d[0]) - self.head + math.pi) % (2 * math.pi) - math.pi, 1.0 / max(float(np.hypot(*d)), 0.3)

    def sense(self):
        w = self.w
        out = np.zeros(DX, dtype=np.float32)
        out[0:2] = w.store; out[2] = w._felt[2]
        for j in range(CHROMA):
            if self.now[j] is not None:
                off, size = self.now[j][0], self.now[j][1]
            elif (self.IMAGINE or self.byroad[j]) and self.way[j] is not None:        # 떠올린 길 / 속으로 걸어 본 길의 몇 걸음 앞 ("저리로 가면 된다" / "저쪽은 아직 모른다")
                d = self.way[j] - self.pos
                off, size = (math.atan2(d[1], d[0]) - self.head + math.pi) % (2 * math.pi) - math.pi, 1.0 / max(float(np.hypot(*d)), 0.3)
                if self.kept[j] is None:
                    size = min(size, 0.3)                           # 있으리라 짐작할 뿐인 곳은 가까이 가도 닿았다고 느끼지 않는다
            elif self.kept[j] is not None:
                off, size = self.recall(j)
            else:
                continue
            out[3 + 3 * j:6 + 3 * j] = [1.0 - min(size, 1.0), abs(off) / math.pi, 1.0]      # 작게 보일수록 1에 가깝다 (가까이 크게 보이면 0), 정면에서 벗어남, 아는가
            out[3 + 3 * NS + 2 * j:5 + 3 * NS + 2 * j] = [min(size, 2.0) / 2.0, float(np.clip(off / (FOV / 2), -1, 1))]
            out[3 + 5 * NS + j] = math.sin(off)
        out[3 + 6 * NS] = (self.gx - 0.5) * 2                     # 눈동자가 어디를 향해 있는가 (제 눈의 느낌)
        out[4 + 6 * NS:4 + 6 * NS + CHROMA] = np.minimum(self.fov * 2, 1.0)       # 초점에 그 빛깔이 얼마나 차 있는가
        if self.EAR:                                              # 귀: 맨 끝 네 칸 (높이 둘 x 왼쪽 / 오른쪽 귀)
            out[-4:] = hear(w)
        elif DX > 4 + 6 * NS + CHROMA:                            # 무늬가 옆으로 밀린 정도, 무늬가 얼마나 뚜렷한가, 왼쪽 / 가운데 / 오른쪽의 밝기
            out[4 + 6 * NS + CHROMA:] = self.flow
        return out


class PathIntegrator:
    """제 걸음을 더해 가는 타고난 셈 (곤충의 중심 복합체처럼): 태어난 자리를 0으로, 몸이 느낀 앞으로 / 옆으로 / 돈 것을 더한다. 세계의 좌표를 읽지 않는다.
    feel(w) -> 속마음에 주는 몸의 느낌 [6]: 방금 앞으로 / 옆으로 / 돈 것, 그리고 더해 온 자리 (x, y)와 보는 쪽 (sin, cos)은 아니고 (x/12, y/12, 보는 쪽의 sin) 대신
    (x/12, y/12, cos, sin) 를 준다 - 무엇이 어디 있었는지를 묶는 것은 속 상태가 배운다."""
    def __init__(self):
        self.pos, self.head = np.zeros(2), 0.0

    def feel(self, w):
        f, t, s_ = w._felt[0], w._felt[1], w._felt[5]
        self.head += t
        self.pos = self.pos + f * np.array([math.cos(self.head), math.sin(self.head)]) + s_ * np.array([-math.sin(self.head), math.cos(self.head)])
        return np.array([f / 0.4, s_ / 0.4, t / 0.42, self.pos[0] / 12, self.pos[1] / 12, math.cos(self.head), math.sin(self.head)], dtype=np.float32)


def hear(w):
    """귀 둘 (몸의 왼쪽과 오른쪽). 샘마다 제 높이의 소리를 낸다 (마실 것과 먹을 것이 다른 높이. 높이를 가르는 것은 타고난 것이고 이름이 아니다).
    들리는 크기: 멀수록 작고 (거리의 제곱에 반비례에 가깝게), 사이에 낀 벽 칸 하나마다 크게 준다 - 그래서 소리는 문으로 새어 나온다.
    소리가 나는 쪽 귀가 조금 더 크게 듣는다 (머리의 그늘). 거리도 쪽도 주지 않는다: 두 귀의 크기만 준다.
    -> [마실 것 왼쪽, 마실 것 오른쪽, 먹을 것 왼쪽, 먹을 것 오른쪽] (0~1)"""
    side = np.array([-math.sin(w.head), math.cos(w.head)])            # 오른쪽 (몸 기준. 화면에서 오른쪽이 +)
    out = []
    for kind in (WATER, FOOD):
        at = np.argwhere(w.map == kind)
        if not len(at):
            out += [0.0, 0.0]; continue
        src = at[0] + 0.5
        for ear in (-1, 1):
            p = w.pos + ear * 0.25 * side
            d = src - p; r = float(np.hypot(*d))
            cells = set()
            for f in np.linspace(0, 1, max(int(r / 0.2), 2))[1:-1]:
                q = p + f * d
                if w.map[int(q[0]), int(q[1])] == 1:
                    cells.add((int(q[0]), int(q[1])))
            shade = 0.6 + 0.4 * max(0.0, float(np.dot(ear * side, d / max(r, 1e-6))))
            out.append(shade * 0.35 ** len(cells) / (1.0 + (r / 2.0) ** 2))
    return np.sqrt(np.clip(out, 0.0, 1.0)).astype(np.float32)      # 크기의 제곱근 (로그로 느끼게 했더니 8칸 밖의 소리가 바로 곁의 소리와 거의 같게 들렸다: 0.69 대 0.74)


def look_retina(ws, eyes, gaze=None):
    img = pictures(ws)
    for i, e in enumerate(eyes):
        e.step(0.0 if gaze is None else gaze[i]); e.take(img[i])


def look_all(ws, eyes, step=0.08):
    """모든 몸의 눈이 한 번 본다 (시선을 한꺼번에 쏜다)."""
    n = len(ws)
    maps = np.stack([w.map for w in ws]); pos = np.stack([w.pos for w in ws]); head = np.array([w.head for w in ws])
    offs = (np.arange(RAYS) / (RAYS - 1) - 0.5) * FOV
    ang = head[:, None] + offs[None]
    dx, dy = np.cos(ang), np.sin(ang)
    kind = np.zeros((n, RAYS), dtype=np.int8); dist = np.zeros((n, RAYS)); live = np.ones((n, RAYS), dtype=bool)
    idx = np.arange(n)[:, None]
    for r in np.arange(step, 23.0, step):
        cx = (pos[:, 0:1] + dx * r).astype(int).clip(0, 15); cy = (pos[:, 1:2] + dy * r).astype(int).clip(0, 15)
        m = maps[idx, cx, cy]
        hit = live & (m != 0)
        kind[hit] = m[hit]; dist[hit] = r; live &= ~hit
        if not live.any():
            break
    for i, e in enumerate(eyes):
        e.step(); e.take(kind[i], dist[i], offs)


LG = math.log1p(16.0)


class Seen:
    """제 몸이 본 것만 안다: 눈앞(시야 안, 가리는 것 없이)에 들어온 샘의 자리를, 제 걸음과 돈 것을 더해 간 제 자리에서 적어 둔다 (세계의 좌표를 읽지 않는다).
    무엇이 샘인지는 아직 세계가 알려 준다 (그림에서 알아내는 것은 다음 일이다)."""
    def __init__(self, w):
        self.w, self.pos, self.head = w, np.zeros(2), 0.0
        self.at = [None, None]

    def look(self):
        w = self.w
        moved, turned, side = w._felt[0], w._felt[1], w._felt[5]   # 몸이 느낀 것: 앞으로 간 거리, 돈 각, 옆으로 간 거리
        self.head += turned
        self.pos = self.pos + moved * np.array([math.cos(self.head), math.sin(self.head)]) + side * np.array([-math.sin(self.head), math.cos(self.head)])
        for j, kind in enumerate((WATER, FOOD)):
            src = np.argwhere(w.map == kind)[0] + 0.5
            d = src - w.pos; r = float(np.hypot(*d))
            off = (math.atan2(d[1], d[0]) - w.head + math.pi) % (2 * math.pi) - math.pi
            if abs(off) > FOV / 2:
                continue
            pts = w.pos + np.linspace(0, 1, max(int(r / 0.2), 2))[1:-1, None] * d
            if all(w.map[int(q[0]), int(q[1])] in (0, kind) for q in pts):
                self.at[j] = self.pos + r * np.array([math.cos(self.head + off), math.sin(self.head + off)])


def sense(w, known):
    if isinstance(known, Seen):
        out = np.zeros(DX, dtype=np.float32)
        out[0:2] = w.store; out[2] = w._felt[2]
        c, s = math.cos(known.head), math.sin(known.head)
        for j in range(2):
            if known.at[j] is not None:
                dx, dy = known.at[j] - known.pos
                put(out, j, dx * c + dy * s, -dx * s + dy * c)
        return out
    out = np.zeros(DX, dtype=np.float32)
    out[0:2] = w.store; out[2] = w._felt[2]
    c, s = math.cos(w.head), math.sin(w.head)
    for j, kind in enumerate((WATER, FOOD)):
        if known[j]:
            src = np.argwhere(w.map == kind)[0] + 0.5
            dx, dy = src - w.pos
            put(out, j, dx * c + dy * s, -dx * s + dy * c)       # 앞으로, 옆으로 (몸 기준)
    return out


def lives(net, n, seed, T, explore=0.0, near=False, relief=True, learn=True, walls=0, awake=None, recent=8, loose=0.0, seen=False, curious=0.0, roam=False, seek_tool=False, seek=False, hold=0.0, home=None, spots=0, carry=None, tire=0.0, layout=None, slow=1.0):
    rng = np.random.default_rng(seed); gen = torch.Generator().manual_seed(seed)
    ws = [Room(seed=seed * 1000 + i if layout is not None or home is None else home, walls=walls, roam=roam, spots=spots, layout=layout) for i in range(n)]      # home: 모두 같은 하나의 세계에서, 같은 자리에서 태어난다
    for w in ws:
        w.store = 0.5 + 0.5 * rng.random(2)
        w.drain = (1 / 60.0) / slow                              # slow: 저장고가 몇 배 느리게 주는가 (느리게 긴박해진다. 오래 돌아다닐 수 있다)
        if near:                                                # 처음 몇 번은 샘을 마주 보고 태어난다 (첫 풀림을 겪을 수 있게)
            src = np.argwhere(w.map == (WATER if rng.random() < 0.5 else FOOD))[0] + 0.5
            for ang in rng.permutation(8) * math.pi / 4:
                p = src - 1.0 * np.array([math.cos(ang), math.sin(ang)])
                if w.map[int(p[0]), int(p[1])] == 0:
                    w.pos, w.head = p, float(ang); break
    eye = seen in ("eye", "retina"); ret = seen == "retina"
    kn = [Retina(w) if ret else Eye(w) if eye else Seen(w) if seen else (True, True) for w in ws]
    if carry is not None and ret:                              # 앞선 삶에서 적어 둔 자리들을 이어 든다 (같은 세계, 같은 태어난 자리여야 제 걸음의 자리가 맞는다)
        for e, (sp, t_, mp, bk, rd, lk) in zip(kn, carry):
            e.spots, e.t, e.back = [[[q.copy(), a_, b_, c_] for q, a_, b_, c_ in l] for l in sp], t_, list(bk)
            if rd is not None and e.road is not None:
                e.road = rd.copy(); e.road.newlife(); e.link = lk.copy()
            if mp is not None and e.map is not None:
                e.map = mp.copy()
    look_retina(ws, kn) if ret else look_all(ws, kn) if eye else [k.look() for k in kn if seen]
    X, A, X2, OK = [], [], [], []
    turning, side = np.zeros(n, dtype=int), np.zeros(n, dtype=int)
    keepon, usual = torch.ones(n), torch.full((n,), 0.05)
    futile = torch.zeros(n, M)
    fills = np.zeros(n); pulled = toward = 0
    for t in range(T):
        x = torch.tensor(np.stack([k.sense() if eye else sense(w, k) for w, k in zip(ws, kn)]))
        a = torch.tensor(rng.integers(0, M, n))
        if relief:
            with torch.no_grad():
                sc, strong = net.pull(x)
            if tire:                                            # 헛됨이 쌓인 행동은 그만큼 덜 끌린다 (내다본 풀림이 오지 않은 만큼 쌓이고, 새어 나간다)
                sc = sc - tire * futile
            if hold and t:                                      # 붙드는 의도: 하던 것을 이어 하는 쪽으로 끌림이 더해진다. 방금 내다본 것과 다르게 되었으면 (놀랐으면) 놓는다
                sc = sc + hold * keepon[:, None] * torch.nn.functional.one_hot(A[-1], M)
            pick = sc.argmax(1) if not loose else torch.multinomial(torch.softmax(sc / loose, 1), 1, generator=gen)[:, 0]      # loose: 끌림이 비슷비슷하면 고르는 것도 느슨하다 (끌림이 뚜렷하면 그대로 간다)
            a = torch.where(strong, pick, a)
            if seek:                                            # 끌릴 데가 없으면 (떠올린 것이 지금 어디 있는지 모르면) 찾아냈던 순간들에 기댄다
                sees = net.sees(x); sk = net.seek(x)
                lost = strong & ~sees
                if sk is not None:
                    a = torch.where(lost, torch.multinomial(torch.softmax(sk / max(loose, 1e-3), 1), 1, generator=gen)[:, 0], a)
                else:
                    a = torch.where(lost, torch.tensor(rng.integers(0, M, n)), a)
            if curious:                                         # 제 내다보기가 틀릴 곳에 끌리는 것을 더한다 (끌림이 뚜렷하면 묻히고, 끌릴 데가 없으면 이것이 이끈다)
                a = torch.multinomial(torch.softmax((sc + curious * net.wonder(x)) / max(loose, 1e-3), 1), 1, generator=gen)[:, 0]
        if seek_tool and eye and not ret:                                   # 재는 도구 (존재의 것이 아니다): 모자란 것의 샘을 본 적이 없으면 막힐 때까지 앞으로 가고 막히면 돈다. 찾기에 얼마나 여지가 있는지 보려고
            for i, w in enumerate(ws):
                j = tint(COLOR[WATER + int(w.store.argmin())])
                if w.store.min() < 0.6 and kn[i].now[j] is None and not kn[i].at[j]:
                    turning[i] = (turning[i] - 1) if turning[i] > 0 else (int(rng.integers(3, 9)) if (w._felt[2] or rng.random() < 0.03) else 0)
                    a[i] = (2 + side[i]) if turning[i] > 0 else 1
                    if turning[i] == 0:
                        side[i] = int(rng.integers(0, 2))
        if explore:
            a = torch.where(torch.tensor(rng.random(n) < explore), torch.tensor(rng.integers(0, M, n)), a)
        live = np.array([not w.dead for w in ws])
        e0 = np.stack([w.store.copy() for w in ws])
        low = e0.argmin(1)
        d0 = np.array([np.linalg.norm(np.argwhere(w.map == (WATER, FOOD)[low[i]])[0] + 0.5 - w.pos) for i, w in enumerate(ws)])
        for i, w in enumerate(ws):
            w.set_motor(*(MOTOR[int(a[i])] if int(a[i]) < len(MOTOR) else MOTOR[0])); w.advance(TICK)
        e1 = np.stack([w.store for w in ws])
        d1 = np.array([np.linalg.norm(np.argwhere(w.map == (WATER, FOOD)[low[i]])[0] + 0.5 - w.pos) for i, w in enumerate(ws)])
        look_retina(ws, kn, [(-1.0 if int(v) == len(MOTOR) else 1.0 if int(v) == len(MOTOR) + 1 else 0.0) for v in a]) if ret else look_all(ws, kn) if eye else [k.look() for k in kn if seen]
        x2 = torch.tensor(np.stack([k.sense() if eye else sense(w, k) for w, k in zip(ws, kn)]))
        if tire:
            more, eased = net.unmet(x, a, x2)
            futile = futile * LEAK
            futile[torch.arange(n), a] += more
            futile[eased] = 0                                   # 풀리면 다 비운다
        if hold:
            with torch.no_grad():
                err = ((net.foresee(x).gather(1, a[:, None, None].expand(-1, 1, DX))[:, 0] - x2) ** 2).mean(1).sqrt()
            usual = 0.95 * usual + 0.05 * err
            keepon = (err < 2.0 * usual).float()
        rose = ((e1 - e0) > 0.05).any(1) & live
        fills += rose
        far = live & (e0.min(1) < 0.5) & (d0 > 2.0)
        pulled += int(far.sum()); toward += int((far & (d1 < d0 - 1e-6)).sum())
        if seek and relief and learn:
            found = lost & net.sees(x2) & torch.tensor(live)
            if found.any():
                net.keep_found(x[found], a[found])
        if learn and rose.any():
            net.keep(torch.tensor((e1 - e0)[rose], dtype=torch.float32), x[torch.tensor(rose)], a[torch.tensor(rose)])
        X.append(x); A.append(a); X2.append(x2); OK.append(torch.tensor(live, dtype=torch.float32))
        if awake is not None:                                   # 사는 동안에도 배운다: 방금 겪은 몇 걸음에서 곧바로 (awake = 그 배움을 하는 것)
            if hasattr(awake, "keep"):                           # plastic.Plastic: 방금 것과 남겨 둔 옛 것을 섞어 배우고, 느린 사본에 매여 있다
                awake.step(torch.stack(X[-recent:], 1), torch.stack(A[-recent:], 1), torch.stack(X2[-recent:], 1), torch.stack(OK[-recent:], 1))
            else:
                loss = net.loss(torch.stack(X[-recent:], 1), torch.stack(A[-recent:], 1), torch.stack(X2[-recent:], 1), torch.stack(OK[-recent:], 1))
                awake.zero_grad(); loss.backward(); awake.step()
    span = np.array([w.clock for w in ws])
    if ret:
        lives.memory = [(e.spots, e.t, e.map, e.back, e.road, e.link) for e in kn]
        lives.roads = float(np.mean([e.road.used for e in kn])) if kn[0].road is not None else 0.0
        lives.known = float(np.mean([(e.map.cell != 0).mean() for e in kn])) if kn[0].map is not None else 0.0
    st = dict(seconds_lived=round(float(span.mean()), 1), whole_at_end=round(float(np.mean([not w.dead for w in ws])), 3), refills=round(float(fills.mean()), 2),
              closer_when_lacking_and_far=round(toward / max(pulled, 1), 3))
    return (torch.stack(X, 1), torch.stack(A, 1), torch.stack(X2, 1), torch.stack(OK, 1)), st


def run(rounds, n=64, T=600, updates=1500, seed=1, walls=0, net=None, small=4096, seen=False, loose=0.0, curious=0.0, seek=False, roam=False, save="", tire=0.0):
    torch.manual_seed(seed)
    net = net or Core(DX, M, INSIDE, PLACE)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    store, t0 = [], time.time()
    for r in range(1, rounds + 1):
        batch, _ = lives(net, n, seed * 100 + r, T, explore=0.3, near=r <= 3 and int(net.rel_n) < net.k, walls=walls, seen=seen, loose=loose, curious=curious, seek=seek, roam=roam, tire=tire)
        store = (store + [batch])[-12:]
        flat = [torch.cat([d[i].flatten(0, 1) for d in store]) for i in range(4)]
        for _ in range(updates):                                # 살아 본 걸음들에서 조금씩 집어 여러 번 배운다
            i = torch.randint(0, len(flat[0]), (small,))
            loss = net.loss(*[f[i] for f in flat])
            opt.zero_grad(); loss.backward(); opt.step()
        if r % 5 == 0 or r == rounds:
            _, s = lives(net, 64, 9999, T, learn=False, walls=walls, seen=seen, loose=loose, curious=curious, seek=seek, roam=roam, tire=tire)
            n_ = int(net.rel_n)
            print(f"round {r:3d} | {time.time() - t0:5.0f}s | foresee {loss.item():.4f} | relief memories {n_}: which store {net.rel_de[:n_].argmax(1).bincount(minlength=2).tolist()}, acts {net.rel_a[:n_].bincount(minlength=M).tolist()} | {s}", flush=True)
            if save:                                            # 자라는 중에도 들여다볼 수 있게 (뷰어가 따라온다)
                torch.save(net.state_dict(), save)
    return net


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--walls", type=int, default=0)
    a = ap.parse_args()
    net = run(a.rounds, walls=a.walls)
    for name, kw in (("as grown", {}), ("relief memory not used (wanders)", dict(relief=False))):
        _, s = lives(net, 128, 4242, 600, learn=False, walls=a.walls, **kw)
        print(f"  {name:34s} {s}")
