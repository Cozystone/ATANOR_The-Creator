"""윈도 쪽에서 MineDojo 세계 서버(mc_server.py, WSL2)에 붙는 몸. 존재는 이것으로만 세계를 겪는다.
  w = MCWorld(); obs = w.obs  # 몸마다: ret [27, 16, 16] (정본 눈: 가운데 + 고리 4 + 센 빛깔 4), gaze (x, y), health / food / air (0~1), hurt, done, 제 걸음 (앞, 옆, 돈 각) 은 step이 계산
  obs = w.step(acts, who="존재", label="...")
"""
import math, os, pickle, socket, struct
import numpy as np

M = 7
ACTS = ["가만히", "앞으로", "왼쪽으로 돌기", "오른쪽으로 돌기", "때리기", "쓰기", "뛰며 앞으로"]

# 갈래 몸 (창립자 2026-11-22: 자유도를 넓혀야 창발의 폭도 넓다): 다리 x 목 x 손을 한꺼번에. 몸짓 번호 = 다리*15 + 목*3 + 손 (세계 서버 ATANOR_MC_BRANCH=1)
LEGS = ["서 있음", "앞", "뒤", "왼옆", "오른옆", "뛰며 앞"]
NECK = ["그대로", "왼쪽 30도", "오른쪽 30도", "위 15도", "아래 15도"]
HANDS = ["빈손", "때림", "씀"]
BM = len(LEGS) * len(NECK) * len(HANDS)
BRANCH = [(l, k, h) for l in range(len(LEGS)) for k in range(len(NECK)) for h in range(len(HANDS))]
BCODES = np.zeros((BM, len(LEGS) + len(NECK) + len(HANDS)), np.float32)      # 몸짓의 부호: 갈래마다 한 칸씩 켬 (알맹이가 갈래의 공통을 나눠 배운다)
for _i, (_l, _k, _h) in enumerate(BRANCH):
    BCODES[_i, _l] = BCODES[_i, len(LEGS) + _k] = BCODES[_i, len(LEGS) + len(NECK) + _h] = 1
BNAMES = [f"{LEGS[l]}/{NECK[k]}/{HANDS[h]}" for l, k, h in BRANCH]
# 옛 7가지로 비춰 보기 (상상의 내다보기가 옛 7가지로 배워져 있어서, 재는 것에도): 손이 먼저, 그다음 목의 좌우, 그다음 다리
BOLD7 = np.array([5 if h == 2 else 4 if h == 1 else 2 if k == 1 else 3 if k == 2 else 6 if l == 5 else 1 if l == 1 else 0 for l, k, h in BRANCH])


def _send(s, obj):
    b = pickle.dumps(obj, protocol=4); s.sendall(struct.pack("!Q", len(b)) + b)


def _recvall(s, n):
    buf = bytearray()
    while len(buf) < n:
        c = s.recv(min(1 << 20, n - len(buf)))
        if not c:
            raise ConnectionError("closed")
        buf += c
    return bytes(buf)


def _recv(s):
    n = struct.unpack("!Q", _recvall(s, 8))[0]; return pickle.loads(_recvall(s, n))


class MCWorld:
    def __init__(self, host="127.0.0.1", port=None, timeout=180):          # 세계가 3분 넘게 답이 없으면 멈춘 것으로 보고 오류를 낸다 (지키는 스크립트가 다시 띄움)
        port = port or int(os.environ.get("ATANOR_MC_PORT", "8790"))          # 시험 세계는 다른 포트에
        self.s = socket.create_connection((host, port), timeout=timeout); self.s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        hello = _recv(self.s); self.n = hello["n"]; self.obs = hello["obs"]
        self.felt = np.zeros((self.n, 3), dtype=np.float32)

    def _moved(self, old, new):
        """제 걸음 감각 (몸이 느끼는 것): 바라보던 쪽으로 얼마나 / 옆으로 얼마나 / 얼마나 돌았나. 절대 위치는 쓰지 않는다 (장부: 위치 변화만 몸 감각으로)."""
        out = np.zeros((self.n, 3), dtype=np.float32)
        for i, (o, q) in enumerate(zip(old, new)):
            d = np.asarray(q["pos"])[[0, 2]] - np.asarray(o["pos"])[[0, 2]]
            yaw = math.radians(o["yaw"]); f = np.array([-math.sin(yaw), math.cos(yaw)])      # Minecraft: yaw 0 = +z
            out[i] = [float(d @ f), float(d @ np.array([-f[1], f[0]])), math.radians(((q["yaw"] - o["yaw"] + 180) % 360) - 180)]
        return out

    def step(self, acts, who="", label="", depth=None):
        _send(self.s, dict(acts=[int(a) for a in acts], who=who, label=label, **({"depth": depth} if depth else {})))
        new = _recv(self.s); self.felt = self._moved(self.obs, new); self.obs = new
        return new

    def reset(self, which):
        _send(self.s, dict(cmd="reset", which=list(which)))
        got = _recv(self.s)
        for i, o in zip(which, got):
            self.obs[i] = o; self.felt[i] = 0
        return got

    def move(self, which, r=400, label=""):
        """실험자가 몸을 땅 위 아무 데로 옮김 (그림을 고르게 모을 때만. 존재가 살 때는 쓰지 않는다)."""
        _send(self.s, dict(cmd="move", which=list(which), r=r, label=label))
        got = _recv(self.s)
        for i, o in zip(which, got):
            self.obs[i] = o; self.felt[i] = 0
        return got

    def feed(self, which, item="bread", qty=8, slot="slot.hotbar.0"):
        """부모가 손에 무언가를 쥐여 준다 (돌봄 단계, 장부, 차츰 뗌): 먹을 것, 또는 도구. slot: 주 손 slot.hotbar.0 / 다른 손 slot.weapon.offhand."""
        _send(self.s, dict(cmd="feed", which=list(which), item=item, qty=qty, slot=slot))
        got = _recv(self.s)
        for i, o in zip(which, got):
            self.obs[i] = o
        return got

    def voice(self, which, wav, text=""):
        """어른의 목소리 (모노 44.1kHz 소리 파형)를 그 몸들의 귀에 들려준다. text는 뷰어에 보일 뿐, 존재는 소리만 듣는다."""
        _send(self.s, dict(cmd="voice", which=list(which), wav=np.asarray(wav, np.float32), text=text))
        return _recv(self.s)

    def pics(self):
        """눈이 받는 것 [n, 27, 16, 16] (가운데 + 고리 4 + 고리의 센 빛깔 4, 각 RGB, 0~1)."""
        return np.stack([o["ret"] for o in self.obs]).astype(np.float32)

    def small(self):
        """깊이 눈이 받는 작은 온 그림 [n, 3, 48, 80] (0~1)."""
        return np.stack([o["small"] for o in self.obs]).transpose(0, 3, 1, 2).astype(np.float32) / 255.0

    def where(self):
        """[n, 5]: 자리 x, y, z, 고개 yaw, pitch (몸이 제 움직임을 느끼는 것 - 깊이 눈의 '내가 얼마나 움직였나'에만 쓴다, 장부)."""
        return np.array([[*np.asarray(o["pos"]).ravel()[:3], o["yaw"], o.get("pitch", 0.0)] for o in self.obs], dtype=np.float32)

    def gaze(self):
        return np.array([o["gaze"] for o in self.obs], dtype=np.float32)

    def inside(self):
        return np.array([[o["health"], o["food"], o["air"]] for o in self.obs], dtype=np.float32)

    def close(self):
        self.s.close()
