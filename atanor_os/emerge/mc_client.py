"""윈도 쪽에서 MineDojo 세계 서버(mc_server.py, WSL2)에 붙는 몸. 존재는 이것으로만 세계를 겪는다.
  w = MCWorld(); obs = w.obs  # 몸마다: ret [27, 16, 16] (정본 눈: 가운데 + 고리 4 + 센 빛깔 4), gaze (x, y), health / food / air (0~1), hurt, done, 제 걸음 (앞, 옆, 돈 각) 은 step이 계산
  obs = w.step(acts, who="존재", label="...")
"""
import math, pickle, socket, struct
import numpy as np

M = 7
ACTS = ["가만히", "앞으로", "왼쪽으로 돌기", "오른쪽으로 돌기", "때리기", "쓰기", "뛰며 앞으로"]


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
    def __init__(self, host="127.0.0.1", port=8790, timeout=600):
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

    def step(self, acts, who="", label=""):
        _send(self.s, dict(acts=[int(a) for a in acts], who=who, label=label))
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

    def pics(self):
        """눈이 받는 것 [n, 27, 16, 16] (가운데 + 고리 4 + 고리의 센 빛깔 4, 각 RGB, 0~1)."""
        return np.stack([o["ret"] for o in self.obs]).astype(np.float32)

    def gaze(self):
        return np.array([o["gaze"] for o in self.obs], dtype=np.float32)

    def inside(self):
        return np.array([[o["health"], o["food"], o["air"]] for o in self.obs], dtype=np.float32)

    def close(self):
        self.s.close()
