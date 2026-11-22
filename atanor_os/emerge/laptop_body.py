"""노트북 몸 (배관 + 안전 울타리만; LAPTOP_BODY.md 0절 '설명표 없는 날것 연결 + 스스로 숙주 체화', 창립자 2026-11-22).
존재가 받는 것: 이름 없는 신호 칸들 (꼴과 크기만)과 이름 없는 몸짓 갈래들. 칸·칸 안의 차례·갈래·갈래 안의 고름 차례는 태어날 때 씨앗으로 섞는다.
진짜 배선 (어느 칸이 어느 장치, 어느 갈래가 어느 레버)은 wiring()으로 재는 쪽에만 준다 - 마음 코드는 이것을 읽지 않는다.
바탕 칸 (보편 계약의 유일한 뜻 표시, 진화된 것) = 하드웨어 울타리: 과열·전원 고갈에 가까워질수록 모자람으로 내려감. 무엇인지는 존재에게 알리지 않음.
울타리 (몸짓을 대신 고르지 않고 바닥만 막음): 밝기 바닥, 소리 크기 상한, 조용한 시간 (23-8시)엔 소리 레버가 소리를 내지 않음, 과열이면 셈을 늦춤.
카메라·마이크: 메모리에서만 (날것을 저장하지 않고 노트북 밖으로 보내지 않음 - 뷰어는 존재가 보는 낮은 해상도만). 접근 권한이 없으면 그 칸은 0 (어둠·고요).
입력 장치는 사건의 수만 센다 (무엇을 쳤는지 읽지 않음)."""
import glob, os, random, struct, subprocess, threading, time
import numpy as np
import torch
from .body_contract import Channel, Contract

H, W = 288, 512                                           # 창립자 2026-11-22 '눈도 더 디테일하게': 중심와 16px가 전의 절반 각도 -> 가운데가 두 배 또렷 (고리는 정본대로 256px까지, 그 밖은 큰세포 길)
SR = 16000


def _read(p, default=0.0):
    try:
        return float(open(p).read().strip().split()[0])
    except Exception:
        return default


class _Proc(threading.Thread):
    """밖의 프로그램이 내는 날 바이트를 받아 마지막 것만 쥔다 (카메라 그림, 마이크 소리)."""

    def __init__(self, cmd, chunk):
        super().__init__(daemon=True); self.cmd, self.chunk, self.last, self.ok, self.p = cmd, chunk, None, False, None

    def run(self):
        while True:
            try:
                self.p = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                while True:
                    b = self.p.stdout.read(self.chunk)
                    if not b or len(b) < self.chunk:
                        break
                    self.last, self.ok = b, True
            except Exception:
                pass
            self.ok = False; time.sleep(30)                                  # 장치가 없거나 권한이 없으면 30초 뒤 다시


class _Speaker(threading.Thread):
    """스피커 하나를 계속 열어 두고 (소리 낼 때마다 프로그램을 새로 띄우지 않음 - 그 셈이 몸을 데웠다), 실제 시간 박자로 20ms 조각을 흘린다.
    낼 소리가 없으면 고요를 흘린다."""

    def __init__(self):
        super().__init__(daemon=True); self.q = b""; self.lock = threading.Lock()

    def say(self, pcm):
        with self.lock:
            self.q = pcm                                                    # 앞 소리가 남았으면 새 소리로 바꿈

    def run(self):
        chunk = SR // 50 * 2; silence = bytes(chunk)
        while True:
            try:
                p = subprocess.Popen(["pw-cat", "--playback", "--raw", "--format", "s16", "--rate", str(SR), "--channels", "1", "-"],
                                     stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
                t0 = time.time(); k = 0
                while True:
                    with self.lock:
                        b, self.q = (self.q[:chunk], self.q[chunk:]) if self.q else (silence, b"")
                    p.stdin.write(b.ljust(chunk, b"\0")); p.stdin.flush(); k += 1
                    d = t0 + k * 0.02 - time.time()
                    if d > 0:
                        time.sleep(d)
            except Exception:
                time.sleep(10)


class _Events(threading.Thread):
    """입력 장치의 사건 수만 센다 (EV_KEY 누름, 터치패드 움직임). 키 번호 (무엇을 쳤나)는 읽고 버린다."""

    def __init__(self, path, kinds):
        super().__init__(daemon=True); self.path, self.kinds, self.n = path, kinds, 0

    def run(self):
        try:
            f = open(self.path, "rb", buffering=0)
            while True:
                b = f.read(24)
                if len(b) == 24:
                    _, _, typ, _, val = struct.unpack("qqHHi", b)
                    if typ in self.kinds and (typ != 1 or val == 1):
                        self.n += 1
        except Exception:
            pass


def _find_input(word):
    try:
        txt = open("/proc/bus/input/devices").read().split("\n\n")
        for blk in txt:
            if word.lower() in blk.lower():
                for tok in blk.split():
                    if tok.startswith("event"):
                        return "/dev/input/" + tok
    except Exception:
        pass
    return None


class LaptopBody:
    def __init__(self, seed=0, quiet=None):                  # 조용한 시간 울타리는 창립자가 풂 (2026-11-22); 소리 크기 상한은 남김
        self.n, self.dev, self.quiet = 1, "cpu", quiet
        rng = random.Random(seed)
        self.thermal = sorted(glob.glob("/sys/class/thermal/thermal_zone*/temp"))
        self.bl = (glob.glob("/sys/class/backlight/*") or [None])[0]
        self.bl_max = _read(self.bl + "/max_brightness", 1.0) if self.bl else 1.0
        self.bat = (glob.glob("/sys/class/power_supply/BAT*") or [None])[0]
        self.ac = (glob.glob("/sys/class/power_supply/A*") or [None])[0]
        self.acc = "/sys/devices/platform/lis3lv02d/position"
        self.lid = (glob.glob("/proc/acpi/button/lid/*/state") or [None])[0]
        self.kb = _Events(_find_input("keyboard") or "/nonexistent", (1,)); self.kb.start()
        self.tp = _Events(_find_input("touchpad") or "/nonexistent", (2, 3)); self.tp.start()
        self.cam = _Proc(["ffmpeg", "-loglevel", "quiet", "-f", "v4l2", "-framerate", "10", "-video_size", "640x360", "-i", "/dev/video0",
                          "-vf", f"scale={W}:{H}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], H * W * 3); self.cam.start()
        self.mic = _Proc(["ffmpeg", "-loglevel", "quiet", "-f", "alsa", "-i", "default", "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"], SR // 5 * 2); self.mic.start()
        self.gap = 0.15; self.vol = 0.3; self.sound_until = 0.0
        self.spk = _Speaker(); self.spk.start()
        # 날 장치 칸 (진짜 이름은 재는 쪽만): (이름, 꼴, 크기, 읽는 함수)
        phys = [("insides: heat margin, energy margin", "vector", 2, self._bedrock, True),
                ("sysfs: thermal zones, battery, ac, mem avail, load", "vector", len(self.thermal) + 4, self._sysfs, False),
                ("accelerometer xyz", "vector", 3, self._accel, False),
                ("lid", "vector", 1, lambda: np.array([1.0 if self.lid and "open" in open(self.lid).read() else 0.0]), False),
                ("input counts: keyboard presses, touchpad events", "vector", 2, self._inputs, False),
                ("lever readback: backlight, volume, gap", "vector", 3, self._readback, False),
                ("microphone 16 bands", "audio", 16, self._hear, False),
                ("camera", "image", None, self._see, False)]
        order = list(range(len(phys))); rng.shuffle(order)
        self.phys = [phys[i] for i in order]
        self.perm = {}                                                       # 칸 안의 차례도 섞음 (벡터·소리)
        for k, p in enumerate(self.phys):
            if p[1] in ("vector", "audio") and not p[4]:
                pm = list(range(p[2])); rng.shuffle(pm); self.perm[k] = pm
        sensors = [Channel("image", (H, W, 3)) if p[1] == "image" else Channel(p[1], (p[2],), bedrock=p[4]) for p in self.phys]
        levers = [("backlight down/hold/up", 3, self._backlight), ("volume down/hold/up", 3, self._volume),
                  ("sound: silent / tone 1-4", 5, self._sound), ("own tempo slow/normal/fast", 3, self._tempo)]
        lo = list(range(len(levers))); rng.shuffle(lo)
        self.levers = [levers[i] for i in lo]
        self.opt = [list(range(lv[1])) for lv in self.levers]
        for o in self.opt:
            rng.shuffle(o)                                                   # 갈래 안의 고름 차례도 섞음 (0 = '그대로'가 아님)
        self.c = Contract(sensors=sensors, motors=[lv[1] for lv in self.levers])
        self.combos, _ = self.c.codes()
        self.dead = torch.zeros(1, dtype=torch.bool)

    # ---- 재는 쪽만 ----
    def wiring(self):
        return dict(sensors=[dict(slot=k, device=p[0], kind=p[1], within_order=self.perm.get(k)) for k, p in enumerate(self.phys)],
                    motors=[dict(branch=j, lever=lv[0], option_order=self.opt[j]) for j, lv in enumerate(self.levers)])

    def heat(self):
        return max([_read(t) / 1000 for t in self.thermal] or [50.0])

    # ---- 읽기 ----
    def _bedrock(self):
        heat = float(np.clip((95.0 - self.heat()) / 30.0, 0, 1))                      # 울타리: 95도에 가까울수록 모자람
        on_ac = _read(self.ac + "/online", 1.0) if self.ac else 1.0
        energy = 1.0 if on_ac >= 1 else float(np.clip(_read(self.bat + "/capacity", 100) / 100, 0, 1))
        return np.array([heat, energy])

    def _sysfs(self):
        mem = dict(l.split(":") for l in open("/proc/meminfo").read().strip().split("\n"))
        avail = float(mem["MemAvailable"].split()[0]) / float(mem["MemTotal"].split()[0])
        return np.array([_read(t) for t in self.thermal] + [_read(self.bat + "/capacity") if self.bat else 0, _read(self.ac + "/online") if self.ac else 0,
                                                             avail, os.getloadavg()[0]])

    def _accel(self):
        try:
            return np.array([float(v) for v in open(self.acc).read().strip("()\n ").split(",")])
        except Exception:
            return np.zeros(3)

    def _inputs(self):
        a, b = self.kb.n, self.tp.n; self.kb.n = self.tp.n = 0
        return np.array([float(a), float(b)])

    def _readback(self):
        return np.array([_read(self.bl + "/brightness") if self.bl else 0, self.vol, self.gap])

    def _hear(self):
        if not self.mic.ok or self.mic.last is None:
            return np.full(16, -9.0)                                         # 고요 (방·마크 귀의 '소리 없음'과 같은 값)
        s = np.frombuffer(self.mic.last, dtype=np.int16).astype(np.float32) / 32768
        sp = np.abs(np.fft.rfft(s * np.hanning(len(s)))) ** 2
        edges = np.geomspace(2, len(sp) - 1, 17).astype(int)
        return np.log(np.array([sp[edges[i]:max(edges[i + 1], edges[i] + 1)].mean() for i in range(16)]) + 1e-4)

    def _see(self):
        if not self.cam.ok or self.cam.last is None:
            return np.zeros((H, W, 3), dtype=np.uint8)                       # 어둠
        return np.frombuffer(self.cam.last, dtype=np.uint8).reshape(H, W, 3)

    def raw(self):
        out = []; self.last_phys = []
        for k, p in enumerate(self.phys):
            v = p[3](); self.last_phys.append(v)                         # 섞기 전 값 (재는 쪽·뷰어만)
            if k in self.perm:
                v = v[self.perm[k]]
            out.append(torch.as_tensor(np.ascontiguousarray(v), dtype=torch.float32)[None])
        return out

    # ---- 레버 (울타리 안에서만) ----
    def _backlight(self, o):
        if not self.bl or o == 1:
            return
        cur = _read(self.bl + "/brightness"); new = int(np.clip(cur + (o - 1) * 0.08 * self.bl_max, 0.10 * self.bl_max, self.bl_max))   # 바닥 10%
        subprocess.run(["sudo", "-n", "/usr/bin/brightnessctl", "-q", "set", str(new)], stderr=subprocess.DEVNULL)

    def _volume(self, o):
        if o == 1:
            return
        self.vol = float(np.clip(self.vol + (o - 1) * 0.05, 0.0, 0.6))                                 # 상한 0.6

    def _sound(self, o):
        hr = time.localtime().tm_hour; q0, q1 = self.quiet or (0, 0)
        if o == 0 or (self.quiet and (hr >= q0 or hr < q1)) or time.time() < self.sound_until:
            return
        f = [330, 440, 587, 784][o - 1]; t = np.arange(int(SR * 0.18)) / SR
        w = (np.sin(2 * np.pi * f * t) * np.hanning(len(t)) * self.vol * 0.5 * 32767).astype(np.int16).tobytes()
        self.spk.say(w); self.sound_until = time.time() + 0.25

    def _tempo(self, o):
        self.gap = [0.5, 0.15, 0.05][o]

    def step(self, act, dt=None):
        a = int(act[0]) if torch.is_tensor(act) else int(act)
        combo = self.combos[a]
        self.last_opts = [self.opt[j][combo[j]] for j in range(len(self.levers))]
        for j, lv in enumerate(self.levers):
            lv[2](self.last_opts[j])                                     # 섞인 고름 -> 진짜 레버 고름 (배관 안에서만)
        if self.heat() > 90:
            self.gap = max(self.gap, 1.0)                                     # 울타리: 과열이면 셈을 늦춤 (존재에게 알리지 않는 법칙)
        return torch.zeros(1, 2, dtype=torch.bool)
