"""A laptop body without hardware, for the LOKI bridge tests and the CPU 'mock laptop' runs (judge-side tool, never in the being).
Same channel kinds, sizes, shuffles and lever fence as atanor_os.emerge.laptop_body.LaptopBody (its raw / step / available / wiring code is reused);
only the devices are synthetic:
  heat: 8 thermal zones lag (time constant 60 s of simulated time) toward a target set by the own-tempo lever (load) + slow outside drift;
  readbacks: backlight (0..255, 8% steps, floor 10%), volume (0..0.6), gap; sound tones raise one microphone band in proportion to volume;
  camera: 288x512 scene lit partly by the screen (backlight) with a slowly drifting bright patch; accelerometer and inputs are noise.
Time: each beat advances simulated time by the gap the being chose (no real sleeping needed: set real_time=False)."""
import random, numpy as np, torch
from atanor_os.emerge import laptop_body as LBm
from atanor_os.emerge.body_contract import Channel, Contract


class _Nerve:
    def __init__(self):
        self.ok, self.last = True, b"x"


class MockLaptop(LBm.LaptopBody):
    def __init__(self, seed=0, quiet=None, world_seed=None):
        self.n, self.dev, self.quiet = 1, "cpu", quiet
        rng = random.Random(seed); self.np = np.random.default_rng(seed + 1000 if world_seed is None else world_seed)
        self.thermal = [f"zone{i}" for i in range(8)]; self.bl, self.bl_max, self.bat, self.ac, self.lid = "bl", 255.0, None, None, None
        self.cam, self.mic = _Nerve(), _Nerve()
        self.T = np.full(8, 60000.0) + self.np.normal(0, 2000, 8); self.bright = 128.0; self.tone = 0; self.clock = 0.0; self.patch = 0.0
        self.gap = 0.15; self.vol = 0.3; self.sound_until = 0.0
        phys = [("insides: heat margin, energy margin", "vector", 2, self._bedrock, True),
                ("sysfs: thermal zones, battery, ac, mem avail, load", "vector", len(self.thermal) + 4, self._sysfs, False),
                ("accelerometer xyz", "vector", 3, self._accel, False),
                ("lid", "vector", 1, lambda: np.array([1.0]), False),
                ("input counts: keyboard presses, touchpad events", "vector", 2, self._inputs, False),
                ("lever readback: backlight, volume, gap", "vector", 3, self._readback, False),
                ("microphone 16 bands", "audio", 16, self._hear, False),
                ("camera", "image", None, self._see, False)]
        order = list(range(len(phys))); rng.shuffle(order)                  # identical shuffle code to LaptopBody
        self.phys = [phys[i] for i in order]
        self.perm = {}
        for k, p in enumerate(self.phys):
            if p[1] in ("vector", "audio") and not p[4]:
                pm = list(range(p[2])); rng.shuffle(pm); self.perm[k] = pm
        sensors = [Channel("image", (LBm.H, LBm.W, 3)) if p[1] == "image" else Channel(p[1], (p[2],), bedrock=p[4]) for p in self.phys]
        levers = [("backlight down/hold/up", 3, self._backlight), ("volume down/hold/up", 3, self._volume),
                  ("sound: silent / tone 1-4", 5, self._sound), ("own tempo slow/normal/fast", 3, self._tempo)]
        lo = list(range(len(levers))); rng.shuffle(lo)
        self.levers = [levers[i] for i in lo]
        self.opt = [list(range(lv[1])) for lv in self.levers]
        for o in self.opt:
            rng.shuffle(o)
        self.c = Contract(sensors=sensors, motors=[lv[1] for lv in self.levers])
        self.combos, _ = self.c.codes()
        self.dead = torch.zeros(1, dtype=torch.bool)

    # ---- synthetic devices ----
    def heat(self):
        return float(self.T.max() / 1000)

    def _sysfs(self):
        load = {0.5: 0.6, 0.15: 1.6, 0.05: 3.4}.get(self.gap, 0.3)
        return np.concatenate([np.round(self.T / 1000) * 1000, [0, 0, 0.6 + self.np.normal(0, 0.01), load + self.np.normal(0, 0.2)]])

    def _accel(self):
        return np.array([174.0, -3.0, 1154.0]) + self.np.normal(0, [22, 14, 48])

    def _inputs(self):
        return self.np.poisson(0.1, 2).astype(float)

    def _readback(self):
        return np.array([self.bright, self.vol, self.gap])

    def _hear(self):
        if not self.mic.ok:
            return np.full(16, -9.0)
        b = -3.0 + self.np.normal(0, 0.7, 16)
        if self.tone:
            b[5 + 2 * self.tone] += 6.0 * self.vol / 0.6
        return b

    def _see(self):
        if not self.cam.ok:
            return np.zeros((LBm.H, LBm.W, 3), dtype=np.uint8)
        y, x = np.mgrid[0:LBm.H, 0:LBm.W]
        base = 50 + 70 * self.bright / 255.0
        cx = (0.5 + 0.35 * np.sin(self.patch)) * LBm.W
        img = base + 80 * np.exp(-((x - cx) ** 2 + (y - LBm.H * 0.4) ** 2) / (2 * 40.0 ** 2))
        img = np.stack([img, img * 0.9, img * 0.8], 2) + self.np.normal(0, 3, (LBm.H, LBm.W, 3))
        return np.clip(img, 0, 255).astype(np.uint8)

    def _backlight(self, o):
        if o != 1:
            self.bright = float(np.clip(self.bright + (o - 1) * 0.08 * self.bl_max, 0.10 * self.bl_max, self.bl_max))

    def _volume(self, o):
        if o != 1:
            self.vol = float(np.clip(self.vol + (o - 1) * 0.05, 0.0, 0.6))

    def _sound(self, o):
        self.tone = int(o)

    def step(self, act, dt=None):
        out = super().step(act, dt)
        dt = self.gap; self.clock += dt; self.patch += 0.02 * dt / 0.15
        load = {0.5: 0.2, 0.15: 0.55, 0.05: 0.95}.get(self.gap, 0.1)
        target = 48000 + 38000 * load + 4000 * np.sin(self.clock / 900) + np.arange(8) * 600
        self.T += (target - self.T) * (1 - np.exp(-dt / 60.0)) + self.np.normal(0, 150, 8)
        return out

    def wiring(self):
        return super().wiring()
