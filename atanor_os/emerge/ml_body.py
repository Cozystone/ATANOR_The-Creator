"""MineLand body (several bodies on one Minecraft server) under the universal body contract - nerves only, no meanings matched (founder 2026-11-23:
'다리를 대 주지 말고 적응력을 보자'). The world declares only signal slots, motor branches and which slot is the bedrock; slot order, order within a slot,
branch order and option order are shuffled per body seed (as on the laptop), so nothing lines up with the room by hand. The being finds its body by living.
Physical slots (judge side only): bedrock [life, food, oxygen] / 20; hurt (life lost this step, life units, clipped 5); own motion like a vestibular
organ (forward, sideways, up/down, turning, nodding - 2026-11-23);
head tilt; eye picture 160 x 256. Physical levers: legs (stand, forward, back, left, right, sprint forward), head (still, yaw -30, yaw +30, pitch -15, pitch +15),
hand (still, use, attack, dig) - MineLand's own low-level options. Care (recorded): a stone pickaxe in hand, inventory kept on death.
  B = MineLandBody(seed=1); B.raw(); B.step(act_tensor); B.dead; B.reset(which)"""
import hashlib, json, math, socket, struct
import numpy as np, torch
from .body_contract import Channel, Contract

LEGS = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 1, 0), (0, 2, 0), (1, 0, 1)]          # (forward/back, left/right, jump/sneak/sprint): last = jump forward (as the old Minecraft body; was sprint by my mistake)
HEAD = [(12, 12), (12, 10), (12, 14), (11, 12), (13, 12)]                           # (pitch, yaw) units of 15 degrees around 12
HAND = [0, 1, 3, 8]                                                               # 0 still, 1 use, 3 attack, 8 dig (2026-11-23: with a pickaxe given as care)
NECK_EVERY = 4
NECK_LIMIT = math.radians(60)                                                       # neck range (body trait): beyond this a nod stops                                                                      # resting neck tone (body trait): 0 = off                                                                    # 0 still, 1 use, 3 attack


def _recvall(s, n):
    buf = bytearray()
    while len(buf) < n:
        c = s.recv(n - len(buf))
        if not c:
            raise ConnectionError("world closed")
        buf += c
    return bytes(buf)


class MineLandBody:
    def __init__(self, seed=1, host="127.0.0.1", port=8792, device="cuda", path=False):
        self.dev = device
        self.s = socket.create_connection((host, port), timeout=600); self.s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        hello, pics = self._recv(); self.n, self.H, self.W = hello["n"], hello["h"], hello["w"]; self.valley, self.names = hello["valley"], hello["names"]
        self._take(hello["obs"], pics); self.health = None
        self.s.settimeout(90)                                       # a step that does not come back in 90 s is a hung world (the guard handles it)
        rng = np.random.default_rng(seed)
        self.phys = [("bedrock", "vector", 3, True), ("hurt", "vector", 1, False), ("motion", "vector", 5, False), ("tilt", "vector", 1, False),
                     ("eye", "image", (self.H, self.W, 3), False)]
        order = list(range(len(self.phys))); rng.shuffle(order); self.order = order
        self.perm = {}
        for k, i in enumerate(order):
            p = self.phys[i]
            if p[1] == "vector" and p[2] > 1:
                pm = list(range(p[2])); rng.shuffle(pm); self.perm[k] = pm
        self.levers = [("legs", 6), ("head", 5), ("hand", 4)]
        lo = list(range(3)); rng.shuffle(lo); self.lever_order = lo
        self.opt = [list(range(self.levers[j][1])) for j in lo]
        for o in self.opt:
            rng.shuffle(o)
        sensors = [Channel(self.phys[i][1], (self.phys[i][2],) if self.phys[i][1] == "vector" else self.phys[i][2], self.phys[i][3]) for i in order]
        if path:                                                        # MC_GOING_PROTOCOL.md: the body declares where forward, sideways and turn sit in its motion signal
            k = order.index(2); pm = self.perm.get(k, list(range(5))); sensors[k] = Channel("vector", (5,), False, motion=(pm.index(0), pm.index(1), pm.index(3)))
        self.c = Contract(sensors=sensors, motors=[self.levers[j][1] for j in lo])
        self.combos, _ = self.c.codes()
        self.dead = torch.zeros(self.n, dtype=torch.bool, device=device)
        self.felt = np.zeros((self.n, 5), np.float32); self.hurt = np.zeros(self.n, np.float32)

    def wiring_id(self):
        key = json.dumps([self.order, {str(k): v for k, v in self.perm.items()}, self.lever_order, self.opt])
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def judge_wiring(self):
        return dict(sensors=[dict(slot=k, device=self.phys[i][0], within_order=self.perm.get(k)) for k, i in enumerate(self.order)],
                    motors=[dict(branch=b, lever=self.levers[j][0], option_order=self.opt[b]) for b, j in enumerate(self.lever_order)])

    # ---- wire ----
    def _recv(self):
        hl, nb = struct.unpack("!II", _recvall(self.s, 8)); obj = json.loads(_recvall(self.s, hl)); blobs = []
        for _ in range(nb):
            (ln,) = struct.unpack("!I", _recvall(self.s, 4)); blobs.append(_recvall(self.s, ln))
        return obj, blobs

    def _send(self, obj):
        head = json.dumps(obj).encode(); self.s.sendall(struct.pack("!II", len(head), 0) + head)

    def _take(self, obs, pics):
        self.obs = obs
        self.pics = [np.frombuffer(b, np.uint8).reshape(self.H, self.W, 3) for b in pics]

    # ---- body ----
    def lowlevel(self, act):
        """combo number -> MineLand's 8 low-level numbers (through the shuffled branch and option order)."""
        cb = self.combos[int(act)]; phys = [0, 0, 0]
        for b, j in enumerate(self.lever_order):
            phys[j] = self.opt[b][cb[b]]
        lg, hd, hn = LEGS[phys[0]], HEAD[phys[1]], HAND[phys[2]]
        return [lg[0], lg[1], lg[2], hd[0], hd[1], hn, 0, 0]

    def relax(self, ll, i):
        """Body trait (designer-given, 2026-11-23, founder approved): the neck has a resting tone. When this step's head option is 'still',
        every NECK_EVERY steps the head sinks one 15-degree unit back toward level (head weight + muscle tone). Not a reflex: no reaction to anything
        sensed, only the body's own mechanics while the being does not move its head."""
        if ll[3] == 12 and ll[4] == 12:
            self.still[i] += 1
            if self.still[i] % NECK_EVERY == 0:
                p = self.obs[i]["pitch"]
                if p > 0.2:
                    ll[3] = 11
                elif p < -0.2:
                    ll[3] = 13
        else:
            self.still[i] = 0
        return ll

    def step(self, act, label=None, text=None):
        old = self.obs
        if not hasattr(self, "still"):
            self.still = [0] * self.n
        lls = [self.relax(self.lowlevel(a), i) if NECK_EVERY else self.lowlevel(a) for i, a in enumerate(act.tolist())]
        for i, ll in enumerate(lls):                                   # joint limit (body trait, like a human neck): no nod past +-60 degrees
            if abs(self.obs[i]["pitch"] + math.radians((ll[3] - 12) * 15)) > NECK_LIMIT and ll[3] != 12:
                past = abs(self.obs[i]["pitch"] + math.radians((ll[3] - 12) * 15)) > abs(self.obs[i]["pitch"])
                if past:
                    ll[3] = 12
        self._send(dict(cmd="step", acts=lls, label=label, text=text)); o, p = self._recv(); self._take(o["obs"], p); self.health = o.get("health")
        for i, (a, b) in enumerate(zip(old, self.obs)):
            dx, dz = b["pos"][0] - a["pos"][0], b["pos"][2] - a["pos"][2]; yaw = a["yaw"]
            fwd = -dx * math.sin(yaw) - dz * math.cos(yaw)                   # mineflayer: looking along (-sin yaw, -cos yaw) in (x, z)
            side = dx * math.cos(yaw) - dz * math.sin(yaw)
            turn = (b["yaw"] - a["yaw"] + math.pi) % (2 * math.pi) - math.pi
            tilt = b["pitch"] - a["pitch"]
            self.felt[i] = (fwd, side, b["pos"][1] - a["pos"][1], turn, tilt)        # vestibular-like: forward, sideways, up/down, turning, nodding
            self.hurt[i] = max(0.0, a["life"] - b["life"])
            if b["life"] <= 0 or math.hypot(dx, dz) > 12:                  # died (respawned elsewhere)
                self.dead[i] = True
        return None

    def raw(self):
        o = self.obs; d = self.dev
        phys = [np.array([[q["life"] / 20, q["food"] / 20, q["oxygen"] / 20] for q in o], np.float32),
                np.clip(self.hurt[:, None], 0, 5), self.felt.copy(), np.array([[q["pitch"] / (math.pi / 2)] for q in o], np.float32).clip(-1, 1),
                np.stack(self.pics).astype(np.float32)]
        out = []
        for k, i in enumerate(self.order):
            v = phys[i]
            if k in self.perm:
                v = v[:, self.perm[k]]
            out.append(torch.tensor(v, device=d, dtype=torch.float32))
        return out

    def judge_cmd(self, cmds, settle=2):
        self._send(dict(cmd="exec", cmds=list(cmds), settle=settle)); o, p = self._recv(); self._take(o["obs"], p); self.health = o.get("health", getattr(self, "health", None))

    def reset(self, which):
        """the judge's hand: a dead body is healed, fed and put back at the valley centre (rebirth, not the being's doing)."""
        v = self.valley; cmds = []
        for i in torch.nonzero(which)[:, 0].tolist():
            nm = self.names[i]
            cmds += [f"effect give {nm} minecraft:instant_health 1 10", f"effect give {nm} minecraft:saturation 1 10",
                     f"tp {nm} {v['x'] + 0.5} {v['y']} {v.get('home', v['z']) + 0.5} 0 0"]          # rebirth place (founder 2026-11-25 '리스폰 포인트 바꿔': place A, off the pit)
            self.dead[i] = False
        if cmds:
            self.judge_cmd(cmds, settle=3)
        self.felt[which.cpu().numpy()] = 0
