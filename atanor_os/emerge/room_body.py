"""방 세계를 보편 몸 계약으로 (body_contract): 날 그림 (마크와 같은 사진기: 256 x 160, 세로 시야 70도에서 아래 44줄을 잘라 낸 꼴, 수평선은 102줄)과
신호 갈래 (바탕의 모자람, 맞음, 제 걸음, 살갗, 고개 기울기, 귀), 움직임 갈래 [6, 5, 3]. 존재는 이 계약만 안다 (방이라는 것을 모른다).
움직임: 다리 (서 있음, 앞, 뒤, 왼옆, 오른옆, 뛰며 앞 = 앞) x 목 (그대로, 왼 30도, 오른 30도, 위, 아래 - 방에는 고개를 들 수 없어 위·아래는 아무 일 없음)
 x 손 (빈손, 때림 - 방에는 때릴 것이 아직 없음, 씀 = 바로 앞 샘을 마시거나 먹음).
'것' 체계 (창립자 2026-11-22: 쫓아오는 적, 놀잇감, 동료를 차례로): Things가 종류·자리·움직임을 들고, 그림과 살갗에 함께 나온다 (다음 단계에서 채움).
"""
import math, torch
from . import room3d_gpu as G
from .body_contract import Channel, Contract

H, W, HORIZON = 160, 256, 102
FY = 0.5 * 204 / math.tan(math.radians(35.0))               # 마크와 같은 초점 거리 (px)
EYE = 1.6                                                    # 눈 높이 (칸)
WALL_H, SPRING_H = 2.0, 1.0


def contract():
    return Contract(sensors=[Channel("vector", (3,), bedrock=True), Channel("vector", (1,)), Channel("vector", (3,)), Channel("vector", (20,)),
                             Channel("vector", (1,)), Channel("audio", (16,)), Channel("image", (H, W, 3))], motors=[6, 5, 3])


class RoomBody:
    default_cues = False                                                # a run can switch distal cues on for every room body it makes
    default_eye = EYE                                                   # eye height (cells); founder 2026-11-24: lower it from the next world (1.1), checked first

    def __init__(self, n, rows, drain=1 / 240.0, seed=0, device="cuda", regrow=None):
        self.n, self.dev = n, device
        self.w = G.World(n, rows, device, seed=seed, drain=drain, regrow=regrow)
        self.c = contract()
        self.combos, _ = self.c.codes()
        cb = torch.tensor(self.combos, device=device)              # [90, 3]
        self.legs, self.neck, self.hand = cb[:, 0], cb[:, 1], cb[:, 2]
        self.cols = torch.atan((torch.arange(W, device=device) + 0.5 - W / 2) / FY)      # 열마다 시선 각
        self.rows = torch.arange(H, device=device).float()[:, None] + 0.5 - HORIZON
        self.prev_store = self.w.store.clone(); self.felt = torch.zeros(n, 3, device=device)
        # 어둠 (R1: 세계가 기억을 요구하게 - 불이 꺼지면 눈으로는 아무것도 보이지 않고, 샘이 어디 있는지는 들고 있는 것으로만 안다). 몸마다 다른 때에.
        self.dark_every, self.dark_len = 0, 0
        self.dark_phase = torch.randint(0, 10 ** 6, (n,), device=device); self.t = 0
        # 먼 단서 (2026-11-24, 세계의 성질 - 행동 규칙 아님; 창립자 '환경 잘 구성해' 위임, 검토 대상): 벽의 면마다 (북·동·남·서) 빛깔이 조금 다름 - 실제 방의 벽이
        # 저마다 다르게 보이듯. 대칭인 방에서 곳·방향을 눈으로 알 수 있게 (D15: 단서 없는 방은 그림 자체가 곳을 싣지 않음). 기본 꺼짐.
        self.cues = RoomBody.default_cues

    @property
    def dead(self):
        return self.w.dead

    def reset(self, which):
        self.w.reset(which); self.felt[which] = 0; self.prev_store = self.w.store.clone()

    @torch.no_grad()
    def render(self):
        w = self.w
        kind, dist = w.cast(w.head[:, None] + self.cols[None], step=0.05)
        z = (dist * torch.cos(self.cols)[None]).clamp(min=0.05)                       # [n, W]
        hgt = torch.where(kind == G.WALL, torch.full_like(z, WALL_H), torch.full_like(z, SPRING_H))
        eye = RoomBody.default_eye; top = -FY * (hgt - eye) / z; bot = FY * eye / z                                # 화면에서 수평선 위아래로 (px)
        r = self.rows[None]                                                             # [1, H, 1]
        hit = (r >= top[:, None, :]) & (r <= bot[:, None, :]) & (dist[:, None, :] < 90)
        col = G.COLOR.to(self.dev)[kind.long()] * (1.0 - z / 14.0).clamp(0.3, 1.0)[..., None]
        if self.cues:                                                                   # which face of the wall the ray met (from the ray direction and the hit cell's side)
            ang = w.head[:, None] + self.cols[None]; dxr, dyr = torch.cos(ang), torch.sin(ang)
            hx = w.pos[:, 0, None] + dxr * dist; hy = w.pos[:, 1, None] + dyr * dist
            px, py = hx - dxr * 0.05, hy - dyr * 0.05
            xface = hx.floor() != px.floor()                                             # crossed a vertical cell boundary -> an east / west face
            face = torch.where(xface, torch.where(dxr > 0, 3, 1), torch.where(dyr > 0, 0, 2))     # 0 north face, 1 east, 2 south, 3 west
            tint = torch.tensor([[1.15, 0.95, 0.80], [0.80, 1.10, 0.85], [0.80, 0.90, 1.20], [1.10, 0.85, 1.10]], device=self.dev)[face]
            col = torch.where((kind == G.WALL)[..., None], col * tint, col)
        sky, gnd = G.SKY.to(self.dev), G.GROUND.to(self.dev)
        base = torch.where((r < 0)[..., None], sky, gnd).expand(self.n, H, W, 3)
        img = torch.where(hit[..., None], col[:, None].expand(-1, H, -1, -1), base)
        if self.dark_every:
            img = img * (~self.dark())[:, None, None, None].float()
        return (img.clamp(0, 1) * 255.0)

    def dark(self):
        """지금 불이 꺼져 있는 몸 [n]."""
        if not self.dark_every:
            return torch.zeros(self.n, dtype=torch.bool, device=self.dev)
        return ((self.t + self.dark_phase) % self.dark_every) < self.dark_len

    def raw(self):
        """계약 차례의 날 신호 목록."""
        w, n, d = self.w, self.n, self.dev
        st = w.store
        bed = torch.stack([torch.ones(n, device=d), st[:, 1], st[:, 0]], 1)            # 바탕 셋 (빈 칸, 먹이, 물)
        ang = w.head[:, None] + torch.tensor([0.0, math.pi / 2, math.pi, -math.pi / 2], device=d)[None]    # 앞·오른쪽·뒤·왼쪽 (방의 head는 + 가 왼쪽으로 돎이 아니라 각 증가)
        _, near = w.cast(ang, step=0.02, far=0.8)
        touch = (1.0 - (near - 0.15) / 0.6).clamp(0, 1)
        skin = torch.zeros(n, 20, device=d); skin[:, 0:4] = touch; skin[:, 4:8] = touch; skin[:, 8] = 1.0
        lit = (~self.dark()).float()
        skin[:, 15] = 0.5 * lit; skin[:, 16] = lit; skin[:, 17] = lit; skin[:, 18] = 0.5      # 밝음·하늘빛·해 (어둠이면 0)·따뜻함, 비 없음
        ear = torch.full((n, 16), -9.0, device=d)                                       # 고요 (마크 귀의 소리 없음과 같은 값)
        return [bed, torch.zeros(n, 1, device=d), self.felt.clone(), skin, torch.zeros(n, 1, device=d), ear, self.render()]

    @torch.no_grad()
    def step(self, act, dt=None):
        """act [n] 몸짓 번호 (갈래 조합). dt: 흐른 실제 시간 (초) - 훈련 밖의 삶은 틱이 아니라 연속된 시간 (창립자 2026-11-22); 없으면 연습의 한 틱 (0.2초).
        돌려줌: 채워졌는가 [n, 2]."""
        TICK = float(dt) if dt is not None else G.TICK
        w = self.w; alive = ~w.dead
        lg, nk, hd = self.legs[act], self.neck[act], self.hand[act]
        fwd = torch.where((lg == 1) | (lg == 5), 1.0, torch.where(lg == 2, -1.0, 0.0))
        side = torch.where(lg == 3, -1.0, torch.where(lg == 4, 1.0, 0.0))
        turn = torch.where(nk == 1, -1.0, torch.where(nk == 2, 1.0, 0.0)) * (math.radians(30) / (G.TURN * G.TICK))      # 고개 돌림은 한 번의 몸짓 (30도), 걷기는 흐른 시간만큼
        do = hd == 2
        was = w.pos.clone(); h0 = w.head.clone()
        hh, t = 1 / 60.0, 0.0
        while t < TICK - 1e-9:
            dd = min(hh, TICK - t); t += dd
            w.head = torch.where(alive, (w.head + turn * G.TURN * dd * (G.TICK / TICK)) % (2 * math.pi), w.head)
            c, s = torch.cos(w.head), torch.sin(w.head)
            stp = G.SPEED * dd * alive
            mv = torch.stack([fwd * c - side * s, fwd * s + side * c], 1) * stp[:, None]
            nx = w.pos + mv
            for ax in (0, 1):
                trial = w.pos.clone(); trial[:, ax] = nx[:, ax]
                ok = w.cell(trial) == G.FLOOR
                w.pos = torch.where(ok[:, None], trial, w.pos)
        w.store = w.store - w.drain * TICK * alive[:, None]
        filled = torch.zeros(self.n, 2, dtype=torch.bool, device=self.dev)
        if do.any():
            k, dist = w.cast(w.head[:, None], step=0.02, far=3.0); k, dist = k[:, 0], dist[:, 0]
            for j, kind in enumerate((G.WATER, G.FOOD)):
                hit = do & alive & (k == kind) & (dist <= G.REACH)
                if hit.any():
                    if w.regrow is None:
                        filled[:, j] = hit; w.store[hit, j] = 1.0
                    else:                                                   # 샘의 고갈 (세계의 성질): 남은 만큼만 준다
                        give = torch.minimum(w.stock[:, j], 1.0 - w.store[:, j]) * hit
                        w.store[:, j] += give; w.stock[:, j] -= give; filled[:, j] = hit & (give > 0.01)
                    w._move(hit, j, kind)
        if w.regrow is not None:
            w.stock = (w.stock + w.regrow * TICK).clamp(max=1.0)
        w.clock += TICK * alive
        self.t += 1
        w.dead |= (w.store <= 0).any(1); w.store.clamp_(min=0)
        dp = w.pos - was; c, s = torch.cos(h0), torch.sin(h0)
        self.felt = torch.stack([dp[:, 0] * c + dp[:, 1] * s, -dp[:, 0] * s + dp[:, 1] * c, ((w.head - h0 + math.pi) % (2 * math.pi)) - math.pi], 1)
        return filled
