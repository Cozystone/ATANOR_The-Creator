"""몸에 새 감각을 더하고, 같은 존재가 그 감각과 함께 다시 자라게 한다 (GPU).

닿음 (touch): 앞 / 왼쪽 / 오른쪽마다, 몸 둘레 가까이에 무엇이 있는지 [n, 3] (0 = 없음, 1 = 몸에 닿아 눌림). 수염처럼 닿기 조금 전부터 느낀다.
  부딪힌 순간만 아는 한 비트(senses의 2번 자리) 대신, 닿아 있는 동안 내내, 어느 쪽인지까지.
깊이 (depth): 창립자의 정본 depth_learner(읽기 전용)를 제 걸음으로 배운 깊이 눈으로, 눈앞을 5갈래로 나눠 갈래마다 벽까지의 가까움 [n, 5].
  참 깊이는 쓰지 않는다 (판정에만).
expand: 얼린 알맹이의 입력에 새 감각 자리를 붙인다. 옛 무게는 그대로, 새 자리로 들어오는 이음은 0에서 시작 (새 감각이 생긴 같은 존재).
grow: 원래 기를 때와 같은 꼴 (삶을 살고, 30%는 아무렇게나 해 보고, 걸음마다 내다보기를 배우고, 풀린 순간을 든다)을 GPU에서.
이 감각들은 고름의 규칙이 아니다: 내다보기에 들어갈 뿐이고, 끌림은 그대로다.
"""
import math, os, importlib.util, torch
from . import room3d_gpu as G

NEAR = 0.6          # 수염의 길이 (칸). 몸 반지름 둘레에서 이만큼 안쪽이면 느낀다


def touch(world):
    ang = world.head[:, None] + torch.tensor([0.0, -math.pi / 2, math.pi / 2], device=world.dev)[None]
    _, dist = world.cast(ang, step=0.02, far=NEAR + 0.2)
    return (1.0 - (dist - 0.15) / NEAR).clamp(0, 1)


class DepthSense:
    """정본 깊이 눈 (제 걸음으로 배운 무게). 그림은 바늘구멍 사진기로 그린다 (정본이 가정하는 것)."""
    def __init__(self, path, device, width=16, PH=48, PW=80, canon=None):
        canon = canon or os.environ.get("ATANOR_DEPTH_PATH", "depth_learner")
        def load(nm):
            sp = importlib.util.spec_from_file_location("canon_depth_" + nm, os.path.join(canon, nm + ".py")); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
        self.SS, MD = load("selfsup"), load("model")
        self.net = MD.DepthNet(width=width).to(device); self.net.load_state_dict(torch.load(path, map_location=device)); self.net.eval()
        self.dev, self.PH, self.PW = device, PH, PW
        self.f = 0.5 * PW / math.tan(0.5 * G.FOV)
        self.offs = torch.atan((torch.arange(PW, device=device) + 0.5 - PW / 2) / self.f)
        self.rows = torch.arange(PH, device=device).float()[:, None] + 0.5 - PH / 2

    @torch.no_grad()
    def picture(self, w):
        kind, dist = w.cast(w.head[:, None] + self.offs[None], step=0.05)
        z = dist * torch.cos(self.offs)[None]; half = 0.5 * self.f / z.clamp(min=0.05)
        wall = self.rows.abs()[None] <= half[:, None, :]
        col = G.COLOR.to(self.dev)[kind.long()] * (1.0 - z / 12.0).clamp(0.25, 1.0)[..., None]
        base = torch.where((self.rows < 0)[None, :, :, None], G.SKY.to(self.dev), G.GROUND.to(self.dev)).expand(w.n, self.PH, self.PW, 3)
        return torch.where(wall[..., None], col[:, None].expand(-1, self.PH, -1, -1), base).permute(0, 3, 1, 2).clamp(0, 1)

    @torch.no_grad()
    def __call__(self, w, chunk=512):
        pic = self.picture(w)
        d = torch.cat([self.SS.bounded_log_depth(self.net(pic[i:i + chunk]), 0.1, 30.0).exp() for i in range(0, w.n, chunk)])     # [n, PH, PW]
        band = d[:, self.PH // 2 - 4:self.PH // 2 + 4]                          # 눈높이 띠
        sec = band.reshape(w.n, 8, 5, self.PW // 5).median(-1).values.median(1).values      # [n, 5] 갈래마다 깊이
        return 1.0 / (1.0 + sec)                                               # 가까움 (0~1)


def expand(net, extra):
    """알맹이에 새 감각 extra개를 붙인다 (옛 무게 그대로, 새 이음은 0)."""
    from .core import Core
    dx, m = net.dx, net.m
    new = Core(dx + extra, m, net.inside, net.place, k=net.k, d=net.f[0].out_features, beta=net.beta).to(net.heavy.device)
    with torch.no_grad():
        W = net.f[0].weight
        new.f[0].weight.copy_(torch.cat([W[:, :dx], torch.zeros(W.shape[0], extra, device=W.device), W[:, dx:]], 1)); new.f[0].bias.copy_(net.f[0].bias)
        new.f[2].load_state_dict(net.f[2].state_dict())
        new.f[4].weight.zero_(); new.f[4].weight[:dx].copy_(net.f[4].weight); new.f[4].bias.zero_(); new.f[4].bias[:dx].copy_(net.f[4].bias)
        gw = net.g[0].weight
        new.g[0].weight.copy_(torch.cat([gw, torch.zeros(gw.shape[0], extra, device=gw.device)], 1)); new.g[0].bias.copy_(net.g[0].bias)
        new.g[2].load_state_dict(net.g[2].state_dict())
        for b in ("rel_de", "rel_a", "rel_n", "rel_s", "rel_c", "fnd_a", "fnd_n"):
            getattr(new, b).copy_(getattr(net, b))
        new.rel_x.zero_(); new.rel_x[:, :dx] = net.rel_x; new.fnd_x.zero_(); new.fnd_x[:, :dx] = net.fnd_x
    new.also, new.far, new.like = net.also, net.far, net.like
    return new
