"""마크의 깊이 눈: 창립자의 정본 depth_learner (읽기 전용으로 불러 씀)를 존재 제 걸음의 그림으로 배운다. 이름표 없음, 미리 배운 무게 없음.
그림: 서버가 보내는 작은 온 그림 48x80 (HUD 잘라 낸 것). 사진기: Minecraft 기본 세로 시야 70도 (가정, 장부), 아래를 잘라 냈으니 가운데가 위로 치우침.
내가 얼마나 움직였나: 자리와 고개 (yaw, pitch)의 바뀜 - 몸이 느끼는 것 (장부). 깊이는 그 움직임으로 그림이 어떻게 흘러야 하는지 맞추며 배운다 (정본 selfsup).
감각: 눈높이 띠를 5갈래로 나눠 갈래마다 가까움 1 / (1 + 깊이) [n, 5]."""
import math, os, importlib.util, torch

CANON = os.environ.get("ATANOR_DEPTH_PATH", "depth_learner")
H0, CROP, W0 = 204, 44, 256           # 서버가 그리는 그림, 잘라 낸 줄
PH, PW = 48, 80
VFOV = 70.0                           # Minecraft 기본 (가정)
EYE_H = 1.62                          # 발에서 눈까지 (Minecraft 몸의 성질; 자리 바뀜에서는 지워진다)


def canon(nm):
    sp = importlib.util.spec_from_file_location("canon_depth_" + nm, os.path.join(CANON, nm + ".py")); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m


def intrinsics(device):
    fy0 = 0.5 * H0 / math.tan(math.radians(VFOV) / 2)
    sx, sy = PW / W0, PH / (H0 - CROP)
    return torch.tensor([[fy0 * sx, 0, W0 / 2 * sx], [0, fy0 * sy, H0 / 2 * sy], [0, 0, 1]], dtype=torch.float32, device=device)


def rot(yaw, pitch):
    """세계 -> 사진기 (x 오른쪽, y 아래, z 앞) [B, 3, 3]. Minecraft: yaw 0 = +z를 봄, pitch + = 아래를 봄."""
    y, p = torch.deg2rad(yaw), torch.deg2rad(pitch)
    f = torch.stack([-torch.sin(y) * torch.cos(p), -torch.sin(p), torch.cos(y) * torch.cos(p)], -1)
    r = torch.stack([-torch.cos(y), torch.zeros_like(y), -torch.sin(y)], -1)
    d = torch.cross(f, r, dim=-1)
    return torch.stack([r, d, f], 1)


def pose(wt, ws):
    """wt, ws [B, 5] (x, y, z, yaw, pitch): 표적 사진기의 점을 원천 사진기로 옮기는 4x4."""
    Rt, Rs = rot(wt[:, 3], wt[:, 4]), rot(ws[:, 3], ws[:, 4])
    M = torch.zeros(len(wt), 4, 4, device=wt.device); M[:, 3, 3] = 1
    M[:, :3, :3] = Rs @ Rt.transpose(1, 2); M[:, :3, 3] = (Rs @ (wt[:, :3] - ws[:, :3])[..., None])[..., 0]
    return M


class DepthEye:
    def __init__(self, path, device, width=16):
        self.SS, MD = canon("selfsup"), canon("model")
        self.net = MD.DepthNet(width=width).to(device); self.net.load_state_dict(torch.load(path, map_location=device)); self.net.eval()

    @torch.no_grad()
    def depth(self, pic):
        return self.SS.bounded_log_depth(self.net(pic), 0.1, 30.0).exp()             # [n, PH, PW]

    @torch.no_grad()
    def __call__(self, pic):
        return self.near(self.depth(pic))

    @torch.no_grad()
    def near(self, d):
        cy = int(H0 / 2 * PH / (H0 - CROP))                                           # 눈높이 줄 (가운데가 위로 치우침)
        band = d[:, max(cy - 4, 0):cy + 4]
        sec = band.reshape(len(d), band.shape[1], 5, PW // 5).median(-1).values.median(1).values
        return 1.0 / (1.0 + sec)


def picture(d, near5):
    """뷰어에 보일 깊이 그림 (PNG 바이트): 깊이 [PH, PW] (가까울수록 밝게, 4배 크게) + 아래에 5갈래 가까움 막대."""
    import io, numpy as np
    from PIL import Image
    v = (1.0 / (1.0 + d.float().cpu().numpy()))
    v = (v - v.min()) / max(v.max() - v.min(), 1e-6)
    rgb = np.stack([v ** 0.8, v ** 1.5, 1 - v], -1)
    im = np.asarray(Image.fromarray((rgb * 255).astype(np.uint8)).resize((PW * 4, PH * 4), Image.NEAREST)).copy()
    bars = np.full((40, PW * 4, 3), 235, np.uint8)
    for k, q in enumerate(near5.float().cpu().numpy()):
        x0 = k * PW * 4 // 5 + 4; x1 = (k + 1) * PW * 4 // 5 - 4; h = int(36 * float(q))
        bars[40 - h:, x0:x1] = (200, 80, 40)
    b = io.BytesIO(); Image.fromarray(np.concatenate([im, bars], 0)).save(b, format="PNG"); return b.getvalue()
