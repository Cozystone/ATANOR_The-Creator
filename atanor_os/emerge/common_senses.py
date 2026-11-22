"""공통 접면 (창립자 2026-11-22: 하나의 존재 파일이 방과 마크를 오간다): 날것의 감각 -> 느낌 x. 어느 세계든 같은 차례, 같은 눈, 같은 고르기.
날것 (raw, 몸 n개, 모두 torch, 같은 기기):
  ret   [n, 27, 16, 16]  작은세포 갈래: 정본 중심와 눈 (가운데 + 고리 4 + 고리의 센 빛깔 4), 0~1
  small [n, 3, 48, 80]   큰세포 갈래: 온 시야를 고르게 줄인 그림, 0~1 (깊이 눈)
  ins   [n, 3]           몸 안 (바탕의 모자람 셋, 1 = 가득). 마크: 체력·배부름·숨 / 방: 1 (빈 칸)·먹이·물
  hurt  [n]              이번 걸음에 맞은 양 (0~5)
  felt  [n, 3]           제 걸음 (앞, 옆, 돎)
  gaze  [n, 2]           초점 자리 (0~1)
  bug   [n]              초점 바로 앞의 작은 움직이는 것
  ear   [n, 16]          귀 (왼·오른 x 띠 8, log10 세기) - 소리가 없는 세계는 고요 (-9)
  skin  [n, 20]          살갗 (닿음 8, 발밑, 물에 잠김 2, 아픔의 쪽 4, 밝음 등 5) - 없는 것은 0
  pitch [n]              고개 기울기 (도, + = 아래)
느낌 x 차례: 몸 안 3 | 맞음 1 | 제 걸음 3 | 초점 자리 2 | 작은 움직임 1 | 눈 요약 64 | 초점 요약 32 | 깊이 5 | 귀 16 | 살갗 20 | 고개 1  (= 148)
눈 요약·초점 요약·귀는 제 겪음의 평균과 흩어짐으로 고르게 하며, 그 통계도 존재 파일에 든다 (세계를 옮겨도 이어짐).
"""
import torch, torch.nn as nn

OUT = "mc"
Z, ZF = 64, 32


class Enc(nn.Module):                                        # 장면 요약 (JEPA, mc_eyejudge.py와 같은 꼴)
    def __init__(s, z=Z):
        super().__init__()
        s.net = nn.Sequential(nn.Conv2d(27, 64, 4, 2, 1), nn.ELU(), nn.Conv2d(64, 64, 4, 2, 1), nn.ELU(), nn.Conv2d(64, 64, 4, 2, 1), nn.ELU(), nn.Flatten(), nn.Linear(256, 256), nn.ELU(), nn.Linear(256, z))

    def forward(s, x):
        return s.net(x)


def focus_enc():
    return nn.Sequential(nn.Conv2d(6, 48, 3, 1, 1), nn.ELU(), nn.Conv2d(48, 64, 4, 2, 1), nn.ELU(), nn.Conv2d(64, 64, 4, 2, 1), nn.ELU(), nn.Flatten(), nn.Linear(64 * 16, 256), nn.ELU(), nn.Linear(256, ZF))


class Stat:
    """제 겪음으로 고르게 (처음 본 것으로 평균과 흩어짐을 잡고 천천히 따라감)."""
    """흩어짐 = 제 평균에서 벗어난 제곱의 평균 (몸들 사이 + 시간 위). 2026-11-23 고침: 전에는 몸들 사이 흩어짐만 쟀다 -> 몸이 하나 (노트북)면
    늘 0이라 폭이 바닥값에 붙어, 모든 칸이 '평균보다 위/아래' 두 값으로 찼다. 겪은 수 c가 적을 때는 빨리 따라감 (1/c, 바닥 0.001)."""
    def __init__(self, floor):
        self.m, self.v, self.floor, self.c = None, None, floor, 0

    @property
    def s(self):
        return self.v.sqrt() + self.floor

    def __call__(self, v, unbiased=True):
        if self.m is None:
            self.m = v.mean(0); self.v = v.var(0, unbiased=False)
        r = max(0.001, 1.0 / (self.c + 1)); self.c += 1
        self.v = (1 - r) * self.v + r * ((v - self.m) ** 2).mean(0)
        self.m = (1 - r) * self.m + r * v.mean(0)
        return ((v - self.m) / self.s).clamp(-4, 4) / 4

    def state(self):
        return None if self.m is None else (self.m.cpu(), self.s.cpu(), self.v.cpu(), self.c)

    def load(self, st, dev):
        if st is not None:
            self.m = st[0].to(dev)
            if len(st) > 2:
                self.v, self.c = st[2].to(dev), st[3]
            else:                                                  # 옛 파일: 폭만 있음. 바닥에 붙은 폭은 아무것도 모르는 것이니 처음부터 빨리 배움
                self.v = (st[1].to(dev) - self.floor).clamp(min=0) ** 2
                self.c = 1000 if bool((self.v > 0).any()) else 0


class Senses:
    def __init__(self, dev="cuda", focus=True, depth="", ear=False, skin=False, neck=False):
        self.dev, self.focus, self.depth, self.ear, self.skin, self.neck = dev, focus, depth, ear, skin, neck
        ck = torch.load(f"{OUT}/mc_jepa.pt", map_location=dev)
        self.E = Enc().to(dev); self.E.load_state_dict(ck["E"]); self.E.eval()
        self.Pr = nn.Sequential(nn.Linear(Z + 7, 256), nn.ELU(), nn.Linear(256, 256), nn.ELU(), nn.Linear(256, Z)).to(dev); self.Pr.load_state_dict(ck["P"]); self.Pr.eval()
        if focus:
            fck = torch.load(f"{OUT}/mc_focus.pt", map_location=dev)
            self.FE = focus_enc().to(dev); self.FE.load_state_dict({k.replace("net.", ""): v for k, v in fck["E"].items()}); self.FE.eval()
        if depth:
            from .mc_depth import DepthEye
            self.DE = DepthEye(depth, dev)
        self.zs, self.fs, self.es = Stat(1e-3), Stat(1e-3), Stat(0.5)
        self.dx = 10 + Z + (ZF if focus else 0) + (5 if depth else 0) + (16 if ear else 0) + (20 if skin else 0) + (1 if neck else 0)
        self.z = None; self.dd = None; self.near = None

    @torch.no_grad()
    def __call__(self, raw):
        z = self.E(raw["ret"]); self.z = z
        parts = [raw["ins"], (raw["hurt"].clamp(max=5.0) / 5)[:, None], raw["felt"].clamp(-1, 1), raw["gaze"] * 2 - 1, (raw["bug"] * 8).clamp(max=1.0)[:, None], self.zs(z)]
        if self.focus:
            pic = raw["ret"]; parts.append(self.fs(self.FE(torch.cat([pic[:, 0:3], pic[:, 15:18]], 1))))
        if self.depth:
            self.dd = self.DE.depth(raw["small"]); self.near = self.DE.near(self.dd); parts.append(self.near)
        if self.ear:
            parts.append(self.es(raw["ear"], unbiased=False))
        if self.skin:
            parts.append(raw["skin"])
        if self.neck:
            parts.append((raw["pitch"] / 90.0).clamp(-1, 1)[:, None])
        return torch.cat(parts, 1).float()

    def state(self):
        return dict(z=self.zs.state(), f=self.fs.state(), e=self.es.state())

    def load(self, st):
        self.zs.load(st.get("z"), self.dev); self.fs.load(st.get("f"), self.dev); self.es.load(st.get("e"), self.dev)
