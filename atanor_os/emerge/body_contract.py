"""보편 몸 계약 (창립자 2026-11-22). 어떤 세계·장치든 신호만 내놓는다:
  감각 갈래 Channel(kind, shape, bedrock): kind = "image" (날 그림 [H, W, 3] 0~255), "vector" (수 묶음), "audio" (소리 세기 묶음).
  bedrock = 바탕의 모자람 (1 = 가득). 허락된 뜻 표시는 이것 하나 (진화된 것). 이름표 (체력, 양, 다리)는 없다.
  움직임 갈래 motors = [갈래마다 고를 수의 개수] (예: [6, 5, 3]). 어느 갈래가 무엇을 움직이는지는 존재가 움직여 보며 안다.
존재 쪽 기관 (Organs): 계약을 읽고 갈래마다 맞는 기관을 붙인다.
  image -> 작은세포 갈래 (정본 중심와, GPU) -> 장면 요약 + 초점 요약 + 초점 자리 + 바로 앞 작은 움직임
         -> 큰세포 갈래 (고르게 줄인 그림) -> 깊이 눈 (눈높이 다섯 갈래의 가까움)
  vector/audio -> 제 겪음으로 고르게 (bedrock 갈래는 그대로, 맨 앞에)
  움직임 -> 갈래 부호 (갈래마다 한 칸씩 켬): 알맹이가 갈래의 공통을 나눠 배운다.
느낌 x = [바탕 | 다른 수 묶음들 (계약 차례) | 그림 기관들]. 계약이 같으면 같은 존재 파일이 어느 세계에서나 그대로 돈다.
새 갈래가 생기면 (새 장치) 알맹이를 키워 (body_gpu.expand처럼 새 이음은 0에서) 붙일 수 있다 - 다음 일.
"""
from dataclasses import dataclass, field, asdict
import itertools, numpy as np, torch, torch.nn.functional as F
from .common_senses import Enc, focus_enc, Stat, OUT, Z, ZF
from .eye_gpu import RetinaGPU


@dataclass
class Channel:
    kind: str                     # "image" | "vector" | "audio"
    shape: tuple
    bedrock: bool = False


@dataclass
class Contract:
    sensors: list
    motors: list

    def signature(self):
        return [(c.kind, tuple(c.shape), c.bedrock) for c in self.sensors], list(self.motors)

    def codes(self):
        """몸짓 = 갈래마다 하나씩 고른 조합. 부호 = 갈래마다 한 칸 켬."""
        combos = list(itertools.product(*[range(k) for k in self.motors]))
        C = np.zeros((len(combos), sum(self.motors)), np.float32)
        for i, cb in enumerate(combos):
            o = 0
            for k, c in zip(self.motors, cb):
                C[i, o + c] = 1; o += k
        return combos, C


class Organs:
    def __init__(self, contract, n, dev="cuda", depth=""):
        self.c, self.n, self.dev = contract, n, dev
        self.bed = [i for i, ch in enumerate(contract.sensors) if ch.bedrock]
        self.vec = [i for i, ch in enumerate(contract.sensors) if not ch.bedrock and ch.kind in ("vector", "audio")]
        self.img = [i for i, ch in enumerate(contract.sensors) if ch.kind == "image"]
        self.stats = {i: Stat(0.5 if contract.sensors[i].kind == "audio" else 1e-3) for i in self.vec}
        ck = torch.load(f"{OUT}/mc_jepa.pt", map_location=dev)
        self.E = Enc().to(dev); self.E.load_state_dict(ck["E"]); self.E.eval()
        fck = torch.load(f"{OUT}/mc_focus.pt", map_location=dev)
        self.FE = focus_enc().to(dev); self.FE.load_state_dict({k.replace("net.", ""): v for k, v in fck["E"].items()}); self.FE.eval()
        self.DE = None
        if depth:
            from .mc_depth import DepthEye
            self.DE = DepthEye(depth, dev)
        self.ret = {i: RetinaGPU(n, *contract.sensors[i].shape[:2], device=dev) for i in self.img}
        self.zs = {i: Stat(1e-3) for i in self.img}; self.fs = {i: Stat(1e-3) for i in self.img}
        self.prev = {i: None for i in self.img}
        self.dx = sum(int(np.prod(contract.sensors[i].shape)) for i in self.bed + self.vec) + len(self.img) * (Z + ZF + 2 + 1 + (5 if depth else 0))
        self.inside = slice(0, sum(int(np.prod(contract.sensors[i].shape)) for i in self.bed))
        self.z = None

    def reset(self, which):
        for r in self.ret.values():
            r.reset(which)
        for i in self.img:
            if self.prev[i] is not None:
                self.prev[i][which] = -1

    def _bug(self, i, g):
        """바로 앞 (시야 한가운데)의 작은 움직이는 것: 가운데가 둘레보다 더 바뀐 만큼 (개구리 망막의 벌레 탐지; 제 몸이 돌면 둘레도 바뀌어 지워짐)."""
        p = self.prev[i]; self.prev[i] = g.clone()
        if p is None:
            return torch.zeros(self.n, device=self.dev)
        H, W = g.shape[1:]; cy, cx = H // 2, W // 2
        d = (g - p).abs(); fresh = p[:, 0, 0] < 0
        c = d[:, cy - 12:cy + 12, cx - 12:cx + 12].mean((1, 2))
        ring = d[:, cy - 36:cy + 36, cx - 36:cx + 36]; sur = (ring.sum((1, 2)) - d[:, cy - 12:cy + 12, cx - 12:cx + 12].sum((1, 2))) / (ring[0].numel() - 24 * 24)
        return torch.where(fresh, torch.zeros_like(c), ((c - sur).clamp(min=0) / 64.0 * 8).clamp(max=1.0))

    @torch.no_grad()
    def __call__(self, raw):
        """raw: 갈래마다 텐서 [n, *shape] (계약 차례의 목록). -> 느낌 x [n, dx]."""
        parts = [raw[i].reshape(self.n, -1).float() for i in self.bed]
        for i in self.vec:
            parts.append(self.stats[i](raw[i].reshape(self.n, -1).float(), unbiased=self.c.sensors[i].kind != "audio"))
        for i in self.img:
            fr = raw[i].float()                                            # [n, H, W, 3] 0~255
            r = self.ret[i].look(fr)
            z = self.E(r); self.z = z
            zf = self.FE(torch.cat([r[:, 0:3], r[:, 15:18]], 1))
            gaze = torch.stack([self.ret[i].cx, self.ret[i].cy], 1) * 2 - 1
            bug = self._bug(i, fr.mean(3))
            parts += [self.zs[i](z), self.fs[i](zf), gaze, bug[:, None]]
            if self.DE is not None:
                small = F.interpolate(fr.permute(0, 3, 1, 2) / 255.0, size=(48, 80), mode="bilinear", align_corners=False)
                parts.append(self.DE(small))
        return torch.cat(parts, 1)

    def state(self):
        return dict(vec={i: s.state() for i, s in self.stats.items()}, z={i: s.state() for i, s in self.zs.items()}, f={i: s.state() for i, s in self.fs.items()})

    def load(self, st):
        for i, s in self.stats.items():
            s.load(st["vec"].get(i), self.dev)
        for i in self.img:
            self.zs[i].load(st["z"].get(i), self.dev); self.fs[i].load(st["f"].get(i), self.dev)
