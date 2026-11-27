"""정본 중심와 눈 (packages/eye/fovea.py, 읽기 전용)을 몸 여럿에 한꺼번에 GPU로 - 같은 셈 (공통 접면: 방과 마크가 같은 눈 길을 쓰게).
정본의 RetinaSpec(fovea_px=16, rings=4, ring_px=16, falloff=2, ring_peak=True), 고정점 관성 0.45, 놀람 격자 12를 그대로 따른다.
  ret = RetinaGPU(n, H, W); out = ret.look(rgb)   # rgb [n, H, W, 3] 0~255 float -> [n, 27, 16, 16] (0~1): 가운데 + 고리 4 + 고리의 센 빛깔 4
정본과 다를 수 있는 곳 (재서 적음): 정수 배가 아닌 줄이기 (놀람 격자 160/12)는 영역 평균을 adaptive pooling으로 (cv2 INTER_AREA와 아주 조금 다름)."""
import torch, torch.nn.functional as F

FOVEA, RINGS, RING_PX, FALLOFF, INERTIA, GRID = 16, 4, 16, 2.0, 0.45, 12


class RetinaGPU:
    def __init__(self, n, H, W, device="cuda"):
        self.n, self.H, self.W, self.dev = n, H, W, device
        self.cx = torch.full((n,), 0.5, device=device); self.cy = torch.full((n,), 0.5, device=device)
        self.prev = None
        self.boxes = []
        # A1 (founder 2026-11-27, "build gaze attention now"): top-down attention learned from the bedrock only. The eye keeps a short trace
        # of the colours in view; when the body's insides rise (a relief), the colours that were in view before it gain relevance (and lose
        # it when in view without relief). The gaze then goes to the centre of (picture change + learned relevance of what is where), each
        # part normalised - the eyes go to what has mattered, not only to what moved. No colour or object is named. Off unless ATANOR_ATTEND=1.
        import os as _os
        self.attend = _os.environ.get("ATANOR_ATTEND", "0") == "1"
        self.aw = torch.zeros(3, device=device); self.ae = torch.zeros(n, 3, device=device); self.ab = torch.zeros(3, device=device); self.phi = None

    def reset(self, which):
        self.cx[which] = 0.5; self.cy[which] = 0.5
        if self.prev is not None:
            self.prev[which] = -1.0                                  # 다음에 놀람 0 (정본: 처음 본 그림은 놀람 없음)

    def _crop(self, img, x0, y0, h, w):
        """img [n, H, W, 3], 몸마다 다른 (x0, y0)에서 h x w 잘라 냄 -> [n, h, w, 3]."""
        iy = (y0[:, None] + torch.arange(h, device=self.dev)[None]).clamp(0, self.H - 1)
        ix = (x0[:, None] + torch.arange(w, device=self.dev)[None]).clamp(0, self.W - 1)
        b = torch.arange(self.n, device=self.dev)[:, None, None]
        return img[b, iy[:, :, None], ix[:, None, :]]

    def _area(self, patch, h, w):
        x = patch.permute(0, 3, 1, 2)                               # [n, 3, H, W]
        H, W = x.shape[2:]
        if H % h == 0 and W % w == 0:
            return F.avg_pool2d(x, (H // h, W // w))
        return F.adaptive_avg_pool2d(x, (h, w))

    def _peak(self, patch, h, w):
        a = patch
        opp = torch.stack([(a[..., k] - 0.5 * (a[..., (k + 1) % 3] + a[..., (k + 2) % 3])).clamp(min=0) for k in range(3)], 1)   # [n, 3, H, W]
        H, W = opp.shape[2:]
        bh, bw = max(1, H // h), max(1, W // w)
        o = opp[:, :, :bh * h, :bw * w]
        return F.max_pool2d(o, (bh, bw))

    @torch.no_grad()
    def look(self, rgb):
        n, H, W = self.n, self.H, self.W
        px, py = self.cx * W, self.cy * H
        half = FOVEA // 2
        x0 = (px - half).clamp(0, max(W - FOVEA, 0)).long(); y0 = (py - half).clamp(0, max(H - FOVEA, 0)).long()
        fov = self._crop(rgb, x0, y0, FOVEA, FOVEA).permute(0, 3, 1, 2)          # [n, 3, 16, 16]
        self.boxes = [(x0, y0, FOVEA, FOVEA)]
        lv, pk = [fov], []
        extent = FOVEA
        for r in range(RINGS):
            extent = int(extent * FALLOFF); h, w = min(extent, H), min(extent, W)
            rx = (px - w / 2).clamp(0, max(W - w, 0)).long(); ry = (py - h / 2).clamp(0, max(H - h, 0)).long()
            patch = self._crop(rgb, rx, ry, h, w)
            lv.append(self._area(patch, RING_PX, RING_PX)); pk.append(self._peak(patch, RING_PX, RING_PX))
            self.boxes.append((rx, ry, h, w))
            if w >= W and h >= H:
                break
        while len(lv) < 5:
            lv.append(lv[-1])
        while len(pk) < 4:
            pk.append(pk[-1])
        out = torch.cat(lv[:5] + pk[:4], 1) / 255.0                               # [n, 27, 16, 16]
        # 놀람: 앞 그림과의 차이 (격자 12) -> 그 무게 중심으로 고정점을 옮김 (관성 0.45)
        g = F.adaptive_avg_pool2d(rgb.mean(3)[:, None], (GRID, GRID))[:, 0]
        if self.prev is None:
            self.prev = g.clone(); s = torch.zeros_like(g)
        else:
            fresh = self.prev[:, 0, 0] < 0
            s = (g - self.prev).abs(); s[fresh] = 0; self.prev = g.clone()
        if self.attend:
            phi = F.adaptive_avg_pool2d(rgb.permute(0, 3, 1, 2) / 255.0, (GRID, GRID)).permute(0, 2, 3, 1)      # [n, 12, 12, 3] colour of each place in view
            self.phi = phi; m = phi.mean((1, 2))
            self.ae = 0.9 * self.ae + 0.1 * m; self.ab = 0.999 * self.ab + 0.001 * m.mean(0)                     # what was in view lately; its usual
            rel = ((phi - self.ab) * self.aw).sum(3).clamp(min=0)                                                   # learned relevance of each place now
            rs = rel.sum((1, 2), keepdim=True); ss = s.sum((1, 2), keepdim=True)
            s = s / ss.clamp(min=1e-9) * (ss > 1e-6) + rel / rs.clamp(min=1e-9) * (rs > 1e-6)
        tot = s.sum((1, 2))
        gy, gx = torch.meshgrid(torch.arange(GRID, device=self.dev, dtype=torch.float32), torch.arange(GRID, device=self.dev, dtype=torch.float32), indexing="ij")
        ty = (s * gy).sum((1, 2)) / tot.clamp(min=1e-9) / (GRID - 1); tx = (s * gx).sum((1, 2)) / tot.clamp(min=1e-9) / (GRID - 1)
        mv = tot > 1e-6
        self.cx = torch.where(mv, INERTIA * self.cx + (1 - INERTIA) * tx, self.cx); self.cy = torch.where(mv, INERTIA * self.cy + (1 - INERTIA) * ty, self.cy)
        return out.clamp(0, 1)

    @torch.no_grad()
    def relief(self, rose):
        """A1: rose [n] bool - the body's insides rose this beat. What was in view before gains relevance; in view without relief, loses a little."""
        if not self.attend:
            return
        d = (self.ae - self.ab)                                                                     # [n, 3]
        r = rose.float()[:, None]
        self.aw += 0.05 * (r * d).mean(0) - 0.0005 * ((1 - r) * d).mean(0)
