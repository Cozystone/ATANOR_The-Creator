"""상상 기관: 눈을 거꾸로 뒤집은 것 + 행동을 받아 다음 속 상태를 내다보는 한 걸음. 세계의 이름도 규칙도 없다 (그림과 제 행동만 받는다).
  encode: 그림 [B, 3, 40, 64] (0~1) -> 속 상태 z의 (평균, 로그 흩어짐) [B, Z]
  decode: z -> 그림
  next:   (z, 행동) -> 다음 z의 (평균, 로그 흩어짐)
첫 판 (2026-11-21, experiments/wm0.py): 눈 감고 행동만 넣어 그린 1~8걸음 앞이 '세상이 그대로'보다 2.2~3.5배 가깝다.

Imagine: 상상으로 고르기. 모자람이 풀린 순간의 z를 모자람마다 크기 고정으로 든다 (겪은 것, 가장 오래된 것부터 내준다).
모자랄 때 행동 줄 여러 개를 속으로 굴려 보고, 그려 본 앞날이 '그 모자람이 풀린 순간들'에 가장 가까이 가는 줄의 첫 행동을 고른다.
"""
import torch, torch.nn as nn, torch.nn.functional as F


class WM(nn.Module):
    def __init__(s, z=32, m=5):
        super().__init__()
        s.enc = nn.Sequential(nn.Conv2d(3, 16, 4, 2, 1), nn.ELU(), nn.Conv2d(16, 32, 4, 2, 1), nn.ELU(), nn.Conv2d(32, 64, 4, 2, 1), nn.ELU(), nn.Flatten(), nn.Linear(64 * 5 * 8, 2 * z))
        s.dec_in = nn.Linear(z, 64 * 5 * 8)
        s.dec = nn.Sequential(nn.ELU(), nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ELU(), nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ELU(), nn.ConvTranspose2d(16, 3, 4, 2, 1))
        s.step = nn.Sequential(nn.Linear(z + m, 256), nn.ELU(), nn.Linear(256, 256), nn.ELU(), nn.Linear(256, 2 * z))
        s.m, s.z = m, z

    def encode(s, x):
        mu, lv = s.enc(x).chunk(2, 1); return mu, lv.clamp(-6, 4)

    def decode(s, z):
        return torch.sigmoid(s.dec(s.dec_in(z).view(-1, 64, 5, 8)))

    def next(s, z, a):
        d, lv = s.step(torch.cat([z, F.one_hot(a, s.m).float()], 1)).chunk(2, 1); return z + d, lv.clamp(-6, 4)


class Imagine:
    def __init__(s, wm, n, kinds=2, keep=64, K=48, H=8, device="cuda", seed=0):
        s.wm, s.n, s.K, s.H, s.dev = wm, n, K, H, device
        s.rel = torch.full((n, kinds, keep, wm.z), float("nan"), device=device); s.cnt = torch.zeros(n, kinds, dtype=torch.long, device=device)
        s.gen = torch.Generator(device=device).manual_seed(seed + 99)
        s.keep = keep; s.weigh = False

    @torch.no_grad()
    def remember(s, z, filled):
        """모자람 j가 풀린 몸들: 그 순간의 z를 든다 (크기 고정, 돌려 쓰기)."""
        for j in range(filled.shape[1]):
            b = filled[:, j].nonzero()[:, 0]
            if len(b):
                k = s.cnt[b, j] % s.keep; s.rel[b, j, k] = z[b]; s.cnt[b, j] += 1

    @torch.no_grad()
    def choose(s, z, j, who):
        """who [n]인 몸들이 모자람 j [n]을 풀려 할 때 고를 행동 [n] (-1 = 떠올릴 풀림이 없음), 그리고 그려 본 가장 가까운 거리 [n]."""
        out = torch.full((s.n,), -1, dtype=torch.long, device=s.dev); best = torch.full((s.n,), float("inf"), device=s.dev)
        b = (who & (s.cnt[torch.arange(s.n, device=s.dev), j] > 0)).nonzero()[:, 0]
        if not len(b):
            return out, best
        B, K, H = len(b), s.K, s.H
        seq = torch.randint(0, s.wm.m, (B, K, H), device=s.dev, generator=s.gen)
        seq[:, :s.wm.m, :] = torch.arange(s.wm.m, device=s.dev)[None, :, None]      # 한 행동을 줄곧 하는 줄도 늘 넣는다
        tgt = s.rel[b, j[b]]                                                       # [B, keep, Z]
        zz = z[b, None].expand(B, K, -1).reshape(B * K, -1)
        d = torch.full((B, K), float("inf"), device=s.dev)
        if s.weigh:                                                                # 알맹이의 끌림과 같은 얼개: 풀린 순간마다 늘 같았던 속 상태의 갈래만 무겁게 (흩어짐의 역수)
            mu = torch.nanmean(tgt, 1); var = torch.nanmean((tgt - mu[:, None]) ** 2, 1).nan_to_num(1.0)
            wt = 1.0 / (var + 0.05); wt = wt / wt.mean(1, keepdim=True)
            far_ = lambda q: (wt[:, None] * (q - mu[:, None]) ** 2).sum(-1)       # [B, K]
        else:
            far_ = lambda q: torch.cdist(q, tgt.nan_to_num(1e6)).min(-1).values
        for h in range(H):
            zz = s.wm.next(zz, seq[:, :, h].reshape(-1))[0]
            d = torch.minimum(d, far_(zz.view(B, K, -1)))                          # 그 걸음의 그림이 풀린 순간에 얼마나 가까운가
        k = d.argmin(1)
        d0 = far_(z[b, None])[:, 0]                                                # 지금 이대로가 이미 얼마나 가까운가
        nearer = d[torch.arange(B, device=s.dev), k] < d0                         # 그려 본 앞날 가운데 지금보다 가까워지는 것이 있을 때만 상상으로 고른다
        out[b] = torch.where(nearer, seq[torch.arange(B, device=s.dev), k, 0], torch.full_like(k, -1)); best[b] = d[torch.arange(B, device=s.dev), k]
        return out, best
