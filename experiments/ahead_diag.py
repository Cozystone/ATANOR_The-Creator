"""Can the memory organ's 'future fill' head (ahead: h, feeling now, act about to be taken -> how much the insides will rise within 1/4/16/32 steps)
serve as imagination? Raise a being (inner 64, relief-tied, dark rooms, 4 lives), then one life without learning, recording h, raw feeling, act, death.
  1. accuracy: predicted vs actual future rise (sum over bedrock dims) per horizon - correlation, and AUC for 'any fill within j steps'
  2. in the dark only (where it matters)
  3. does it depend on the act? spread of the prediction across the 90 acts vs across moments; and the actual future rise for moments where the taken act
     was in the head's top 10% vs bottom 50% (same moment's ranking) - a weak counterfactual check
  python ahead_diag.py"""
import torch, numpy as np
from atanor_os.emerge.room_body import RoomBody
from atanor_os.emerge.body_contract import Organs
from atanor_os.emerge.mind import Mind
from atanor_os.emerge.room_vary import condition
dev, N, T = "cuda", 128, 900
torch.manual_seed(3)
O = Mi = None
for life in range(5):
    B = RoomBody(N, *condition(1 + life % 8 if life < 4 else 2), seed=300 + life, device=dev); B.dark_every, B.dark_len = 60, 20
    if O is None:
        O = Organs(B.c, N, dev=dev, depth="mc/mc_depth.pt"); Mi = Mind(O, B.c, N, dev=dev, inner=64)
    O.reset(torch.ones(N, dtype=torch.bool, device=dev)); Mi.begin(torch.ones(N, dtype=torch.bool, device=dev)); Mi.wo.on = life >= 1
    learn = life < 4; xr = O(B.raw()); x = Mi.feel(xr); R = []
    for t in range(T):
        h0 = Mi.h.clone(); dk = B.dark().clone()
        act = Mi.choose(x); a0 = ~B.dead.clone(); B.step(act); died = a0 & B.dead; cut = B.dead & ~died
        if not learn:
            R.append((h0, xr.clone(), act.clone(), a0.clone(), dk))
        xr2 = O(B.raw()); Mi.carry(xr, act, cut | died); x2 = Mi.feel(xr2)
        if learn:
            Mi.after(x, act, x2, died, cut)
        else:
            Mi.prev = (x, act); Mi.cut = cut; Mi.steps += 1
        x, xr = x2, xr2
        if B.dead.all():
            break
    print("life", life + 1, "whole", float((~B.dead).float().mean()), flush=True)

Hh = torch.stack([r[0] for r in R]); X = torch.stack([r[1] for r in R]); A = torch.stack([r[2] for r in R]); AL = torch.stack([r[3] for r in R]); DK = torch.stack([r[4] for r in R])
ins = Mi.o.inside; Tn = len(X); HOR = Mi.HOR; nH = len(HOR)


def corr(a, b):
    a, b = a - a.mean(), b - b.mean(); return float((a * b).sum() / (a.norm() * b.norm() + 1e-9))


def auc(s, y):
    s, y = s.cpu().numpy(), y.cpu().numpy().astype(bool)
    if y.sum() == 0 or (~y).sum() == 0:
        return float("nan")
    r = np.argsort(np.argsort(s)).astype(float)
    return float((r[y].mean() - (y.sum() - 1) / 2) / (~y).sum())


with torch.no_grad():
    J = max(HOR); ts = torch.arange(0, Tn - J - 1, device=dev)
    pred = Mi.ahead(torch.cat([Hh[ts], X[ts], Mi.Cc[A[ts]]], 2).reshape(-1, Hh.shape[2] + X.shape[2] + Mi.Cc.shape[1])).view(len(ts), N, nH, -1).sum(3)
    for hi, j in enumerate(HOR):
        win = torch.stack([X[ts + k][:, :, ins] for k in range(j + 1)])                       # [j+1, T, N, ni]
        act_rise = ((win.max(0).values - X[ts][:, :, ins]).clamp(min=0) * 10).sum(2)        # [T, N]
        ok = torch.stack([AL[ts + k] for k in range(j + 1)]).all(0)
        p, y = pred[:, :, hi][ok], act_rise[ok]; d = DK[ts][ok]
        print(f"horizon {j:2d}: corr {corr(p, y):+.3f} | AUC any fill {auc(p, y > 0.05):.3f} (fill share {float((y > 0.05).float().mean()):.3f}) | "
              f"in the dark: corr {corr(p[d], y[d]):+.3f}, AUC {auc(p[d], y[d] > 0.05):.3f}", flush=True)
    # act dependence at sampled moments
    tt = torch.arange(0, Tn - J - 1, 7, device=dev); M = Mi.Cc.shape[0]
    hs, xs = Hh[tt].reshape(-1, Hh.shape[2]), X[tt].reshape(-1, X.shape[2]); okm = AL[tt].reshape(-1)
    hs, xs = hs[okm], xs[okm]
    allp = torch.stack([Mi.ahead(torch.cat([hs, xs, Mi.Cc[torch.full((len(hs),), a, device=dev)]], 1)).view(len(hs), nH, -1).sum(2) for a in range(M)], 2)   # [S, nH, M]
    for hi, j in enumerate(HOR):
        P = allp[:, hi]; across_acts = float(P.std(1).mean()); across_moments = float(P.mean(1).std())
        taken = A[tt].reshape(-1)[okm]; rank = (P < P.gather(1, taken[:, None])).float().mean(1)     # share of acts the taken act beats
        win = torch.stack([X[tt + k][:, :, ins] for k in range(j + 1)]); y = ((win.max(0).values - X[tt][:, :, ins]).clamp(min=0) * 10).sum(2).reshape(-1)[okm]
        top, low = rank >= 0.9, rank < 0.5
        print(f"horizon {j:2d}: spread across acts {across_acts:.4f} vs across moments {across_moments:.4f} | actual rise when taken act in head's top 10%: "
              f"{float(y[top].mean()) if top.any() else float('nan'):.3f} (n {int(top.sum())}) vs bottom half {float(y[low].mean()) if low.any() else float('nan'):.3f} (n {int(low.sum())})", flush=True)
print("DONE")
