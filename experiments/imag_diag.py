"""Why did value-judged imagination fail? Raise a being (inner 64, relief-tied, dark rooms, 4 lives), then live one more life without learning,
recording feelings x_t and acts. For k = 1, 2, 4, 6: roll the core forecast from x_t along the ACTUAL acts a_t..a_t+k-1 and compare
  predicted wholeness change  whole(z_k) - whole(x_t)   vs   actual  whole(x_t+k) - whole(x_t)   (correlation, and on moments where it changed)
  feeling error |z_k - x_t+k| relative to |x_t+k - x_t|.
Also: does the imagined ranking of first acts agree with the value's own ranking (Q) at the same moment?  (rank corr)
  python imag_diag.py"""
import torch
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
        O = Organs(B.c, N, dev=dev, depth="mc/mc_depth.pt"); Mi = Mind(O, B.c, N, dev=dev, inner=64, imagine=0.0)
    O.reset(torch.ones(N, dtype=torch.bool, device=dev)); Mi.begin(torch.ones(N, dtype=torch.bool, device=dev)); Mi.wo.on = life >= 1
    learn = life < 4; xr = O(B.raw()); x = Mi.feel(xr); XS, AS, DS = [], [], []
    for t in range(T):
        act = Mi.choose(x); a0 = ~B.dead.clone(); B.step(act); died = a0 & B.dead; cut = B.dead & ~died
        xr2 = O(B.raw()); Mi.carry(xr, act, cut | died); x2 = Mi.feel(xr2)
        if learn:
            Mi.after(x, act, x2, died, cut)
        else:
            Mi.prev = (x, act); Mi.cut = cut; Mi.steps += 1
            XS.append(x.clone()); AS.append(act.clone()); DS.append(B.dead.clone())
        x, xr = x2, xr2
        if B.dead.all():
            break
    print("life", life + 1, "whole", float((~B.dead).float().mean()), flush=True)

X = torch.stack(XS); A = torch.stack(AS); D = torch.stack(DS)                    # [T, N, dx], [T, N] (act index), [T, N]
print("act tensor", A.shape, A.dtype)
Tn = len(X)


def corr(a, b):
    a, b = a - a.mean(), b - b.mean(); return float((a * b).sum() / (a.norm() * b.norm() + 1e-9))


with torch.no_grad():
    for k in (1, 2, 4, 6, 12):
        ts = torch.arange(0, Tn - k, 3, device=dev)
        z = X[ts].reshape(-1, X.shape[2]); ok = ~(D[ts:ts + 1].squeeze(0) if False else D[ts]).reshape(-1)
        for j in range(k):
            alive = ~D[ts + j + 1].reshape(-1); ok = ok & alive
            a = A[ts + j].reshape(-1)
            z = z + Mi.net.f(torch.cat([z, Mi.net.codes[a]], 1))
        xk = X[ts + k].reshape(-1, X.shape[2]); x0 = X[ts].reshape(-1, X.shape[2])
        pw = (Mi.wo.whole(z) - Mi.wo.whole(x0))[ok]; aw = (Mi.wo.whole(xk) - Mi.wo.whole(x0))[ok]
        ch = aw.abs() > 1e-3
        fe = float((z - xk)[ok].norm(dim=1).mean() / ((xk - x0)[ok].norm(dim=1).mean() + 1e-9))
        print(f"k {k:2d}: whole change corr {corr(pw, aw):+.3f} (on changed moments {corr(pw[ch], aw[ch]) if ch.sum() > 10 else float('nan'):+.3f}, n {int(ch.sum())}) | "
              f"feeling error / actual change {fe:.2f} | pred change sd {float(pw.std()):.4f} vs actual {float(aw.std()):.4f}", flush=True)
    # imagined ranking vs value ranking at sampled moments
    ts = torch.arange(0, Tn, 25, device=dev); xs = X[ts].reshape(-1, X.shape[2])[: 2048]
    Mi.horizon, Mi.rolls = 6, 2; im = Mi.imagined(xs); q = Mi.wo.v(xs)
    def rk(m): return m.argsort(1).argsort(1).float()
    r = torch.tensor([corr(rk(im)[i], rk(q)[i]) for i in range(len(xs))])
    im1 = []
    Mi.horizon = 1; im1 = Mi.imagined(xs)
    r1 = torch.tensor([corr(rk(im1)[i], rk(q)[i]) for i in range(len(xs))])
    print(f"rank corr imagined(6) vs Q: {float(r.mean()):+.3f} | imagined(1) vs Q: {float(r1.mean()):+.3f} | acts {q.shape[1]}")
print("DONE")
