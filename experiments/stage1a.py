"""Roadmap 3, stage 1a: a forecaster with an inner state, trained to foresee several steps ahead with eyes shut (actions only).
Senses = only what the eye sees now. Questions:
  (1) wipe test - erase the inner state at the moment the wanted thing leaves view: does foreseeing its reappearance get worse?
  (2) does the inner state carry where the out-of-view thing is (a linear read-out, for measuring only)?"""
import sys, time, math, numpy as np, torch, torch.nn as nn
sys.argv = sys.argv[:2]
exec(open(__file__.replace("stage1a.py", "stage0.py"), encoding="utf-8").read().split("gen = torch.Generator().manual_seed(3)")[0])   # same recording, same models

K = 8
class Inner(nn.Module):
    def __init__(self, h=128):
        super().__init__(); self.h = h
        self.g = nn.GRUCell(DX + M + 1, h); self.o = nn.Sequential(nn.Linear(h, 128), nn.GELU(), nn.Linear(128, DX))
    def run(self, X, A, blank, h0=None, wipe=None):
        B, T = A.shape; h = torch.zeros(B, self.h) if h0 is None else h0; outs, hs = [], []
        for t in range(T):
            if wipe is not None:
                h = h * (1 - wipe[:, t:t + 1])
            x = X[:, t] * (1 - blank[:, t:t + 1])
            h = self.g(torch.cat([x, nn.functional.one_hot(A[:, t], M).float(), blank[:, t:t + 1]], -1), h)
            outs.append(self.o(h)); hs.append(h)
        return torch.stack(outs, 1), torch.stack(hs, 1)      # out[:, t] foresees X[:, t+1]


gen = torch.Generator().manual_seed(3)
m = Inner(); opt = torch.optim.Adam(m.parameters(), lr=1e-3); W = 64; t0 = time.time()
for it in range(3000):
    r = torch.randint(0, Xtr.shape[0], (32,), generator=gen); s = int(torch.randint(0, Xtr.shape[1] - 1 - W, (1,), generator=gen))
    X, A, Y = Xtr[r, s:s + W], Atr[r, s:s + W], Xtr[r, s + 1:s + W + 1]
    bl = blanks((32, W), 0.35, gen)                                   # eyes shut a third of the time: it must carry the world in its state
    out, _ = m.run(X, A, bl)
    loss = ((out - Y) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
print(f"trained inner-state forecaster: {time.time() - t0:.0f}s, loss {loss.item():.4f}", flush=True)

pl = slice(3, 3 + 3 * R.NS)
seen = Xte[..., 3 + 2:3 + 3 * R.NS:3]
with torch.no_grad():
    bl0 = torch.zeros(Ate.shape)
    out, hs = m.run(Xte[:, :-1], Ate, bl0)
    leave = (seen[:, :-1] > 0) & (seen[:, 1:] == 0)                    # the thing leaves view at this tick
    wipe = leave.any(-1).float()
    out_w, _ = m.run(Xte[:, :-1], Ate, bl0, wipe=wipe)
    Y = Xte[:, 1:]
    back = ((seen[:, 1:] > 0) & (seen[:, :-1] == 0)).any(-1)          # it comes back into view next tick
    # only reappearances within 12 ticks after a leave (so a wipe sits between)
    recent = torch.zeros_like(back)
    for d in range(1, 13):
        recent[:, d:] |= leave.any(-1)[:, :-d]
    msk = back & recent
    e = lambda o: float(((o - Y)[..., pl] ** 2).mean(-1)[msk].mean())
    print(f"(1) foreseeing a reappearance within 12 ticks of leaving view (n={int(msk.sum())}): kept state {e(out):.4f} | state wiped when it left {e(out_w):.4f}", flush=True)
    # (2) linear read-out of the out-of-view thing's off-front (from the next time it is seen) out of the state, measured only
    j = 2; tgt = Xte[:, 1:, 3 + 3 * j + 1]; vis = seen[:, 1:, j] > 0
    nxt = torch.full_like(tgt, float('nan'))
    for t in range(tgt.shape[1] - 2, -1, -1):
        nxt[:, t] = torch.where(vis[:, t + 1], tgt[:, t + 1], nxt[:, t + 1])
    gone = (seen[:, :-1, j] == 0) & ~torch.isnan(nxt)
    H = hs[gone]; yv = nxt[gone]; n = H.shape[0]; cut = n * 2 // 3
    Hb = torch.cat([H, torch.ones(n, 1)], 1)
    wls = torch.linalg.lstsq(Hb[:cut], yv[:cut, None]).solution
    pred = (Hb[cut:] @ wls)[:, 0]
    r2 = 1 - float(((pred - yv[cut:]) ** 2).mean() / yv[cut:].var())
    print(f"(2) blue out of view: how far off the front it will be when next seen, read linearly from the state (held-out R^2) {r2:.2f}  (n={n})", flush=True)
print("DONE")
