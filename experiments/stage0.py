"""Roadmap 3, stage 0 (diagnostic, nothing built into the being): does the world demand memory?
Record lives in the home world. Senses given to the forecasters are ONLY what the eye sees now (no kept places, no map).
Compare a one-step forecaster without history (as the being has now) with one that carries an inner state (GRU),
on the steps right after the wanted thing has left view, and in 'eyes shut' stretches (inputs blanked)."""
import sys, time, math, numpy as np, torch, torch.nn as nn
from atanor_os.emerge import room3d_live as R
from atanor_os.emerge.core import Core
torch.set_num_threads(4); torch.manual_seed(0)
R.set_slots(R.CHROMA, 1 + R.CHROMA); R.set_motors(False); R.Retina.reflex = True; R.Retina.OPPONENT = "rgb"; R.Retina.POOL = "mean"
R.Retina.SPOTS, R.Retina.MAP, R.Retina.IMAGINE = 6, True, True
net = Core(R.DX, R.M, R.INSIDE, R.PLACE); net.load_state_dict(torch.load(sys.argv[1]), strict=False)


def now_only(e):
    """What the eye gives this moment, nothing remembered."""
    kept, way = e.kept, e.way
    e.kept, e.way = [None] * R.CHROMA, [None] * R.CHROMA
    x = e.sense(); e.kept, e.way = kept, way
    return x


def record(n, T, seed):
    from atanor_os.emerge.room3d import Room
    ws = [Room(seed=seed * 100 + i, roam=True, layout=R.HOME) for i in range(n)]
    for w in ws:
        w.drain = 1 / 240.0
    eyes = [R.Retina(w) for w in ws]; R.look_retina(ws, eyes)
    gen = torch.Generator().manual_seed(seed); fut = torch.zeros(n, R.M)
    XS, AS = [], []
    for t in range(T):
        xs = torch.tensor(np.stack([e.sense() for e in eyes]))
        XS.append(np.stack([now_only(e) for e in eyes]))
        with torch.no_grad():
            sc, strong = net.pull(xs); sc = sc - 4.0 * fut
        a = torch.multinomial(torch.softmax(sc / 0.05, 1), 1, generator=gen)[:, 0]
        a = torch.where(strong, a, torch.randint(0, R.M, (n,), generator=gen))
        for i, w in enumerate(ws):
            if w.dead:
                w.dead = False; w.store[:] = 0.8                 # keep the record going (we study the senses, not survival)
            w.set_motor(*R.MOTOR[int(a[i])]); w.advance(R.TICK)
        R.look_retina(ws, eyes)
        x2 = torch.tensor(np.stack([e.sense() for e in eyes]))
        more, eased = net.unmet(xs, a, x2); fut = fut * R.LEAK; fut[torch.arange(n), a] += more; fut[eased] = 0
        AS.append(a.numpy())
    XS.append(np.stack([now_only(e) for e in eyes]))
    return torch.tensor(np.stack(XS, 1)), torch.tensor(np.stack(AS, 1))       # [n, T+1, dx], [n, T]


t0 = time.time()
Xtr, Atr = record(48, 1200, 1)
Xte, Ate = record(16, 1200, 2)
print(f"recorded {Xtr.shape[0] + Xte.shape[0]} lives x {Xtr.shape[1] - 1} ticks in {time.time() - t0:.0f}s", flush=True)
DX, M = R.DX, R.M


class MLP(nn.Module):
    def __init__(self):
        super().__init__(); self.f = nn.Sequential(nn.Linear(DX + M, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, DX))

    def forward(self, X, A, blank):
        x = X * (1 - blank[..., None])
        return x + self.f(torch.cat([x, nn.functional.one_hot(A, M).float()], -1))


class GRU(nn.Module):
    def __init__(self, h=128):
        super().__init__(); self.g = nn.GRU(DX + M + 1, h, batch_first=True); self.o = nn.Sequential(nn.Linear(h + DX, 128), nn.GELU(), nn.Linear(128, DX))

    def forward(self, X, A, blank):
        x = X * (1 - blank[..., None])
        hs, _ = self.g(torch.cat([x, nn.functional.one_hot(A, M).float(), blank[..., None]], -1))
        return x + self.o(torch.cat([hs, x], -1))


def blanks(shape, p, gen):
    """'eyes shut' stretches: runs of 4-16 ticks with no input."""
    b = torch.zeros(shape)
    n = int(shape[0] * shape[1] * p / 10)
    s = torch.randint(0, shape[1], (n,), generator=gen); r = torch.randint(0, shape[0], (n,), generator=gen); L = torch.randint(4, 17, (n,), generator=gen)
    for i in range(n):
        b[r[i], s[i]:s[i] + L[i]] = 1
    return b


gen = torch.Generator().manual_seed(3)
models = {"no history (as now)": MLP(), "inner state (GRU)": GRU()}
W = 64
for name, m in models.items():
    opt = torch.optim.Adam(m.parameters(), lr=1e-3); t0 = time.time()
    for it in range(2500):
        r = torch.randint(0, Xtr.shape[0], (32,), generator=gen); s = int(torch.randint(0, Xtr.shape[1] - 1 - W, (1,), generator=gen))
        X, A, Y = Xtr[r, s:s + W], Atr[r, s:s + W], Xtr[r, s + 1:s + W + 1]
        bl = blanks((32, W), 0.15, gen)
        loss = ((m(X, A, bl) - Y) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"trained {name}: {time.time() - t0:.0f}s, last loss {loss.item():.4f}", flush=True)

# evaluation on held-out lives: whole sequences
seen = Xte[..., 5::3][..., :R.NS] if False else Xte[..., 3 + 2:3 + 3 * R.NS:3]          # 'known' flag per colour slot (now only)
bl_te = blanks(Ate.shape, 0.15, torch.Generator().manual_seed(9))
with torch.no_grad():
    out = {k: m(Xte[:, :-1], Ate, bl_te) for k, m in models.items()}
Y = Xte[:, 1:]
pl = slice(3, 3 + 3 * R.NS)
gone = (seen[:, :-1] == 0) & torch.stack([torch.nn.functional.max_pool1d(seen[:, :-1, j][:, None].float(), 21, 1, 10)[:, 0] for j in range(R.NS)], -1).bool()      # out of view now, in view within 10 ticks around
back = gone.any(-1) & (seen[:, 1:] > 0).any(-1)                                    # and it comes back into view next tick: can it foresee where?
masks = {"all ticks": torch.ones_like(back), "eyes shut stretches": bl_te.bool(), "wanted thing just out of view and reappearing next tick": back}
for mk, m in masks.items():
    print(f"{mk} (n={int(m.sum())}):", " | ".join(f"{k}: place-sense error {float(((out[k] - Y)[..., pl] ** 2).mean(-1)[m].mean()):.4f}" for k in models), flush=True)
print("DONE")
