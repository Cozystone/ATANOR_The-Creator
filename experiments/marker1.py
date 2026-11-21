"""Self marker 1: does the being tell its own doing from the world's doing?
In the room the world sometimes moves a source by itself (not after eating). Among moments when what it sees changes a lot,
compare those it caused (its own turn or step) with those the world caused (a source jumped). Two readings, both from its
own forecast, nothing added:
  surprise = how wrong its forecast of the next feeling was
  agency   = how much its forecast differs between what it did and standing still (efference-copy-like: 'this change was mine')
Score: how well each separates own from world changes (AUC, 0.5 = cannot tell)."""
import sys, math, numpy as np, torch
from atanor_os.emerge import room3d_live as R
from atanor_os.emerge.room3d import Room, WATER, FOOD
from atanor_os.emerge.core import Core
torch.set_num_threads(4)
R.set_slots(R.CHROMA, 1 + R.CHROMA); R.set_motors(False); R.Retina.reflex = True; R.Retina.OPPONENT = "rgb"; R.Retina.POOL = "mean"
R.Retina.SPOTS, R.Retina.MAP, R.Retina.IMAGINE = 1, False, False
net = Core(R.DX, R.M, R.INSIDE, R.PLACE); net.load_state_dict(torch.load(sys.argv[1]), strict=False)
pl = slice(3, 3 + 3 * R.NS)


def now_only(e):
    kept, way = e.kept, e.way
    e.kept, e.way = [None] * R.CHROMA, [None] * R.CHROMA
    x = e.sense(); e.kept, e.way = kept, way
    return x


rows = []
rng = np.random.default_rng(0); gen = torch.Generator().manual_seed(0)
for life in range(24):
    w = Room(seed=life, walls=0); w.drain = 1 / 100000.0
    eye = R.Retina(w); R.look_retina([w], [eye])
    for t in range(800):
        x = torch.tensor(now_only(eye))[None]
        with torch.no_grad():
            sc, strong = net.pull(x)
        a = int(torch.multinomial(torch.softmax(sc[0] / 0.05, 0), 1, generator=gen)) if bool(strong[0]) else int(rng.integers(0, R.M))
        world = False
        if rng.random() < 0.03:                                     # the world moves a source by itself
            k = WATER if rng.random() < 0.5 else FOOD
            free = np.argwhere(w.map == 0); free = free[np.abs(free + 0.5 - w.pos).max(1) > 1.5]
            w.map[w.map == k] = 0; w.map[tuple(free[rng.integers(len(free))])] = k; world = True
        w.set_motor(*R.MOTOR[a]); w.advance(R.TICK); R.look_retina([w], [eye])
        x2 = torch.tensor(now_only(eye))[None]
        with torch.no_grad():
            fo = net.foresee(x)[0]
        change = float(((x2[0] - x[0])[pl] ** 2).sum())
        surprise = float(((fo[a] - x2[0])[pl] ** 2).sum())
        agency = float(((fo[a] - fo[0])[pl] ** 2).sum())
        rows.append((change, surprise, agency, world, a in (1, 2, 3)))
r = np.array(rows, dtype=float)
big = r[:, 0] > np.quantile(r[:, 0], 0.8)
own = big & (r[:, 3] == 0) & (r[:, 4] == 1); wor = big & (r[:, 3] == 1)


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean())


print(f"big changes: caused by its own move {int(own.sum())}, by the world {int(wor.sum())}")
print(f"surprise separates them (world > own): AUC {auc(r[wor, 1], r[own, 1]):.2f}")
print(f"agency separates them (own > world):   AUC {auc(r[own, 2], r[wor, 2]):.2f}")
