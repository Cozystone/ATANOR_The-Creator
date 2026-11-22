"""Does the bedrock value (action-value form) learn a disposition that carries to rooms never seen?
  python room_carry.py <seed> <train_lives>
Raise A: one value, trained only in HOME (drain 1/240). Raise B: one value, trained across 8 varied rooms (room_vary seeds 1-8, rotating).
Test: 8 unseen varied rooms (seeds 101-108), learning frozen; compare no value / A / B: whole lives and fills per body (mean over the 8)."""
import sys, time, torch
from atanor_os.emerge import room3d_live as R
from atanor_os.emerge.core import Core, Worth
from atanor_os.emerge.life_gpu import Life
from atanor_os.emerge.room_vary import condition
R.set_slots(R.CHROMA, 1 + R.CHROMA)
SEED, TL = int(sys.argv[1]), int(sys.argv[2])
dev, N, T = "cuda", 512, 900
torch.manual_seed(SEED)
net = Core(R.DX, R.M, R.INSIDE, R.PLACE); net.load_state_dict(torch.load("models/room3d_ret4r.pt", map_location=dev), strict=False)
net = net.to(dev).eval(); net.far = 0.3


def live(rows, drain, W, seed, learn=True):
    L = Life(net, N, rows, dev, seed=seed, pool="mean", drain=drain); L.dishab = True
    if W is not None:
        L.worth = W; L.worth_every = 20 if learn else 10 ** 9
    w, f = L.live(T)
    return float(w.mean()), float(f.mean())


def train(varied):
    W = Worth(R.DX, R.INSIDE, lam=2.0, m=R.M).to(dev)
    t0 = time.time()
    for life in range(TL):
        W.on = life >= 1
        rows, drain = condition(1 + life % 8) if varied else (R.HOME, 1 / 240.0)
        w, f = live(rows, drain, W, SEED * 1000 + life)
        print(f"  train {'varied' if varied else 'single'} life {life + 1}: whole {w:.3f} fills {f:.2f} earned {W.earned():.2f} ({time.time() - t0:.0f}s)", flush=True)
    return W


WA = train(False); WB = train(True)
res = {"none": [], "single-room value": [], "varied-rooms value": []}
for s in range(101, 109):
    rows, drain = condition(s)
    for nm, W in (("none", None), ("single-room value", WA), ("varied-rooms value", WB)):
        res[nm].append(live(rows, drain, W, 50_000 + s, learn=False))
for nm, v in res.items():
    print(f"TEST unseen rooms, {nm:20s}: whole {sum(x[0] for x in v) / len(v):.3f} | fills {sum(x[1] for x in v) / len(v):.2f} | per room fills {[round(x[1], 2) for x in v]}", flush=True)
print("DONE")
