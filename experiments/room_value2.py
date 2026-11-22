"""Bedrock-only value (founder decision 2026-11-22): the being learns V(x) = expected discounted change of its OWN wholeness
(wholeness = 1 - mean squared deficit of the insides, the fixed first bedrock; death = wholeness gone, from then on nothing).
No other reward, no designed weights between drives. Choosing: core pull - futility (unchanged) + lam * (V(foresee(x, act)) - V(x)),
scaled by the spread of that across acts. Life 1 learns only (lam 0), from life 2 the value takes part. lam, gamma on the ledger.
  python room_value2.py <lam> <seed> <lives> <n>"""
import sys, time, torch, torch.nn as nn
from atanor_os.emerge import room3d_live as R
from atanor_os.emerge.core import Core
from atanor_os.emerge.life_gpu import Life
R.set_slots(R.CHROMA, 1 + R.CHROMA)
LAM, SEED, LIVES, N = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
T, GAMMA, dev = 900, 0.97, "cuda"
torch.manual_seed(SEED)
net = Core(R.DX, R.M, R.INSIDE, R.PLACE); net.load_state_dict(torch.load("models/room3d_ret4r.pt", map_location=dev), strict=False)
net = net.to(dev).eval(); net.far = 0.3
mk = lambda: nn.Sequential(nn.Linear(R.DX, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 1)).to(dev)
V, Vt = mk(), mk(); Vt.load_state_dict(V.state_dict()); opt = torch.optim.Adam(V.parameters(), 1e-3)
whole = lambda x: 1 - ((1 - x[..., R.INSIDE]).clamp(min=0) ** 2).mean(-1)
orig_pull = net.pull
use = {"on": False}


def pull(x):
    sc, strong = orig_pull(x)
    if LAM and use["on"]:
        with torch.no_grad():
            adv = V(net.foresee(x))[..., 0] - V(x)
            adv = (adv - adv.mean(1, keepdim=True)) / (adv.std(1, keepdim=True) + 1e-4)
        sc = sc + LAM * adv; strong = torch.ones_like(strong)
    return sc, strong


net.pull = pull
L = Life(net, N, R.HOME, dev, seed=SEED, pool="mean"); L.dishab = True
BUF = 300000
bx = torch.zeros(BUF, R.DX, device=dev); bx2 = torch.zeros(BUF, R.DX, device=dev); br = torch.zeros(BUF, device=dev); bd = torch.zeros(BUF, device=dev); bi = 0; bfull = False
t0 = time.time()
for life in range(LIVES):
    use["on"] = life >= 1
    L.begin(); px = None
    for tick in range(T):
        L.tick()
        x, a, live0 = L.prev
        if px is not None:
            alive = pl > 0
            died = alive & L.world.dead
            r = torch.where(died, -whole(px), whole(x) - whole(px))       # the only reward: own wholeness changing (death: all of it gone)
            k = int(alive.sum()); idx = (torch.arange(k, device=dev) + bi) % BUF
            bx[idx], bx2[idx], br[idx], bd[idx] = px[alive], x[alive], r[alive], died[alive].float()
            bi = (bi + k) % BUF; bfull = bfull or bi < k
        px, pl = x, live0
        if tick % 20 == 0 and (bfull or bi > 4096):
            top = BUF if bfull else bi
            for _ in range(4):
                j = torch.randint(0, top, (2048,), device=dev)
                with torch.no_grad():
                    tg = br[j] + GAMMA * Vt(bx2[j])[:, 0] * (1 - bd[j])
                loss = ((V(bx[j])[:, 0] - tg) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
            if tick % 200 == 0:
                Vt.load_state_dict(V.state_dict())
        if L.world.dead.all():
            break
    print(f"lam {LAM} seed {SEED} life {life + 1}{' (learning only)' if life == 0 else ''}: whole {float((~L.world.dead).float().mean()):.3f} | fills/body {float(L.fills.mean()):.2f} | {time.time() - t0:.0f}s", flush=True)
print("DONE")
