"""Why does the bedrock value rise then collapse? Per life: (1) how well V predicts the wholeness change that really came
(correlation of V(x_t) with the realised discounted return G_t, on this life's own steps; not trained on yet this life);
(2) V's mean level; (3) the advantage spread; (4) share of choices where the value's lean overrides the core's own top pick;
(5) whole lives, fills. One being-population, lam 2, scaling mode argv[3].
  python room_worth_diag.py <seed> <lives> <scene|typical>"""
import sys, time, torch
from atanor_os.emerge import room3d_live as R
from atanor_os.emerge.core import Core, Worth
from atanor_os.emerge.life_gpu import Life
R.set_slots(R.CHROMA, 1 + R.CHROMA)
SEED, LIVES, MODE = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
dev = "cuda"; torch.manual_seed(SEED); N, T = 512, 900
net = Core(R.DX, R.M, R.INSIDE, R.PLACE); net.load_state_dict(torch.load("models/room3d_ret4r.pt", map_location=dev), strict=False)
net = net.to(dev).eval(); net.far = 0.3
L = Life(net, N, R.HOME, dev, seed=SEED, pool="mean"); L.dishab = True
W = Worth(R.DX, R.INSIDE, lam=2.0).to(dev); W.scale_mode = MODE; L.worth = W
orig = net.pull; stat = dict(over=0, n=0, adv=0.0)


def pull(x):                                                # watch how often the value changes the core's own top pick
    sc, st = orig(x)
    lw = W.lean(net, x) if W.on else None
    if lw is not None:
        stat["over"] += int(((sc + lw).argmax(1) != sc.argmax(1)).sum()); stat["n"] += len(x); stat["adv"] += float(lw.abs().mean()) * len(x)
    return sc, st


net.pull = pull
W.lean_orig = W.lean
W.lean = lambda core, x: None                               # the life loop adds nothing itself; we add it inside pull below
def pull2(x):
    sc, st = orig(x)
    lw = W.lean_orig(net, x) if W.on else None
    if lw is not None:
        stat["over"] += int(((sc + lw).argmax(1) != sc.argmax(1)).sum()); stat["n"] += len(x); stat["adv"] += float(lw.abs().mean()) * len(x)
        sc = sc + lw; st = torch.ones_like(st)
    return sc, st
net.pull = pull2
t0 = time.time()
for life in range(LIVES):
    W.on = life >= 1; stat.update(over=0, n=0, adv=0.0)
    L.begin(); xs, ws, al = [], [], []
    with torch.no_grad():
        for tick in range(T):
            L.tick()
            x = L.prev[0]; xs.append(x[:64].clone()); ws.append(torch.where(L.world.dead[:64], torch.zeros(64, device=dev), W.whole(x[:64]))); al.append((~L.world.dead[:64]).float())
            if L.world.dead.all():
                break
        X = torch.stack(xs); Wh = torch.stack(ws); A = torch.stack(al)             # [T, 64]
        Tn = len(X); r = torch.zeros(Tn, 64, device=dev); r[:-1] = Wh[1:] - Wh[:-1]
        G = torch.zeros(Tn, 64, device=dev); g = torch.zeros(64, device=dev)
        for t in range(Tn - 1, -1, -1):
            g = r[t] + W.gamma * g * A[t]; G[t] = g
        ok = A[:-60] > 0; v = W.v(X[:-60].reshape(-1, R.DX))[:, 0].reshape(-1, 64)
        vv, gg = v[ok], G[:-60][ok]
        corr = float(torch.corrcoef(torch.stack([vv, gg]))[0, 1]) if len(vv) > 10 else float("nan")
    print(f"{MODE} seed {SEED} life {life + 1}: whole {float((~L.world.dead).float().mean()):.3f} fills {float(L.fills.mean()):.2f} | V predicts own return r={corr:.3f}, V mean {float(vv.mean()):.3f}, return mean {float(gg.mean()):.3f} | "
          f"value overrides core's pick {stat['over'] / max(stat['n'], 1):.2f}, |lean| {stat['adv'] / max(stat['n'], 1):.2f} | trust {W.trust():.2f} | {time.time() - t0:.0f}s", flush=True)
print("DONE")
