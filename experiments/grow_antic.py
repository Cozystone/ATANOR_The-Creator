"""Anticipated-body sense (founder's line 2026-11-22: no score, no maximiser; a prediction of the body's own raw signals, used only as a sense).
  python grow_antic.py <model> <mode none|antic> <rounds> <n> <T> <upd> <seed>
Organ A(x): for each inside dim separately (room: 2 stores), two time scales (gamma 0.88 ~ 8 ticks, 0.97 ~ 32 ticks): the expected
exponentially-weighted future level of that store, learned by bootstrapping (target = (1-g) * level' + g * A(x'), death = body empty from then).
Sense given to the being: A(x) - level now (how my body tends to go from here), 4 numbers. Choosing is unchanged (pure core + widened futility +
dishabituation): the relief memory and the forecast decide what to do with the new sense. Growing as in grow_body.py (only new connections learn).
mode none: identical growing procedure, no new sense (baseline). Judge every 4 rounds without random tries: whole, refills, wall-pushing."""
import sys, time, torch, torch.nn as nn
from atanor_os.emerge import room3d_live as R
from atanor_os.emerge import room3d_gpu as G
from atanor_os.emerge.core import Core
from atanor_os.emerge.life_gpu import Life
from atanor_os.emerge.body_gpu import expand
R.set_slots(R.CHROMA, 1 + R.CHROMA)
mode, ROUNDS, n, T, UPD, seed = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
torch.manual_seed(seed); dev = "cuda"
net = Core(R.DX, R.M, R.INSIDE, R.PLACE); net.load_state_dict(torch.load(sys.argv[1]), strict=False); net = net.to(dev)
DX0, NI = R.DX, R.INSIDE.stop - R.INSIDE.start
GAM = torch.tensor([0.88] * NI + [0.97] * NI, device=dev)
E = 2 * NI if mode == "antic" else 0
mk = lambda: nn.Sequential(nn.Linear(DX0, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 2 * NI)).to(dev)
A, At = mk(), mk(); At.load_state_dict(A.state_dict()); optA = torch.optim.Adam(A.parameters(), 1e-3)


def antic(w, x):
    with torch.no_grad():
        lv = x[:, R.INSIDE].repeat(1, 2)
        return (A(x[:, :DX0]) - lv).clamp(-1, 1)


extra = antic if E else None
if E:
    net = expand(net, E)
net.far = 0.3
masks = {}
if E:
    m0 = torch.zeros_like(net.f[0].weight); m0[:, DX0:DX0 + E] = 1; masks[net.f[0].weight] = m0
    m4 = torch.zeros_like(net.f[4].weight); m4[DX0:DX0 + E] = 1; masks[net.f[4].weight] = m4
    b4 = torch.zeros_like(net.f[4].bias); b4[DX0:DX0 + E] = 1; masks[net.f[4].bias] = b4
    mg = torch.zeros_like(net.g[0].weight); mg[:, DX0:DX0 + E] = 1; masks[net.g[0].weight] = mg
for p in net.parameters():
    p.requires_grad_(p in masks)
for p, m in masks.items():
    p.register_hook(lambda g, m=m: g * m)
opt = torch.optim.Adam([p for p in masks], lr=3e-4) if masks else None


def life(nn_, seed_, tries, learn, record):
    L = Life(net, nn_, R.HOME, dev, seed=seed_, pool="mean"); L.extra = extra; L.extra_x = True; L.dishab = True; L.try_p = tries; L.learn = learn; L.record = record
    return L


def judge(rounds_done):
    net.eval()
    L = life(512, 9000 + seed, 0.0, False, None); L.begin()
    bump = torch.zeros(512, device=dev); at = torch.zeros(512, device=dev)
    with torch.no_grad():
        for t in range(900):
            L.tick(); al = ~L.world.dead
            bump += ((L.world.felt[:, 2] > 0) & al).float(); at += al.float()
            if L.world.dead.all():
                break
    print(f"{mode:6s} seed {seed} round {rounds_done:2d}: whole {float((~L.world.dead).float().mean()):.3f} refills {float(L.fills.mean()):.2f} pushing-into-wall {float(bump.sum() / at.sum()):.3f} | relief memories {int(net.rel_n)}", flush=True)


judge(0)
store, t0 = [], time.time(); loss = torch.tensor(0.0); la = torch.tensor(0.0)
for r in range(1, ROUNDS + 1):
    net.eval()
    L = life(n, seed * 1000 + r, 0.3, True, []); L.begin()
    with torch.no_grad():
        for t in range(T):
            L.tick()
            if L.world.dead.all():
                break
    X, Ac, X2, OK = [torch.cat([rec[i] for rec in L.record]) for i in range(4)]
    store = (store + [(X, Ac, X2, OK)])[-8:]
    flat = [torch.cat([s[i] for s in store]) for i in range(4)]
    if E:                                                     # the anticipation organ learns first (its target is the body's own future, nothing else)
        for k in range(UPD):
            i = torch.randint(0, len(flat[0]), (4096,), device=dev)
            x, x2, ok = flat[0][i][:, :DX0], flat[2][i][:, :DX0], flat[3][i]
            dead_next = (x2[:, R.INSIDE].sum(1) <= 0.0) | (ok < 0.5)
            lv2 = x2[:, R.INSIDE].repeat(1, 2) * (~dead_next).float()[:, None]
            with torch.no_grad():
                tg = (1 - GAM) * lv2 + GAM * At(x2) * (~dead_next).float()[:, None]
            la = ((A(x) - tg) ** 2).mean(); optA.zero_grad(); la.backward(); optA.step()
            if k % 50 == 0:
                At.load_state_dict(A.state_dict())
        with torch.no_grad():                                 # the sense changes as the organ learns: refresh it in the stored steps
            for s in store:
                s[0][:, DX0:DX0 + E] = antic(None, s[0]); s[2][:, DX0:DX0 + E] = antic(None, s[2])
        flat = [torch.cat([s[i] for s in store]) for i in range(4)]
    net.train()
    for _ in range(UPD if opt is not None else 0):
        i = torch.randint(0, len(flat[0]), (4096,), device=dev)
        loss = net.loss(flat[0][i], flat[1][i], flat[2][i], flat[3][i])
        opt.zero_grad(); loss.backward(); opt.step()
    if r % 4 == 0 or r == ROUNDS:
        print(f"  round {r} foresee {float(loss):.4f} anticipation {float(la):.4f} {time.time() - t0:.0f}s", flush=True)
        judge(r)
print("DONE")
