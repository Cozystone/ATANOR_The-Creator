"""The same being (ret4r) gets new body senses and grows on with them (GPU). One sense set per run (argv[2]: none / touch / touch+depth).
Growing = the old way: live (30% tried at random), learn to foresee from every step, keep relief moments. Old weights kept, new inputs start at 0.
Judge every 4 rounds (no random tries; widened futility 0.3 + dishabituation, the adopted way of living): whole, refills, share of ticks
spent pushing into a wall (bumped), longest spin."""
import sys, time, torch
from atanor_os.emerge import room3d_live as R
from atanor_os.emerge.core import Core
from atanor_os.emerge.life_gpu import Life
from atanor_os.emerge.body_gpu import touch, DepthSense, expand
R.set_slots(R.CHROMA, 1 + R.CHROMA)
mode, ROUNDS, n, T, UPD, seed = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
torch.manual_seed(seed)
net = Core(R.DX, R.M, R.INSIDE, R.PLACE); net.load_state_dict(torch.load(sys.argv[1]), strict=False); net = net.cuda()
dep = DepthSense("models/depth0.pt", "cuda") if "depth" in mode else None
extra = None
if mode == "touch":
    extra = touch
elif mode == "touch+depth":
    extra = lambda w: torch.cat([touch(w), dep(w)], 1)
E = 0 if extra is None else (3 if mode == "touch" else 8)
if E:
    net = expand(net, E)
net.far = 0.3
# Only the new sense's own connections learn (old connections kept as they were: relearning the old forecast on these lives broke the being, growwhy*.py).
DX0 = R.DX
masks = {}
if E:
    m0 = torch.zeros_like(net.f[0].weight); m0[:, DX0:DX0 + E] = 1; masks[net.f[0].weight] = m0                 # new inputs into the forecast
    m4 = torch.zeros_like(net.f[4].weight); m4[DX0:DX0 + E] = 1; masks[net.f[4].weight] = m4                     # forecasting the new senses themselves
    b4 = torch.zeros_like(net.f[4].bias); b4[DX0:DX0 + E] = 1; masks[net.f[4].bias] = b4
    mg = torch.zeros_like(net.g[0].weight); mg[:, DX0:DX0 + E] = 1; masks[net.g[0].weight] = mg
for p in net.parameters():
    p.requires_grad_(p in masks)
for p, m in masks.items():
    p.register_hook(lambda g, m=m: g * m)
opt = torch.optim.Adam([p for p in masks], lr=3e-4) if masks else None
out = f"models/body_{mode.replace('+', '_')}_s{seed}.pt"


def judge(rounds_done):
    net.eval()
    L = Life(net, 512, R.HOME, "cuda", seed=9000 + seed, pool="mean"); L.extra = extra; L.dishab = True
    L.begin(); bump = torch.zeros(512, device="cuda"); at = torch.zeros(512, device="cuda"); run = torch.zeros(512, device="cuda"); best = torch.zeros(512, device="cuda")
    with torch.no_grad():
        for t in range(900):
            a = L.tick()[0]; al = ~L.world.dead
            bump += ((L.world.felt[:, 2] > 0) & al).float(); at += al.float()
            tr = ((a == 2) | (a == 3)) & al; run = torch.where(tr, run + 1, torch.zeros_like(run)); best = torch.maximum(best, run)
            if L.world.dead.all(): break
    print(f"{mode:12s} seed {seed} round {rounds_done:2d}: whole {float((~L.world.dead).float().mean()):.3f} refills {float(L.fills.mean()):.2f} pushing-into-wall {float(bump.sum() / at.sum()):.3f} spin {float(best.mean()):.0f} | relief memories {int(net.rel_n)}", flush=True)


judge(0)
store, t0 = [], time.time(); loss = torch.tensor(0.0)
for r in range(1, ROUNDS + 1):
    net.eval()
    L = Life(net, n, R.HOME, "cuda", seed=seed * 1000 + r, pool="mean"); L.extra = extra; L.dishab = True; L.try_p = 0.3; L.learn = True; L.record = []
    L.begin()
    with torch.no_grad():
        for t in range(T):
            L.tick()
            if L.world.dead.all(): break
    X, A, X2, OK = [torch.cat([rec[i] for rec in L.record]) for i in range(4)]
    store = (store + [(X, A, X2, OK)])[-8:]
    flat = [torch.cat([s[i] for s in store]) for i in range(4)]
    net.train()
    for _ in range(UPD if opt is not None else 0):
        i = torch.randint(0, len(flat[0]), (4096,), device="cuda")
        loss = net.loss(flat[0][i], flat[1][i], flat[2][i], flat[3][i])
        opt.zero_grad(); loss.backward(); opt.step()
    if r % 4 == 0 or r == ROUNDS:
        print(f"  round {r} foresee {float(loss):.4f} {time.time() - t0:.0f}s", flush=True)
        judge(r); torch.save(net.state_dict(), out)
print("DONE")
