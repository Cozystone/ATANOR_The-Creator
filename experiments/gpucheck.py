import time, math, numpy as np, torch
from atanor_os.emerge import room3d_live as R, room3d_gpu as G
from atanor_os.emerge.room3d import Room
from atanor_os.emerge.core import Core
R.set_slots(R.CHROMA, 1 + R.CHROMA); R.set_motors(False); R.Retina.reflex = True; R.Retina.OPPONENT = "rgb"; R.Retina.POOL = "max"
R.Retina.SPOTS, R.Retina.MAP, R.Retina.IMAGINE = 1, False, False
# 1. same state, compare senses (gpu vs cpu, now-only)
w = G.World(8, R.HOME, "cuda", seed=3)
torch.manual_seed(0)
for _ in range(30):
    w.step(torch.randint(0, 5, (8,), device="cuda"))
xg = G.senses(w, G.Eye(8, "cuda", pool="max"))
diffs = []
for i in range(8):
    r = Room(seed=0, roam=True, layout=R.HOME); r.map = w.map[i].cpu().numpy().astype(np.int8); r.pos = w.pos[i].double().cpu().numpy(); r.head = float(w.head[i]); r.store = w.store[i].double().cpu().numpy(); r._felt = w.felt[i].double().cpu().numpy()
    e = R.Retina(r); R.look_retina([r], [e])
    e.kept = [None] * R.CHROMA
    xc = e.sense()
    diffs.append(np.abs(xc - xg[i].cpu().numpy())[:21].max())
print("max sense difference per body (first look, fovea at the middle):", np.round(diffs, 3))
# 2. speed
for n in (256, 2048, 8192):
    w = G.World(n, R.HOME, "cuda"); eye = G.Eye(n, "cuda")
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(20):
        x = G.senses(w, eye); w.step(torch.randint(0, 5, (n,), device="cuda"))
    torch.cuda.synchronize(); dt = (time.time() - t0) / 20
    print(f"{n} bodies: {dt * 1000:.0f} ms a tick -> {n / dt:.0f} body-ticks a second")
# cpu reference
ws = [Room(seed=i, roam=True, layout=R.HOME) for i in range(64)]; eyes = [R.Retina(x) for x in ws]
t0 = time.time()
for _ in range(10):
    R.look_retina(ws, eyes); [e.sense() for e in eyes]; [x.set_motor(1, 0, False) or x.advance(0.2) for x in ws]
dt = (time.time() - t0) / 10
print(f"CPU now: 64 bodies {dt * 1000:.0f} ms a tick -> {64 / dt:.0f} body-ticks a second")
