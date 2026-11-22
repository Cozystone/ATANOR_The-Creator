"""Same test as room_value2.py, but through the core organ (core.Worth) wired into life_gpu.Life (the shared main-line loop).
  python room_worth.py <lam> <seed> <lives> <n>   (lam 0 = no Worth)"""
import sys, time, torch
from atanor_os.emerge import room3d_live as R
from atanor_os.emerge.core import Core, Worth
from atanor_os.emerge.life_gpu import Life
R.set_slots(R.CHROMA, 1 + R.CHROMA)
LAM, SEED, LIVES, N = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
dev = "cuda"; torch.manual_seed(SEED)
net = Core(R.DX, R.M, R.INSIDE, R.PLACE); net.load_state_dict(torch.load("models/room3d_ret4r.pt", map_location=dev), strict=False)
net = net.to(dev).eval(); net.far = 0.3
L = Life(net, N, R.HOME, dev, seed=SEED, pool="mean"); L.dishab = True
if LAM:
    L.worth = Worth(R.DX, R.INSIDE, lam=LAM, m=(R.M if (len(sys.argv) > 5 and sys.argv[5] == "q") else None)).to(dev); L.worth.prio = len(sys.argv) > 5 and sys.argv[5] == "prio"; L.worth.trust_on = len(sys.argv) > 5 and sys.argv[5] == "trust"; L.worth.scale_mode = "scene" if (len(sys.argv) > 5 and sys.argv[5] == "scene") else "typical"
t0 = time.time()
for life in range(LIVES):
    if LAM:
        L.worth.on = life >= 1
    w_, f = L.live(900)
    print(f"{sys.argv[5] if len(sys.argv) > 5 else 'uniform'} earned {L.worth.earned() if LAM else 0:.2f} worth lam {LAM} seed {SEED} life {life + 1}: whole {float(w_.mean()):.3f} | fills/body {float(f.mean()):.2f} | {time.time() - t0:.0f}s", flush=True)
print("DONE")
