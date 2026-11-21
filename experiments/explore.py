import sys, time, torch
from atanor_os.emerge import room3d_live as R
from atanor_os.emerge.core import Core
from atanor_os.emerge.life_gpu import Life
R.set_slots(R.CHROMA, 1 + R.CHROMA)
net = Core(R.DX, R.M, R.INSIDE, R.PLACE); net.load_state_dict(torch.load(sys.argv[1]), strict=False); net = net.cuda().eval()
n, LIVES, T = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
g = torch.arange(n, device="cuda") % 3                     # 0 none, 1 roads, 2 roads + fresh
L = Life(net, n, R.HOME, "cuda", seed=5, roads=g >= 1, explore=g == 2, pool="mean")
t0 = time.time()
for life in range(LIVES):
    whole, fills = L.live(T)
    print(f"life {life + 1} | " + " | ".join(f"{nm}: whole {float(whole[g == i].mean()):.3f} refills {float(fills[g == i].mean()):.2f}" for i, nm in enumerate(("none", "roads", "roads + less-walked"))) + f" | {time.time() - t0:.0f}s", flush=True)
