import sys, time, torch
from atanor_os.emerge import room3d_live as R
from atanor_os.emerge.core import Core
from atanor_os.emerge.life_gpu import Life
R.set_slots(R.CHROMA, 1 + R.CHROMA)
net = Core(R.DX, R.M, R.INSIDE, R.PLACE); net.load_state_dict(torch.load(sys.argv[1]), strict=False); net = net.cuda().eval()
n, LIVES, T = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
half = torch.arange(n, device="cuda") >= n // 2
L = Life(net, n, R.HOME, "cuda", seed=int(sys.argv[5]) if len(sys.argv) > 5 else 0, roads=half, pool="mean")
t0 = time.time()
for life in range(LIVES):
    whole, fills = L.live(T)
    print(f"life {life + 1} | no roads: whole {float(whole[~half].mean()):.3f} refills {float(fills[~half].mean()):.2f} | roads: whole {float(whole[half].mean()):.3f} refills {float(fills[half].mean()):.2f} | roads recalled {float(L.roads.used[half].mean()):.0f} | {time.time() - t0:.0f}s", flush=True)
