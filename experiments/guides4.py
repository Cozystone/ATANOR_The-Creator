"""With walk-on in place: which guide helps or hurts? Groups: none / roads only / less-walked only / both. 3 lives."""
import sys, time, torch
from atanor_os.emerge import room3d_live as R
from atanor_os.emerge.core import Core
from atanor_os.emerge.life_gpu import Life
from atanor_os.emerge.roads_gpu import RoadsGPU
R.set_slots(R.CHROMA, 1 + R.CHROMA)
net = Core(R.DX, R.M, R.INSIDE, R.PLACE); net.load_state_dict(torch.load(sys.argv[1]), strict=False); net = net.cuda().eval()
n, LIVES, T = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
g = torch.arange(n, device="cuda") % 4
L = Life(net, n, R.HOME, "cuda", seed=int(sys.argv[5]), roads=(g == 1) | (g == 3), explore=(g == 2) | (g == 3), pool="mean")
L.roads = RoadsGPU(n, "cuda", L=2000); L.walkon = True
t0 = time.time()
for life in range(LIVES):
    whole, fills = L.live(T)
    print(f"life {life+1} | " + " | ".join(f"{nm}: whole {float(whole[g == i].mean()):.3f} refills {float(fills[g == i].mean()):.2f}" for i, nm in enumerate(("none", "roads", "less-walked", "both"))) + f" | {time.time()-t0:.0f}s", flush=True)
print("DONE")
