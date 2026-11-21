"""How much does the being spin in place? Longest run of consecutive turns, share of ticks turning, with explore on/off."""
import sys, torch
from atanor_os.emerge import room3d_live as R
from atanor_os.emerge.core import Core
from atanor_os.emerge.life_gpu import Life
from atanor_os.emerge.roads_gpu import RoadsGPU
R.set_slots(R.CHROMA, 1 + R.CHROMA)
net = Core(R.DX, R.M, R.INSIDE, R.PLACE); net.load_state_dict(torch.load(sys.argv[1]), strict=False); net = net.cuda().eval()
n, T, hold = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 0
ar = torch.arange(n, device="cuda"); ex = ar % 2 == 0
L = Life(net, n, R.HOME, "cuda", seed=5, roads=torch.ones(n, dtype=torch.bool, device="cuda"), explore=ex, pool="mean")
L.roads = RoadsGPU(n, "cuda", L=2000)
if hold:
    L.walkon = True

for life in range(2):
    L.begin(); run = torch.zeros(n, device="cuda"); best = torch.zeros(n, device="cuda"); turns = torch.zeros(n, device="cuda"); alive_t = torch.zeros(n, device="cuda")
    for t in range(T):
        a = L.tick()[0]
        al = ~L.world.dead
        tr = ((a == 2) | (a == 3)) & al
        run = torch.where(tr, run + 1, torch.zeros_like(run)); best = torch.maximum(best, run)
        turns += tr.float(); alive_t += al.float()
    for nm, m in (("explore", ex), ("no explore", ~ex)):
        print(f"life {life+1} {nm}: turning share {float((turns[m] / alive_t[m].clamp(min=1)).mean()):.2f}, longest spin {float(best[m].mean()):.1f} ticks (max {float(best[m].max()):.0f}), whole {float((~L.world.dead[m]).float().mean()):.3f}, refills {float(L.fills[m].mean()):.2f}", flush=True)
