"""No hand rules (no walk-on, no roads, no less-walked). Only the core's futility, widened (far): an act also tires when it was foreseen
to bring the recalled relief moment nearer and did not. Does the being learn by itself that turning in place is futile? far = 0 / 0.3 / 1 / 3."""
import sys, time, torch
from atanor_os.emerge import room3d_live as R
from atanor_os.emerge.core import Core
from atanor_os.emerge.life_gpu import Life
R.set_slots(R.CHROMA, 1 + R.CHROMA)
n, LIVES, T, seed = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
for far in (0.0, 0.3, 1.0, 3.0):
    net = Core(R.DX, R.M, R.INSIDE, R.PLACE); net.load_state_dict(torch.load(sys.argv[1]), strict=False); net = net.cuda().eval(); net.far = far
    L = Life(net, n, R.HOME, "cuda", seed=seed, pool="mean")
    out = []; t0 = time.time()
    for life in range(LIVES):
        run = torch.zeros(n, device="cuda"); best = torch.zeros(n, device="cuda"); turns = torch.zeros(n, device="cuda"); at = torch.zeros(n, device="cuda")
        L.begin()
        for t in range(T):
            a = L.tick()[0]; al = ~L.world.dead; tr = ((a == 2) | (a == 3)) & al
            run = torch.where(tr, run + 1, torch.zeros_like(run)); best = torch.maximum(best, run); turns += tr.float(); at += al.float()
            if L.world.dead.all(): break
        L.lives += 1
        out.append(f"L{life+1} whole {float((~L.world.dead).float().mean()):.3f} refills {float(L.fills.mean()):.2f} turning {float((turns / at.clamp(min=1)).mean()):.2f} longest spin {float(best.mean()):.0f}")
    print(f"far {far}: " + " | ".join(out) + f" | {time.time()-t0:.0f}s", flush=True)
print("DONE")
