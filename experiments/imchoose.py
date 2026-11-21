"""Choosing by imagination (no hand rules). Three ways of living, same world and seed, 3 lives:
 A pure core | B core + widened futility 0.3 | C imagination: when lacking, roll act sequences in the imagination organ and take the first act of the one
 whose imagined scene comes nearest to the scenes just before that lack was relieved (own lived memory, fixed size). If nothing imagined comes nearer
 than now, the core chooses as before. Relief scenes are kept across lives (bounded)."""
import sys, time, torch
from atanor_os.emerge import room3d_live as R
from atanor_os.emerge import room3d_gpu as G
from atanor_os.emerge.core import Core
from atanor_os.emerge.life_gpu import Life
from atanor_os.emerge.eyewm import WM, Imagine
R.set_slots(R.CHROMA, 1 + R.CHROMA)
n, LIVES, T, seed = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
wm = WM().cuda().eval(); wm.load_state_dict(torch.load("models/wm0.pt"))
modes = sys.argv[6].split(",") if len(sys.argv) > 6 else ["core", "far", "imagine"]
for mode in modes:
    net = Core(R.DX, R.M, R.INSIDE, R.PLACE); net.load_state_dict(torch.load(sys.argv[1]), strict=False); net = net.cuda().eval()
    net.far = 0.3 if mode == "far" else 0.0
    L = Life(net, n, R.HOME, "cuda", seed=seed, pool="mean")
    im = Imagine(wm, n, K=48, H=8, seed=seed); im.weigh = mode == "imagine_w"
    out = []; t0 = time.time()
    for life in range(LIVES):
        L.begin(); run = torch.zeros(n, device="cuda"); best = torch.zeros(n, device="cuda"); turns = torch.zeros(n, device="cuda"); at = torch.zeros(n, device="cuda"); used = torch.zeros(n, device="cuda")
        zprev = None
        for t in range(T):
            w = L.world
            if mode.startswith("imagine"):
                with torch.no_grad():
                    z = wm.encode(w.picture().permute(0, 3, 1, 2).clamp(0, 1))[0]
                lack = (w.store.min(1).values < 0.75) & ~w.dead
                pick, _ = im.choose(z, w.store.argmin(1), lack)
                L.override = lambda a, pick=pick: torch.where(pick >= 0, pick, a)
            a, sc, strong, filled = L.tick()
            if mode.startswith("imagine"):
                im.remember(z, filled); used += (pick >= 0).float()
            al = ~L.world.dead; tr = ((a == 2) | (a == 3)) & al
            run = torch.where(tr, run + 1, torch.zeros_like(run)); best = torch.maximum(best, run); turns += tr.float(); at += al.float()
            if L.world.dead.all(): break
        L.lives += 1
        out.append(f"L{life+1} whole {float((~L.world.dead).float().mean()):.3f} refills {float(L.fills.mean()):.2f} turning {float((turns / at.clamp(min=1)).mean()):.2f} spin {float(best.mean()):.0f}" + (f" imagined {float((used / at.clamp(min=1)).mean()):.2f}" if mode.startswith("imagine") else ""))
    print(f"{mode:8s}: " + " | ".join(out) + f" | {time.time()-t0:.0f}s", flush=True)
print("DONE")
