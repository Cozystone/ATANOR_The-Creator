"""Dishabituation test (no hand rules): accumulated futility is released when something comes that differs from the forecast more than the being's
own usual (running mean + 2 sd of its forecast error). Groups: core / core + dishab / widened futility 0.3 / widened 0.3 + dishab. 3 lives.
Also: when lacking and a colour is in view, how often does it walk forward (approach)?"""
import sys, time, torch
from atanor_os.emerge import room3d_live as R
from atanor_os.emerge.core import Core
from atanor_os.emerge.life_gpu import Life
R.set_slots(R.CHROMA, 1 + R.CHROMA)
n, LIVES, T, seed = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
for nm, far, dh in (("core", 0.0, False), ("core+dishab", 0.0, True), ("far.3", 0.3, False), ("far.3+dishab", 0.3, True)):
    net = Core(R.DX, R.M, R.INSIDE, R.PLACE); net.load_state_dict(torch.load(sys.argv[1]), strict=False); net = net.cuda().eval(); net.far = far
    L = Life(net, n, R.HOME, "cuda", seed=seed, pool="mean"); L.dishab = dh
    out = []; t0 = time.time()
    for life in range(LIVES):
        L.begin(); run = torch.zeros(n, device="cuda"); best = torch.zeros(n, device="cuda"); seenl = torch.zeros(n, device="cuda"); fw = torch.zeros(n, device="cuda"); nov = torch.zeros(n, device="cuda"); at = torch.zeros(n, device="cuda")
        for t in range(T):
            a = L.tick()[0]; al = ~L.world.dead; tr = ((a == 2) | (a == 3)) & al
            run = torch.where(tr, run + 1, torch.zeros_like(run)); best = torch.maximum(best, run)
            s = L.seen_now.any(1) & (L.world.store.min(1).values < 0.75) & al
            seenl += s.float(); fw += (s & (a == 1)).float(); nov += L.novel.float(); at += al.float()
            if L.world.dead.all(): break
        L.lives += 1
        out.append(f"L{life+1} whole {float((~L.world.dead).float().mean()):.3f} refills {float(L.fills.mean()):.2f} spin {float(best.mean()):.0f} fwd-when-seen {float(fw.sum() / seenl.sum().clamp(min=1)):.2f} novel {float(nov.sum() / at.sum()):.3f}")
    print(f"{nm:13s} seed {seed}: " + " | ".join(out) + f" | {time.time()-t0:.0f}s", flush=True)
print("DONE")
