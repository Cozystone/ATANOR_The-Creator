import sys, time, torch
from atanor_os.emerge import room3d_live as R
from atanor_os.emerge.core import Core
from atanor_os.emerge.social_gpu import Social
R.set_slots(R.CHROMA, 1 + R.CHROMA)
net = Core(R.DX, R.M, R.INSIDE, R.PLACE); net.load_state_dict(torch.load(sys.argv[1]), strict=False); net = net.cuda().eval()
n, LIVES, ALONE, T = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
ar = torch.arange(n, device="cuda"); learner = ar % 2 == 0; grp = (ar // 2) % 3        # 0: follows + writes what it sees, 1: follows only, 2: neither
L = Social(net, n, R.HOME, "cuda", seed=13, watch_mask=learner & (grp == 0), roads=learner, explore=learner, pool="mean")
L.roads = __import__("atanor_os.emerge.roads_gpu", fromlist=["RoadsGPU"]).RoadsGPU(n, "cuda", L=2000)
L.follow = learner & (grp <= 1)
names = ("follows + writes what it sees", "follows only", "neither")
t0 = time.time()
for life in range(LIVES + ALONE):
    L.alone = life >= LIVES
    whole, fills = L.live(T)
    print(f"life {life + 1} {'ALONE' if L.alone else 'with elder'} | " + " | ".join(f"{nm}: whole {float(whole[learner & (grp == i)].mean()):.3f} refills {float(fills[learner & (grp == i)].mean()):.2f}" + (f" (watched {float(L.watched[learner & (grp == i)].mean()):.1f})" if i == 0 else "") for i, nm in enumerate(names)) + f" | {time.time() - t0:.0f}s", flush=True)
print("DONE")
