import sys, time, torch
from atanor_os.emerge import room3d_live as R
from atanor_os.emerge.core import Core
from atanor_os.emerge.social_gpu import Social
R.set_slots(R.CHROMA, 1 + R.CHROMA)
net = Core(R.DX, R.M, R.INSIDE, R.PLACE); net.load_state_dict(torch.load(sys.argv[1]), strict=False); net = net.cuda().eval()
n, LIVES, ALONE, T = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
ar = torch.arange(n, device="cuda"); learner = ar % 2 == 0
watch = learner & ((ar // 2) % 2 == 0)                    # half the learners write what they see the elder do
ex = learner if (len(sys.argv) > 6 and sys.argv[6] == "explore") else None
L = Social(net, n, R.HOME, "cuda", seed=11, watch_mask=watch, roads=learner, explore=ex, pool="mean")
t0 = time.time()
for life in range(LIVES + ALONE):
    L.alone = life >= LIVES
    whole, fills = L.live(T)
    a, b = learner & watch, learner & ~watch
    print(f"life {life + 1} {'ALONE' if L.alone else 'with elder'} | watches the elder: whole {float(whole[a].mean()):.3f} refills {float(fills[a].mean()):.2f} (watched {float(L.watched[a].mean()):.1f}) | does not: whole {float(whole[b].mean()):.3f} refills {float(fills[b].mean()):.2f} | elder refills {float(fills[~learner].mean()):.2f} | {time.time() - t0:.0f}s", flush=True)
