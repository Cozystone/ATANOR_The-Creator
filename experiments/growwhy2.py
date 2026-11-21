"""Why does growing on break the being? One round of lives with 30% random tries (old senses), then:
 A as it was (nothing changed) | B only the relief memory takes the new moments | C only the forecast relearns (relief memory as it was)
 | D both (what grow_body did) | E forecast relearns mixed: new steps + steps of the being living its own way (no random tries)."""
import sys, copy, torch
from atanor_os.emerge import room3d_live as R
from atanor_os.emerge.core import Core
from atanor_os.emerge.life_gpu import Life
R.set_slots(R.CHROMA, 1 + R.CHROMA)
base = Core(R.DX, R.M, R.INSIDE, R.PLACE); base.load_state_dict(torch.load(sys.argv[1]), strict=False); base = base.cuda(); base.far = 0.3


def gather(net, try_p, learn, seed, n=512, T=600):
    L = Life(net, n, R.HOME, "cuda", seed=seed, pool="mean"); L.dishab = True; L.try_p = try_p; L.learn = learn; L.record = []
    L.begin()
    with torch.no_grad():
        for t in range(T):
            L.tick()
    return [torch.cat([r[i] for r in L.record]) for i in range(4)]


def judge(net, nm):
    net.eval()
    L = Life(net, 512, R.HOME, "cuda", seed=9001, pool="mean"); L.dishab = True
    w, f = L.live(900)
    print(f"{nm:44s}: whole {float(w.mean()):.3f} refills {float(f.mean()):.2f} | relief memories by store {net.rel_de[:int(net.rel_n)].argmax(1).bincount(minlength=2).tolist()} acts {net.rel_a[:int(net.rel_n)].bincount(minlength=5).tolist()}", flush=True)


def train(net, data, UPD=1500):
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4); net.train()
    for _ in range(UPD):
        i = torch.randint(0, len(data[0]), (4096,), device="cuda")
        loss = net.loss(*[d[i] for d in data]); opt.zero_grad(); loss.backward(); opt.step()


data = gather(copy.deepcopy(base), 0.3, False, 1001)
rose = ((data[2][:, 0:2] - data[0][:, 0:2]) > 0.05).any(1)
print(f"steps {len(rose)}, of which the body filled: {int(rose.sum())} ({float(rose.float().mean()) * 100:.3f}%)", flush=True)


def train_bal(net, data, share, UPD=1500):
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4); net.train()
    pos = rose.nonzero()[:, 0]; k = int(4096 * share)
    for _ in range(UPD):
        i = torch.cat([torch.randint(0, len(data[0]), (4096 - k,), device="cuda"), pos[torch.randint(0, len(pos), (k,), device="cuda")]])
        loss = net.loss(*[d[i] for d in data]); opt.zero_grad(); loss.backward(); opt.step()


for share in (0.01, 0.05):
    ng = copy.deepcopy(base); train_bal(ng, data, share); judge(ng, f"G relearned, filling moments {int(share * 100)}% of each batch")
print("DONE")
