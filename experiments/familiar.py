"""One fixed world, sources that move among a few fixed places. Does the being do better as the world grows familiar (places kept across lives)?"""
import sys, numpy as np, torch
from atanor_os.emerge import room3d_live as R
from atanor_os.emerge.core import Core
R.set_slots(R.CHROMA, 1 + R.CHROMA); R.set_motors(False); R.Retina.reflex = True; R.Retina.OPPONENT = "rgb"
R.Retina.POOL = sys.argv[2]
net = Core(R.DX, R.M, R.INSIDE, R.PLACE); net.load_state_dict(torch.load(sys.argv[1]), strict=False)
for home, walls in ((77, 3), (78, 5)):
    kw = dict(learn=False, loose=0.05, walls=walls, seen="retina", roam=True, home=home, spots=4)
    R.Retina.SPOTS = 1
    a = [R.lives(net, 48, 500 + k, 600, **kw)[1] for k in range(3)]
    print(f"world {home} (walls {walls}) | one place kept, forgotten each life: whole {np.mean([s['whole_at_end'] for s in a]):.2f} refills {np.mean([s['refills'] for s in a]):.2f}", flush=True)
    R.Retina.SPOTS = 6
    a = [R.lives(net, 48, 500 + k, 600, **kw)[1] for k in range(3)]
    print(f"                      | six places kept, forgotten each life: whole {np.mean([s['whole_at_end'] for s in a]):.2f} refills {np.mean([s['refills'] for s in a]):.2f}", flush=True)
    mem, out = None, []
    for life in range(8):
        s = R.lives(net, 48, 600 + life, 600, carry=mem, **kw)[1]; mem = R.lives.memory
        out.append((s['whole_at_end'], s['refills']))
    print("                      | six places kept and carried on, life by life (whole / refills): " + "  ".join(f"{a:.2f}/{b:.1f}" for a, b in out), flush=True)
    print("                      | places held at the end (red, green, blue), mean:", np.mean([[len(l) for l in sp] for sp, _ in mem], 0).round(1).tolist(), flush=True)
