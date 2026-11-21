import sys, torch
from atanor_os.emerge import room3d_live as R
from atanor_os.emerge.core import Core
R.set_slots(R.CHROMA, 1 + R.CHROMA); R.set_motors(False); R.Retina.reflex = True; R.Retina.OPPONENT = "rgb"; R.Retina.POOL = sys.argv[2]
net = Core(R.DX, R.M, R.INSIDE, R.PLACE); net.load_state_dict(torch.load(sys.argv[1]), strict=False)
for walls, roam in ((0, False), (3, True)):
    for g in (0.0, 1.0, 4.0, 16.0):
        b, s = R.lives(net, 64, 4242, 600, learn=False, loose=0.05, walls=walls, seen="retina", roam=roam, tire=g)
        X, A, X2, OK = b
        lack = (X[..., :2].min(-1).values < 0.5) & (OK > 0)
        rose = ((X2[..., :2] - X[..., :2]) > 0.05).any(-1)
        do = lack & (A == 4)
        print(f"walls {walls} {'move' if roam else 'stay'} | tire {g} | whole {s['whole_at_end']} refills {s['refills']} | 'do' share of lacking steps {float(do.sum() / lack.sum()):.2f} | 'do' presses that filled {float((do & rose).sum() / do.sum().clamp(min=1)):.3f}", flush=True)
