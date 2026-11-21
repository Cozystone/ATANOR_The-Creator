import sys, torch
from atanor_os.emerge import room3d_live as R
from atanor_os.emerge.core import Core
R.set_slots(R.CHROMA, 1 + R.CHROMA); R.set_motors(False); R.Retina.reflex = True; R.Retina.OPPONENT = "rgb"
net = Core(R.DX, R.M, R.INSIDE, R.PLACE); net.load_state_dict(torch.load(sys.argv[1]), strict=False)
rev = {1: 1, 2: 3, 3: 2}
for walls, roam in ((0, False), (0, True), (3, False), (3, True)):
    for h in (0.0, 0.05, 0.15, 0.4):
        b, s = R.lives(net, 96, 4242, 600, learn=False, loose=0.05, walls=walls, seen="retina", roam=roam, hold=h)
        A, X, OK = b[1], b[0], b[3]
        lack = (X[..., :2].min(-1).values < 0.5) & (OK > 0)
        flip = ((A[:, 1:] == 2) & (A[:, :-1] == 3)) | ((A[:, 1:] == 3) & (A[:, :-1] == 2))
        same = A[:, 1:] == A[:, :-1]
        m = lack[:, 1:]
        print(f"walls {walls} {'move' if roam else 'stay'} | hold {h} | whole {s['whole_at_end']} refills {s['refills']} | left-right flips {float(flip[m].float().mean()):.3f} | same act as before {float(same[m].float().mean()):.2f} | 'do' share {float((A[:, 1:][m] == 4).float().mean()):.2f}", flush=True)
