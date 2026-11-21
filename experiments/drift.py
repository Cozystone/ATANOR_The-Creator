"""How far does the place it keeps by its own steps drift from where it truly is?"""
import sys, math, numpy as np, torch
from atanor_os.emerge import room3d_live as R
from atanor_os.emerge.room3d import Room
from atanor_os.emerge.core import Core
R.set_slots(R.CHROMA, 1 + R.CHROMA); R.set_motors(False); R.Retina.reflex = True; R.Retina.OPPONENT = "rgb"; R.Retina.POOL = "mean"
R.Retina.SPOTS, R.Retina.MAP, R.Retina.IMAGINE = 6, True, True
net = Core(R.DX, R.M, R.INSIDE, R.PLACE); net.load_state_dict(torch.load(sys.argv[1]), strict=False)
gen = torch.Generator().manual_seed(1); out = {150: [], 500: [], 1500: []}
for life in range(8):
    w = Room(seed=life, roam=True, layout=R.HOME); w.drain = 1 / 100000.0
    p0, h0 = w.pos.copy(), w.head
    eye = R.Retina(w); R.look_retina([w], [eye]); walked = 0.0
    for t in range(1, 1501):
        x = torch.tensor(eye.sense())[None]
        with torch.no_grad():
            sc, strong = net.pull(x)
        a = int(torch.multinomial(torch.softmax(sc[0] / 0.05, 0), 1, generator=gen)) if bool(strong[0]) else int(torch.randint(0, R.M, (1,), generator=gen))
        q = w.pos.copy(); w.set_motor(*R.MOTOR[a]); w.advance(R.TICK); R.look_retina([w], [eye]); walked += float(np.hypot(*(w.pos - q)))
        if t in out:
            d = w.pos - p0; c, s = math.cos(-h0), math.sin(-h0)
            true = np.array([d[0] * c - d[1] * s, d[0] * s + d[1] * c])
            out[t].append((float(np.hypot(*(eye.pos - true))), walked, abs(((eye.head - (w.head - h0)) + math.pi) % (2 * math.pi) - math.pi)))
for t, v in out.items():
    v = np.array(v); print(f"after {t} ticks ({t * R.TICK:.0f} s): place off by {np.median(v[:, 0]):.2f} cells (worst {v[:, 0].max():.2f}) after walking {np.median(v[:, 1]):.0f} cells | heading off by {np.degrees(np.median(v[:, 2])):.1f} deg")
