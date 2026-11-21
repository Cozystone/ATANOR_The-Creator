"""After it has just emptied a source: how long does it stay near, and what keeps it there?"""
import sys, math, numpy as np, torch
from atanor_os.emerge import room3d_live as R
from atanor_os.emerge.room3d import Room, WATER
from atanor_os.emerge.core import Core
R.set_slots(R.CHROMA, 1 + R.CHROMA); R.set_motors(False); R.Retina.reflex = True; R.Retina.OPPONENT = "rgb"; R.Retina.POOL = "mean"
R.Retina.SPOTS, R.Retina.MAP, R.Retina.IMAGINE = 6, True, True
net = Core(R.DX, R.M, R.INSIDE, R.PLACE); net.load_state_dict(torch.load(sys.argv[1]), strict=False)
gen = torch.Generator().manual_seed(1); ev = []
for life in range(10):
    w = Room(seed=life, roam=True, layout=R.HOME); w.drain = 1 / 240.0; w.store = np.array([0.6, 0.6])
    eye = R.Retina(w); R.look_retina([w], [eye]); fut = torch.zeros(1, R.M); cur = None
    for t in range(2000):
        if w.dead: break
        x = torch.tensor(eye.sense())[None]
        with torch.no_grad():
            sc, strong = net.pull(x); sc = sc - 4.0 * fut
        a = int(torch.multinomial(torch.softmax(sc[0] / 0.05, 0), 1, generator=gen)) if bool(strong[0]) else int(torch.randint(0, R.M, (1,), generator=gen))
        e0 = w.store.copy(); p0 = w.pos.copy()
        w.set_motor(*R.MOTOR[a]); w.advance(R.TICK); R.look_retina([w], [eye])
        more, eased = net.unmet(x, torch.tensor([a]), torch.tensor(eye.sense())[None]); fut = fut * R.LEAK; fut[0, a] += float(more[0])
        if bool(eased[0]): fut.zero_()
        k = int((w.store - e0).argmax())
        if (w.store - e0).max() > 0.05:
            if cur: ev.append(cur)
            j = 2 if k == 0 else 0
            cur = dict(at=p0.copy(), eat=eye.pos.copy(), j=j, t=t, near=0, left=None, alive_near=[], lack_other=float(1 - w.store.min()))
        elif cur and cur["left"] is None:
            d = float(np.hypot(*(w.pos - cur["at"])))
            if d > 5.0: cur["left"] = t - cur["t"]
            if t - cur["t"] in (5, 25, 75):
                cur["alive_near"].append(sum(1 for sp in eye.spots[cur["j"]] if sp[1] > sp[2] and np.hypot(*(sp[0] - cur["eat"])) < 4.0))
    if cur: ev.append(cur)
left = [e["left"] for e in ev if e["left"] is not None]
print("refills", len(ev), "| left the place (5 cells away) after, ticks: median", np.median(left) if left else None, "| never left before the next event or the end:", sum(e["left"] is None for e in ev))
print("places of that colour still believed full within 4 cells of where it ate, at 5 / 25 / 75 ticks after (mean):", np.mean([e["alive_near"] for e in ev if len(e["alive_near"]) == 3], 0).round(2).tolist())
