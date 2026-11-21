"""Ear: can a being with no built memory (eye now only) find what it has never seen, by sound through the doors?
Same being (ret4r) widened for four ear senses (zero start), relief memory emptied, lives in the home world with ears and learns its forecast
from that; then compared with ears on / ears muted. The pull may also weigh the ear senses (core.also) - how loud it was at relief moments."""
import sys, time, numpy as np, torch
from atanor_os.emerge import room3d_live as R
from atanor_os.emerge.room3d import Room
from atanor_os.emerge.core import Core
torch.set_num_threads(4); torch.manual_seed(0)
R.set_slots(R.CHROMA, 1 + R.CHROMA); R.set_motors(False); R.Retina.reflex = True; R.Retina.OPPONENT = "rgb"; R.Retina.POOL = "mean"
R.Retina.SPOTS, R.Retina.MAP, R.Retina.IMAGINE = 1, False, False
old = torch.load(sys.argv[1]); dx0 = R.DX
R.set_slots(R.CHROMA, 1 + R.CHROMA + 4); R.Retina.EAR = True
net = Core(R.DX, R.M, R.INSIDE, R.PLACE); sd = net.state_dict()
for k, v in old.items():
    if k not in sd:
        continue
    t = sd[k].clone()
    if k == "f.0.weight":
        t.zero_(); t[:, :dx0] = v[:, :dx0]; t[:, R.DX:] = v[:, dx0:]
    elif k in ("f.4.weight",):
        t.zero_(); t[:dx0] = v
    elif k in ("f.4.bias", "heavy", "pl"):
        t[:dx0] = v
    elif v.shape == t.shape:
        t = v.clone()
    sd[k] = t
sd["rel_n"].zero_(); sd["rel_c"].zero_()
net.load_state_dict(sd, strict=False)
EAR = slice(R.DX - 4, R.DX)


def now_only(e):
    kept, way = e.kept, e.way
    e.kept, e.way = [None] * R.CHROMA, [None] * R.CHROMA
    x = e.sense(); e.kept, e.way = kept, way
    return x


def live(n, T, seed, ears=True, learn=False, use=True):
    ws = [Room(seed=seed * 100 + i, roam=True, layout=R.HOME) for i in range(n)]
    for w in ws:
        w.drain = 1 / 240.0; w.store = np.array([0.7, 0.7])
    eyes = [R.Retina(w) for w in ws]; R.look_retina(ws, eyes)
    gen = torch.Generator().manual_seed(seed); fut = torch.zeros(n, R.M); fills = np.zeros(n); D = ([], [], [], [])
    net.also = EAR if use else None
    for t in range(T):
        x = torch.tensor(np.stack([now_only(e) for e in eyes]))
        if not ears:
            x[:, EAR] = 0
        with torch.no_grad():
            sc, strong = net.pull(x); sc = sc - 4.0 * fut
        a = torch.multinomial(torch.softmax(sc / 0.05, 1), 1, generator=gen)[:, 0]
        a = torch.where(strong, a, torch.randint(0, R.M, (n,), generator=gen))
        if learn:
            a = torch.where(torch.rand(n, generator=gen) < 0.2, torch.randint(0, R.M, (n,), generator=gen), a)
        live_ = torch.tensor([not w.dead for w in ws]); e0 = np.stack([w.store.copy() for w in ws])
        for i, w in enumerate(ws):
            w.set_motor(*R.MOTOR[int(a[i])]); w.advance(R.TICK)
        R.look_retina(ws, eyes)
        x2 = torch.tensor(np.stack([now_only(e) for e in eyes]))
        if not ears:
            x2[:, EAR] = 0
        e1 = np.stack([w.store for w in ws]); rose = torch.tensor(((e1 - e0) > 0.05).any(1)) & live_
        fills += rose.numpy()
        with torch.no_grad():
            more, eased = net.unmet(x, a, x2)
        fut = fut * R.LEAK; fut[torch.arange(n), a] += more; fut[eased] = 0
        if learn:
            if rose.any():
                net.keep(torch.tensor((e1 - e0), dtype=torch.float32)[rose], x[rose], a[rose])
            D[0].append(x); D[1].append(a); D[2].append(x2); D[3].append(live_.float())
            for w in ws:
                if w.dead:
                    w.dead = False; w.store[:] = 0.8
    if learn:
        return [torch.stack(d, 1) for d in D]
    return dict(whole=float(np.mean([not w.dead for w in ws])), refills=float(fills.mean()), seconds=float(np.mean([w.clock for w in ws])))


t0 = time.time()
opt = torch.optim.AdamW(net.parameters(), lr=5e-4)
for r in range(6):                                                   # live with ears, learn the forecast from it (sound included), keep relief moments
    data = live(48, 600, 10 + r, learn=True)
    flat = [d.flatten(0, 1) for d in data]
    for _ in range(800):
        i = torch.randint(0, len(flat[0]), (4096,))
        loss = net.loss(*[f[i] for f in flat]); opt.zero_grad(); loss.backward(); opt.step()
    print(f"round {r + 1} | {time.time() - t0:.0f}s | foresee {loss.item():.4f} | relief memories {int(net.rel_n)} | loudness at relief moments (water L, R, food L, R) {net.rel_x[:int(net.rel_n), EAR].mean(0).numpy().round(2).tolist()}", flush=True)
torch.save(net.state_dict(), sys.argv[2])
for name, kw in (("ears, the pull weighs what it heard", dict()), ("ears, the pull ignores sound", dict(use=False)), ("ears muted", dict(ears=False, use=False))):
    out = [live(24, 1500, 70 + k, **kw) for k in range(2)]
    print(f"{name} | whole {np.mean([o['whole'] for o in out]):.2f} | refills {np.mean([o['refills'] for o in out]):.2f} | seconds {np.mean([o['seconds'] for o in out]):.0f} of 300 | {time.time() - t0:.0f}s", flush=True)
print("DONE")
