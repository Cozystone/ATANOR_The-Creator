"""Roadmap 3, stage 1b: the inner state steers. In the home world, compare
  A  hand-built memory (kept places + map + walking it over in the head) - today's being
  B  no memory at all: only what the eye sees now, one-step pull (the floor)
  C  no hand-built memory: what the eye sees now + the inner state, choosing by action chunks foreseen with eyes shut
  C' as C, but the inner state wiped every 4 ticks (wipe test)
"""
import sys, time, math, numpy as np, torch
from atanor_os.emerge import room3d_live as R
from atanor_os.emerge.room3d import Room
from atanor_os.emerge.core import Core
from atanor_os.emerge.inner import Inner, train_inner, chunk_pull
torch.set_num_threads(4); torch.manual_seed(0)
R.set_slots(R.CHROMA, 1 + R.CHROMA); R.set_motors(False); R.Retina.reflex = True; R.Retina.OPPONENT = "rgb"; R.Retina.POOL = "mean"
net = Core(R.DX, R.M, R.INSIDE, R.PLACE); net.load_state_dict(torch.load(sys.argv[1]), strict=False)
SAVE = sys.argv[2]
K, LAM = 6, float(sys.argv[3]) if len(sys.argv) > 3 else 1.0


def now_only(e):
    kept, way = e.kept, e.way
    e.kept, e.way = [None] * R.CHROMA, [None] * R.CHROMA
    x = e.sense(); e.kept, e.way = kept, way
    return x


PIS = {}


def felt(ws):
    return torch.tensor(np.stack([PIS.setdefault(id(w), R.PathIntegrator()).feel(w) for w in ws]), dtype=torch.float32)


def felt_old(ws):
    """몸이 느낀 제 움직임: 앞으로 간 거리, 옆으로 간 거리, 돈 각 (방금 한 걸음)."""
    return torch.tensor(np.stack([[w._felt[0] / 0.4, w._felt[5] / 0.4, w._felt[1] / 0.42] for w in ws]), dtype=torch.float32)


def live(mode, n, T, seed, inner=None, record=False, wipe=0):
    hand = mode == "A"
    R.Retina.SPOTS, R.Retina.MAP, R.Retina.IMAGINE = (6, True, True) if hand else (1, False, False)
    ws = [Room(seed=seed * 100 + i, roam=True, layout=R.HOME) for i in range(n)]
    PIS.clear()
    for w in ws:
        w.drain = 1 / 240.0; w.store = np.array([0.7, 0.7])
    eyes = [R.Retina(w) for w in ws]; R.look_retina(ws, eyes)
    gen = torch.Generator().manual_seed(seed); fut = torch.zeros(n, R.M)
    h = inner.zero(n) if inner is not None else None
    fills = np.zeros(n); XS, AS, TR, MS = [], [], [], []
    for t in range(T):
        xs = torch.tensor(np.stack([e.sense() if hand else now_only(e) for e in eyes]))
        if record:
            XS.append(np.stack([now_only(e) for e in eyes]))
            tr = []
            for w in ws:                                                         # truth, for measuring only: where blue and red are, body-centred (sin, cos of the bearing, distance)
                row = []
                for kind in (3, 2):
                    d = np.argwhere(w.map == kind)[0] + 0.5 - w.pos; b = math.atan2(d[1], d[0]) - w.head
                    row += [math.sin(b), math.cos(b), float(np.hypot(*d))]
                tr.append(row)
            TR.append(tr)
        with torch.no_grad():
            sc, strong = net.pull(xs)
            if inner is not None:
                if wipe and t % wipe == 0:
                    h = inner.zero(n)
                mu, wt, scale = net._last
                sc = sc + LAM * chunk_pull(inner, h, xs, mu, wt, scale, K, keep=(4,))
                strong = strong | (scale > 0)
            sc = sc - 4.0 * fut
        a = torch.multinomial(torch.softmax(sc / 0.05, 1), 1, generator=gen)[:, 0]
        a = torch.where(strong, a, torch.randint(0, R.M, (n,), generator=gen))
        live_ = np.array([not w.dead for w in ws]); e0 = np.stack([w.store.copy() for w in ws])
        for i, w in enumerate(ws):
            w.set_motor(*R.MOTOR[int(a[i])]); w.advance(R.TICK)
        R.look_retina(ws, eyes)
        x2 = torch.tensor(np.stack([e.sense() if hand else now_only(e) for e in eyes]))
        with torch.no_grad():
            if inner is not None:
                h, _ = inner.step(h, xs, a, motion=felt(ws) if inner.mo else None)
            more, eased = net.unmet(xs, a, x2)
        fut = fut * R.LEAK; fut[torch.arange(n), a] += more; fut[eased] = 0
        fills += ((np.stack([w.store for w in ws]) - e0) > 0.05).any(1) & live_
        if record:
            AS.append(a.numpy()); MS.append(felt(ws).numpy())
            for w in ws:
                if w.dead:
                    w.dead = False; w.store[:] = 0.8
    if record:
        XS.append(np.stack([now_only(e) for e in eyes]))
        return torch.tensor(np.stack(XS, 1)), torch.tensor(np.stack(AS, 1)), torch.tensor(np.stack(TR, 1), dtype=torch.float32), torch.tensor(np.stack(MS, 1))
    return dict(whole=round(float(np.mean([not w.dead for w in ws])), 3), refills=round(float(fills.mean()), 2), seconds=round(float(np.mean([w.clock for w in ws])), 0))


t0 = time.time()
X, A, TRU, MO = live("A", 64, 1200, 1, record=True)                                       # experience to learn the inner state from (lived by today's being)
print(f"recorded {X.shape[0]} x {A.shape[1]} ticks: {time.time() - t0:.0f}s", flush=True)
inner = train_inner(Inner(R.DX, R.M, mo=7), X[:48], A[:48], MO=MO[:48], iters=6000, shut_p=0.5, shut_len=(8, 41), window=96, place=R.PLACE, rare=20.0, log=lambda s: print(s, flush=True))
torch.save(inner.state_dict(), SAVE)
# measure: wipe test and whether the state holds where hidden things are (truth used for measuring only, never fed back)
with torch.no_grad():
    Xe, Ae, Te = X[48:], A[48:], TRU[48:]
    h = inner.zero(Xe.shape[0]); H = []
    for t in range(Ae.shape[1]):
        h, _ = inner.step(h, Xe[:, t], Ae[:, t], motion=MO[48:, t]); H.append(h)
    H = torch.stack(H, 1)
    H = H[:, :-1]; seenb = Xe[:, 1:-1, 3 + 3 * 2 + 2] > 0
    for j, nm in ((0, "blue"), ):
        m = ~seenb; Hm = H[m]; tg = Te[:, 1:][m][:, 0:2]
        nn_ = Hm.shape[0]; cut = nn_ * 2 // 3; Hb = torch.cat([Hm, torch.ones(nn_, 1)], 1)
        wls = torch.linalg.lstsq(Hb[:cut], tg[:cut]).solution; pr = Hb[cut:] @ wls
        r2 = 1 - float(((pr - tg[cut:]) ** 2).sum() / ((tg[cut:] - tg[cut:].mean(0)) ** 2).sum())
        print(f"probe: {nm} out of view, its true bearing (sin, cos) read linearly from the inner state: held-out R^2 {r2:.2f} (n={nn_})", flush=True)

torch.save(inner.state_dict(), SAVE)
print(f"inner trained: {time.time() - t0:.0f}s", flush=True)
for name, mode, kw, lam in (("C  eye now only + inner state (feels own motion and its summed path), chunks 0.1", "C", dict(inner=inner), 0.1),
                       ("C' same, inner state wiped every 4 ticks", "C", dict(inner=inner, wipe=4), 0.1),
                       ("C  chunks 0.3", "C", dict(inner=inner), 0.3)):
    LAM = lam
    out = [live(mode, 24, 1500, 50 + k, **kw) for k in range(2)]
    print(f"{name} | whole {np.mean([o['whole'] for o in out]):.2f} | refills {np.mean([o['refills'] for o in out]):.2f} | seconds {np.mean([o['seconds'] for o in out]):.0f} of 300 | {time.time() - t0:.0f}s", flush=True)
print("DONE")
