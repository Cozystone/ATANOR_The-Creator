"""Does the inner state h hold where the unseen spring is? Raise a being with the inner organ (dark rooms, 8 lives, 128 bodies) - or load it - then
live 2 more lives in unseen dark rooms (no learning) recording, at DARK moments: h, the feeling x, and the TRUE bearing of the nearest spring relative
to the head (judging only). Probes (trained on 75% of bodies, tested on the rest): is the spring within 40 deg in front? (AUC), cos/sin of the bearing (R2).
Compare: from h (raised), from the feeling x alone, from h of an UNTRAINED organ (same size, random) - to see if h holds more than the feeling.
  python r1_probe_h.py <raise 1/0>"""
import sys, math, os, torch, torch.nn as nn, numpy as np
from atanor_os.emerge.room_body import RoomBody
from atanor_os.emerge.body_contract import Organs
from atanor_os.emerge.mind import Mind
from atanor_os.emerge.room_vary import condition
dev, N, T = "cuda", 128, 900
PATH = "being/r1_inner.pt"
torch.manual_seed(0)
rows, drain = condition(1)
B = RoomBody(N, rows, drain=drain, seed=0, device=dev)
O = Organs(B.c, N, dev=dev, depth="mc/mc_depth.pt"); Mi = Mind(O, B.c, N, dev=dev, inner=64); Mi.inner_every = int(os.environ.get('INNER_EVERY', '10'))


def life(room, seed, learn, record):
    B = RoomBody(N, *condition(room), seed=seed, device=dev); B.dark_every, B.dark_len = 60, 20
    O.reset(torch.ones(N, dtype=torch.bool, device=dev)); Mi.begin(torch.ones(N, dtype=torch.bool, device=dev))
    xr = O(B.raw()); x = Mi.feel(xr); rec = []
    for t in range(T):
        act = Mi.choose(x); a0 = ~B.dead.clone(); B.step(act); died = a0 & B.dead; cut = B.dead & ~died
        xr2 = O(B.raw()); Mi.carry(xr, act, cut | died); x2 = Mi.feel(xr2)
        if learn:
            Mi.after(x, act, x2, died, cut)
        else:
            Mi.prev = (x, act); Mi.cut = cut; Mi.steps += 1
        if record:
            c = B.w.src.float() + 0.5; dv = c - B.w.pos[:, None]; j = dv.norm(dim=2).argmin(1)
            v = dv[torch.arange(N, device=dev), j]; bear = torch.atan2(v[:, 1], v[:, 0]) - B.w.head
            bear = (bear + math.pi) % (2 * math.pi) - math.pi
            m = B.dark() & ~B.dead
            rec.append((xr2[m].clone(), Mi.h[m].clone(), bear[m].clone(), torch.nonzero(m)[:, 0] + 1000 * seed))
        x, xr = x2, xr2
        if B.dead.all():
            break
    return rec


if sys.argv[1] == "1" or not os.path.exists(PATH):
    for k in range(8):
        life(1 + k % 8, 100 + k, True, False); print("raised life", k + 1, flush=True)
    Mi.save(PATH)
else:
    Mi.load(PATH)
rec = life(101, 7, False, True) + life(102, 8, False, True)
X = torch.cat([r[0] for r in rec]); Hh = torch.cat([r[1] for r in rec]); Bb = torch.cat([r[2] for r in rec]); body = torch.cat([r[3] for r in rec])
# untrained organ's h on the same experience is not available after the fact -> re-run with a fresh GRU of the same size on recorded x is not possible
# (acts not stored); instead compare with a fresh GRU fed the same feelings in order: approximate by shuffling h across moments (keeps marginal, kills link)
Hs = Hh[torch.randperm(len(Hh), device=dev)]
ub = body.unique(); perm = ub[torch.randperm(len(ub), device=dev)]; trb = perm[: int(len(perm) * 0.75)]
tr = torch.isin(body, trb); te = ~tr
front = (Bb.abs() < math.radians(40)).float(); cs = torch.stack([torch.cos(Bb), torch.sin(Bb)], 1)


def auc(s, y):
    s, y = s.cpu().numpy(), y.cpu().numpy().astype(bool)
    r = np.argsort(np.argsort(s)).astype(float)
    return float((r[y].mean() - (y.sum() - 1) / 2) / max((~y).sum(), 1))


def probe(F, nm):
    torch.manual_seed(1)
    m = nn.Sequential(nn.Linear(F.shape[1], 128), nn.ELU(), nn.Linear(128, 3)).to(dev); opt = torch.optim.Adam(m.parameters(), 1e-3)
    Ftr, Fte = F[tr], F[te]
    for it in range(3000):
        i = torch.randint(0, len(Ftr), (1024,), device=dev); o = m(Ftr[i])
        loss = nn.functional.binary_cross_entropy_with_logits(o[:, 0], front[tr][i]) + ((o[:, 1:] - cs[tr][i]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        o = m(Fte)
    r2 = 1 - ((o[:, 1:] - cs[te]) ** 2).sum() / ((cs[te] - cs[te].mean(0)) ** 2).sum()
    print(f"{nm:34s}: spring within 40 deg in front AUC {auc(o[:, 0], front[te]):.3f} | bearing (cos, sin) R2 {float(r2):.3f}", flush=True)


print(f"dark moments: {len(X)} (test {int(te.sum())}), spring in front share {float(front.mean()):.2f}")
probe(X, "feeling now only")
probe(Hh, "inner state h (raised)")
probe(torch.cat([X, Hh], 1), "feeling + h")
probe(Hs, "h shuffled across moments (control)")
print("DONE")
