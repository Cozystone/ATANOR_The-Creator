"""R1 cheap diagnostic: does a state carried over time hold what is out of view?
Record: the room-raised being lives in 4 varied rooms (128 bodies, 600 ticks each, no learning), we keep its feelings x (148), acts, and - for JUDGING
only - whether a spring (water or food) is in view (any column of the 256-column cast hits a spring within 12 cells).
Question asked at moments when NO spring is in view now but one WAS in view within the last 20 steps: will a spring be in view again within the
next 10 steps? (needs memory of where it was + of how I turned)
Compare on held-out bodies:
  now-only:  MLP(x_t)
  carried:   GRU over (x, act code) up to t, then MLP(h_t)      (same budget)
AUC (0.5 = chance). Also the same for moments with no recent sighting (control: memory should not matter there)."""
import torch, torch.nn as nn, numpy as np, time
from atanor_os.emerge.room_body import RoomBody
from atanor_os.emerge.body_contract import Organs
from atanor_os.emerge.mind import Mind
from atanor_os.emerge.room_vary import condition
from atanor_os.emerge import room3d_gpu as G
dev, N, T = "cuda", 128, 600
torch.manual_seed(0)
X, A, V = [], [], []
for room in (1, 2, 3, 4):
    rows, drain = condition(room)
    B = RoomBody(N, rows, drain=drain, seed=room, device=dev)
    if room == 1:
        O = Organs(B.c, N, dev=dev, depth="mc/mc_depth.pt"); Mi = Mind(O, B.c, N, dev=dev)
        Mi.load("being/host/atanor_born_in_rooms.pt"); Mi.wo.on = True; _, C = B.c.codes(); C = torch.tensor(C, device=dev)
    O.reset(torch.ones(N, dtype=torch.bool, device=dev)); Mi.begin(torch.ones(N, dtype=torch.bool, device=dev))
    xs, acs, vs = [], [], []
    x = O(B.raw())
    for t in range(T):
        act = Mi.choose(x)
        kind, dist = B.w.cast(B.w.head[:, None] + B.cols[None], step=0.1, far=12.0)
        vis = (((kind == G.WATER) | (kind == G.FOOD)) & (dist < 12)).any(1)
        xs.append(x); acs.append(act); vs.append(vis)
        a0 = ~B.dead.clone(); B.step(act); x2 = O(B.raw()); died = a0 & B.dead
        Mi.prev = (x, act); Mi.cut = B.dead & ~died; x = x2
    X.append(torch.stack(xs)); A.append(torch.stack(acs)); V.append(torch.stack(vs))
    print(f"room {room} recorded, spring in view {float(torch.stack(vs).float().mean()):.2f}", flush=True)
X = torch.cat(X, 1); A = torch.cat(A, 1); V = torch.cat(V, 1).float()           # [T, 4N, ...]
Tn, NB, dx = X.shape
inp = torch.cat([X, C[A]], 2)                                                    # feeling + what I did
recent = torch.zeros_like(V); soon = torch.zeros_like(V)
for k in range(1, 21):
    recent[k:] = torch.maximum(recent[k:], V[:-k])
for k in range(1, 11):
    soon[:-k] = torch.maximum(soon[:-k], V[k:])
mem_q = (V == 0) & (recent > 0); ctl_q = (V == 0) & (recent == 0)
mem_q[:30] = False; ctl_q[:30] = False; mem_q[-11:] = False; ctl_q[-11:] = False
perm = torch.randperm(NB, device=dev); tr, te = perm[: int(NB * 0.75)], perm[int(NB * 0.75):]


def auc(s, y):
    s, y = s.detach().cpu().numpy(), y.cpu().numpy().astype(bool)
    if y.sum() == 0 or (~y).sum() == 0:
        return float("nan")
    r = np.argsort(np.argsort(s)).astype(float)
    return float((r[y].mean() - (y.sum() - 1) / 2) / (~y).sum())


class Now(nn.Module):
    def __init__(s):
        super().__init__(); s.m = nn.Sequential(nn.Linear(inp.shape[2], 128), nn.ELU(), nn.Linear(128, 1))
    def forward(s, z):
        return s.m(z)[..., 0]


class Carried(nn.Module):
    def __init__(s):
        super().__init__(); s.g = nn.GRU(inp.shape[2], 64); s.m = nn.Sequential(nn.Linear(64, 128), nn.ELU(), nn.Linear(128, 1))
    def forward(s, z):
        h, _ = s.g(z); return s.m(h)[..., 0]


res = {}
for nm, M in (("now-only", Now), ("carried state", Carried)):
    torch.manual_seed(1); m = M().to(dev); opt = torch.optim.Adam(m.parameters(), 2e-3)
    for it in range(6000):
        b = tr[torch.randint(len(tr), (64,), device=dev)]
        out = m(inp[:, b]); q = (mem_q[:, b] | ctl_q[:, b]).float()
        loss = (nn.functional.binary_cross_entropy_with_logits(out, soon[:, b], reduction="none") * q).sum() / q.sum()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        out = m(inp[:, te])
    res[nm] = (auc(out[mem_q[:, te]], soon[:, te][mem_q[:, te]]), auc(out[ctl_q[:, te]], soon[:, te][ctl_q[:, te]]))
    print(f"{nm:14s}: spring will be back within 10 steps | after a recent sighting AUC {res[nm][0]:.3f} | with no recent sighting AUC {res[nm][1]:.3f}", flush=True)
print(f"cases: after a recent sighting {int(mem_q[:, te].sum())} (back within 10: {float(soon[:, te][mem_q[:, te]].mean()):.2f}); no recent sighting {int(ctl_q[:, te].sum())}")
print("DONE")
