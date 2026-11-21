"""Imagination organ with a continuing inner state (RSSM, as in Dreamer): h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}]), prior p(z_t | h_t), posterior q(z_t | h_t, eye_t),
picture redrawn from [h, z]. Trained only on the eye's pictures and own acts. No world rules, no world names.
Cheap diagnostic first (theory first): when a source WAS seen in the last 40 ticks but is OUT OF VIEW now, does a linear probe read its
egocentric side (cos, sin of bearing) from h, better than from the picture-now state (the frame encoder alone)?
The actor that gathers pictures: the pure core with widened futility 0.3 (no hand rules; it moves more than the spinning core)."""
import sys, time, math, torch, torch.nn as nn, torch.nn.functional as F
from atanor_os.emerge import room3d_live as R
from atanor_os.emerge import room3d_gpu as G
from atanor_os.emerge.core import Core
from atanor_os.emerge.life_gpu import Life
torch.manual_seed(0)
R.set_slots(R.CHROMA, 1 + R.CHROMA)
net = Core(R.DX, R.M, R.INSIDE, R.PLACE); net.load_state_dict(torch.load(sys.argv[1]), strict=False); net = net.cuda().eval(); net.far = 0.3
NB, T, STEPS, HD = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
dev = "cuda"; Z = 32
L = Life(net, NB, R.HOME, dev, seed=3, pool="mean"); L.begin()
P = torch.zeros(T, NB, 3, G.PH, G.PW, dtype=torch.uint8, device=dev); A = torch.zeros(T, NB, dtype=torch.long, device=dev)
POS = torch.zeros(T, NB, 2, device=dev); HEAD = torch.zeros(T, NB, device=dev); SRC = torch.zeros(T, NB, 2, 2, device=dev); DEAD = torch.zeros(T, NB, dtype=torch.bool, device=dev)
with torch.no_grad():
    for t in range(T):
        w = L.world
        P[t] = (w.picture().permute(0, 3, 1, 2).clamp(0, 1) * 255).to(torch.uint8)
        POS[t] = w.pos; HEAD[t] = w.head; SRC[t] = w.src.float() + 0.5; DEAD[t] = w.dead
        A[t] = L.tick()[0]
img = lambda t, b: P[t, b].float() / 255
# in view now: a pixel strongly of that kind's colour (water blue, food red)
def inview(p):
    r, g, b = p[:, 0], p[:, 1], p[:, 2]
    return torch.stack([((b - torch.maximum(r, g)) > 0.15).flatten(1).any(1), ((r - torch.maximum(g, b)) > 0.15).flatten(1).any(1)], 1)
print(f"gathered {NB} x {T}; acts {[round(float((A == k).float().mean()), 2) for k in range(5)]}", flush=True)


class RSSM(nn.Module):
    def __init__(s, m=5):
        super().__init__()
        s.enc = nn.Sequential(nn.Conv2d(3, 16, 4, 2, 1), nn.ELU(), nn.Conv2d(16, 32, 4, 2, 1), nn.ELU(), nn.Conv2d(32, 64, 4, 2, 1), nn.ELU(), nn.Flatten(), nn.Linear(64 * 5 * 8, 256), nn.ELU())
        s.gru = nn.GRUCell(Z + m, HD)
        s.prior = nn.Sequential(nn.Linear(HD, 256), nn.ELU(), nn.Linear(256, 2 * Z))
        s.post = nn.Sequential(nn.Linear(HD + 256, 256), nn.ELU(), nn.Linear(256, 2 * Z))
        s.dec_in = nn.Linear(HD + Z, 64 * 5 * 8)
        s.dec = nn.Sequential(nn.ELU(), nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ELU(), nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ELU(), nn.ConvTranspose2d(16, 3, 4, 2, 1))
        s.m = m

    def decode(s, h, z):
        return torch.sigmoid(s.dec(s.dec_in(torch.cat([h, z], 1)).view(-1, 64, 5, 8)))

    def stat(s, o):
        mu, lv = o.chunk(2, 1); return mu, lv.clamp(-6, 4)


rs = RSSM().to(dev); opt = torch.optim.Adam(rs.parameters(), 3e-4)
print("params", sum(p.numel() for p in rs.parameters()), flush=True)
tr_b, te_b = torch.arange(0, NB * 3 // 4, device=dev), torch.arange(NB * 3 // 4, NB, device=dev)
SEQ, BS = 48, 32


def kl(m1, l1, m2, l2):
    return 0.5 * (l2 - l1 + (l1.exp() + (m1 - m2) ** 2) / l2.exp() - 1).sum(1)


t0 = time.time()
for it in range(STEPS):
    b = tr_b[torch.randint(len(tr_b), (BS,), device=dev)]; t = int(torch.randint(0, T - SEQ, (1,)))
    h = torch.zeros(BS, HD, device=dev); z = torch.zeros(BS, Z, device=dev); a = torch.zeros(BS, dtype=torch.long, device=dev)
    loss = 0
    for k in range(SEQ):
        h = rs.gru(torch.cat([z, F.one_hot(a, 5).float()], 1), h)
        pm, pl = rs.stat(rs.prior(h)); x = img(t + k, b); qm, ql = rs.stat(rs.post(torch.cat([h, rs.enc(x)], 1)))
        z = qm + torch.randn_like(qm) * (0.5 * ql).exp()
        ok = (~DEAD[t + k, b]).float()
        loss = loss + ((F.mse_loss(rs.decode(h, z), x, reduction="none").mean((1, 2, 3)) * 100 + kl(qm, ql, pm, pl).clamp(min=1.0)) * ok).mean()
        a = A[t + k, b]
    opt.zero_grad(); (loss / SEQ).backward(); torch.nn.utils.clip_grad_norm_(rs.parameters(), 100); opt.step()
    if it % 1000 == 0:
        print(f"  it {it} loss {float(loss / SEQ):.3f} {time.time() - t0:.0f}s", flush=True)

# ---- diagnostic on held-out bodies ----
rs.eval()
feats = {"h (continuing)": [], "picture now": [], "z (posterior)": []}; tgt = []; kindl = []
with torch.no_grad():
    b = te_b; B = len(b)
    h = torch.zeros(B, HD, device=dev); z = torch.zeros(B, Z, device=dev); a = torch.zeros(B, dtype=torch.long, device=dev)
    last = torch.full((B, 2), -1e9, device=dev)
    for t in range(T):
        x = img(t, b); e = rs.enc(x)
        h = rs.gru(torch.cat([z, F.one_hot(a, 5).float()], 1), h)
        qm, ql = rs.stat(rs.post(torch.cat([h, e], 1))); z = qm
        iv = inview(x); last = torch.where(iv, torch.full_like(last, float(t)), last)
        for j in range(2):
            m = ~iv[:, j] & (t - last[:, j] <= 40) & (t - last[:, j] >= 3) & ~DEAD[t, b]
            if m.any() and t > 20:
                d = SRC[t, b, j] - POS[t, b]; ang = torch.atan2(d[:, 1], d[:, 0]) - HEAD[t, b]
                feats["h (continuing)"].append(h[m]); feats["picture now"].append(e[m]); feats["z (posterior)"].append(qm[m])
                tgt.append(torch.stack([torch.cos(ang[m]), torch.sin(ang[m])], 1)); kindl.append(torch.full((int(m.sum()),), j, device=dev))
        a = A[t, b]
Y = torch.cat(tgt); K_ = torch.cat(kindl)
print(f"probe cases (source seen 3-40 ticks ago, out of view now): {len(Y)}", flush=True)
perm = torch.randperm(len(Y), device=dev); half = len(Y) // 2; i1, i2 = perm[:half], perm[half:]
for nm, fl in feats.items():
    X = torch.cat([torch.cat(fl), F.one_hot(K_, 2).float()], 1); X = torch.cat([X, torch.ones(len(X), 1, device=dev)], 1)
    W = torch.linalg.solve(X[i1].T @ X[i1] + 1e-1 * torch.eye(X.shape[1], device=dev), X[i1].T @ Y[i1])
    P_ = X[i2] @ W
    err = torch.remainder(torch.atan2(P_[:, 1], P_[:, 0]) - torch.atan2(Y[i2, 1], Y[i2, 0]) + math.pi, 2 * math.pi) - math.pi
    side = ((P_[:, 1] > 0) == (Y[i2, 1] > 0)).float().mean()
    print(f"{nm:16s}: mean angle error {float(err.abs().mean()) * 180 / math.pi:.0f} deg (chance 90) | left/right right {float(side):.2f}", flush=True)
torch.save(rs.state_dict(), "models/wm1.pt")
print("DONE")
