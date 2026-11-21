"""Minecraft eye: pixel-redraw imagination (eyewm.WM) vs JEPA-style summary-space imagination, both from the canon fovea eye's input (27x16x16 + gaze).
JEPA: online encoder E, target encoder = slow copy of E (EMA), predictor (z, act) -> next z, rolled K steps in the summary space; loss = distance between
predicted and target summaries (normalised) + variance/covariance terms that stop every summary collapsing to one point (VICReg). No pixels redrawn.
Judge (no labels from people; all from the being's own body): from the summary at time t (held-out last 20% of time), linear probes for
  blocked  : it pressed forward (act 1 or 6) and barely moved (felt forward < 0.05)
  water    : air drops within the next 20 steps
  hurt     : health drops / got hurt within the next 20 steps
  distance : how far it walks forward over the next 10 steps (R^2)
Baseline 'no eye': gaze + inside only.  python mc_eyejudge.py STEPS"""
import sys, time, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from atanor_os.emerge.eyewm import WM
OUT = "mc"
torch.manual_seed(0); dev = "cuda"; STEPS = int(sys.argv[1]); K = 8
d = np.load(f"{OUT}/frames.npz")
P = torch.tensor(d["P"]).to(dev); A = torch.tensor(d["A"]).to(dev); DN = torch.tensor(d["DN"]).to(dev)
G = torch.tensor(d["G"]).to(dev); IN = torch.tensor(d["IN"]).to(dev); FE = torch.tensor(d["FE"]).to(dev); HU = torch.tensor(d["HU"]).to(dev)
T, NB = A.shape; cut = int(T * 0.8)
print(f"data {T} steps x {NB} bodies; breaks {int(DN.sum())}", flush=True)
f = lambda t, b: P[t, b].float()


def spans(lo, hi, bs):
    b = torch.randint(NB, (bs,), device=dev); t = torch.randint(lo, hi - K - 1, (bs,), device=dev)
    keep = ~torch.stack([DN[t + k, b] for k in range(1, K + 1)]).any(0)
    return b[keep], t[keep]


# ---- A: pixel-redraw imagination
wm = WM(z=32, m=7, ch=27, hw=(16, 16)).to(dev); opt = torch.optim.Adam(wm.parameters(), 3e-4); t0 = time.time()
for it in range(STEPS):
    b, t = spans(0, cut, 64); x0 = f(t, b); mu, lv = wm.encode(x0); z = mu + torch.randn_like(mu) * (0.5 * lv).exp()
    loss = F.mse_loss(wm.decode(z), x0) * 100 + 1e-3 * (0.5 * (mu ** 2 + lv.exp() - 1 - lv)).sum(1).mean(); zi = z
    for k in range(1, K + 1):
        zm, zlv = wm.next(zi, A[t + k - 1, b]); tm, _ = wm.encode(f(t + k, b))
        loss = loss + 0.5 * ((zm - tm.detach()) ** 2 / zlv.exp() + zlv).sum(1).mean() * 1e-2 + F.mse_loss(wm.decode(zm), f(t + k, b)) * 100 / K
        zi = zm + torch.randn_like(zm) * (0.5 * zlv).exp()
    opt.zero_grad(); loss.backward(); opt.step()
    if it % 3000 == 0:
        print(f"  pixel it {it} loss {float(loss):.3f} {time.time() - t0:.0f}s", flush=True)
wm.eval(); torch.save(wm.state_dict(), f"{OUT}/mc_wm_pixel.pt")


# ---- B: JEPA-style summary-space imagination
class Enc(nn.Module):
    def __init__(s, z=64):
        super().__init__()
        s.net = nn.Sequential(nn.Conv2d(27, 64, 4, 2, 1), nn.ELU(), nn.Conv2d(64, 64, 4, 2, 1), nn.ELU(), nn.Conv2d(64, 64, 4, 2, 1), nn.ELU(), nn.Flatten(), nn.Linear(256, 256), nn.ELU(), nn.Linear(256, z))
    def forward(s, x):
        return s.net(x)


Z = 64
E = Enc(Z).to(dev); Et = Enc(Z).to(dev); Et.load_state_dict(E.state_dict())
for p in Et.parameters():
    p.requires_grad_(False)
Pr = nn.Sequential(nn.Linear(Z + 7, 256), nn.ELU(), nn.Linear(256, 256), nn.ELU(), nn.Linear(256, Z)).to(dev)
opt = torch.optim.Adam(list(E.parameters()) + list(Pr.parameters()), 3e-4); t0 = time.time()


def vicreg(z):
    std = torch.sqrt(z.var(0) + 1e-4); v = F.relu(1 - std).mean()
    zc = z - z.mean(0); cov = (zc.T @ zc) / (len(z) - 1); off = cov - torch.diag(torch.diag(cov))
    return v, (off ** 2).sum() / z.shape[1]


for it in range(STEPS):
    b, t = spans(0, cut, 128); z = E(f(t, b)); zi = z; loss = 0
    for k in range(1, K + 1):
        zi = zi + Pr(torch.cat([zi, F.one_hot(A[t + k - 1, b], 7).float()], 1))
        with torch.no_grad():
            tg = Et(f(t + k, b))
        loss = loss + ((F.normalize(zi, dim=1) - F.normalize(tg, dim=1)) ** 2).sum(1).mean() / K
    v, c = vicreg(z); loss = loss + 1.0 * v + 0.04 * c
    opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        for p, q in zip(E.parameters(), Et.parameters()):
            q.mul_(0.99).add_(0.01 * p)
    if it % 3000 == 0:
        print(f"  jepa it {it} loss {float(loss):.3f} (var {float(v):.3f} cov {float(c):.3f}) {time.time() - t0:.0f}s", flush=True)
E.eval(); torch.save({"E": E.state_dict(), "P": Pr.state_dict()}, f"{OUT}/mc_jepa.pt")

# JEPA imagination check: predicted summary k steps ahead vs 'the summary stays' (cosine to the true next summary)
with torch.no_grad():
    for k_ in (1, 4, 8):
        sims = [[], []]
        for _ in range(30):
            b, t = spans(cut, T, 256); z = E(f(t, b)); zi = z
            for k in range(1, k_ + 1):
                zi = zi + Pr(torch.cat([zi, F.one_hot(A[t + k - 1, b], 7).float()], 1))
            tg = E(f(t + k_, b)); sims[0].append(F.cosine_similarity(zi, tg)); sims[1].append(F.cosine_similarity(z, tg))
        print(f"jepa k={k_}: imagined summary cosine {float(torch.cat(sims[0]).mean()):.3f} vs summary-stays {float(torch.cat(sims[1]).mean()):.3f}", flush=True)

# ---- probes from the body's own future
H = 20
idx = [(t, b) for t in range(1, T - H - 1) for b in range(NB)]
tt = torch.tensor([i[0] for i in idx], device=dev); bb = torch.tensor([i[1] for i in idx], device=dev)
ok = ~torch.stack([DN[tt + k, bb] for k in range(1, H + 1)]).any(0); tt, bb = tt[ok], bb[ok]
fwd_act = (A[tt, bb] == 1) | (A[tt, bb] == 6)
blocked = fwd_act & (FE[tt, bb, 0] < 0.05)
air_drop = torch.stack([IN[tt + k, bb, 2] < IN[tt, bb, 2] - 0.01 for k in range(1, H + 1)]).any(0)
air_full = IN[tt, bb, 2] > 0.99                              # 'water' asks only from full air: will it go under within 20 steps (the body sense alone cannot know)
hurt = torch.stack([(IN[tt + k, bb, 0] < IN[tt, bb, 0] - 0.01) | (HU[tt + k - 1, bb] > 0) for k in range(1, H + 1)]).any(0)
dist = torch.stack([FE[tt + k, bb, 0].clamp(min=0) for k in range(0, 10)]).sum(0)
test = tt >= cut
print(f"probe cases {len(tt)} (test {int(test.sum())}); base rates: blocked|fwd {float(blocked[fwd_act].float().mean()):.3f}, water {float(air_drop.float().mean()):.3f}, hurt {float(hurt.float().mean()):.3f}", flush=True)


@torch.no_grad()
def feats(name):
    out = []
    for i in range(0, len(tt), 4096):
        t_, b_ = tt[i:i + 4096], bb[i:i + 4096]; x = f(t_, b_)
        base = torch.cat([G[t_, b_], IN[t_, b_]], 1)
        if name == "no eye":
            out.append(base)
        elif name == "pixel imagination":
            out.append(torch.cat([wm.encode(x)[0], base], 1))
        else:
            out.append(torch.cat([E(x), base], 1))
    return torch.cat(out)


def auc(score, y):
    y = y.float(); o = score.argsort(); r = torch.empty_like(score); r[o] = torch.arange(len(score), device=dev, dtype=score.dtype)
    npos = y.sum(); nneg = len(y) - npos
    return float((r[y > 0].sum() - npos * (npos - 1) / 2) / (npos * nneg).clamp(min=1))


def logistic(X, y, tr, te):
    m = X[tr].mean(0); s = X[tr].std(0) + 1e-5; Xn = (X - m) / s
    w = torch.zeros(X.shape[1], device=dev, requires_grad=True); b0 = torch.zeros(1, device=dev, requires_grad=True)
    o = torch.optim.LBFGS([w, b0], max_iter=200)
    pw = (len(y[tr]) - y[tr].sum()) / y[tr].sum().clamp(min=1)
    def cl():
        o.zero_grad(); l = F.binary_cross_entropy_with_logits(Xn[tr] @ w + b0, y[tr].float(), pos_weight=pw) + 1e-3 * (w ** 2).sum(); l.backward(); return l
    o.step(cl)
    with torch.no_grad():
        return auc(Xn[te] @ w + b0, y[te])


def ridge(X, y, tr, te):
    m = X[tr].mean(0); s = X[tr].std(0) + 1e-5; Xn = torch.cat([(X - m) / s, torch.ones(len(X), 1, device=dev)], 1)
    W = torch.linalg.solve(Xn[tr].T @ Xn[tr] + 0.05 * len(Xn[tr]) * torch.eye(Xn.shape[1], device=dev), Xn[tr].T @ y[tr])      # strong shrinkage (new places at test)
    pr = Xn[te] @ W; return float(1 - ((pr - y[te]) ** 2).mean() / y[te].var())


tr = ~test
for nm in ("no eye", "pixel imagination", "summary-space (JEPA)"):
    X = feats(nm)
    fa = fwd_act
    af = air_full
    r = dict(blocked=logistic(X[fa], blocked[fa], tr[fa], test[fa]), water=logistic(X[af], air_drop[af], tr[af], test[af]), hurt=logistic(X, hurt, tr, test), distance=ridge(X, dist, tr, test))
    print(f"{nm:22s}: AUC blocked {r['blocked']:.3f} | going under water {r['water']:.3f} | hurt {r['hurt']:.3f} (test cases {int((hurt & test).sum())}) | distance R2 {r['distance']:.3f}", flush=True)
print("DONE")
