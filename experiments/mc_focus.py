"""Step 1 (founder): sharper vision - a summary of the FOCUS alone (the canon eye's fovea, 16x16 full resolution + the strongest colours
of the first ring), learned the same JEPA way (predict the next focus summary from the current one and the act; target = slow copy; VICReg).
Then the nameless-groups check: k-means on focus summaries, the most typical focus patches per group (enlarged), so a person can see
whether things (sheep, zombie, tree trunk, leaves, water, sand...) separate - which the whole-view summary did not.
Also the body probes again (blocked / going under water / hurt / distance) from [whole-view summary + focus summary] vs whole-view alone."""
import sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
OUT = "mc"
torch.manual_seed(0); dev = "cuda"; STEPS = int(sys.argv[1]); K = 4
d = np.load(f"{OUT}/frames.npz")
P = torch.tensor(d["P"]).to(dev); A = torch.tensor(d["A"]).to(dev); DN = torch.tensor(d["DN"]).to(dev)
T, NB = A.shape; cut = int(T * 0.8)
foc = lambda t, b: torch.cat([P[t, b, 0:3], P[t, b, 15:18]], 1).float()     # fovea RGB + first ring's peak colours (6 x 16 x 16)


class FEnc(nn.Module):
    def __init__(s, z=32):
        super().__init__()
        s.net = nn.Sequential(nn.Conv2d(6, 48, 3, 1, 1), nn.ELU(), nn.Conv2d(48, 64, 4, 2, 1), nn.ELU(), nn.Conv2d(64, 64, 4, 2, 1), nn.ELU(), nn.Flatten(), nn.Linear(64 * 16, 256), nn.ELU(), nn.Linear(256, z))
    def forward(s, x):
        return s.net(x)


Z = 32
E = FEnc(Z).to(dev); Et = FEnc(Z).to(dev); Et.load_state_dict(E.state_dict())
for p in Et.parameters():
    p.requires_grad_(False)
Pr = nn.Sequential(nn.Linear(Z + 7, 256), nn.ELU(), nn.Linear(256, Z)).to(dev)
opt = torch.optim.Adam(list(E.parameters()) + list(Pr.parameters()), 3e-4)


def vicreg(z):
    std = torch.sqrt(z.var(0) + 1e-4); v = F.relu(1 - std).mean()
    zc = z - z.mean(0); cov = (zc.T @ zc) / (len(z) - 1); off = cov - torch.diag(torch.diag(cov))
    return v, (off ** 2).sum() / z.shape[1]


if STEPS == 0:
    ck = torch.load(f"{OUT}/mc_focus.pt"); E.load_state_dict(ck["E"]); Pr.load_state_dict(ck["P"])
for it in range(STEPS):
    b = torch.randint(NB, (256,), device=dev); t = torch.randint(0, cut - K - 1, (256,), device=dev)
    keep = ~torch.stack([DN[t + k, b] for k in range(1, K + 1)]).any(0); b, t = b[keep], t[keep]
    z = E(foc(t, b)); zi = z; loss = 0
    for k in range(1, K + 1):
        zi = zi + Pr(torch.cat([zi, F.one_hot(A[t + k - 1, b], 7).float()], 1))
        with torch.no_grad():
            tg = Et(foc(t + k, b))
        loss = loss + ((F.normalize(zi, dim=1) - F.normalize(tg, dim=1)) ** 2).sum(1).mean() / K
    v, c = vicreg(z); loss = loss + v + 0.04 * c
    opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        for p, q in zip(E.parameters(), Et.parameters()):
            q.mul_(0.99).add_(0.01 * p)
    if it % 3000 == 0:
        print(f"  focus jepa it {it} loss {float(loss):.3f} var {float(v):.3f}", flush=True)
E.eval()
if STEPS:
    torch.save({"E": E.state_dict(), "P": Pr.state_dict()}, f"{OUT}/mc_focus.pt")

# nameless groups of the focus summary
PP = d["P"].reshape(-1, 27, 16, 16)
idx = np.random.default_rng(0).choice(len(PP), 20000, replace=False)
X = torch.tensor(PP[idx]).float().to(dev)
Xf = torch.cat([X[:, 0:3], X[:, 15:18]], 1)
with torch.no_grad():
    Zf = torch.cat([E(Xf[i:i + 2048]) for i in range(0, len(Xf), 2048)])
Zf = (Zf - Zf.mean(0)) / (Zf.std(0) + 1e-5)
G_ = 16; C = Zf[torch.randperm(len(Zf))[:G_]].clone()
for _ in range(60):
    lab = torch.cdist(Zf, C).argmin(1); C = torch.stack([Zf[lab == k].mean(0) if (lab == k).any() else C[k] for k in range(G_)])
lab = torch.cdist(Zf, C).argmin(1).cpu().numpy(); dist = torch.cdist(Zf, C).min(1).values.cpu().numpy()
fov = X[:, 0:3].cpu().numpy(); whole = X[:, 12:15].cpu().numpy()
rows = []
for k in np.argsort(-np.bincount(lab, minlength=G_)):
    m = np.where(lab == k)[0]; m = m[np.argsort(dist[m])][:8]
    tiles = []
    for i in m:
        f_ = np.asarray(Image.fromarray((fov[i].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)).resize((72, 72), Image.NEAREST))
        w_ = np.asarray(Image.fromarray((whole[i].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)).resize((36, 36), Image.NEAREST))
        tile = f_.copy(); tile[:36, :36] = w_                     # focus patch big, whole view small in the corner
        tiles.append(np.concatenate([tile, np.full((72, 3, 3), 255, np.uint8)], 1))
    while len(tiles) < 8:
        tiles.append(np.full((72, 75, 3), 255, np.uint8))
    rows.append(np.concatenate([np.full((72, 4, 3), 255, np.uint8)] + tiles, 1)); rows.append(np.full((4, rows[-1].shape[1], 3), 255, np.uint8))
    print(f"focus group {k}: {(lab == k).mean() * 100:.1f}%", flush=True)
Image.fromarray(np.concatenate(rows, 0)).save("D:/ATANOR-experiments/2026-09-19/developmental-research/docs_tmp/mc_focus_groups.png")
print("DONE")
