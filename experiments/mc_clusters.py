"""Are there nameless groupings inside the JEPA eye's summary? k-means (no labels) on the summaries of the gathered pictures,
then show for each group a few of its pictures (the canon eye's widest ring = whole view, coarse 16x16, enlarged) so a person can see what it lumped together."""
import numpy as np, torch, torch.nn as nn
from PIL import Image
OUT = "mc"
dev = "cuda"; torch.manual_seed(0)


class Enc(nn.Module):
    def __init__(s, z=64):
        super().__init__()
        s.net = nn.Sequential(nn.Conv2d(27, 64, 4, 2, 1), nn.ELU(), nn.Conv2d(64, 64, 4, 2, 1), nn.ELU(), nn.Conv2d(64, 64, 4, 2, 1), nn.ELU(), nn.Flatten(), nn.Linear(256, 256), nn.ELU(), nn.Linear(256, z))
    def forward(s, x):
        return s.net(x)


d = np.load(f"{OUT}/frames.npz"); P = d["P"]                     # [T, NB, 27, 16, 16]
T, NB = P.shape[:2]
idx = np.random.default_rng(0).choice(T * NB, 20000, replace=False)
X = torch.tensor(P.reshape(T * NB, 27, 16, 16)[idx]).float().to(dev)
E = Enc().to(dev); E.load_state_dict(torch.load(f"{OUT}/mc_jepa.pt", map_location=dev)["E"]); E.eval()
with torch.no_grad():
    Z = torch.cat([E(X[i:i + 2048]) for i in range(0, len(X), 2048)])
Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-5)
K = 12
C = Z[torch.randperm(len(Z))[:K]].clone()
for _ in range(50):
    lab = torch.cdist(Z, C).argmin(1)
    C = torch.stack([Z[lab == k].mean(0) if (lab == k).any() else C[k] for k in range(K)])
lab = torch.cdist(Z, C).argmin(1).cpu().numpy(); dist = torch.cdist(Z, C).min(1).values.cpu().numpy()
whole = X[:, 12:15].cpu().numpy()                                 # ring 4 (widest = whole view), RGB 0..1
fov = X[:, 0:3].cpu().numpy()
rows = []
for k in np.argsort(-np.bincount(lab, minlength=K)):
    m = np.where(lab == k)[0]
    m = m[np.argsort(dist[m])][:8]                                 # the 8 most typical members
    tiles = []
    for i in m:
        a = (whole[i].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        tiles.append(np.asarray(Image.fromarray(a).resize((96, 96), Image.NEAREST)))
    while len(tiles) < 8:
        tiles.append(np.zeros((96, 96, 3), np.uint8))
    row = np.concatenate([np.full((96, 6, 3), 255, np.uint8)] + [np.concatenate([t, np.full((96, 4, 3), 255, np.uint8)], 1) for t in tiles], 1)
    rows.append(row); rows.append(np.full((6, row.shape[1], 3), 255, np.uint8))
    print(f"group {k}: {int((lab == k).sum())} pictures ({(lab == k).mean() * 100:.1f}%)")
Image.fromarray(np.concatenate(rows, 0)).save("D:/ATANOR-experiments/2026-09-19/developmental-research/docs_tmp/mc_jepa_groups.png")
print("saved")
