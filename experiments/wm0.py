"""Imagination organ, cheap first look (theory first). Inverse eye + Gaussian latent state:
encoder: picture 40x64 -> z (mean, var), decoder: z -> picture, step: (z, act) -> next z (mean, var).
Trained only on what the eye received while the being lived (its own acts). No world rules inside.
Question: imagining k steps ahead with the eyes shut (only acts fed in), is the imagined picture closer to what came than
'the world stays as it is' (copy the last picture)? Split by act (turning shifts the scene, walking grows it)."""
import sys, time, torch, torch.nn as nn, torch.nn.functional as F
from atanor_os.emerge import room3d_live as R
from atanor_os.emerge import room3d_gpu as G
from atanor_os.emerge.core import Core
from atanor_os.emerge.life_gpu import Life
torch.manual_seed(0)
R.set_slots(R.CHROMA, 1 + R.CHROMA)
net = Core(R.DX, R.M, R.INSIDE, R.PLACE); net.load_state_dict(torch.load(sys.argv[1]), strict=False); net = net.cuda().eval()
NB, T, Z, STEPS = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
dev = "cuda"
# 1. live and keep what the eye received + own acts
L = Life(net, NB, R.HOME, dev, seed=3, pool="mean")     # (2026-11-21 첫 판의 그림은 손 규칙(길의 기억, 덜 걸어 본 쪽, 걸어가기)을 켠 몸이 모았다. 그것들은 걷어 냈다)
L.begin()
P = torch.zeros(T + 1, NB, 3, G.PH, G.PW, dtype=torch.uint8, device=dev); A = torch.zeros(T, NB, dtype=torch.long, device=dev); D = torch.zeros(T + 1, NB, dtype=torch.bool, device=dev)
with torch.no_grad():
    for t in range(T):
        P[t] = (L.world.picture().permute(0, 3, 1, 2).clamp(0, 1) * 255).to(torch.uint8); D[t] = L.world.dead
        A[t] = L.tick()[0]
    P[T] = (L.world.picture().permute(0, 3, 1, 2).clamp(0, 1) * 255).to(torch.uint8); D[T] = L.world.dead
print(f"lived {NB} bodies x {T} ticks; acts share {[round(float((A == k).float().mean()), 2) for k in range(5)]}", flush=True)
tr_b, te_b = torch.arange(0, NB * 3 // 4), torch.arange(NB * 3 // 4, NB)


class WM(nn.Module):
    def __init__(s, z=Z, m=5):
        super().__init__()
        s.enc = nn.Sequential(nn.Conv2d(3, 16, 4, 2, 1), nn.ELU(), nn.Conv2d(16, 32, 4, 2, 1), nn.ELU(), nn.Conv2d(32, 64, 4, 2, 1), nn.ELU(), nn.Flatten(), nn.Linear(64 * 5 * 8, 2 * z))
        s.dec_in = nn.Linear(z, 64 * 5 * 8)
        s.dec = nn.Sequential(nn.ELU(), nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ELU(), nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ELU(), nn.ConvTranspose2d(16, 3, 4, 2, 1))
        s.step = nn.Sequential(nn.Linear(z + m, 256), nn.ELU(), nn.Linear(256, 256), nn.ELU(), nn.Linear(256, 2 * z))
        s.m = m

    def encode(s, x):
        mu, lv = s.enc(x).chunk(2, 1); return mu, lv.clamp(-6, 4)

    def decode(s, z):
        return torch.sigmoid(s.dec(s.dec_in(z).view(-1, 64, 5, 8)))

    def next(s, z, a):
        d, lv = s.step(torch.cat([z, F.one_hot(a, s.m).float()], 1)).chunk(2, 1); return z + d, lv.clamp(-6, 4)


wm = WM().to(dev); opt = torch.optim.Adam(wm.parameters(), 3e-4)
print("params", sum(p.numel() for p in wm.parameters()), flush=True)
K = 8
f = lambda t, b: P[t, b].float() / 255


def batch(bs, src):
    b = src[torch.randint(len(src), (bs,))].to(dev); t = torch.randint(0, T - K, (bs,), device=dev)
    ok = ~D[t + K, b]
    return b[ok], t[ok]


t0 = time.time()
for it in range(STEPS):
    b, t = batch(64, tr_b)
    x0 = f(t, b); mu, lv = wm.encode(x0); z = mu + torch.randn_like(mu) * (0.5 * lv).exp()
    loss = F.mse_loss(wm.decode(z), x0) * 100 + 1e-3 * (0.5 * (mu ** 2 + lv.exp() - 1 - lv)).sum(1).mean()
    zi = z
    for k in range(1, K + 1):                                # imagine k steps with eyes shut, compare with what the eye then got
        zm, zlv = wm.next(zi, A[t + k - 1, b]); tm, tlv = wm.encode(f(t + k, b))
        loss = loss + 0.5 * ((zm - tm.detach()) ** 2 / zlv.exp() + zlv).sum(1).mean() * 1e-2 + F.mse_loss(wm.decode(zm), f(t + k, b)) * 100 / K
        zi = zm + torch.randn_like(zm) * (0.5 * zlv).exp()
    opt.zero_grad(); loss.backward(); opt.step()
    if it % 2000 == 0:
        print(f"  it {it} loss {float(loss):.3f} {time.time() - t0:.0f}s", flush=True)

# 2. judge on held-out bodies: imagined k-step picture error vs copy-last, by the act taken at the first step
wm.eval()
with torch.no_grad():
    res = {k: [[], [], []] for k in (1, 2, 4, 8)}
    for _ in range(40):
        b, t = batch(256, te_b)
        x0 = f(t, b); z = wm.encode(x0)[0]
        for k in range(1, K + 1):
            z = wm.next(z, A[t + k - 1, b])[0]
            if k in res:
                tgt = f(t + k, b)
                e_im = ((wm.decode(z) - tgt) ** 2).mean((1, 2, 3)); e_cp = ((x0 - tgt) ** 2).mean((1, 2, 3))
                res[k][0].append(e_im); res[k][1].append(e_cp); res[k][2].append(A[t, b])
    for k, (ei, ec, a) in res.items():
        ei, ec, a = torch.cat(ei), torch.cat(ec), torch.cat(a)
        moved = ec > 1e-4                                          # only where the picture really changed
        line = f"k={k}: imagined {float(ei[moved].mean()):.4f} vs copy-last {float(ec[moved].mean()):.4f} (imagined better in {float((ei < ec)[moved].float().mean()):.2f} of changing cases)"
        for nm, s in (("turn", (a == 2) | (a == 3)), ("forward", a == 1)):
            m = moved & s
            if m.any():
                line += f" | first act {nm}: {float(ei[m].mean()):.4f} vs {float(ec[m].mean()):.4f}"
        print(line, flush=True)
    recon = ((wm.decode(wm.encode(f(t, b))[0]) - f(t, b)) ** 2).mean()
    print(f"eyes-open reconstruction error {float(recon):.4f}")
torch.save(wm.state_dict(), "models/wm0.pt")
print("DONE")
