"""Depth from own motion, with no labels (canon depth_learner/selfsup.py + model.DepthNet, imported read-only).
The camera motion between frames comes from the body's own sense of how it moved and turned (no PoseNet).
Pictures are rendered as a pinhole camera (what canon's warp assumes). True depth is used ONLY to judge, never to train.
Stage A (sanity): with the TRUE depth, does warping with the felt motion beat 'no warp'? (else the geometry is wrong)
Stage B: learn depth from the being's own walking pictures; judge on held-out bodies: rank agreement with true depth, error after median scaling, raw error."""
import sys, os, math, time, importlib.util, torch, torch.nn.functional as F
from atanor_os.emerge import room3d_live as R
from atanor_os.emerge import room3d_gpu as G
from atanor_os.emerge.core import Core
from atanor_os.emerge.life_gpu import Life

CANON = os.environ.get("ATANOR_DEPTH_PATH", "C:/<founder-folder>/27., ATANOR DEMO/packages/depth_learner")
def load(name):
    spec = importlib.util.spec_from_file_location("canon_depth_" + name, os.path.join(CANON, name + ".py")); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
SS = load("selfsup"); MD = load("model")

R.set_slots(R.CHROMA, 1 + R.CHROMA)
NB, T, STEPS, W_ = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
PH, PW = 48, 80
dev = "cuda"; torch.manual_seed(0)
FOVD = math.degrees(G.FOV)
f = 0.5 * PW / math.tan(0.5 * G.FOV)
K = SS.intrinsics(PH, PW, FOVD, device=dev)
xs = (torch.arange(PW, device=dev) + 0.5 - PW / 2)                     # pixel x relative to centre
offs = torch.atan(xs / f)                                               # pinhole: column angle
rows = torch.arange(PH, device=dev).float()[:, None] + 0.5 - PH / 2     # pixel y relative to centre (down +)


@torch.no_grad()
def render(w):
    """Pinhole picture [n, 3, PH, PW] and true depth along the view axis [n, PH, PW] (nan = sky)."""
    kind, dist = w.cast(w.head[:, None] + offs[None], step=0.05)
    z = dist * torch.cos(offs)[None]                                    # [n, PW]
    half = 0.5 * f / z.clamp(min=0.05)                                  # wall from -0.5 to +0.5 around eye height
    wall = rows.abs()[None] <= half[:, None, :]                         # [n, PH, PW]
    shade = (1.0 - z / 12.0).clamp(0.25, 1.0)
    col = G.COLOR.to(dev)[kind.long()] * shade[..., None]               # [n, PW, 3]
    sky, gnd = G.SKY.to(dev), G.GROUND.to(dev)
    base = torch.where((rows < 0)[None, :, :, None], sky, gnd).expand(w.n, PH, PW, 3)
    img = torch.where(wall[..., None], col[:, None].expand(-1, PH, -1, -1), base)
    floor_z = (0.5 * f / rows.clamp(min=1e-3)).expand(PH, PW)
    depth = torch.where(wall, z[:, None].expand(-1, PH, -1), torch.where((rows > 0)[None], floor_z[None], torch.full_like(wall, float("nan"), dtype=torch.float)))
    return img.permute(0, 3, 1, 2).clamp(0, 1), depth


def pose(pt, ht, ps, hs):
    """4x4 taking a point in the target camera (x right, y down, z forward) to the source camera, from where the body was and faced."""
    ft = torch.stack([torch.cos(ht), torch.sin(ht)], 1); rt = torch.stack([-torch.sin(ht), torch.cos(ht)], 1)
    fs = torch.stack([torch.cos(hs), torch.sin(hs)], 1); rs = torch.stack([-torch.sin(hs), torch.cos(hs)], 1)
    d = pt - ps; B = len(pt)
    M = torch.zeros(B, 4, 4, device=dev)
    M[:, 0, 0] = (rt * rs).sum(1); M[:, 0, 2] = (ft * rs).sum(1); M[:, 1, 1] = 1
    M[:, 2, 0] = (rt * fs).sum(1); M[:, 2, 2] = (ft * fs).sum(1)
    M[:, 0, 3] = (d * rs).sum(1); M[:, 2, 3] = (d * fs).sum(1); M[:, 3, 3] = 1
    return M


# gather: the pure core with widened futility walks (no hand rules); keep pinhole pictures, true depth (judging only), where it was and faced
net = Core(R.DX, R.M, R.INSIDE, R.PLACE); net.load_state_dict(torch.load(sys.argv[1]), strict=False); net = net.cuda().eval(); net.far = 0.3
L = Life(net, NB, R.HOME, dev, seed=4, pool="mean"); L.begin()
IM = torch.zeros(T, NB, 3, PH, PW, dtype=torch.uint8, device=dev); DP = torch.zeros(T, NB, PH, PW, dtype=torch.float16, device=dev)
PS = torch.zeros(T, NB, 2, device=dev); HD = torch.zeros(T, NB, device=dev); DEAD = torch.zeros(T, NB, dtype=torch.bool, device=dev)
for t in range(T):
    im, dp = render(L.world); IM[t] = (im * 255).to(torch.uint8); DP[t] = dp.half(); PS[t] = L.world.pos; HD[t] = L.world.head; DEAD[t] = L.world.dead
    L.tick()
print(f"gathered {NB} x {T} pinhole pictures {PH}x{PW}, fov {FOVD:.0f} deg", flush=True)
img = lambda t, b: IM[t, b].float() / 255

# Stage A: geometry sanity with TRUE depth
b = torch.arange(min(NB, 256), device=dev); t = 50
tgt, src = img(t, b), img(t + 1, b); dtrue = DP[t, b].float().nan_to_num(30.0).clamp(0.05, 30)[:, None]
P = pose(PS[t, b], HD[t, b], PS[t + 1, b], HD[t + 1, b])
moved = ((PS[t + 1, b] - PS[t, b]).norm(dim=1) > 0.05) | ((HD[t + 1, b] - HD[t, b]).abs() > 0.05)
e_w = SS.photometric(SS.warp(src, dtrue, P, K), tgt).mean((1, 2, 3)); e_i = SS.photometric(src, tgt).mean((1, 2, 3))
e_wrong = SS.photometric(SS.warp(src, dtrue * 3.0, P, K), tgt).mean((1, 2, 3))
print(f"stage A (bodies that moved: {int(moved.sum())}): warp with true depth {float(e_w[moved].mean()):.4f} | no warp {float(e_i[moved].mean()):.4f} | warp with 3x wrong depth {float(e_wrong[moved].mean()):.4f}", flush=True)

# Stage B: learn depth from own walking
dn = MD.DepthNet(width=W_).to(dev); opt = torch.optim.Adam(dn.parameters(), 2e-4)
print("depth net params", sum(p.numel() for p in dn.parameters()), flush=True)
tr_b, te_b = torch.arange(0, NB * 3 // 4, device=dev), torch.arange(NB * 3 // 4, NB, device=dev)
DMIN, DMAX = 0.1, 30.0
t0 = time.time()
for it in range(STEPS):
    bb = tr_b[torch.randint(len(tr_b), (24,), device=dev)]; tt = torch.randint(1, T - 1, (24,), device=dev)
    ok = ~DEAD[tt + 1, bb]
    bb, tt = bb[ok], tt[ok]
    x = IM[tt, bb].float() / 255
    nb = [IM[tt - 1, bb].float() / 255, IM[tt + 1, bb].float() / 255]
    ps = [pose(PS[tt, bb], HD[tt, bb], PS[tt - 1, bb], HD[tt - 1, bb]), pose(PS[tt, bb], HD[tt, bb], PS[tt + 1, bb], HD[tt + 1, bb])]
    d = SS.bounded_log_depth(dn(x), DMIN, DMAX).exp()[:, None]
    out = SS.selfsup_loss(x, nb, d, ps, K)
    opt.zero_grad(); out["loss"].backward(); opt.step()
    if it % 1000 == 0:
        print(f"  it {it} loss {float(out['loss']):.4f} kept {float(out['kept']):.2f} {time.time() - t0:.0f}s", flush=True)

# judge on held-out bodies (true depth only here)
dn.eval(); rk, rel_s, rel_r, rel_c, n_ = 0, 0, 0, 0, 0
with torch.no_grad():
    for t in range(5, T, 7):
        bb = te_b[~DEAD[t, te_b]]
        if not len(bb): continue
        pr = SS.bounded_log_depth(dn(img(t, bb)), DMIN, DMAX).exp(); tr = DP[t, bb].float()
        for i in range(len(bb)):
            v = ~torch.isnan(tr[i]); p_, q_ = pr[i][v], tr[i][v]
            if v.sum() < 50: continue
            a_ = p_.argsort().argsort().float(); c_ = q_.argsort().argsort().float()
            rk += float(torch.corrcoef(torch.stack([a_, c_]))[0, 1])
            s = q_.median() / p_.median(); rel_s += float(((p_ * s - q_).abs() / q_).mean()); rel_r += float(((p_ - q_).abs() / q_).mean())
            rel_c += float(((q_.median() - q_).abs() / q_).mean()); n_ += 1
print(f"held-out pictures {n_}: rank agreement with true depth {rk / n_:.3f} (0 = none, 1 = perfect) | error after median scaling {rel_s / n_:.3f} | raw error (body-scale) {rel_r / n_:.3f} | constant-depth baseline {rel_c / n_:.3f}", flush=True)
torch.save(dn.state_dict(), "models/depth0.pt")
print("DONE")
