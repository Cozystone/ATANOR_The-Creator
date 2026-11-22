"""Minecraft depth eye from own walking (canon depth_learner selfsup + DepthNet, read-only). No labels, no pretrained weights.
  python depth_mc.py <depthframes.npz> <steps>
Stage A (geometry sanity, before learning): with a constant depth, does warping by the felt motion (right sign of yaw / pitch) beat the
  flipped-sign geometry on frames where the body turned? If not, the camera model is wrong.
Stage B: learn depth; judge (no true depth in MineDojo) by:
  - photometric error after warping, held-out bodies, learned depth vs constant depth (does the depth explain how the picture flows?)
  - 'blocked ahead' AUC: nearness of the middle sector vs whether the next forward step moved the body < 0.05 (the world's answer)."""
import sys, time, numpy as np, torch
from atanor_os.emerge import mc_depth as D
SS, MD = D.canon("selfsup"), D.canon("model")
dev = "cuda"; torch.manual_seed(0)
d = np.load(sys.argv[1]); STEPS = int(sys.argv[2]); FAR_W = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0; FAR_M = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0; STILL = float(sys.argv[6]) if len(sys.argv) > 6 else 9.0; OUTP = sys.argv[4] if len(sys.argv) > 4 else "mc/mc_depth.pt"
S = torch.tensor(d["S"]).to(dev).permute(0, 1, 4, 2, 3).contiguous()      # [T, n, 3, 48, 80] uint8
Wh = torch.tensor(d["W"]).to(dev); J = torch.tensor(d["J"]).to(dev)
T, NB = J.shape
K = D.intrinsics(dev)
img = lambda t, b: S[t, b].float() / 255
jump = (Wh[1:, :, :3] - Wh[:-1, :, :3]).norm(dim=-1) > 2.0                # hand moved / died: do not link
link = ~(J[:-1] | jump)                                                    # link[t, b]: frame t -> t+1 is one continuous move
print(f"frames {T} x {NB}, linked pairs {float(link.float().mean()):.2f}", flush=True)

# Stage A
t_ = torch.arange(1, T - 1, device=dev)
turn = (Wh[1:, :, 3] - Wh[:-1, :, 3]).abs()
cand = torch.nonzero(link & (turn > 3) & (turn < 40))[:512]
tt, bb = cand[:, 0], cand[:, 1]
tgt, src = img(tt, bb), img(tt + 1, bb); dc = torch.full((len(tt), 1, D.PH, D.PW), 8.0, device=dev)
def err(ws):
    return SS.photometric(SS.warp(src, dc, D.pose(Wh[tt, bb], ws), K), tgt).mean((1, 2, 3)).mean()
w1 = Wh[tt + 1, bb].clone(); flip = Wh[tt, bb].clone(); flip[:, 3] = 2 * Wh[tt, bb, 3] - w1[:, 3]; flip[:, :3] = w1[:, :3]; flip[:, 4] = w1[:, 4]
print(f"stage A ({len(tt)} turning pairs): warp by felt turn {float(err(w1)):.4f} | flipped turn {float(err(flip)):.4f} | no warp {float(SS.photometric(src, tgt).mean()):.4f}", flush=True)
pt = torch.nonzero(link & ((Wh[1:, :, 4] - Wh[:-1, :, 4]).abs() > 3))[:512]
if len(pt) > 20:
    tt, bb = pt[:, 0], pt[:, 1]; tgt, src = img(tt, bb), img(tt + 1, bb); dc = torch.full((len(tt), 1, D.PH, D.PW), 8.0, device=dev)
    w1 = Wh[tt + 1, bb].clone(); fl = w1.clone(); fl[:, 4] = 2 * Wh[tt, bb, 4] - w1[:, 4]
    print(f"stage A pitch ({len(tt)} pairs): felt {float(err(w1)):.4f} | flipped {float(err(fl)):.4f}", flush=True)

# Stage B
dn = MD.DepthNet(width=16).to(dev); opt = torch.optim.Adam(dn.parameters(), 2e-4)
trb = torch.arange(0, max(NB - 1, 1), device=dev); teb = torch.tensor([NB - 1], device=dev)
ok_mid = link[:-1] & link[1:]                                              # t-1 -> t -> t+1 all continuous: index by t-1
cand = torch.nonzero(ok_mid); cand = cand[torch.isin(cand[:, 1], trb)]
t0 = time.time()
for it in range(STEPS):
    c = cand[torch.randint(len(cand), (24,), device=dev)]; tt, bb = c[:, 0] + 1, c[:, 1]
    x = img(tt, bb); nb = [img(tt - 1, bb), img(tt + 1, bb)]
    ps = [D.pose(Wh[tt, bb], Wh[tt - 1, bb]), D.pose(Wh[tt, bb], Wh[tt + 1, bb])]
    dd = SS.bounded_log_depth(dn(x), 0.1, 30.0).exp()[:, None]
    out = SS.selfsup_loss(x, nb, dd, ps, K)
    if FAR_W:                                                  # 아주 멀다고 쳐도 흐름이 똑같이 (또는 더 잘) 설명되는 화소는 멀다고 본다 (하늘, 먼 산): 흐르지 않는 곳의 가장 단순한 설명
        with torch.no_grad():
            far = torch.full_like(dd, 30.0)
            e_far = torch.cat([SS.photometric(SS.warp(s_, far, p_, K), x) for s_, p_ in zip(nb, ps)], 1).min(1, keepdim=True)[0]
            e_now = torch.cat([SS.photometric(SS.warp(s_, dd, p_, K), x) for s_, p_ in zip(nb, ps)], 1).min(1, keepdim=True)[0]
            e_id = torch.cat([SS.photometric(s_, x) for s_ in nb], 1).min(1, keepdim=True)[0]
            m_far = ((e_far + FAR_M < e_now) & (e_id < STILL)).float()   # 멀다고 칠 때 더 잘 설명되고, 그 화소가 앞뒤 그림 사이에 거의 바뀌지 않을 때만 (하늘처럼 흐르지 않는 곳)
        out["loss"] = out["loss"] + FAR_W * (m_far * (np.log(30.0) - torch.log(dd))).mean()
    opt.zero_grad(); out["loss"].backward(); opt.step()
    if it % 2000 == 0:
        print(f"  it {it} loss {float(out['loss']):.4f} kept {float(out['kept']):.2f} {time.time() - t0:.0f}s", flush=True)
dn.eval()
torch.save(dn.state_dict(), OUTP)
# sky check (judging only): daytime frames' top band where the colour is sky-like (blue above red/green, bright) - mean predicted depth there vs the rest
with torch.no_grad():
    c = torch.nonzero(ok_mid)[:2000]; tt, bb = c[:, 0] + 1, c[:, 1]
    sky_d, rest_d = [], []
    for i in range(0, len(tt), 64):
        x = img(tt[i:i + 64], bb[i:i + 64]); dd = SS.bounded_log_depth(dn(x), 0.1, 30.0).exp()
        sky = (x[:, 2] > x[:, 0] + 0.08) & (x[:, 2] > x[:, 1]) & (x.mean(1) > 0.45)
        sky_d.append(dd[sky]); rest_d.append(dd[~sky])
    sky_d, rest_d = torch.cat(sky_d), torch.cat(rest_d)
    print(f"sky-coloured pixels {len(sky_d)}: median depth {float(sky_d.median()):.1f} | other pixels median {float(rest_d.median()):.1f} (bounds 0.1-30)", flush=True)

with torch.no_grad():
    c = torch.nonzero(ok_mid); c = c[torch.isin(c[:, 1], teb)][:1500]; tt, bb = c[:, 0] + 1, c[:, 1]
    mv = (Wh[tt + 1, bb, :3] - Wh[tt, bb, :3]).norm(dim=-1) > 0.05
    e_l, e_c, e_i = [], [], []
    for i in range(0, len(tt), 64):
        t2, b2 = tt[i:i + 64], bb[i:i + 64]; x = img(t2, b2); s2 = img(t2 + 1, b2); P = D.pose(Wh[t2, b2], Wh[t2 + 1, b2])
        dd = SS.bounded_log_depth(dn(x), 0.1, 30.0).exp()[:, None]
        e_l.append(SS.photometric(SS.warp(s2, dd, P, K), x).mean((1, 2, 3)))
        e_c.append(SS.photometric(SS.warp(s2, torch.full_like(dd, float(dd.median())), P, K), x).mean((1, 2, 3)))
        e_i.append(SS.photometric(s2, x).mean((1, 2, 3)))
    e_l, e_c, e_i = torch.cat(e_l), torch.cat(e_c), torch.cat(e_i)
    print(f"held-out body, pairs that moved ({int(mv.sum())}): warp error learned depth {float(e_l[mv].mean()):.4f} | constant depth {float(e_c[mv].mean()):.4f} | no warp {float(e_i[mv].mean()):.4f}", flush=True)
print("DONE")
