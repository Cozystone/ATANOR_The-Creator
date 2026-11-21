"""Minecraft eye, first look: gather what the canon eye receives (fovea + 4 rings + ring peaks, 27x16x16, and where the gaze is) with a test motion (NOT the being; labelled on the viewer), then learn the
inverse eye + Gaussian step (eyewm.WM, 7 acts) from them, and judge like wm0: imagined k steps ahead (eyes shut, acts only) vs 'the world stays'.
  python mc_eye.py gather N      -> mc/frames.npz
  python mc_eye.py learn STEPS   -> mc/mc_wm0.pt"""
import os, sys, time
import numpy as np
OUT = "mc"
os.makedirs(OUT, exist_ok=True)

if sys.argv[1] == "gather":
    from atanor_os.emerge.mc_client import MCWorld
    N = int(sys.argv[2]); w = MCWorld(); rng = np.random.default_rng(0)
    P = np.zeros((N + 1, w.n, 27, 16, 16), np.float16); G = np.zeros((N + 1, w.n, 2), np.float32); A = np.zeros((N, w.n), np.int64); IN = np.zeros((N + 1, w.n, 3), np.float32)
    FE = np.zeros((N, w.n, 3), np.float32); HU = np.zeros((N, w.n), np.float32); DN = np.zeros((N + 1, w.n), bool)
    P[0] = w.pics(); G[0] = w.gaze(); IN[0] = w.inside(); t0 = time.time(); cur = np.ones(w.n, np.int64); hold = np.zeros(w.n, np.int64)
    for t in range(N):
        for i in range(w.n):                                   # test motion: keeps an act for a few steps (so the eye sees walking and turning, not jitter)
            if hold[i] <= 0:
                cur[i] = rng.choice(7, p=[0.08, 0.42, 0.14, 0.14, 0.08, 0.06, 0.08]); hold[i] = rng.integers(2, 10)
            hold[i] -= 1
        A[t] = cur
        obs = w.step(cur, who="시험 몸짓 (존재 아님): 눈을 기르려고 그림을 모으는 중", label=f"그림 모으기 {t + 1} / {N}")
        P[t + 1] = w.pics(); G[t + 1] = w.gaze(); IN[t + 1] = w.inside(); FE[t] = w.felt; HU[t] = [o["hurt"] for o in obs]; DN[t + 1] = [o["done"] for o in obs]
        dead = [i for i in range(w.n) if obs[i]["done"]]
        if (t + 1) % 300 == 0:                                 # the experimenter carries the body to another surface spot (diverse pictures; not the being's choice)
            w.move(range(w.n), r=400, label=f"그림 모으기 {t + 1} / {N} (실험자가 몸을 옮김)"); P[t + 1] = w.pics(); G[t + 1] = w.gaze(); DN[t + 1] = True
        if dead:
            w.reset(dead); P[t + 1, dead] = w.pics()[dead]
        if t % 500 == 0:
            print(f"  {t} steps {time.time() - t0:.0f}s, health {IN[t + 1, :, 0].mean():.2f} food {IN[t + 1, :, 1].mean():.2f} air {IN[t + 1, :, 2].mean():.2f}", flush=True)
    np.savez_compressed(f"{OUT}/frames.npz", P=P, G=G, A=A, IN=IN, FE=FE, HU=HU, DN=DN)
    print(f"gathered {N} x {w.n} in {time.time() - t0:.0f}s; deaths {int(DN.sum())}; hurt events {int((HU > 0).sum())}; air below full {(IN[:, :, 2] < 0.99).mean():.3f}", flush=True)

else:
    import torch, torch.nn.functional as F
    from atanor_os.emerge.eyewm import WM
    torch.manual_seed(0); dev = "cuda"
    d = np.load(f"{OUT}/frames.npz"); P = torch.tensor(d["P"]).to(dev); A = torch.tensor(d["A"]).to(dev); DN = torch.tensor(d["DN"]).to(dev)
    T, NB = A.shape; K = 8; STEPS = int(sys.argv[2])
    cut = int(T * 0.8)                                         # learn on the first 80% of the time, judge on the last 20% (unseen stretches)
    wm = WM(z=32, m=7, ch=27, hw=(16, 16)).to(dev); opt = torch.optim.Adam(wm.parameters(), 3e-4)
    f = lambda t, b: P[t, b].float()
    ok_run = lambda t, b: ~DN[t + 1:t + K + 1, b].any(0)
    t0 = time.time()
    for it in range(STEPS):
        b = torch.randint(NB, (64,), device=dev); t = torch.randint(0, cut - K - 1, (64,), device=dev)
        keep = ~torch.stack([DN[t + k, b] for k in range(1, K + 1)]).any(0); b, t = b[keep], t[keep]
        x0 = f(t, b); mu, lv = wm.encode(x0); z = mu + torch.randn_like(mu) * (0.5 * lv).exp()
        loss = F.mse_loss(wm.decode(z), x0) * 100 + 1e-3 * (0.5 * (mu ** 2 + lv.exp() - 1 - lv)).sum(1).mean()
        zi = z
        for k in range(1, K + 1):
            zm, zlv = wm.next(zi, A[t + k - 1, b]); tm, _ = wm.encode(f(t + k, b))
            loss = loss + 0.5 * ((zm - tm.detach()) ** 2 / zlv.exp() + zlv).sum(1).mean() * 1e-2 + F.mse_loss(wm.decode(zm), f(t + k, b)) * 100 / K
            zi = zm + torch.randn_like(zm) * (0.5 * zlv).exp()
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 2000 == 0:
            print(f"  it {it} loss {float(loss):.3f} {time.time() - t0:.0f}s", flush=True)
    wm.eval(); res = {k: [[], [], []] for k in (1, 2, 4, 8)}
    with torch.no_grad():
        for _ in range(60):
            b = torch.randint(NB, (128,), device=dev); t = torch.randint(cut, T - K - 1, (128,), device=dev)
            keep = ~torch.stack([DN[t + k, b] for k in range(1, K + 1)]).any(0); b, t = b[keep], t[keep]
            x0 = f(t, b); z = wm.encode(x0)[0]
            for k in range(1, K + 1):
                z = wm.next(z, A[t + k - 1, b])[0]
                if k in res:
                    tg = f(t + k, b)
                    res[k][0].append(((wm.decode(z) - tg) ** 2).mean((1, 2, 3))); res[k][1].append(((x0 - tg) ** 2).mean((1, 2, 3))); res[k][2].append(A[t, b])
    for k, (ei, ec, a) in res.items():
        ei, ec, a = torch.cat(ei), torch.cat(ec), torch.cat(a); mv = ec > 1e-4
        line = f"k={k}: imagined {float(ei[mv].mean()):.4f} vs world-stays {float(ec[mv].mean()):.4f} (imagined better in {float((ei < ec)[mv].float().mean()):.2f})"
        for nm, s in (("turn", (a == 2) | (a == 3)), ("forward", (a == 1) | (a == 6))):
            m = mv & s
            if m.any():
                line += f" | first act {nm}: {float(ei[m].mean()):.4f} vs {float(ec[m].mean()):.4f}"
        print(line, flush=True)
    torch.save(wm.state_dict(), f"{OUT}/mc_wm0.pt"); print("DONE")
