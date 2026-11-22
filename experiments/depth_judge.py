"""Judge the Minecraft depth eye against lidar rays (judging only; the eye never sees them). One body on port 8791, random walk.
Per step: the eye's 5 eye-level sectors (depth) vs the 5 rays' block distance at the same angles.
Reports: rank agreement across the 5 sectors (Spearman, mean over steps), and whether the eye's nearest sector is the ray's nearest,
and correlation of the middle sector with the middle ray over time. Baseline: a constant guess (chance for 'nearest' = 0.2)."""
import sys, numpy as np, torch
from atanor_os.emerge.mc_client import MCWorld
from atanor_os.emerge.mc_depth import DepthEye, PH, PW, H0, CROP
w = MCWorld(port=8791); E = DepthEye(sys.argv[1], "cuda")
rng = np.random.default_rng(0)
P, R = [], []
act = 1
for t in range(int(sys.argv[2])):
    if rng.random() < 0.15:
        act = int(rng.choice([0, 1, 1, 1, 2, 3, 6]))
    o = w.step([act])[0]
    if t == 0: print("ray yaws", np.round(o["rays"].get("ray_yaw", []), 3).tolist(), "judge reach", o.get("judge", {}).get("reach"), flush=True)
    d = E.depth(torch.tensor(w.small(), device="cuda"))
    cy = int(H0 / 2 * PH / (H0 - CROP)); band = d[:, max(cy - 4, 0):cy + 4]
    sec = band.reshape(1, band.shape[1], 5, PW // 5).median(-1).values.median(1).values[0].cpu().numpy()
    r = np.asarray(o["rays"]["block_distance"], np.float32)[:5]
    P.append(sec); R.append(r)
P, R = np.array(P), np.array(R)
ok = (R < 29).all(1)                                                     # all five rays hit something within range
rk = lambda v: np.argsort(np.argsort(v, 1), 1).astype(float)
sp = [np.corrcoef(a, b)[0, 1] for a, b in zip(rk(P[ok]), rk(R[ok])) if a.std() > 0 and b.std() > 0]
near = float((P[ok].argmin(1) == R[ok].argmin(1)).mean())
mid = np.corrcoef(np.log(P[ok, 2]), np.log(R[ok, 2]))[0, 1]
print(f"steps {len(P)}, all rays hit {int(ok.sum())} | rank agreement across sectors {np.mean(sp):.3f} (0 = none) | nearest sector right {near:.2f} (chance 0.20) | middle sector vs ray over time r = {mid:.3f} (log)")
Rf = R[:, ::-1]
sp2 = [np.corrcoef(a, b)[0, 1] for a, b in zip(rk(P[ok]), rk(Rf[ok])) if a.std() > 0 and b.std() > 0]
print(f"rays in reversed order: rank agreement {np.mean(sp2):.3f} | nearest right {float((P[ok].argmin(1) == Rf[ok].argmin(1)).mean()):.2f}")
for k in range(5):
    print(f"  eye sector {k} vs ray {k}: r = {np.corrcoef(np.log(P[ok, k]), np.log(R[ok, k]))[0, 1]:.3f} | vs ray {4 - k}: r = {np.corrcoef(np.log(P[ok, k]), np.log(R[ok, 4 - k]))[0, 1]:.3f}")
print("DONE")
