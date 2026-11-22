"""L2 check on run 37b: did the being's own forecast come to use the elder's sound to foresee the scene?
For steps right after the elder spoke (t in [s, s+6]) vs ordinary steps: forecast error of the next eye summary (dims 10:74) and focus summary
(74:106) from (x_t, a_t) with the real ear features vs ear features swapped in from random other moments. If the forecast leans on the sound,
swapping hurts more right after speech than at ordinary moments."""
import numpy as np, torch
from atanor_os.emerge.core import Core
OUT = "mc"
d = np.load(f"{OUT}/steplog_being37b_elder.npz", allow_pickle=True)
X = torch.tensor(d["X"]).float().cuda(); A = torch.tensor(d["A"]).long().cuda(); T, n, dx = X.shape
SP = d["SPOKE"] if "SPOKE" in d.files else np.zeros((0, 2), int); txt = d["SPOKE_TEXT"] if "SPOKE_TEXT" in d.files else []
print(f"log {T} x {n}, dx {dx}, elder spoke {len(SP)} times; words: {sorted(set(str(t).split(': ')[-1] for t in txt))[:15]}")
net = Core(dx, 7, slice(0, 3), slice(0, 0)).cuda(); net.load_state_dict(torch.load(f"{OUT}/being37b_elder.pt"), strict=False); net.eval()
EAR = slice(106, 122); EYE = slice(10, 106)
after = np.zeros((T, n), bool)
for s, b in SP:
    after[int(s):int(s) + 7, int(b)] = True
rng = np.random.default_rng(0)


def err(mask, swap):
    tt, bb = np.nonzero(mask[:-1]);
    if len(tt) > 3000:
        k = rng.choice(len(tt), 3000, replace=False); tt, bb = tt[k], bb[k]
    x = X[tt, bb].clone(); x2 = X[tt + 1, bb]; a = A[tt, bb]
    if swap:
        j = rng.integers(0, T, len(tt)); k = rng.integers(0, n, len(tt)); x[:, EAR] = X[j, k][:, EAR]
    with torch.no_grad():
        fo = net.foresee(x).gather(1, a[:, None, None].expand(-1, 1, dx))[:, 0]
    return float(((fo[:, EYE] - x2[:, EYE]) ** 2).mean()), len(tt)


for nm, m in (("right after the elder spoke", after), ("ordinary moments", ~after)):
    e0, k = err(m, False); e1, _ = err(m, True)
    print(f"{nm:28s} ({k} steps): eye-forecast error real sound {e0:.4f} | swapped sound {e1:.4f} | worse by {100 * (e1 - e0) / e0:+.1f}%")
print("DONE")
