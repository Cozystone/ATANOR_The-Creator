"""Offline diagnostic for the TD fear value (EMERGE_PLAN 2026-11-22 '돌파 연구').
  python diag_fear.py <steplog.npz> <layout: skin | all | base>
Learn V_h(x) = expected discounted future harm (TD, gamma 0.95) on the first 70% of time; on the last 30% judge how well each score
predicts 'harm within the next k steps' (AUC), k = 1, 5, 10, 20:
  - V_h (all senses), and V_h without each sense group (ablation: which group carries the warning)
  - episodic harm memory as now: closeness to the last 32 harm moments (core's weighting), and to ALL past harm moments (upper bound)
  - body state only (low health)
Also restricted to harm from outside (air full, not starving), which is the hard case (zombies, falls), and in steps that are not
already hurt in the previous 5 steps (true early warning, not 'still being hit')."""
import sys, numpy as np, torch, torch.nn as nn
torch.manual_seed(0); dev = "cuda"
d = np.load(sys.argv[1]); lay = sys.argv[2]
X = torch.tensor(d["X"]).float().to(dev); A = d["A"]; Dm = d["D"]; K = d["K"]; P = d["P"]
T, n, dx = X.shape
groups = {"inside": slice(0, 3), "hurt": slice(3, 4), "felt": slice(4, 7), "gaze+bug": slice(7, 10), "eye summary": slice(10, 74), "focus": slice(74, 106)}
o = 106
if lay == "all":
    groups["depth"] = slice(o, o + 5); o += 5; groups["ear"] = slice(o, o + 16); o += 16
if lay in ("all", "skin"):
    groups["skin: touch"] = slice(o, o + 11); groups["skin: pain side"] = slice(o + 11, o + 15); groups["skin: light"] = slice(o + 15, o + 20); o += 20
assert o == dx, (o, dx)
jump = np.zeros((T, n), bool); jump[:-1] = np.linalg.norm(P[1:, :, [0, 2]] - P[:-1, :, [0, 2]], axis=-1) > 15
term = jump | K                                                       # no bootstrapping across death / hand moves
H = (Dm > 0.01).astype(np.float32)
cut = int(T * 0.7)
print(f"log {T} steps x {n} bodies, dx {dx}, harm steps {int(H.sum())} (train {int(H[:cut].sum())}, test {int(H[cut:].sum())})", flush=True)


def ahead(k):
    y = np.zeros((T, n), bool)
    for j in range(k):
        y[:T - j] |= H[j:] > 0
    return y


def auc(score, y, mask):
    s, yy = score[mask], y[mask]
    if yy.sum() == 0 or (~yy).sum() == 0:
        return float("nan")
    r = np.argsort(np.argsort(s)).astype(float)
    return float((r[yy].mean() - (yy.sum() - 1) / 2) / (~yy).sum())


def fit_v(cols):
    keep = torch.zeros(dx, device=dev); keep[cols] = 1
    V = nn.Sequential(nn.Linear(dx, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 1)).to(dev)
    Vt = nn.Sequential(nn.Linear(dx, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 1)).to(dev); Vt.load_state_dict(V.state_dict())
    opt = torch.optim.Adam(V.parameters(), 1e-3)
    Dt = torch.tensor(Dm, device=dev); Tm = torch.tensor(term, device=dev)
    for it in range(6000):
        t = torch.randint(0, cut - 1, (512,), device=dev); b = torch.randint(0, n, (512,), device=dev)
        x, x2 = X[t, b] * keep, X[t + 1, b] * keep
        with torch.no_grad():
            tg = Dt[t, b] + 0.95 * Vt(x2)[:, 0] * (~Tm[t, b]).float()
        loss = ((V(x)[:, 0] - tg) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        if it % 200 == 0:
            Vt.load_state_dict(V.state_dict())
    with torch.no_grad():
        return torch.cat([V(X[i:i + 2000].reshape(-1, dx) * keep).reshape(-1, n) for i in range(0, T, 2000)]).cpu().numpy()


def episodic(last=32):
    """closeness to harm moments seen so far (core avoid(): weights 1/var of the harm moments over non-inside dims)."""
    other = np.ones(dx, bool); other[0:3] = False
    out = np.full((T, n), -1e9, np.float32); mem = []
    Xn = X.cpu().numpy()
    for t in range(T):
        if len(mem) > 1:
            hx = np.stack(mem[-last:] if last else mem)
            q = 1.0 / (hx[:, other].var(0) + 0.02); q = q / q.mean()
            dist = (((Xn[t][:, None, other] - hx[None, :, other]) ** 2) * q).sum(-1) / other.sum()
            out[t] = -dist.min(1)
        for b in np.where(H[t] > 0)[0]:
            mem.append(Xn[t, b])
    return out


test = np.zeros((T, n), bool); test[cut:] = True
calm = np.ones((T, n), bool)
for j in range(1, 6):
    calm[j:] &= H[:-j] == 0                                           # not hurt in the previous 5 steps: early warning, not 'still being hit'
outside = (X[:, :, 2] > 0.99).cpu().numpy() & (X[:, :, 1] > 0.3).cpu().numpy()
scores = {"V_h all senses": fit_v(list(range(dx)))}
for g, sl in groups.items():
    cols = [c for c in range(dx) if not (sl.start <= c < sl.stop)]
    scores[f"V_h without {g}"] = fit_v(cols)
scores["episodic last 32"] = episodic(32)
scores["episodic all"] = episodic(0)
scores["low health only"] = -X[:, :, 0].cpu().numpy()
print(f"{'score':28s} " + " ".join(f"{'k=' + str(k):>7s}" for k in (1, 5, 10, 20)) + " | outside+calm k=5 k=10 k=20", flush=True)
for nm, s in scores.items():
    row = [auc(s, ahead(k), test) for k in (1, 5, 10, 20)]
    row2 = [auc(s, ahead(k) & outside, test & calm & outside) for k in (5, 10, 20)]
    print(f"{nm:28s} " + " ".join(f"{v:7.3f}" for v in row) + " | " + " ".join(f"{v:6.3f}" for v in row2), flush=True)
print("DONE")
