"""D28 (atanor_os/GRAIN_THEORY.md, registered before running): false shortcuts in the being's place map, and the body's own continuity.
  PYTHONPATH=. python -m tools.d28_shortcut OUT.json 9101_0 ..."""
import json, sys, time
import numpy as np, torch
from scipy.stats import spearmanr
from tools.need_fork import t_room, DEV
from tools import d25_grain as D25
from tools.d26_map import kmeans, GAMMA
from tools.d27_place import code, SAVE


def value(U, Lf, rise, k, min_share=0.0, shuffle_seed=None):
    same = (Lf[1:] == Lf[:-1]) & (U[1:] < k) & (U[:-1] < k); a, b = U[:-1][same], U[1:][same]; C = np.zeros((k, k)); np.add.at(C, (a, b), 1); C = C + C.T
    if min_share:
        C = np.where(C >= min_share * C.sum(1, keepdims=True), C, 0); C = np.minimum(C, C.T) * (C > 0) + 0 * C; np.fill_diagonal(C, np.maximum(C.diagonal(), 1))
    Tm = C / np.maximum(C.sum(1, keepdims=True), 1); u = U.reshape(-1); ok = u < k; cnt = np.bincount(u[ok], minlength=k).astype(float); r = np.zeros((k, 2))
    for c in (0, 1):
        r[:, c] = np.bincount(u[ok], weights=rise[..., c].reshape(-1)[ok].astype(float), minlength=k) / np.maximum(cnt, 1)
    if shuffle_seed is not None:
        r = r[np.random.default_rng(shuffle_seed).permutation(k)]
    return np.linalg.solve(np.eye(k) - GAMMA * Tm, r)


def smooth(U, Lf, k, h=4):
    T, n = U.shape; out = U.copy()
    for i in range(n):
        for t in range(T):
            a, b = max(0, t - h), min(T, t + h + 1); w = U[a:b, i][(Lf[a:b, i] == Lf[t, i]) & (U[a:b, i] < k)]
            if len(w):
                out[t, i] = np.bincount(w).argmax()
    return out


def main():
    out = sys.argv[1]; res = []; t0 = time.time()
    for lay in sys.argv[2:]:
        seed, mirror = map(int, lay.split("_")); rows, info = t_room(seed, bool(mirror)); sd = seed * 10 + mirror; z = torch.load(f"{SAVE}/{lay}.pt", weights_only=False)
        X, P, St, Lf, ins, Acts, M = z["X"], z["P"], z["St"], z["Lf"], z["ins"], z["acts"], z["M"]; T, n, dx = X.shape
        rise = np.zeros((T, n, 2), bool); rise[:-1] = ((St[1:] - St[:-1]) > 1e-4) & (Lf[1:] == Lf[:-1])[..., None]
        Xo = X.clone(); Xo[:, :, ins] = 0; mu, sg = Xo.reshape(-1, dx).mean(0), Xo.reshape(-1, dx).std(0) + 1e-6; Xo = (Xo - mu) / sg
        g = np.array([list(r) for r in rows]); cy, cx = np.clip(P[..., 1].astype(int), 0, g.shape[0] - 1), np.clip(P[..., 0].astype(int), 0, g.shape[1] - 1)
        dmap = {0: D25.dist_map(rows, "w"), 1: D25.dist_map(rows, "f")}; dist = [dmap[c][cy, cx].reshape(-1).astype(float) for c in (0, 1)]
        Z, valid, _ = code("one", Xo, Acts, Lf, M, sd); rec = dict(layout=lay, k={})
        for k in (64, 256):
            U = kmeans(Z.reshape(T * n, -1), k, sd).cpu().numpy().reshape(T, n); u = U.reshape(-1); far = np.zeros(T * n, bool); cyf, cxf = cy.reshape(-1), cx.reshape(-1)
            for q in range(k):
                m = np.nonzero(u == q)[0]
                if len(m) < 2:
                    continue
                cells = cyf[m] * 100 + cxf[m]; top = np.bincount(cells).argmax(); ty, tx = top // 100, top % 100
                dq = D25.dist_map([("".join("x" if (yy, xx) == (ty, tx) else ch for xx, ch in enumerate(row))) for yy, row in enumerate(rows)], "x")[cyf[m], cxf[m]]; far[m[dq > 3]] = True
            Up = U.copy().reshape(-1); Up[far] = k; Up = Up.reshape(T, n); Us = smooth(U, Lf, k)
            fs = (cyf * 100 + cxf); r = dict(far_share=float(far.mean()))
            def w(Uv, **kw):
                V = value(Uv, Lf, rise, k, **kw); uu = Uv.reshape(-1); ok = uu < k
                return [round(float(spearmanr(V[uu[ok], c], -dist[c][ok])[0]), 2) for c in (0, 1)]
            r["as is"] = w(U); r["judge-purged"] = w(Up); r["continuity"] = w(Us); r["continuity + rare links dropped"] = w(Us, min_share=0.01)
            r["continuity shuffled"] = w(Us, shuffle_seed=sd); r["continuity + rare shuffled"] = w(Us, min_share=0.01, shuffle_seed=sd)
            r["flips removed by continuity"] = float((Us != U).mean())
            rec["k"][str(k)] = r; print(lay, "k", k, r, round(time.time() - t0), flush=True)
        res.append(rec); json.dump(dict(protocol="atanor_os/GRAIN_THEORY.md (D28)", rows=res), open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
