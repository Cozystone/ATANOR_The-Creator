"""D29 (atanor_os/GRAIN_THEORY.md, registered before running): landmarks laid down by novelty; does the map over them know the way?
  PYTHONPATH=. python -m tools.d29_landmarks OUT.json 9101_0 ..."""
import json, sys, time
import numpy as np, torch
from scipy.stats import spearmanr
from tools.need_fork import t_room, DEV
from tools import d25_grain as D25
from tools.d27_place import code, SAVE
from tools.d28_shortcut import value


def landmarks(F, n_keep, seed, pool=40000):
    """Farthest-first over the being's own sensed states: a moment is kept when it is unlike every landmark kept so far."""
    g = torch.Generator(device=F.device).manual_seed(seed); S = F[torch.randperm(len(F), generator=g, device=F.device)[:pool]]
    keep = [0]; d = torch.cdist(S, S[:1])[:, 0]
    for _ in range(n_keep - 1):
        j = int(d.argmax()); keep.append(j); d = torch.minimum(d, torch.cdist(S, S[j:j + 1])[:, 0])
    L = S[keep]
    return torch.cat([torch.cdist(F[j:j + 8192], L).argmin(1) for j in range(0, len(F), 8192)]).cpu().numpy()


def main():
    out = sys.argv[1]; res = []; t0 = time.time()
    for lay in sys.argv[2:]:
        seed, mirror = map(int, lay.split("_")); rows, info = t_room(seed, bool(mirror)); sd = seed * 10 + mirror; z = torch.load(f"{SAVE}/{lay}.pt", weights_only=False)
        X, P, St, Lf, ins, Acts, M = z["X"], z["P"], z["St"], z["Lf"], z["ins"], z["acts"], z["M"]; T, n, dx = X.shape
        rise = np.zeros((T, n, 2), bool); rise[:-1] = ((St[1:] - St[:-1]) > 1e-4) & (Lf[1:] == Lf[:-1])[..., None]
        Xo = X.clone(); Xo[:, :, ins] = 0; mu, sg = Xo.reshape(-1, dx).mean(0), Xo.reshape(-1, dx).std(0) + 1e-6; Xo = (Xo - mu) / sg
        g = np.array([list(r) for r in rows]); cy, cx = np.clip(P[..., 1].astype(int), 0, g.shape[0] - 1), np.clip(P[..., 0].astype(int), 0, g.shape[1] - 1)
        cell = (cy * 100 + cx).reshape(-1); cells, ci = np.unique(cell, return_inverse=True); Dc = [D25.dist_map(rows, c)[cells // 100, cells % 100].astype(float) for c in "wf"]
        stem = (np.abs(cells % 100 + .5 - info["x0"]) <= 2.5) & (cells // 100 >= 3); st_state = stem[ci]
        Z, _, _ = code("one", Xo, Acts, Lf, M, sd); feats = {"raw": Xo.reshape(T * n, dx).to(DEV), "learned": Z.reshape(T * n, -1)}; rec = dict(layout=lay, time_in_stem=float(st_state.mean()), maps={})
        for fn, Fm in feats.items():
            for k in (512, 128):
                U = landmarks(Fm, k, sd).reshape(T, n); u = U.reshape(-1); r = {}
                # judge-side description: landmarks per zone, and false links
                home = np.array([np.bincount(ci[u == q]).argmax() if (u == q).any() else -1 for q in range(k)]); r["landmarks_in_stem_share"] = float(stem[home[home >= 0]].mean())
                dq = np.array([D25.dist_map([("".join("x" if (yy, xx) == (cells[home[q]] // 100, cells[home[q]] % 100) else ch for xx, ch in enumerate(row))) for yy, row in enumerate(rows)], "x") if home[q] >= 0 else None for q in range(k)], dtype=object)
                far = np.zeros(T * n, bool)
                for q in range(k):
                    m = np.nonzero(u == q)[0]
                    if len(m):
                        far[m] = dq[q][cells[ci[m]] // 100, cells[ci[m]] % 100] > 3
                r["far_member_share"] = float(far.mean())
                for shuf in (False, True):
                    V = value(U, Lf, rise, k, shuffle_seed=sd if shuf else None); key = "shuffled" if shuf else "map"; r[key] = {}
                    for c, nm in ((0, "thirsty"), (1, "hungry")):
                        vc = np.bincount(ci, weights=V[u, c]) / np.bincount(ci); dist_s = Dc[c][ci]
                        r[key][nm] = dict(per_cell=round(float(spearmanr(vc, -Dc[c])[0]), 2), per_cell_outside_stem=round(float(spearmanr(vc[~stem], -Dc[c][~stem])[0]), 2), per_state=round(float(spearmanr(V[u, c], -dist_s)[0]), 2))
                rec["maps"][f"{fn} {k}"] = r; print(lay, fn, k, r, round(time.time() - t0), flush=True)
        res.append(rec); json.dump(dict(protocol="atanor_os/GRAIN_THEORY.md (D29)", rows=res), open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
