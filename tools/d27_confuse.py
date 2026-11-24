"""D27 follow-up named in its registered reading ('none passes -> report what is being confused, and whether the room gives the eye enough').
Judge-side description only, on the saved D27 stretches.  PYTHONPATH=. python -m tools.d27_confuse OUT.json 9101_0 ..."""
import json, sys, time
import numpy as np, torch
from scipy.stats import spearmanr
from tools.need_fork import t_room, DEV
from tools import d25_grain as D25
from tools.d26_map import kmeans, map_value, K
from tools.d27_place import code, SAVE


def way(U, k, Lf, rise, dmap, cy, cx, vf):
    V = map_value(U, Lf, rise, k, True); u = U.reshape(-1)
    return [float(spearmanr(V[u, c][vf], -dmap[c][cy, cx].reshape(-1)[vf].astype(float))[0]) for c in (0, 1)], V


def main():
    out = sys.argv[1]; res = []; t0 = time.time()
    for lay in sys.argv[2:]:
        seed, mirror = map(int, lay.split("_")); rows, info = t_room(seed, bool(mirror)); sd = seed * 10 + mirror; z = torch.load(f"{SAVE}/{lay}.pt", weights_only=False)
        X, P, St, Lf, ins, Acts, M = z["X"], z["P"], z["St"], z["Lf"], z["ins"], z["acts"], z["M"]; T, n, dx = X.shape
        rise = np.zeros((T, n, 2), bool); rise[:-1] = ((St[1:] - St[:-1]) > 1e-4) & (Lf[1:] == Lf[:-1])[..., None]
        Xo = X.clone(); Xo[:, :, ins] = 0; mu, sg = Xo.reshape(-1, dx).mean(0), Xo.reshape(-1, dx).std(0) + 1e-6; Xo = (Xo - mu) / sg
        g = np.array([list(r) for r in rows]); cy, cx = np.clip(P[..., 1].astype(int), 0, g.shape[0] - 1), np.clip(P[..., 0].astype(int), 0, g.shape[1] - 1)
        dmap = {0: D25.dist_map(rows, "w"), 1: D25.dist_map(rows, "f")}; dw, df = dmap[0][cy, cx].reshape(-1), dmap[1][cy, cx].reshape(-1); pos = P.reshape(-1, 2); allv = np.ones(T * n, bool)
        rec = dict(layout=lay, time_share=dict(at_water=float((dw <= 2).mean()), at_food=float((df <= 2).mean())), rises=rise.reshape(-1, 2).sum(0).tolist())
        # 1. is coarseness the problem? true-position units of growing size (contiguous by construction)
        rec["true position units"] = {}
        for k in (40, 20, 10, 6):
            U = kmeans(torch.as_tensor(pos, device=DEV).float(), k, sd).cpu().numpy(); u = U
            spread = float(np.mean([np.sqrt(((pos[u == q] - pos[u == q].mean(0)) ** 2).sum(1).mean()) for q in range(k) if (u == q).sum() > 5]))
            rec["true position units"][str(k)] = dict(spread=spread, way=way(U.reshape(T, n), k, Lf, rise, dmap, cy, cx, allv)[0])
        # 2. the being's code ('one' and 'carried'): where the relief record sits, and what a unit mixes
        for kind in ("one", "carried"):
            Z, valid, _ = code(kind, Xo, Acts, Lf, M, sd); vf = valid.reshape(-1); U = np.full(T * n, K, int)
            U[vf] = kmeans(Z.reshape(T * n, -1)[torch.as_tensor(vf, device=DEV)], K, sd).cpu().numpy(); w, V = way(U.reshape(T, n), K + 1, Lf, rise, dmap, cy, cx, vf); r = dict(way=w)
            far = 0; tot = 0; leak = [0.0, 0.0]; mass = [0.0, 0.0]; rs = rise.reshape(-1, 2)
            for q in range(K):
                m = U == q
                if m.sum() < 5:
                    continue
                # far alias: states of the unit more than 3 path-cells from the unit's commonest cell
                cells = cy.reshape(-1)[m] * 100 + cx.reshape(-1)[m]; top = np.bincount(cells).argmax(); ty, tx = top // 100, top % 100
                dq = D25.dist_map([("".join("x" if (yy, xx) == (ty, tx) else ch for xx, ch in enumerate(row))) for yy, row in enumerate(rows)], "x")[cy.reshape(-1)[m], cx.reshape(-1)[m]]
                far += int((dq > 3).sum()); tot += int(m.sum())
                for c, dn, do in ((0, dw, df), (1, df, dw)):                   # relief record of channel c: how much of it sits in units that are mostly at the OTHER spring
                    rate = rs[m, c].mean(); mass[c] += rate; leak[c] += rate * float((do[m] <= 2).mean() > (dn[m] <= 2).mean() and (do[m] <= 2).mean() > 0.3)
            r["far_alias_share"] = far / max(1, tot); r["relief_record_at_wrong_spring"] = [leak[c] / max(mass[c], 1e-9) for c in (0, 1)]
            # where the map's best unit truly is: median path distance of its states to the needed spring / to the other spring
            for c, nm in ((0, "thirsty"), (1, "hungry")):
                q = int(V[:K, c].argmax()); m = U == q; r["best unit " + nm] = dict(to_needed=float(np.median((dw if c == 0 else df)[m])), to_other=float(np.median((df if c == 0 else dw)[m])), states=int(m.sum()))
            # 3. does the code hold place at all: nearest-neighbour read-out of the true cell (half the bodies teach, half are asked)
            Zf = Z.reshape(T, n, -1); a = Zf[:, :32].reshape(-1, Zf.shape[2])[::7]; b = Zf[:, 32:].reshape(-1, Zf.shape[2])[::29]
            pa = torch.as_tensor(P[:, :32].reshape(-1, 2)[::7], device=DEV).float(); pb = P[:, 32:].reshape(-1, 2)[::29]
            nn_ = torch.cat([torch.cdist(b[j:j + 2048], a).argmin(1) for j in range(0, len(b), 2048)]); err = np.sqrt(((pa[nn_].cpu().numpy() - pb) ** 2).sum(1))
            r["nearest_neighbour_place_error_cells"] = dict(median=float(np.median(err)), share_over_3=float((err > 3).mean()))
            rec[kind] = r
        xa = Xo[:, :32].reshape(-1, dx)[::7].to(DEV); xb = Xo[:, 32:].reshape(-1, dx)[::29].to(DEV); nn_ = torch.cat([torch.cdist(xb[j:j + 2048], xa).argmin(1) for j in range(0, len(xb), 2048)])
        err = np.sqrt(((P[:, :32].reshape(-1, 2)[::7][nn_.cpu().numpy()] - P[:, 32:].reshape(-1, 2)[::29]) ** 2).sum(1)); rec["raw sense nearest neighbour error"] = dict(median=float(np.median(err)), share_over_3=float((err > 3).mean()))
        print(lay, json.dumps(rec)[:1900], round(time.time() - t0), flush=True); res.append(rec); json.dump(dict(note="D27 follow-up (description)", rows=res), open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
