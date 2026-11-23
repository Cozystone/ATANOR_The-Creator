"""LOKI v4 effect readout through ATANOR's room body loop (protocol: atanor_os/LOKI_V4_ROOM_PROTOCOL.md, frozen before measuring).
  PYTHONPATH=. python -m tools.loki.room_effect train  OUTDIR
  PYTHONPATH=. python -m tools.loki.room_effect confirm OUTDIR  -> OUTDIR/room_confirm.json
Every sensation the organs see comes from RoomBody.raw() -> Organs (feeling x) -> LokiBridge.features. The judge alone names commands and forks the world."""
import copy, json, os, sys, time
import numpy as np, torch
from atanor_os.emerge.room_body import RoomBody
from atanor_os.emerge.room_vary import make_room
from atanor_os.emerge.body_contract import Organs
from atanor_os.emerge.inner import Inner, train_inner
from atanor_os.emerge.loki_bridge import LokiBridge

DEV = "cuda" if torch.cuda.is_available() else "cpu"
G = 96


def make(seed, n):
    B = RoomBody(n, make_room(seed), drain=0.0, seed=seed, device=DEV)
    return B, Organs(B.c, n, dev=DEV, depth="")


def cmds(B):
    ix = {tuple(c): i for i, c in enumerate(B.combos)}
    return dict(stand=ix[(0, 0, 0)], fwd=ix[(1, 0, 0)], left=ix[(0, 1, 0)], right=ix[(0, 2, 0)], use=ix[(0, 0, 2)])


def feats(LB, B, O):
    raw = B.raw(); x = O(raw)
    return np.stack([LB.features(raw, x, body=j)[0] for j in range(B.n)])


def step(B, a):
    B.step(torch.as_tensor(np.asarray(a), device=DEV))


def switching(rng, n, T, C):
    five = np.array([C["fwd"], C["left"], C["right"], C["use"], C["stand"]]); act = np.full(n, C["stand"]); out = []
    for _ in range(T):
        act = np.where(rng.random(n) < .42, rng.choice(five, size=n), act); out.append(act.copy())
    return np.array(out)                                                   # [T, n]


def feed(LB, frames, acts):
    LB.mem.cut(); LB.mem.observe(frames[0])
    for f, a in zip(frames[1:], acts):
        LB.mem.observe(f, LB.motor(int(a)))


def arms(B, O, LB, plans):
    """plans [2, n, 20] -> endpoint features [2, n, width] and the arm streams [2, 21, n, width] (judge forks)."""
    ends, streams = [], []
    for arm in range(2):
        Bc, Oc = copy.deepcopy(B), copy.deepcopy(O); fr = [feats(LB, Bc, Oc)]
        for t in range(plans.shape[2]):
            step(Bc, plans[arm, :, t]); fr.append(feats(LB, Bc, Oc))
        streams.append(np.array(fr)); ends.append(fr[-1])
    return np.array(ends), np.array(streams)


def make_plans(rng, n, counts, flips, C):
    base = rng.choice([C["left"], C["right"], C["stand"]], size=(n, 20)); P = np.stack([base.copy(), base.copy()])
    for j in range(n):
        s = rng.choice(20, int(counts[j]), replace=False)
        P[int(flips[j]), j, s] = C["fwd"]; P[1 - int(flips[j]), j, s] = C["stand"]
    return P


def train(out):
    os.makedirs(out, exist_ok=True); torch.manual_seed(803)
    B0, O0 = make(1, 1); C = cmds(B0)
    full = LokiBridge(B0.c, O0); noeff = LokiBridge(B0.c, O0)
    X, A = [], []; t0 = time.time()
    for seed in range(1, 13):                                              # lit streams
        B, O = make(seed, 1); rng = np.random.default_rng(seed); acts = switching(rng, 1, 208, C)
        fr = [feats(full, B, O)[0]]
        for t in range(208):
            step(B, acts[t]); fr.append(feats(full, B, O)[0])
        fr = np.array(fr)
        for m in (full, noeff):
            feed(m, fr, acts[:, 0])
        X += [fr[:81, -G:], fr[81:162, -G:]]; A += [acts[:80, 0], acts[81:161, 0]]
        print("stream", seed, round(time.time() - t0), flush=True)
    npairs = 0
    for seed in range(1, 13):                                              # paired forks (simulator only)
        n = 24; B, O = make(seed, n); rng = np.random.default_rng(seed + 800); acts = []
        fr = [feats(full, B, O)]
        act = np.full(n, C["stand"]); five = np.array([C["fwd"], C["left"], C["right"], C["use"], C["stand"]])
        for t in range(80):
            act = np.where(rng.random(n) < .42, rng.choice(five, size=n), act); acts.append(act.copy())
            step(B, act); fr.append(feats(full, B, O))
            if t in (39, 79):
                P = make_plans(rng, n, rng.integers(2, 19, size=n), rng.integers(2, size=n), C)
                ends, streams = arms(B, O, full, P)
                F, Aa = np.array(fr), np.array(acts)
                for j in range(n):
                    for m in (full, noeff):
                        feed(m, F[:, j], Aa[:, j])
                    full.learn_effect(P[0, j].tolist(), P[1, j].tolist(), ends[0, j], ends[1, j]); npairs += 1
                    for arm in range(2):
                        for m in (full, noeff):
                            feed(m, streams[arm][:, j], P[arm, j])
        F, Aa = np.array(fr), np.array(acts)
        for j in range(n):
            X.append(F[:, j, -G:]); A.append(Aa[:, j])
        print("pairs room", seed, "total", npairs, round(time.time() - t0), flush=True)
    full.mem.save(os.path.join(out, "loki_full.npz")); noeff.mem.save(os.path.join(out, "loki_noeff.npz"))
    inner = Inner(G, len(B0.combos), h=128)
    train_inner(inner, torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(A), dtype=torch.long), iters=2000, window=64, seed=803)
    torch.save(inner.state_dict(), os.path.join(out, "inner.pt"))
    json.dump(dict(pairs=npairs, matrix_bytes=int(full.mem.state_bytes), width=full.width, seconds=round(time.time() - t0)), open(os.path.join(out, "train.json"), "w"))
    print("trained", npairs, "pairs", full.mem.state_bytes, "bytes", flush=True)


@torch.no_grad()
def inner_diff(net, h, frame, P):
    h = h.repeat(2, 1); x = torch.tensor(frame[None], dtype=torch.float32).repeat(2, 1)
    for i in range(P.shape[1]):
        h, y = net.step(h, x if i == 0 else torch.zeros_like(x), torch.tensor(P[:, i]), torch.full((2,), float(i > 0)))
    y = y.numpy().clip(0, 1)
    return y[0] - y[1]


def confirm(out, seeds=range(6101, 6113)):
    B0, O0 = make(1, 1); C = cmds(B0)
    base = {k: LokiBridge(B0.c, O0) for k in ("full", "noeff")}
    for k in base:
        base[k].mem = base[k].L.LokiMemory.load(os.path.join(out, f"loki_{k}.npz"))
    inner0 = Inner(G, len(B0.combos), h=128); inner0.load_state_dict(torch.load(os.path.join(out, "inner.pt")))
    rows = []
    for seed in seeds:
        B, O = make(seed, 1); rng = np.random.default_rng(seed + 17000); acts = switching(np.random.default_rng(seed), 1, 248, C)[:, 0]
        M = {k: copy.deepcopy(v) for k, v in base.items()}; net = copy.deepcopy(inner0)
        fr = [feats(M["full"], B, O)[0]]
        for m in M.values():
            m.mem.cut(); m.mem.observe(fr[0])
        pairs = []; h = None
        for t, a in enumerate(acts):
            step(B, [a]); f = feats(M["full"], B, O)[0]; fr.append(f)
            for m in M.values():
                m.mem.observe(f, m.motor(int(a)))
            if t + 1 == 128:                                               # Inner adapts on the room's own lit past only, then re-reads it
                F = torch.tensor(np.array(fr)[None, :, -G:], dtype=torch.float32); Aa = torch.tensor(acts[None, :128], dtype=torch.long)
                train_inner(net, F, Aa, iters=64, window=64, seed=seed)
                h = net.zero(1)
                with torch.no_grad():
                    for q in range(128):
                        h, _ = net.step(h, F[:, q], Aa[:, q])
            elif h is not None:
                with torch.no_grad():
                    h, _ = net.step(h, torch.tensor(fr[-2][None, -G:], dtype=torch.float32), torch.tensor([int(a)]))
            if t + 1 not in (128, 168, 208, 248):
                continue
            i = len(pairs); P = make_plans(rng, 1, [4 * (i + 1)], [i % 2], C)[:, 0]
            est, paths = {}, {}
            for k, m in M.items():
                before = m.mem.matrix.copy()
                d, paths[k] = m.effect(P[0].tolist(), P[1].tolist())
                assert np.array_equal(before, m.mem.matrix)                 # forecasting is read-only
                est[k] = d
            est["inner"] = np.concatenate([np.zeros(M["full"].nb, np.float32), inner_diff(net, h, f[-G:], P)])
            est["zero"] = np.zeros_like(est["full"])
            ends, _ = arms(B, O, M["full"], P[:, None, :])                   # only now the judge forks and lives both futures
            truth = ends[0, 0] - ends[1, 0]
            nbed, nb = M["full"].nbed, M["full"].nb
            def vec(d):
                q = d[:nb].copy(); q[nbed:] *= 2; return q                 # [0,1] differences -> feeling units
            pairs.append(dict(beat=t + 1, substitutions=4 * (i + 1), identical=bool(not np.any(truth[-G:])), path=paths,
                              grid_mse={k: float(np.mean((e[-G:] - truth[-G:]) ** 2)) for k, e in est.items()},
                              vector_mse={k: float(np.mean((vec(e) - vec(truth)) ** 2)) for k, e in est.items() if k != "inner"},
                              predicted_energy=float(np.mean(est["full"][-G:] ** 2)), true_energy=float(np.mean(truth[-G:] ** 2))))
        mean = {k: float(np.mean([p["grid_mse"][k] for p in pairs])) for k in pairs[0]["grid_mse"]}
        vmean = {k: float(np.mean([p["vector_mse"][k] for p in pairs])) for k in pairs[0]["vector_mse"]}
        rows.append(dict(seed=seed, pairs=pairs, grid=mean, vector=vmean))
        print(seed, {k: round(v, 6) for k, v in mean.items()}, flush=True)
    avg = {k: float(np.mean([r["grid"][k] for r in rows])) for k in rows[0]["grid"]}
    gates = {}
    for k in ("zero", "inner", "noeff"):
        wins = sum(r["grid"]["full"] < r["grid"][k] for r in rows); red = 1 - avg["full"] / avg[k]
        gates[k] = dict(wins=int(wins), reduction=round(red, 4), passed=bool(wins >= 10 and red >= .1), wins_above_10=bool(wins > 10))
    allp = [p for r in rows for p in r["pairs"]]
    split = {lab: {k: float(np.mean([p["grid_mse"][k] for p in allp if p["identical"] == ident])) for k in allp[0]["grid_mse"]}
             for lab, ident in (("nonzero", False), ("identical", True)) if any(p["identical"] == ident for p in allp)}
    res = dict(protocol="atanor_os/LOKI_V4_ROOM_PROTOCOL.md", rooms=list(seeds), mean_grid=avg, gates=gates, passed=all(g["passed"] for g in gates.values()),
               mean_vector={k: float(np.mean([r["vector"][k] for r in rows])) for k in rows[0]["vector"]},
               pairs_nonzero=sum(not p["identical"] for p in allp), pairs_identical=sum(p["identical"] for p in allp), by_identity=split, rows=rows)
    json.dump(res, open(os.path.join(out, "room_confirm.json"), "w"), indent=1)
    print(json.dumps(dict(mean_grid=avg, gates=gates, passed=res["passed"]), indent=1))


if __name__ == "__main__":
    torch.set_num_threads(4)
    {"train": train, "confirm": confirm}[sys.argv[1]](sys.argv[2])
