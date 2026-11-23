"""Need fork test (protocol atanor_os/NEED_FORK_PROTOCOL.md, frozen before running).
  PYTHONPATH=. python -m tools.need_fork OUT.json [--seeds 7101-7112]
The judge builds T rooms, knows which arm holds which spring, places bodies and sets their stores. The being only gets its contract senses."""
import argparse, copy, json, math, os, tempfile, time
import numpy as np, torch
from atanor_os.emerge.room_body import RoomBody
from atanor_os.emerge.body_contract import Organs
from atanor_os.emerge.mind import Mind
from atanor_os.emerge import room3d_gpu as G

BEING = "being/atanor.pt"
DEPTH = "mc/mc_depth.pt"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
K = 16
SRC = ["strong pull", "value", "uniform random", "hold", "try"]


def t_room(seed, mirror=False):
    r = np.random.default_rng(seed); Ls, La, Ld = int(r.integers(3, 7)), int(r.integers(3, 7)), int(r.integers(2, 5)); water_left = bool(r.integers(2))
    cx = La + 2; W = 2 * La + 6; y0 = 1; H = max(y0 + 3 + Ld, y0 + 2 + Ls) + 1
    g = np.full((H, W), "#", dtype="<U1")
    g[y0:y0 + 2, cx - La - 1:cx + La + 3] = "."                                  # crossbar
    g[y0 + 2:y0 + 2 + Ls, cx:cx + 2] = "."                                       # stem
    for c0 in (cx - La - 1, cx + La + 1):                                        # down legs
        g[y0 + 2:y0 + 2 + Ld, c0:c0 + 2] = "."
    yl = y0 + 2 + Ld; g[yl, cx - La - 1] = "w" if water_left else "f"; g[yl, cx - La] = "."
    g[yl, cx + La + 2] = "f" if water_left else "w"; g[yl, cx + La + 1] = "."
    sy = y0 + 1 + Ls; g[sy, cx] = "s"
    if mirror:
        g = g[:, ::-1]; water_left = not water_left
    S = max(g.shape); sq = np.full((S, S), "#", dtype="<U1"); sq[:g.shape[0], :g.shape[1]] = g    # the world wants a square map: pad with wall
    rows = ["".join(row) for row in sq]
    return rows, dict(x0=cx + 1.0, ys=sy + 0.5, water_left=water_left, Ls=Ls, La=La, Ld=Ld)


def place(B, info, which=None):
    w = B.w; idx = torch.arange(B.n, device=DEV) if which is None else which
    w.pos[idx, 0] = info["x0"]; w.pos[idx, 1] = info["ys"]; w.head[idx] = -math.pi / 2


def phase_a(rows, info, seed, cap=3000, drain=1 / 120.0, rebirth=False, trail=False):
    torch.manual_seed(seed)
    B = RoomBody(1, rows, drain=drain, seed=seed, device=DEV); O = Organs(B.c, 1, dev=DEV, depth=DEPTH)
    Mi = Mind(O, B.c, 1, dev=DEV)
    if trail:
        Mi.use_trail()
    Mi.load(BEING)
    if Mi.wo is not None:
        Mi.wo.on = True
    B.w.store[:] = 0.75; place(B, info)
    one = torch.ones(1, dtype=torch.bool, device=DEV); O.reset(one); Mi.begin(one)
    x = O(B.raw()); got = np.zeros(2, int); t = 0; deaths = 0
    for t in range(cap):
        act = Mi.choose(x); alive0 = ~B.dead.clone(); f = B.step(act); got += f[0].cpu().numpy().astype(int)
        died = alive0 & B.dead; x2 = O(B.raw()); Mi.after(x, act, x2, died, B.dead & ~died); x = x2
        if B.dead.all():
            if not rebirth:
                break
            deaths += 1; B.w.dead[:] = False; B.w.store[:] = 0.75; place(B, info); O.reset(one); Mi.begin(one); x = O(B.raw())
        if (got > 0).all():
            break
    return Mi, dict(beats=t + 1, fills_water=int(got[0]), fills_food=int(got[1]), ok=bool((got > 0).all()), died=bool(B.dead.all()), rebirths=deaths,
                    relief=int(Mi.net.rel_n), harm=int(Mi.net.harm_n), trail_points=int(Mi.net.trl_n) if trail else 0)


def phase_b(ckpt, rows, info, seed, cond, T=400, trail=False):
    torch.manual_seed(seed + 1)
    n = 2 * K; B = RoomBody(n, rows, drain=0.0, seed=seed, device=DEV); O = Organs(B.c, n, dev=DEV, depth=DEPTH)
    Mi = Mind(O, B.c, n, dev=DEV)
    if trail:
        Mi.use_trail()
    Mi.load(ckpt)
    if cond == "no-episodic":
        for k in ("rel_n", "fnd_n", "harm_n", "slow_n") + (("trl_n",) if trail else ()):
            getattr(Mi.net, k).zero_()
    if cond == "no-trail":
        Mi.net.trl_n.zero_()
    if cond == "no-value":
        Mi.wo = None
    elif Mi.wo is not None:
        Mi.wo.on = True
    B.w.store[:K, 0], B.w.store[:K, 1] = 0.30, 0.95                             # bodies 0..K-1 lack water (store order: water, food)
    B.w.store[K:, 0], B.w.store[K:, 1] = 0.95, 0.30
    place(B, info); every = torch.ones(n, dtype=torch.bool, device=DEV); O.reset(every); Mi.begin(every)
    choice = np.zeros(n, int); src = np.zeros(5, int); x = O(B.raw()); sims = []
    for t in range(T):
        act = Mi.choose(x); live = choice == 0
        s = Mi.src.cpu().numpy(); src += np.bincount(s[live], minlength=5)
        if trail and live.any() and hasattr(Mi.net, "trail_sim"):
            sims.append(float(Mi.net.trail_sim.cpu().numpy()[live].mean()))
        B.step(act); x2 = O(B.raw())
        Mi.prev = (x, act); Mi.cut = torch.zeros(n, dtype=torch.bool, device=DEV)  # what after() does for choosing, without learning
        px = B.w.pos[:, 0].cpu().numpy(); py = B.w.pos[:, 1].cpu().numpy()
        upper = py < 3.0
        choice = np.where((choice == 0) & upper & (px < info["x0"] - 2.5), -1, choice)
        choice = np.where((choice == 0) & upper & (px > info["x0"] + 2.5), 1, choice)
        x = x2
        if (choice != 0).all():
            break
    need_left = np.array([info["water_left"]] * K + [not info["water_left"]] * K)
    chose = choice != 0; match = (choice == -1) == need_left
    m = float(match[chose].mean()) if chose.any() else 0.5
    return dict(choice_rate=float(chose.mean()), match=m, left_share=float((choice[chose] == -1).mean()) if chose.any() else None,
                match_water_lack=float(match[:K][chose[:K]].mean()) if chose[:K].any() else None,
                match_food_lack=float(match[K:][chose[K:]].mean()) if chose[K:].any() else None,
                decided_by=dict(zip(SRC, src.tolist())), beats=t + 1, trail_sim=float(np.mean(sims)) if sims else None)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--seeds", default="7101-7112")
    ap.add_argument("--drain", type=float, default=1 / 120.0); ap.add_argument("--rebirth", action="store_true"); ap.add_argument("--trail", action="store_true"); a = ap.parse_args()
    s0, s1 = map(int, a.seeds.split("-")); rows_out = []; t0 = time.time()
    conds = ["full", "no-trail", "no-episodic"] if a.trail else ["full", "no-episodic", "no-value"]
    for seed in range(s0, s1 + 1):
        for mirror in (False, True):
            rows, info = t_room(seed, mirror)
            Mi, pa = phase_a(rows, info, seed * 10 + mirror, drain=a.drain, rebirth=a.rebirth, trail=a.trail)
            rec = dict(seed=seed, mirror=mirror, layout=rows, info=info, phase_a=pa)
            if pa["ok"]:
                with tempfile.TemporaryDirectory() as d:
                    ck = os.path.join(d, "ck.pt"); Mi.save(ck)
                    rec["phase_b"] = {c: phase_b(ck, rows, info, seed * 10 + mirror, c, trail=a.trail) for c in conds}
            else:
                rec["phase_b"] = {c: dict(match=0.0, choice_rate=0.0, failed_phase_a=True) for c in conds}
            rows_out.append(rec)
            print(seed, "mirror" if mirror else "drawn", pa, {c: round(v["match"], 2) for c, v in rec["phase_b"].items()},
                  {c: v.get("choice_rate") for c, v in rec["phase_b"].items()}, round(time.time() - t0), flush=True)
    mean = {c: float(np.mean([r["phase_b"][c]["match"] for r in rows_out])) for c in conds}
    pairs = {}
    for r in rows_out:
        pairs.setdefault(r["seed"], []).append(r["phase_b"]["full"]["match"])
    both = sum(all(v > 0.5 for v in vs) for vs in pairs.values())
    gate = dict(mean_full=mean["full"], both_mirrors_over_half=both, over_no_episodic=mean["full"] - mean["no-episodic"],
                passed=bool(mean["full"] >= 0.70 and both >= 9 and mean["full"] - mean["no-episodic"] >= 0.15))
    res = dict(protocol="atanor_os/NEED_FORK_PROTOCOL.md", mean_match=mean, gate=gate, rows=rows_out)
    json.dump(res, open(a.out, "w"), indent=1); print(json.dumps(dict(mean_match=mean, gate=gate), indent=1))


if __name__ == "__main__":
    main()
