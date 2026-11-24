"""D33 (atanor_os/P1_PATH_PROTOCOL.md, registered before running): going to a remembered place, learned from own play with the innate place sense.
Babbling (stores not draining), then goal practice toward places it has been, each round relabelled by hindsight (no reward).
  PYTHONPATH=. python -m tools.d33_play_reach OUT.json 9101_0 ..."""
import json, math, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from tools.need_fork import t_room, place, DEV, DEPTH, K
from tools.a1_channel import V2, junction
from tools.a1_sf import paths_of
from tools import d25_grain as D25
from atanor_os.emerge.room_body import RoomBody
from atanor_os.emerge.body_contract import Organs, Channel, PathOrgan

NP = PathOrgan.N                                                      # 26: place code (24) + heading (2), last columns of the sense


def body(rows, n, seed, organs_state):
    B = RoomBody(n, rows, drain=0.0, seed=seed, device=DEV, regrow=V2["regrow"]); B.c.sensors[2] = Channel("vector", (3,), motion=True)
    O = Organs(B.c, n, dev=DEV, depth=DEPTH); O.load(organs_state); return B, O


def set_organ(B, O, info):
    """Bodies put down by the judge: the organ is given where they are relative to birth (as if they had walked there)."""
    hb = -math.pi / 2; d = B.w.pos - torch.tensor([info["x0"], info["ys"]], device=DEV); c, s = math.cos(hb), math.sin(hb)
    O.path.p = torch.stack([d[:, 0] * c + d[:, 1] * s, -d[:, 0] * s + d[:, 1] * c], 1); O.path.h = (B.w.head - hb + math.pi) % (2 * math.pi) - math.pi


class Skill(nn.Module):
    def __init__(self, dx, M):
        super().__init__(); self.mu = nn.Parameter(torch.zeros(dx), requires_grad=False); self.sd = nn.Parameter(torch.ones(dx), requires_grad=False)
        self.f = nn.Sequential(nn.Linear(dx + NP - 2, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, M))

    def forward(self, x, g):
        return self.f(torch.cat([((x - self.mu) / self.sd.clamp(min=0.05)).clamp(-5, 5), g], 1))   # bounded: near-constant senses must not blow up


def play(rows, info, seed, organs_state, skill, beats=3000, n=64, eps=0.1, pool=None):
    torch.manual_seed(seed); B, O = body(rows, n, seed, organs_state); place(B, info); O.reset(torch.ones(n, dtype=torch.bool, device=DEV))
    B.w.store[:] = 0.75; x = O(B.raw()); M = len(B.legs); X, A, R = [], [], []; g = None; left = torch.zeros(n, device=DEV)
    for t in range(beats):
        if skill is None:
            a = torch.randint(0, M, (n,), device=DEV)
        else:
            new = (left <= 0) | ((x[:, -NP:-2] - g).norm(dim=1) < 0.5 if g is not None else torch.ones(n, dtype=torch.bool, device=DEV))
            if g is None:
                g = pool[torch.randint(0, len(pool), (n,), device=DEV)]
            pick = pool[torch.randint(0, len(pool), (n,), device=DEV)]; g = torch.where(new[:, None], pick, g); left = torch.where(new, torch.full_like(left, 200), left - 1)
            with torch.no_grad():
                a = torch.multinomial(torch.softmax(skill(x, g), 1), 1)[:, 0]          # followed by sampling its own distribution (GCSL)
            a = torch.where(torch.rand(n, device=DEV) < eps, torch.randint(0, M, (n,), device=DEV), a)
        s0 = B.w.store.clone(); X.append(x); A.append(a); B.step(a); R.append((B.w.store - s0 > 0.01)); B.w.store[:] = 0.75; x = O(B.raw())
    return torch.stack(X), torch.stack(A), torch.stack(R)


def train(skill, plays, inside, updates=6000, batch=1024, far=32):
    opt = torch.optim.Adam(skill.f.parameters(), 3e-4); le = 0.0
    for u in range(updates):
        X, A, _ = plays[np.random.randint(len(plays))]; T, n, dx = X.shape
        t = torch.randint(0, T - 1, (batch,), device=DEV); i = torch.randint(0, n, (batch,), device=DEV); t2 = (t + torch.randint(1, far + 1, (batch,), device=DEV)).clamp(max=T - 1)
        x = X[t, i]; x[:, inside] = 0; loss = F.cross_entropy(skill(x, X[t2, i][:, -NP:-2]), A[t, i]); opt.zero_grad(); loss.backward(); opt.step()
        if u >= updates - 200:
            le += float(loss.detach()) / 200
    return le


@torch.no_grad()
def test(skill, rows, info, seed, organs_state, P, start, goals, inside, T=400, eps=0.1):
    """goals: [2, k, 24] relief place codes per inside (0 water, 1 food) or None. Thirsty bodies first K, hungry next K."""
    torch.manual_seed(seed + 13); n = 2 * K; B, O = body(rows, n, seed, organs_state); M = len(B.legs)
    if start == "junction":
        junction(B, info)
    elif start == "stem":
        B.w.pos[:, 0], B.w.pos[:, 1], B.w.head[:] = info["x0"], info["ys"], -math.pi / 2
    else:
        at = {sp: cells[-1] for cells, sp in P.values()}
        for i in range(n):
            px, py, h = at["f"] if i < K else at["w"]; B.w.pos[i, 0], B.w.pos[i, 1], B.w.head[i] = px, py, h
    O.reset(torch.ones(n, dtype=torch.bool, device=DEV)); set_organ(B, O, info)
    B.w.store[:K, 0], B.w.store[:K, 1] = 0.30, 0.95; B.w.store[K:, 0], B.w.store[K:, 1] = 0.95, 0.30; st0 = B.w.store.clone(); x = O(B.raw())
    need = [0] * K + [1] * K
    g = torch.zeros(n, NP - 2, device=DEV) if goals is None else torch.stack([goals[need[i]][torch.randint(0, len(goals[need[i]]), (1,)).item()] for i in range(n)])
    for t in range(T):
        xx = x.clone(); xx[:, inside] = 0; a = torch.multinomial(torch.softmax(skill(xx, g), 1), 1)[:, 0]; a = torch.where(torch.rand(n, device=DEV) < eps, torch.randint(0, M, (n,), device=DEV), a)
        B.step(a); x = O(B.raw())
    r = (B.w.store - st0).cpu().numpy(); got = np.r_[r[:K, 0], r[K:, 1]] > 0.05
    return [float(got[:K].mean()), float(got[K:].mean())]


def main():
    V2["on"] = True; RoomBody.default_cues = True; outp = sys.argv[1]; res = []; t0 = time.time()
    for lay in sys.argv[2:]:
        seed, mirror = map(int, lay.split("_")); assert not 7701 <= seed <= 7712; rows, info = t_room(seed, bool(mirror)); sd = seed * 10 + mirror
        st = torch.load(f"{D25.ROOT}/{lay}/mind.pt", map_location=DEV, weights_only=False)["organs"]; P = paths_of(rows, info)
        plays = [play(rows, info, sd, st, None)]; X0 = plays[0][0]; dx = X0.shape[2]; inside = list(range(3))
        skill = Skill(dx, 90).to(DEV); Xf = X0.reshape(-1, dx); skill.mu.data = Xf.mean(0); skill.sd.data = Xf.std(0) + 1e-3
        rec = dict(layout=lay, rounds=[])
        for rnd in range(6):
            with torch.enable_grad():
                le = train(skill, plays, inside)
            allX = torch.cat([p[0] for p in plays]); allR = torch.cat([p[2] for p in plays])
            goals = [allX[:, :, -NP:-2][allR[..., c]] for c in (0, 1)]                       # where it was when water / food rose, in its own play
            r = dict(round=rnd, loss=round(le, 3), reliefs=[len(q) for q in goals], reach={})
            if min(len(q) for q in goals) > 0:
                for nm, gg in (("right", goals), ("wrong", goals[::-1]), ("none", None)):
                    r["reach"][nm] = {s: test(skill, rows, info, sd, st, P, s, gg, inside) for s in ("junction", "far", "stem")}
            pos_seen = allX[:, :, -NP:-2].reshape(-1, NP - 2); cover = None
            rec["rounds"].append(r); print(lay, "round", rnd, "loss", r["loss"], "reliefs", r["reliefs"], "|", r["reach"], round(time.time() - t0), flush=True)
            if rnd < 5:
                plays.append(play(rows, info, sd + 100 * (rnd + 1), st, skill, pool=pos_seen[torch.randperm(len(pos_seen), device=DEV)[:20000]]))
        res.append(rec); json.dump(dict(protocol="atanor_os/P1_PATH_PROTOCOL.md (D33)", rows=res), open(outp, "w"), indent=1); del plays; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
