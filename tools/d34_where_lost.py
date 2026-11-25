"""D34 (atanor_os/P1_PATH_PROTOCOL.md, registered before running): where the going is lost in the mind's choosing.
On the saved L1 beings (frozen), the L1 fork test with: A the organ alone (gesture drawn from its own skill toward its own next place), B the mind as is,
and the share of beats on which the lean changes the mind's best gesture (how much it can weigh against the other pulls).
  PYTHONPATH=. python -m tools.d34_where_lost OUT.json 9101_0 ... [--dir l1/W1.0]"""
import argparse, json, math
import numpy as np, torch
from tools.need_fork import t_room, DEV, DEPTH
from tools.a1_channel import V2
from tools.a1_sf import paths_of
from tools.d33_play_reach import set_organ
from tools.l1_life import make
from atanor_os.emerge.room_body import RoomBody
from atanor_os.emerge.body_contract import Organs
from atanor_os.emerge.mind import Mind
from atanor_os.emerge.going import Going
HOPS = 1


def load(d, lay, rows, sd, n=64):
    B = make(rows, n, sd, V2["drain"]); O = Organs(B.c, n, dev=DEV, depth=DEPTH); Mi = Mind(O, B.c, n, dev=DEV, value_arm="relief")
    Mi.load(f"{d}/{lay}_mind.pt"); Mi.wo.on = True
    G = Going(O.dx, Mi.M, n, O.inside, DEV); st = torch.load(f"{d}/{lay}_going.pt", map_location=DEV, weights_only=False)
    G.skill.load_state_dict(st["skill"]); G.C, G.Cn, G.T, G.R, G.fitted = st["C"], st["Cn"], st["T"], st["R"], st["fitted"]
    G.mended = st.get("mended", False); G.Vn = st.get("Vn", G.Vn); G.born = st.get("born", False); G.used = st.get("used", G.KU); G.refresh(); Mi.going = G
    return Mi, O


@torch.no_grad()
def forks(Mi, O, rows, info, seed, P, arm, T=400):
    out = {}; n = Mi.n; G = Mi.going; flips = []
    for start in ("far", "stem"):
        torch.manual_seed(seed + 21); B = make(rows, n, seed, 0.0); h = n // 2
        if start == "stem":
            B.w.pos[:, 0], B.w.pos[:, 1], B.w.head[:] = info["x0"], info["ys"], -math.pi / 2
        else:
            at = {sp: cells[-1] for cells, sp in P.values()}
            for i in range(n):
                px, py, hh = at["f"] if i < h else at["w"]; B.w.pos[i, 0], B.w.pos[i, 1], B.w.head[i] = px, py, hh
        B.w.store[:h, 0], B.w.store[:h, 1] = 0.30, 0.95; B.w.store[h:, 0], B.w.store[h:, 1] = 0.95, 0.30
        every = torch.ones(n, dtype=torch.bool, device=DEV); O.reset(every); set_organ(B, O, info); Mi.begin(every); Mi.prev = None
        st0 = B.w.store.clone(); x = O(B.raw())
        for t in range(T):
            if arm.startswith("A"):
                lack, c = G.needs(x); u = G.unit(G.place(x)); v = u
                for _ in range(HOPS):
                    v = G.nxt[c, v]
                a = torch.multinomial(G.toward(x, G.C[v]).exp(), 1)[:, 0]
            else:
                G.W = 0.0; s0, _ = Mi.net.pull(x); G.W = 1.0; gl = G.lean(x)
                flips.append(float((s0.argmax(1) != (s0 + gl).argmax(1)).float().mean()))
                a = Mi.choose(x)
            B.step(a); x = O(B.raw())
        r = (B.w.store - st0).cpu().numpy(); got = np.r_[r[:h, 0], r[h:, 1]] > 0.05
        out[start] = [float(got[:h].mean()), float(got[h:].mean())]
    if flips:
        out["lean_flips_best"] = round(float(np.mean(flips)), 3)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("lays", nargs="+"); ap.add_argument("--dir", default="l1/W1.0"); ap.add_argument("--hops", type=int, nargs="*")
    a = ap.parse_args(); V2["on"] = True; RoomBody.default_cues = True; res = []
    for lay in a.lays:
        seed, mirror = map(int, lay.split("_")); assert not 7701 <= seed <= 7712; rows, info = t_room(seed, bool(mirror)); sd = seed * 10 + mirror; P = paths_of(rows, info)
        row = dict(layout=lay)
        arms = [f"A{k}" for k in a.hops] if a.hops else ["A", "B"]
        for arm in arms:
            global HOPS
            HOPS = int(arm[1:]) if arm[1:] else 1
            Mi, O = load(a.dir, lay, rows, sd); row[arm] = forks(Mi, O, rows, info, sd, P, arm); del Mi; torch.cuda.empty_cache()
        print(json.dumps(row), flush=True); res.append(row); json.dump(dict(protocol="atanor_os/P1_PATH_PROTOCOL.md (D34)", rows=res), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
