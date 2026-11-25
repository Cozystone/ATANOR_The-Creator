"""D34b (atanor_os/P1_PATH_PROTOCOL.md, registered before running): which part of the going organ, as grown in the L1 life, fails.
On the saved L1 beings (frozen, W = 1 life), same fork starts and seeds as D34:
 (a) the map: from each body's start unit, follow its own next-unit table toward the needed inside (up to 64 hops); where does the chain end
     (distance in cells from the needed spring, by the true place each unit centre stands for)?
 (b) the skill with the judge's waypoint: each beat the goal is the place code of the cell 4 cells further along the true route to the needed spring
     (a judge-side probe; tests only whether the skill grown in this life carries a body along short known hops).
  PYTHONPATH=. python -m tools.d34b_organ_parts OUT.json 9101_0 ..."""
import argparse, json, math
import numpy as np, torch
from tools.need_fork import t_room, DEV
from tools.a1_channel import V2
from tools.a1_sf import paths_of
from tools.d33_play_reach import set_organ
from tools.d34_where_lost import load
from tools.l1_life import make
from atanor_os.emerge.room_body import RoomBody
from atanor_os.emerge.body_contract import PathOrgan
from atanor_os.emerge.going import NP


def code_of(xy, info, dev):
    """The innate code (place part) the path organ gives at these true places (birth frame as set_organ uses)."""
    P = PathOrgan(len(xy), dev); hb = -math.pi / 2; d = torch.as_tensor(xy, dtype=torch.float32, device=dev) - torch.tensor([info["x0"], info["ys"]], device=dev)
    c, s = math.cos(hb), math.sin(hb); P.p = torch.stack([d[:, 0] * c + d[:, 1] * s, -d[:, 0] * s + d[:, 1] * c], 1); P.h[:] = 0
    return P(torch.zeros(len(xy), 3, device=dev))[:, :NP - 2]


def routes(rows, info, P):
    by = {sp: cells for cells, sp in P.values()}; stem = [(info["x0"], y + 0.5) for y in range(int(info["ys"]) - 1, 1, -1)]
    xy = lambda cs: [(a, b) for a, b, _ in cs]
    return {("far", "w"): xy(by["f"][::-1]) + xy(by["w"]), ("far", "f"): xy(by["w"][::-1]) + xy(by["f"]),
            ("stem", "w"): stem + xy(by["w"]), ("stem", "f"): stem + xy(by["f"])}


@torch.no_grad()
def run(Mi, O, rows, info, seed, P, T=400):
    G = Mi.going; n = Mi.n; h = n // 2; R = routes(rows, info, P); out = {}
    g = np.array([list(r) for r in rows]); cells = [(x + 0.5, y + 0.5) for y in range(g.shape[0]) for x in range(g.shape[1]) if g[y, x] not in "#"]
    cc = code_of(cells, info, DEV); unit_cell = torch.cdist(G.C, cc).argmin(1).cpu().numpy(); cells = np.array(cells)
    spring = {sp: np.array(cs[-1][:2]) for cs, sp in P.values()}
    for start in ("far", "stem"):
        torch.manual_seed(seed + 21); B = make(rows, n, seed, 0.0)
        if start == "stem":
            B.w.pos[:, 0], B.w.pos[:, 1], B.w.head[:] = info["x0"], info["ys"], -math.pi / 2
        else:
            at = {sp: c[-1] for c, sp in P.values()}
            for i in range(n):
                px, py, hh = at["f"] if i < h else at["w"]; B.w.pos[i, 0], B.w.pos[i, 1], B.w.head[i] = px, py, hh
        B.w.store[:h, 0], B.w.store[:h, 1] = 0.30, 0.95; B.w.store[h:, 0], B.w.store[h:, 1] = 0.95, 0.30
        every = torch.ones(n, dtype=torch.bool, device=DEV); O.reset(every); set_organ(B, O, info); Mi.begin(every); x = O(B.raw())
        lack, c = G.needs(x); u = G.unit(G.place(x)).tolist(); ends = []
        for i in range(n):                                                    # (a) the map's own chain
            v = u[i]
            for _ in range(64):
                w = int(G.nxt[c[i], v])
                if w == v:
                    break
                v = w
            need = "w" if i < h else "f"; ends.append(float(np.linalg.norm(cells[unit_cell[v]] - spring[need])))
        out[start + "_map_end_cells_from_needed_spring"] = [round(float(np.mean(ends[:h])), 2), round(float(np.mean(ends[h:])), 2)]
        out[start + "_map_chain_ends_within_1.5"] = [round(float(np.mean(np.array(ends[:h]) <= 1.5)), 2), round(float(np.mean(np.array(ends[h:]) <= 1.5)), 2)]
        rw = [np.array(R[(start, "w")]), np.array(R[(start, "f")])]; rc = [code_of(r.tolist(), info, DEV) for r in rw]
        st0 = B.w.store.clone()
        for t in range(T):                                                    # (b) the skill along the judge's waypoints
            pos = B.w.pos.cpu().numpy(); goals = torch.zeros(n, NP - 2, device=DEV)
            for half, (r, co) in enumerate(zip(rw, rc)):
                sl = slice(0, h) if half == 0 else slice(h, n); k = np.linalg.norm(pos[sl, None, :] - r[None], axis=2).argmin(1)
                goals[sl] = co[torch.as_tensor(np.minimum(k + 4, len(r) - 1), device=DEV)]
            a = torch.multinomial(G.toward(x, goals).exp(), 1)[:, 0]; B.step(a); x = O(B.raw())
        r = (B.w.store - st0).cpu().numpy(); got = np.r_[r[:h, 0], r[h:, 1]] > 0.05
        out[start + "_skill_waypoints"] = [float(got[:h].mean()), float(got[h:].mean())]
    relief = (G.R / G.Cn.clamp(min=1)[:, None]).cpu().numpy(); top = {nm: cells[unit_cell[relief[:, k].argmax()]].tolist() for k, nm in ((0, "water"), (1, "food"))}
    out["best_relief_unit_place"] = top; out["springs"] = {k: v.tolist() for k, v in spring.items()}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("lays", nargs="+"); ap.add_argument("--dir", default="l1/W1.0")
    a = ap.parse_args(); V2["on"] = True; RoomBody.default_cues = True; res = []
    for lay in a.lays:
        seed, mirror = map(int, lay.split("_")); assert not 7701 <= seed <= 7712; rows, info = t_room(seed, bool(mirror)); sd = seed * 10 + mirror
        Mi, O = load(a.dir, lay, rows, sd); row = dict(layout=lay, **run(Mi, O, rows, info, sd, paths_of(rows, info))); del Mi; torch.cuda.empty_cache()
        print(json.dumps(row), flush=True); res.append(row); json.dump(dict(protocol="atanor_os/P1_PATH_PROTOCOL.md (D34b)", rows=res), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
