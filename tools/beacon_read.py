"""BC1 (atanor_os/WORLD_A_PROTOCOL.md): how much of the picture says "a spring is here", old renderer against the corrected one. No being.
  PYTHONPATH=. python -m tools.beacon_read OUT.json [--lay 9104_0]"""
import argparse, json, math, os
import numpy as np, torch
from tools.need_fork import t_room, DEV
from tools.l1_life import make
from tools.a1_channel import V2
from atanor_os.emerge.room_body import RoomBody
from atanor_os.emerge import room3d_gpu as G


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--lay", default="9104_0"); ap.add_argument("--eye", type=float, default=1.1); a = ap.parse_args()
    RoomBody.default_eye = a.eye; RoomBody.default_cues = True; seed, mirror = map(int, a.lay.split("_")); rows, info = t_room(seed, bool(mirror))
    g = np.array([list(r) for r in rows]); cells = [(x, y) for x in range(g.shape[0]) for y in range(g.shape[1])]
    B0 = make(rows, 1, seed, V2["drain"]); fl = [(x, y) for (x, y) in cells if int(B0.w.map[0, x, y]) == G.FLOOR]; n = len(fl)
    B = make(rows, n, seed, V2["drain"]); w = B.w; pos = torch.tensor(fl, device=DEV).float() + 0.5; w.pos[:] = pos
    out = dict(layout=a.lay, eye=a.eye, floor_cells=n, kinds={})
    for ci, name in ((0, "water"), (1, "food")):
        sp = w.src[:, ci].float() + 0.5; v = sp - pos; d = v.norm(dim=1); to = torch.atan2(v[:, 1], v[:, 0]); res = {}
        for off in (0, 45):
            w.head[:] = to + math.radians(off)
            os.environ["ATANOR_RENDER_FIX"] = "0"; old = B.render().clone(); os.environ["ATANOR_RENDER_FIX"] = "1"; new = B.render().clone()
            m0 = w.map.clone(); sx = w.src[:, ci].long(); w.map[torch.arange(n, device=DEV), sx[:, 0], sx[:, 1]] = G.FLOOR
            os.environ["ATANOR_RENDER_FIX"] = "0"; none_old = B.render().clone(); os.environ["ATANOR_RENDER_FIX"] = "1"; none_new = B.render().clone(); w.map[:] = m0
            H = old.shape[1]; up = torch.arange(H, device=DEV)[None, :, None] < H // 2
            eo = (old - none_old).abs().amax(3); en = (new - none_new).abs().amax(3)       # how strongly each pixel says "a spring is here"
            do = eo > 0.02; dn = en > 0.02; so = eo > 0.25; sn = en > 0.25
            rows_ = []
            for lo, hi in ((0, 2), (2, 4), (4, 6), (6, 9), (9, 99)):
                m = (d >= lo) & (d < hi)
                if not bool(m.any()):
                    continue
                f = lambda t_: round(float(t_.float().mean()), 1)
                rows_.append(dict(dist=f"{lo}-{hi}", views=int(m.sum()), old_px=f(do[m].sum((1, 2))), new_px=f(dn[m].sum((1, 2))), old_strong_px=f(so[m].sum((1, 2))), new_strong_px=f(sn[m].sum((1, 2))), old_strong_above=f((so & up)[m].sum((1, 2))), new_strong_above=f((sn & up)[m].sum((1, 2))), old_energy=f(eo[m].sum((1, 2))), new_energy=f(en[m].sum((1, 2))),
                                  old_strong_seen_share=round(float((so[m].sum((1, 2)) >= 6).float().mean()), 3), new_strong_seen_share=round(float((sn[m].sum((1, 2)) >= 6).float().mean()), 3), old_px_above_horizon=f((do & up)[m].sum((1, 2))), new_px_above_horizon=f((dn & up)[m].sum((1, 2))),
                                  old_seen_share=round(float((do[m].sum((1, 2)) >= 6).float().mean()), 3), new_seen_share=round(float((dn[m].sum((1, 2)) >= 6).float().mean()), 3)))
            res[f"off_{off}"] = rows_
        out["kinds"][name] = res
    os.environ["ATANOR_RENDER_FIX"] = "0"; print(json.dumps(out)); json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
