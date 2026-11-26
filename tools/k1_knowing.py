"""K1 (atanor_os/K1_KNOWING_PROTOCOL.md, registered before running): where the being believes its springs are, against where they are.
Judge side only: the being lives and learns as always. Each going place unit gets a true position (slow mean of the true positions of the bodies
in it); every 50 beats the unit with the largest water (food) value is the believed spring; error = cells to the nearest true spring cell.
  PYTHONPATH=. python -m tools.k1_knowing OUT.json --lay 9105_0 --being s3_a [--beats 3000]"""
import argparse, json, time
import numpy as np, torch
from tools.need_fork import t_room, place, DEV
from tools.a1_channel import V2
from tools.l1_life import make
from tools.s4_kin import being, rebirth
from atanor_os.emerge.room_body import RoomBody


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--lay", default="9105_0"); ap.add_argument("--being", required=True)
    ap.add_argument("--beats", type=int, default=3000); ap.add_argument("--every", type=int, default=50); ap.add_argument("--chan", default="room")   # G1 equivalence: contract
    a = ap.parse_args(); V2["on"] = True; RoomBody.default_cues = True; t0 = time.time(); n = 64
    seed, mirror = map(int, a.lay.split("_")); assert not 7701 <= seed <= 7712; rows, info = t_room(seed, bool(mirror)); sd = seed * 10 + mirror + 41
    torch.manual_seed(sd)
    B = make(rows, n, sd, V2["drain"]); O, Mi = being(B, n, a.being); G = Mi.going
    col = {"water": 0, "food": 1}
    if a.chan == "contract":                                            # G1: the same organ reading the bedrock by contract; its saved room memory moved to the contract's order
        R3 = torch.zeros(G.KU, len(G.ins), device=DEV); R3[:, 1] = G.R[:, 1]; R3[:, 2] = G.R[:, 0]
        G.chan, G.k, G.R = "contract", len(G.ins), R3; G.refresh(); col = {"water": 2, "food": 1}
    every = torch.ones(n, dtype=torch.bool, device=DEV); B.w.store[:] = 0.75; place(B, info); O.reset(every); Mi.begin(every); Mi.prev = None
    g = np.array([list(r) for r in rows]); springs = {k: np.argwhere(g == c)[:, ::-1] + 0.5 for k, c in (("water", "w"), ("food", "f"))}   # (x, y) cell centres
    KU = G.KU; upos = torch.zeros(KU, 2, device=DEV); useen = torch.zeros(KU, device=DEV)
    x = O(B.raw()); first = {"water": None, "food": None}; fills = {"water": 0, "food": 0}; trace = []; deaths = 0
    for t in range(a.beats):
        act = Mi.choose(x); al = ~B.dead.clone(); s0 = B.w.store.clone(); f = B.step(act); x2 = O(B.raw()); died = al & B.dead; deaths += int(died.sum())
        for k, j in (("water", 0), ("food", 1)):                         # the world's store order: 0 water, 1 food (as room_body's fills)
            c = int(f[:, j].bool().sum()); fills[k] += c
            if c and first[k] is None:
                first[k] = t
        with torch.no_grad():                                            # judge side: where each place unit really is
            u = G.unit(G.place(x2)); live = ~B.dead
            if bool(live.any()):
                cnt = torch.bincount(u[live], minlength=KU).float(); sm = torch.zeros(KU, 2, device=DEV).index_add_(0, u[live], B.w.pos[live].float())
                hit = cnt > 0; upos = torch.where(hit[:, None], 0.99 * upos + 0.01 * sm / cnt.clamp(min=1)[:, None] if bool((useen > 0).any()) else sm / cnt.clamp(min=1)[:, None], upos)
                upos = torch.where((hit & (useen == 0))[:, None], sm / cnt.clamp(min=1)[:, None], upos); useen += cnt
        Mi.after(x, act, x2, died, B.dead & ~died); x = x2
        if B.dead.any():
            rebirth(B, O, Mi, info); B.felt[:] = 0; x = O(B.raw())
        if (t + 1) % 1000 == 0:
            G.learn()
            with torch.enable_grad():
                Mi.wo.learn(n=max(1, 2500 // max(1, getattr(Mi.wo, "per_call", 1))))
        if (t + 1) % a.every == 0 and getattr(G, "V", None) is not None:
            row = dict(t=t + 1, fills=dict(fills))
            known = useen > 0
            for k in ("water", "food"):
                j = col[k]; V = G.V[:, j].clone(); V[~known] = -1e9; b = int(V.argmax()); p = upos[b].cpu().numpy()
                d = float(np.min(np.hypot(springs[k][:, 0] - p[0], springs[k][:, 1] - p[1])))
                srt = torch.sort(G.V[known, j], descending=True).values
                row[k] = dict(err=round(d, 2), unit=b, believed=[round(float(p[0]), 1), round(float(p[1]), 1)], v_top=round(float(srt[0]), 4),
                              v_gap=round(float(srt[0] - srt[1]), 4) if len(srt) > 1 else None)
            trace.append(row)
    r = dict(chan=a.chan, need_fills=None, layout=a.lay, being=a.being, beats=a.beats, deaths=deaths, first_fill=first, fills=fills, springs={k: v.tolist() for k, v in springs.items()}, trace=trace,
             seconds=round(time.time() - t0))
    print(json.dumps(dict(layout=a.lay, being=a.being, first_fill=first, deaths=deaths, err=[(q["t"], q["water"]["err"], q["food"]["err"]) for q in trace[::4]])), flush=True)
    json.dump(r, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
