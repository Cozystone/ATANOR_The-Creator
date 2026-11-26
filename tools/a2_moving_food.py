"""World A2 (atanor_os/LOKI_IDEAL_SPEC.md section 5; registered in atanor_os/WORLD_A_PROTOCOL.md): food moves between three fixed dead-end corners
in a fixed cycle (the leg end it starts at -> top-left corner of the crossbar -> top-right corner), `--period` beats each, for every body at once.
It does not jump when eaten (one spot at a time). Water stays. Everything else as the room V2 world. Judge side: when food moved, how long until the
first fill at its new place, fills, deaths, where the being believes the food is (K1 read-out), and visits to the NEXT corner before food comes.
  PYTHONPATH=. python -m tools.a2_moving_food OUT.json --lay 9101_0 --being s3_a [--beats 6000 --period 300]"""
import argparse, json, time
import numpy as np, torch
from tools.need_fork import t_room, place, DEV
from tools.a1_channel import V2
from tools.l1_life import make
from tools.s4_kin import being, rebirth
from atanor_os.emerge.room_body import RoomBody
from atanor_os.emerge.room3d_gpu import FLOOR, FOOD


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--lay", default="9101_0"); ap.add_argument("--being", required=True)
    ap.add_argument("--beats", type=int, default=6000); ap.add_argument("--period", type=int, default=300); ap.add_argument("--still", type=int, default=0); ap.add_argument("--decay", type=float, default=0.0)   # V1 variant: the going organ's fading per beat (0 = as it is, .9995)
    a = ap.parse_args(); V2["on"] = True; RoomBody.default_cues = True; t0 = time.time(); n = 64
    seed, mirror = map(int, a.lay.split("_")); assert not 7701 <= seed <= 7712; rows, info = t_room(seed, bool(mirror)); sd = seed * 10 + mirror + 71
    torch.manual_seed(sd)
    g = np.array([list(r) for r in rows]); H, W = g.shape
    f0 = tuple(int(v) for v in np.argwhere(g == "f")[0][::-1])                    # (x, y)
    corners = [f0, (1, 1), (W - 2, 1)]
    for c in corners[1:]:
        assert g[c[1], c[0]] == ".", (c, g[c[1], c[0]])
    B = make(rows, n, sd, V2["drain"]); O, Mi = being(B, n, a.being); w = B.w; G = Mi.going
    if a.decay:
        G.decay = a.decay

    def put(k):                                                          # food at corner k for every body (the world's clock, not a body's doing)
        c = corners[k]; old = w.src[:, 1].clone()
        w.map[torch.arange(n, device=DEV), old[:, 0], old[:, 1]] = FLOOR
        w.map[:, c[0], c[1]] = FOOD; w.src[:, 1] = torch.tensor(c, device=DEV); w.spots[FOOD] = torch.tensor([c], device=DEV)
    every = torch.ones(n, dtype=torch.bool, device=DEV); B.w.store[:] = 0.75; place(B, info); O.reset(every); Mi.begin(every); Mi.prev = None
    put(0); x = O(B.raw()); k = 0
    KU = G.KU; upos = torch.zeros(KU, 2, device=DEV); useen = torch.zeros(KU, device=DEV)
    cyc, cur = [], None; deaths = 0; fills = [0, 0]; need = 0; trace = []
    since = torch.full((n,), -1, dtype=torch.long, device=DEV); waits, lost = [], 0   # A2-1: per body, beats from food store falling under .7 to its next food fill
    for t in range(a.beats):
        if not a.still and t % a.period == 0:
            k = (t // a.period) % 3; put(k)
            if cur is not None:
                cyc.append(cur)
            nxt = corners[(k + 1) % 3]
            cur = dict(t=t, corner=k, first_fill=None, fills=0, deaths=0, next_visits=0, at_next=nxt)
        act = Mi.choose(x); al = ~B.dead.clone(); s0 = B.w.store.clone(); f = B.step(act); x2 = O(B.raw()); died = al & B.dead
        deaths += int(died.sum()); fills[0] += int(f[:, 0].sum()); fills[1] += int(f[:, 1].sum())
        need += int(((f[:, 0].bool() & (s0[:, 0] < s0[:, 1])) | (f[:, 1].bool() & (s0[:, 1] < s0[:, 0]))).sum())
        hungry = (w.store[:, 1] < 0.7) & ~died; start = hungry & (since < 0); since = torch.where(start, torch.full_like(since, t), since)
        got = f[:, 1].bool() & (since >= 0)
        for i in torch.nonzero(got)[:, 0].tolist():
            waits.append(t - int(since[i]))
        lost += int((died & (since >= 0)).sum()); since = torch.where(got | died, torch.full_like(since, -1), since)
        if cur is not None:
            cur["deaths"] += int(died.sum()); cur["fills"] += int(f[:, 1].sum())
            if cur["first_fill"] is None and bool(f[:, 1].any()):
                cur["first_fill"] = t - cur["t"]
            nx = torch.tensor(cur["at_next"], device=DEV, dtype=torch.float32) + 0.5
            cur["next_visits"] += int(((w.pos - nx).norm(dim=1) < 1.6).sum())
        with torch.no_grad():                                            # judge side: where each place unit really is (as K1)
            u = G.unit(G.place(x2)); live = ~B.dead
            if bool(live.any()):
                cnt = torch.bincount(u[live], minlength=KU).float(); sm = torch.zeros(KU, 2, device=DEV).index_add_(0, u[live], w.pos[live].float()); hit = cnt > 0
                upos = torch.where((hit & (useen == 0))[:, None], sm / cnt.clamp(min=1)[:, None], torch.where(hit[:, None], 0.99 * upos + 0.01 * sm / cnt.clamp(min=1)[:, None], upos)); useen += cnt
        Mi.after(x, act, x2, died, B.dead & ~died); x = x2
        if B.dead.any():
            rebirth(B, O, Mi, info); B.felt[:] = 0; x = O(B.raw())
        if (t + 1) % 1000 == 0:
            G.learn()
            with torch.enable_grad():
                Mi.wo.learn(n=max(1, 2500 // max(1, getattr(Mi.wo, "per_call", 1))))
        if (t + 1) % 50 == 0 and getattr(G, "V", None) is not None:
            V = G.V[:, 1].clone(); V[useen == 0] = -1e9; b = int(V.argmax()); p = upos[b].cpu().numpy()
            dist = [float(np.hypot(c[0] + 0.5 - p[0], c[1] + 0.5 - p[1])) for c in corners]
            trace.append(dict(t=t + 1, food_corner=k, believed_corner=int(np.argmin(dist)), believed_err=round(dist[k], 2)))
    if cur is not None:
        cyc.append(cur)
    lat = [c["first_fill"] if c["first_fill"] is not None else a.period for c in cyc]
    third = max(1, len(lat) // 3)
    r = dict(decay=G.decay, layout=a.lay, being=a.being, beats=a.beats, period=a.period, still=a.still, corners=corners, deaths=deaths, fills=fills, need_fills=need,
             hunger_to_food=dict(n=len(waits), median=float(np.median(waits)) if waits else None, mean=float(np.mean(waits)) if waits else None, first_half_median=float(np.median(waits[:len(waits) // 2])) if len(waits) > 1 else None, last_half_median=float(np.median(waits[len(waits) // 2:])) if len(waits) > 1 else None, died_hungry=lost),
             deaths_per_1000=round(1000 * deaths / (n * a.beats), 3), food_fills_per_1000=round(1000 * fills[1] / (n * a.beats), 2),
             latency_first_third=float(np.mean(lat[:third])), latency_last_third=float(np.mean(lat[-third:])),
             belief_on_food_corner=round(float(np.mean([q["believed_corner"] == q["food_corner"] for q in trace])), 3), cycles=cyc, trace=trace, seconds=round(time.time() - t0))
    print(json.dumps({k_: v for k_, v in r.items() if k_ not in ("cycles", "trace")}), flush=True)
    json.dump(r, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
