"""S0 (atanor_os/M2_SLEEP_PROTOCOL.md): one mind lives world A2 (moving food), then A3 (two rooms), then A2 again, same bodies' mind
throughout. Judge side: place-unit use and redundancy after the first A2; how far A2's units move during A3; A2 deaths and food waits
before vs after the excursion (the A2 seeds are the same for both A2 phases).
  PYTHONPATH=. python -m tools.s0_roundtrip OUT.json --being s3_b [--beats 3000 --rep 0]"""
import argparse, json, time
import numpy as np, torch
from tools.need_fork import t_room, place, DEV
from tools.a1_channel import V2
from tools.l1_life import make
from tools.s4_kin import being, rebirth
from tools.a3_rooms import layout as a3_layout
from atanor_os.emerge.room3d_gpu import FLOOR, FOOD


def phase(kind, Mi, O, n, beats, seed, probe=None):
    """One stretch of life in one world for the same mind; returns deaths, food waits (store < .7 -> food fill)."""
    if kind == "A2":
        rows, info = t_room(9104, False)
        g = np.array([list(r) for r in rows]); H, W = g.shape
        corners = [tuple(int(v) for v in np.argwhere(g == "f")[0][::-1]), (1, 1), (W - 2, 1)]
    else:
        rows, info, _ = a3_layout(False); corners = None
    torch.manual_seed(seed); B = make(rows, n, seed, V2["drain"]); w = B.w; G = Mi.going

    def put(k):
        c = corners[k]; old = w.src[:, 1].clone()
        w.map[torch.arange(n, device=DEV), old[:, 0], old[:, 1]] = FLOOR
        w.map[:, c[0], c[1]] = FOOD; w.src[:, 1] = torch.tensor(c, device=DEV); w.spots[FOOD] = torch.tensor([c], device=DEV)
    every = torch.ones(n, dtype=torch.bool, device=DEV); w.store[:] = 0.75; place(B, info); O.reset(every); Mi.begin(every); Mi.prev = None
    if corners:
        put(0)
    x = O(B.raw()); deaths = 0; waits = []; since = torch.full((n,), -1, dtype=torch.long, device=DEV)
    if probe is not None:
        probe.enter(kind)
    for t in range(beats):
        if corners and t % 300 == 0:
            put((t // 300) % 3)
        act = Mi.choose(x); al = ~B.dead.clone(); f = B.step(act); x2 = O(B.raw()); died = al & B.dead
        deaths += int(died.sum())
        hungry = (w.store[:, 1] < 0.7) & ~died; start = hungry & (since < 0); since = torch.where(start, torch.full_like(since, t), since)
        got = f[:, 1].bool() & (since >= 0)
        for i in torch.nonzero(got)[:, 0].tolist():
            waits.append(t - int(since[i]))
        since = torch.where(got | died, torch.full_like(since, -1), since)
        if probe is not None:
            probe.beat(x, x2, O, Mi.going, died)
        Mi.after(x, act, x2, died, B.dead & ~died); x = x2
        if B.dead.any():
            if probe is not None:
                probe.reborn(B.dead)
            rebirth(B, O, Mi, info); B.felt[:] = 0; x = O(B.raw())
        if (t + 1) % 1000 == 0:
            G.learn()
            with torch.enable_grad():
                Mi.wo.learn(n=max(1, 2500 // max(1, getattr(Mi.wo, "per_call", 1))))
    if probe is not None:
        probe.leave()
    return dict(world=kind, deaths=deaths, food_waits=len(waits), wait_median=float(np.median(waits)) if waits else None)


@torch.no_grad()
def units(G):
    C = G.C[:G.used] if G.used else G.C; d = torch.cdist(C, C) + 1e9 * torch.eye(len(C), device=C.device)
    close = int(((d < G.rad).sum() // 2)); quiet = int(((G.Vn[:len(C)] < 1.0) & (G.R[:len(C)].sum(1) < 1e-3)).sum())
    return dict(used=int(G.used), KU=int(G.KU), pairs_closer_than_birth_radius=close, quiet_units=quiet)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--being", required=True)
    ap.add_argument("--beats", type=int, default=3000); ap.add_argument("--rep", type=int, default=0); ap.add_argument("--bodies", type=int, default=64); ap.add_argument("--mid", default="A3"); ap.add_argument("--witness", type=int, default=0); ap.add_argument("--witness-view", default="picture")
    a = ap.parse_args(); n = a.bodies; t0 = time.time()
    seed_a2 = 91040 + 71 + 1000 * a.rep; seed_a3 = 9301 + a.rep; assert not any(7701 <= s <= 7712 for s in (seed_a2, seed_a3))
    rows, info = t_room(9104, False); B0 = make(rows, n, seed_a2, V2["drain"]); O, Mi = being(B0, n, a.being); G = Mi.going
    probe = None
    if a.witness:                                                        # P1 (M2_SLEEP_PROTOCOL.md): Astra's RouteWitness, read-only beside the going organ
        from tools.witness_probe import WitnessProbe
        probe = WitnessProbe(a.witness, view=a.witness_view)
    out = dict(being=a.being, beats=a.beats, rep=a.rep, bodies=n, phases=[])
    out["phases"].append(phase("A2", Mi, O, n, a.beats, seed_a2, probe)); out["units_after_A2"] = units(G)
    C_a2 = G.C.clone(); Pu_a2 = G.Pu.clone() if getattr(G, "Pu", None) is not None else None
    out["phases"].append(phase(a.mid, Mi, O, n, a.beats, seed_a3 if a.mid == "A3" else seed_a2 + 500, probe)); out["units_after_A3"] = units(G); out["mid"] = a.mid
    out["a2_unit_drift_mean"] = float((G.C - C_a2).norm(dim=1).mean()); out["birth_radius"] = G.rad
    out["a2_unit_drift_over_radius_share"] = float(((G.C - C_a2).norm(dim=1) > G.rad).float().mean())
    if Pu_a2 is not None:
        out["a2_homing_pos_drift_mean"] = float((G.Pu - Pu_a2).norm(dim=1).mean())
    out["phases"].append(phase("A2", Mi, O, n, a.beats, seed_a2, probe))
    if probe is not None:
        out["witness"] = probe.report()
    out["seconds"] = round(time.time() - t0)
    json.dump(out, open(a.out, "w"), indent=1); print(json.dumps(out))


if __name__ == "__main__":
    main()
