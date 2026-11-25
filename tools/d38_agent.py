"""D38 (atanor_os/DEVELOPMENTAL_MAP.md, registered before running): does the being's own forecaster already tell a thing that reacts to it
from one that moves on its own? No new part; the world draws one other of its kind per body, in three kinds (one mind, 64 bodies, split in thirds):
  still      - the other stands where it was put;
  responsive - each beat the other moves by .7 of its body's own last displacement (+ a little noise): it reacts to what the body does;
  yoked      - the other replays a responsive other's moves, but of another body: the same motion, not caused by this body (yoked control).
Measured with the being's own forecaster (Core.foresee, the gesture it chose): error on the eye's part of the sense on beats the other is seen,
minus the same body's error when it is not seen, first and last 1,000 beats. The mind lives and learns as always.
  PYTHONPATH=. python -m tools.d38_agent OUT.json --lay 9101_0 --being s3_a [--beats 5000]"""
import argparse, json, time
import numpy as np, torch
from tools.need_fork import t_room, place, DEV
from tools.a1_channel import V2
from tools.l1_life import make
from tools.s4_kin import being, rebirth
from atanor_os.emerge.room_body import RoomBody, G
from atanor_os.emerge.body_contract import PathOrgan


def put(B, K, idx):
    """the other one and a half cells ahead of its body (where that is floor), else where the body stands."""
    w = B.w; h = w.head; tgt = w.pos + 1.5 * torch.stack([torch.cos(h), torch.sin(h)], 1)     # the world's cell lookup reads every body's own map
    ok = w.cell(tgt) == G.FLOOR; K[idx] = torch.where(ok[:, None], tgt, w.pos + 0.01)[idx]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--lay", default="9101_0"); ap.add_argument("--being", required=True)
    ap.add_argument("--beats", type=int, default=5000); ap.add_argument("--gain", type=float, default=0.7)
    a = ap.parse_args(); V2["on"] = True; RoomBody.default_cues = True; t0 = time.time(); n = 64
    seed, mirror = map(int, a.lay.split("_")); assert not 7701 <= seed <= 7712; rows, info = t_room(seed, bool(mirror)); sd = seed * 10 + mirror + 5
    torch.manual_seed(sd)
    B = make(rows, n, sd, V2["drain"]); O, Mi = being(B, n, a.being)
    every = torch.ones(n, dtype=torch.bool, device=DEV); B.w.store[:] = 0.75; place(B, info); O.reset(every); Mi.begin(every); Mi.prev = None
    grp = torch.arange(n, device=DEV) * 3 // n                          # 0 still, 1 responsive, 2 yoked
    third = int((grp == 1).sum()); yk = torch.nonzero(grp == 2)[:, 0]; src = torch.nonzero(grp == 1)[:, 0][(torch.arange(len(yk), device=DEV) + 1) % third]
    K = torch.zeros(n, 2, device=DEV); put(B, K, torch.arange(n, device=DEV)); B.kin = K.clone()
    nb = O.inside.stop + sum(int(np.prod(B.c.sensors[i].shape)) for i in O.vec); eye = slice(nb, O.dx - (PathOrgan.N if O.path is not None else 0))
    x = O(B.raw()); last = torch.zeros(n, 2, device=DEV)
    E = np.zeros((a.beats, n)); S = np.zeros((a.beats, n), bool); MV = np.zeros((a.beats, n), bool); SE = np.zeros((a.beats, n)); deaths = 0
    for t in range(a.beats):
        act = Mi.choose(x); was = B.w.pos.clone(); al = ~B.dead.clone(); B.step(act); dp = B.w.pos - was
        mv = torch.zeros(n, 2, device=DEV); r = grp == 1
        mv[r] = a.gain * last[r] + 0.03 * torch.randn(int(r.sum()), 2, device=DEV)
        mv[yk] = mv[src]                                                 # the yoked others move as some responsive other moved (not by this body)
        nk = K + mv; ok = B.w.cell(nk) == G.FLOOR; K = torch.where(ok[:, None], nk, K); B.kin = K.clone(); last = dp
        x2 = O(B.raw()); seen = getattr(B, "kin_seen", torch.zeros(n, dtype=torch.bool, device=DEV))
        with torch.no_grad():
            fa = Mi.net.foresee(x); fo = fa.gather(1, act[:, None, None].expand(n, 1, x.shape[1]))[:, 0]
            SE[t] = fa[:, :, eye].var(1).mean(1).cpu().numpy()                                   # D38c: how much its own eye forecast hangs on which act it makes
            E[t] = ((fo - x2)[:, eye] ** 2).mean(1).cpu().numpy()
        S[t] = seen.cpu().numpy(); MV[t] = (dp.norm(dim=1) > 0.05).cpu().numpy()          # D38b: did its own act move it this beat
        died = al & B.dead; deaths += int(died.sum()); Mi.after(x, act, x2, died, B.dead & ~died); x = x2
        if B.dead.any():
            dd = B.dead.clone(); idx = torch.nonzero(dd)[:, 0]; rebirth(B, O, Mi, info); put(B, K, idx); B.kin = K.clone(); last[idx] = 0
            B.felt[:] = 0; x = O(B.raw())
    g = grp.cpu().numpy(); out = {}
    for name, k in (("still", 0), ("responsive", 1), ("yoked", 2)):
        m = g == k; row = {}
        for tag, lo, hi in (("first", 0, 1000), ("last", a.beats - 1000, a.beats)):
            e, s = E[lo:hi][:, m], S[lo:hi][:, m]
            es, eu = (e[s].mean() if s.any() else float("nan")), (e[~s].mean() if (~s).any() else float("nan"))
            mvv = MV[lo:hi][:, m]; em = e[s & mvv].mean() if (s & mvv).any() else float("nan"); est = e[s & ~mvv].mean() if (s & ~mvv).any() else float("nan")
            row[tag] = dict(seen=round(float(es), 5), unseen=round(float(eu), 5), excess=round(float(es - eu), 5), seen_share=round(float(s.mean()), 3),
                            act_sens_seen=round(float(SE[lo:hi][:, m][s].mean()), 7) if s.any() else None, act_sens_unseen=round(float(SE[lo:hi][:, m][~s].mean()), 7), seen_moved=round(float(em), 5), seen_still=round(float(est), 5), still_share=round(float((~mvv[s]).mean()) if s.any() else float("nan"), 3))
        out[name] = row
    r = dict(layout=a.lay, being=a.being, gain=a.gain, groups=out, deaths=deaths, seconds=round(time.time() - t0))
    print(json.dumps(r), flush=True)
    try:
        old = json.load(open(a.out))["rows"]
    except (OSError, ValueError, KeyError):
        old = []
    json.dump(dict(protocol="atanor_os/DEVELOPMENTAL_MAP.md (D38)", rows=old + [r]), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
