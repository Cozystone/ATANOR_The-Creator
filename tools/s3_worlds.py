"""S3 (atanor_os/S3_S4_PROTOCOL.md, registered before running): one being (one mind + one going organ, never reset) lives in rooms A, B, C, then A
again. Childhood (protected, babbling then practice toward its own places) only at the start of the first stay.
  PYTHONPATH=. python -m tools.s3_worlds OUT.json A B C [--beats 6000] [--tag X]"""
import argparse, json, os, time
import numpy as np, torch
from tools.need_fork import t_room, place, DEV, DEPTH, BEING
from tools.a1_channel import V2
from tools.a1_sf import paths_of
from tools.l1_life import make, forks
from atanor_os.emerge.room_body import RoomBody
from atanor_os.emerge.body_contract import Organs
from atanor_os.emerge.mind import Mind
from atanor_os.emerge.going import Going

SAVE = "s3"


VIEW = None; TAG = ""


def stay(Mi, O, rows, info, seed, beats, child=0, babble=0, n=64, room=""):
    torch.manual_seed(seed); G = Mi.going
    B = make(rows, n, seed, V2["drain"]); B.w.store[:] = 0.75; place(B, info); every = torch.ones(n, dtype=torch.bool, device=DEV)
    O.reset(every); Mi.begin(every); Mi.prev = None; x = O(B.raw())
    deaths = np.zeros(beats, int); need = np.zeros(beats, int); goal = None; left = torch.zeros(n, device=DEV)
    for t in range(beats):
        if t < babble:
            act = torch.randint(0, Mi.M, (n,), device=DEV)
        elif t < child:
            u = G.unit(G.place(x)); arrive = goal is not None and (u == goal)
            new = (left <= 0) | (arrive if goal is not None else torch.ones(n, dtype=torch.bool, device=DEV))
            pick = torch.randint(0, G.used if G.born else G.KU, (n,), device=DEV); goal = pick if goal is None else torch.where(new, pick, goal)
            left = torch.where(new, torch.full_like(left, 200), left - 1)
            act = torch.multinomial(G.toward(x, G.C[goal]).exp(), 1)[:, 0]; act = torch.where(torch.rand(n, device=DEV) < 0.1, torch.randint(0, Mi.M, (n,), device=DEV), act)
        else:
            act = Mi.choose(x)
        s0 = B.w.store.clone(); alive0 = ~B.dead.clone(); f = B.step(act)
        need[t] = int(((f[:, 0].bool() & (s0[:, 0] < s0[:, 1])) | (f[:, 1].bool() & (s0[:, 1] < s0[:, 0]))).sum())
        died = alive0 & B.dead; raw2 = B.raw(); x2 = O(raw2); Mi.after(x, act, x2, died, B.dead & ~died); x = x2; deaths[t] = int(died.sum())
        if VIEW is not None and t % 3 == 0:                                # the room viewer (8768): body 0's eye, the judge's map, the memory's context
            if bool(died[0]):
                VIEW.TRAIL.clear()
            sw = [w_ for w_ in getattr(G, "switches", [])][-4:]
            VIEW.push(raw2[-1][0].cpu().numpy(), rows, B.w.pos[0].tolist(), float(B.w.head[0]), B.w.store[0].tolist(), B.w.stock[0].tolist() if hasattr(B.w, "stock") else None,
                      f"{TAG} room {room} - beat {t + 1} / {beats}" + (" - childhood" if t < child else ""),
                      "context in use: " + str(getattr(G, "act", "-")) + "  (contexts opened: " + str(getattr(G, "nctx", 1)) + ")   merges ('I was wrong'): " + str(getattr(G, "merges", 0))
                      + "   held crossings waiting: " + ("yes" if getattr(G, "h_since", -1) >= 0 else "no") + chr(10)
                      + "last switches (beat, from, to): " + str(sw) + chr(10)
                      + f"this stay so far (64 bodies): deaths {int(deaths[:t + 1].sum())}, need fills {int(need[:t + 1].sum())}")
        if t < child:
            B.w.store[:] = 0.75; B.w.dead[:] = False
        if (t + 1) % 1000 == 0:
            G.learn()
            if t >= child:
                with torch.enable_grad():
                    Mi.wo.learn(n=max(1, 2500 // max(1, getattr(Mi.wo, 'per_call', 1))))
        if B.dead.any():
            dd = B.dead.clone(); idx = torch.nonzero(dd)[:, 0]; B.w.dead[idx] = False; B.w.store[idx] = 0.75; place(B, info, which=idx); O.reset(dd); Mi.begin(dd)
            B.felt[:] = 0; x = O(B.raw())                                 # the re-read after rebirth feels no motion (L1d drift fix)
    s = child; k = 1500
    per = lambda a, lo, hi: round(1000 * float(a[lo:hi].sum()) / (n * max(1, hi - lo)), 2)
    return dict(first_deaths=per(deaths, s, s + k), last_deaths=per(deaths, beats - k, beats), first_need=per(need, s, s + k), last_need=per(need, beats - k, beats))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("lays", nargs="+"); ap.add_argument("--beats", type=int, default=6000); ap.add_argument("--tag", default=""); ap.add_argument("--ctx", type=int, default=0); ap.add_argument("--ctx-by", default="both"); ap.add_argument("--hold", type=int, default=0); ap.add_argument("--rep", type=int, default=0); ap.add_argument("--view", action="store_true")
    a = ap.parse_args()
    global VIEW, TAG
    TAG = a.tag
    if a.view:
        from tools import room_live_view as VIEW_
        VIEW_.start(); VIEW = VIEW_
    V2["on"] = True; RoomBody.default_cues = True; t0 = time.time(); d = f"{SAVE}{a.tag}"; os.makedirs(d, exist_ok=True)
    seq = list(a.lays) + [a.lays[0]]; s0_, m0_ = map(int, seq[0].split("_")); rows0, info0 = t_room(s0_, bool(m0_))
    n = 64; B0 = make(rows0, n, 1, V2["drain"]); O = Organs(B0.c, n, dev=DEV, depth=DEPTH); Mi = Mind(O, B0.c, n, dev=DEV, value_arm="relief"); Mi.grow(BEING); Mi.wo.on = True
    Mi.going = Going(O.dx, Mi.M, n, O.inside, DEV, W=1.0, ctx=a.ctx, ctx_by=a.ctx_by, hold=a.hold); res = []
    if a.ctx:
        Mi.going.do_mask = B0.hand == 2                                # the reaching gesture (S3b scores relief expectations on reaching beats)
    for k, lay in enumerate(seq):
        seed, mirror = map(int, lay.split("_")); assert not 7701 <= seed <= 7712; rows, info = t_room(seed, bool(mirror)); sd = seed * 10 + mirror + 1000 * k + 50000 * a.rep
        first = k == 0; b0 = Mi.going.beats; c0 = getattr(Mi.going, "act", None); r = stay(Mi, O, rows, info, sd, a.beats + (3000 if first else 0), child=3000 if first else 0, babble=1000 if first else 0, room=lay)
        fk = forks(Mi, O, rows, info, sd, paths_of(rows, info))
        if a.ctx:
            G = Mi.going; r.update(ctx_start=c0, ctx_end=G.act, switches=[(b - b0, o, w) for b, o, w in G.switches if b > b0], contexts=G.nctx, merges=getattr(G, 'merges', 0))
        r.update(stay=k, layout=lay, forks={s: fk[s] for s in ("far", "stem")}, units_used=int(Mi.going.used), seconds=round(time.time() - t0))
        print(json.dumps(r), flush=True); res.append(r); json.dump(dict(protocol="atanor_os/S3_S4_PROTOCOL.md (S3)", rows=res), open(a.out, "w"), indent=1)
    torch.save(Mi.going.state(), f"{d}/going.pt"); Mi.save(f"{d}/mind.pt")


if __name__ == "__main__":
    main()
