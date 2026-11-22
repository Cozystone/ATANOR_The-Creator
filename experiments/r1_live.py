"""R1 organ test: rooms with darkness (lights out 20 of every 60 steps). Fresh beings raised over 8 lives (varied rooms 1-8), with the inner-state
organ (inner 64) vs without. Per life: whole, fills; IN THE DARK: change of distance to the nearest spring per dark step (negative = goes toward it
without seeing it); at the moment the lights come back: share of bodies with a spring in view.
  python r1_live.py <inner> <seed>"""
import sys, time, torch
from atanor_os.emerge.room_body import RoomBody
from atanor_os.emerge.body_contract import Organs
from atanor_os.emerge.mind import Mind
from atanor_os.emerge.room_vary import condition
from atanor_os.emerge import room3d_gpu as G
INNER, SEED = int(sys.argv[1]), int(sys.argv[2]); IMAG = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
dev, N, T, LIVES = "cuda", 128, 900, 8
torch.manual_seed(SEED)
O = Mi = None; t0 = time.time()
for life in range(LIVES):
    rows, drain = condition(1 + life % 8)
    B = RoomBody(N, rows, drain=drain, seed=SEED * 100 + life, device=dev); B.dark_every, B.dark_len = 60, 20
    if O is None:
        O = Organs(B.c, N, dev=dev, depth="mc/mc_depth.pt"); Mi = Mind(O, B.c, N, dev=dev, inner=INNER, imagine=IMAG); Mi.imagine_mode = sys.argv[4] if len(sys.argv) > 4 else 'roll'
    O.reset(torch.ones(N, dtype=torch.bool, device=dev)); Mi.begin(torch.ones(N, dtype=torch.bool, device=dev))
    if Mi.wo is not None:
        Mi.wo.on = life >= 1
    fills = torch.zeros(N, device=dev); dd_sum, dd_n, face_sum, face_n = 0.0, 0, 0.0, 0
    xr = O(B.raw()); x = Mi.feel(xr)

    def near():
        c = B.w.src.float() + 0.5                                         # [n, 2, 2]
        return (c - B.w.pos[:, None]).norm(dim=2).min(1).values

    for t in range(T):
        dark0 = B.dark(); d0 = near()
        act = Mi.choose(x); a0 = ~B.dead.clone(); f = B.step(act); fills += f.any(1).float(); died = a0 & B.dead
        cut = B.dead & ~died
        xr2 = O(B.raw()); Mi.carry(xr, act, cut | died); x2 = Mi.feel(xr2)
        Mi.after(x, act, x2, died, cut); x, xr = x2, xr2
        al = ~B.dead
        m = dark0 & B.dark() & al & ~f.any(1)
        if m.any():
            dd_sum += float((near() - d0)[m].sum()); dd_n += int(m.sum())
        lit_now = dark0 & ~B.dark() & al
        if lit_now.any():
            kind, dist = B.w.cast(B.w.head[:, None] + B.cols[None], step=0.1, far=12.0)
            vis = (((kind == G.WATER) | (kind == G.FOOD)) & (dist < 12)).any(1)
            face_sum += float(vis[lit_now].float().sum()); face_n += int(lit_now.sum())
        if B.dead.all():
            break
    print(f"inner {INNER} imagine {IMAG} {Mi.imagine_mode} seed {SEED} life {life + 1} (room {1 + life % 8}): whole {float((~B.dead).float().mean()):.3f} | fills {float(fills.mean()):.2f} | "
          f"in the dark: distance to nearest spring {dd_sum / max(dd_n, 1):+.4f}/step | spring in view as lights return {face_sum / max(face_n, 1):.2f} | {time.time() - t0:.0f}s", flush=True)
print("DONE")
