"""방에서 하나의 존재 파일을 기른다 (공통 접면: room_body + body_contract.Organs + mind.Mind). 여러 조건의 방 (room_vary)을 삶마다 돌아가며.
  python -m atanor_os.emerge.room_live --lives 20 --n 256 --out being/atanor.pt [--load 같은 파일]
삶마다: 온전한 삶 (끝까지 산 몸의 몫), 몸당 채움, 값의 얻은 믿음. 존재 파일 (알맹이·값·기억·감각 통계)을 삶마다 저장한다.
"""
import argparse, os, time, torch
from .room_body import RoomBody
from .body_contract import Organs
from .mind import Mind
from .room_vary import condition


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--lives", type=int, default=20); ap.add_argument("--n", type=int, default=256); ap.add_argument("--T", type=int, default=900)
    ap.add_argument("--out", default="being/atanor.pt"); ap.add_argument("--load", default=""); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--depth", default="mc/mc_depth.pt"); ap.add_argument("--first_room", type=int, default=1); ap.add_argument("--device", default="cuda"); ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.manual_seed(a.seed); dev = a.device
    if a.threads:
        torch.set_num_threads(a.threads)
    rows, drain = condition(a.first_room)
    B = RoomBody(a.n, rows, drain=drain, seed=a.seed, device=dev)
    O = Organs(B.c, a.n, dev=dev, depth=a.depth)
    Mi = Mind(O, B.c, a.n, dev=dev)
    if a.load and os.path.exists(a.load):
        print("loaded being at step", Mi.load(a.load), flush=True)
    t0 = time.time()
    for life in range(a.lives):
        rows, drain = condition(a.first_room + life % 8)
        B = RoomBody(a.n, rows, drain=drain, seed=a.seed * 1000 + life, device=dev)
        O.reset(torch.ones(a.n, dtype=torch.bool, device=dev)); Mi.begin(torch.ones(a.n, dtype=torch.bool, device=dev))
        if Mi.wo is not None:
            Mi.wo.on = Mi.steps > 900
        fills = torch.zeros(a.n, device=dev)
        x = O(B.raw())
        for t in range(a.T):
            act = Mi.choose(x)
            alive0 = ~B.dead.clone()
            f = B.step(act); fills += f.any(1).float()
            died = alive0 & B.dead
            x2 = O(B.raw())
            cut = B.dead & ~died                                   # 이미 죽어 있던 몸 (잇지 않음)
            Mi.after(x, act, x2, died, cut)
            x = x2
            if B.dead.all():
                break
        Mi.save(a.out)
        print(f"life {life + 1} (room {a.first_room + life % 8}): whole {float((~B.dead).float().mean()):.3f} | fills/body {float(fills.mean()):.2f} | "
              f"relief memories {int(Mi.net.rel_n)}, harm {int(Mi.net.harm_n)} | value trust earned {Mi.wo.earned() if Mi.wo is not None else 0:.2f} | {time.time() - t0:.0f}s", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
