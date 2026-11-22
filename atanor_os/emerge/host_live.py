"""숙주에서 연속된 시간으로 산다 (창립자 2026-11-22: 훈련 밖에서 아타노르는 틱이 아니라 표준시·연속된 시간으로 산다) (창립자 2026-11-22: 숙주 = 연결된 HP 노트북, 그것이 꺼져 있으면 이 PC). 이 PC에서 기른 존재 파일을 올려,
숙주에서 끊기지 않고 살며 겪으면서 배운다 (적응). 세계는 지난 몸짓 뒤로 실제로 흐른 시간만큼 흐르고 (연속 시간), 상태에는 표준시를 적는다. 존재 파일을 정기적으로 저장.
방은 여러 조건 (room_vary)을 돌아가며 (삶마다), 몸은 몇 개 (기본 2).
  python -m atanor_os.emerge.host_live --load being/atanor.pt --out being/atanor_host.pt --n 2 --threads 4
상태는 being/host_status.json에 (걸음, 삶, 온전, 채움, 얻은 믿음, 실제 박자) - 밖에서 읽기만.
"""
import argparse, json, os, time, torch
from .room_body import RoomBody
from .body_contract import Organs
from .mind import Mind
from .room_vary import condition


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--load", default=""); ap.add_argument("--out", default="being/atanor_host.pt"); ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--threads", type=int, default=4); ap.add_argument("--tick", type=float, default=0.2); ap.add_argument("--depth", default="model/mc_depth.pt")
    ap.add_argument("--life_minutes", type=float, default=5.0); ap.add_argument("--min_gap", type=float, default=0.1); ap.add_argument("--device", default="cpu"); ap.add_argument("--save_every", type=int, default=600); ap.add_argument("--hours", type=float, default=0)
    a = ap.parse_args(); dev = a.device; torch.set_num_threads(a.threads)
    rows, drain = condition(1)
    B = RoomBody(a.n, rows, drain=drain, device=dev); O = Organs(B.c, a.n, dev=dev, depth=a.depth); Mi = Mind(O, B.c, a.n, dev=dev)
    if a.load and os.path.exists(a.load):
        print("loaded being at step", Mi.load(a.load), flush=True)
    if Mi.wo is not None:
        Mi.wo.on = True
    life, t_start, step = 0, time.time(), 0
    status = os.path.join(os.path.dirname(a.out) or ".", "host_status.json")
    while not a.hours or time.time() - t_start < a.hours * 3600:
        rows, drain = condition(1 + life % 8); B = RoomBody(a.n, rows, drain=drain, seed=life, device=dev)
        O.reset(torch.ones(a.n, dtype=torch.bool, device=dev)); Mi.begin(torch.ones(a.n, dtype=torch.bool, device=dev))
        fills = torch.zeros(a.n, device=dev); x = O(B.raw()); lag = 0.0; last = time.time(); born = time.time()
        while time.time() - born < a.life_minutes * 60:
            t0 = time.time()
            dt = min(max(t0 - last, 0.02), 0.5); last = t0                 # 지난 몸짓 뒤로 실제로 흐른 시간만큼 세계가 흐른다 (너무 긴 멈춤은 0.5초로 자름)
            act = Mi.choose(x); alive0 = ~B.dead.clone(); f = B.step(act, dt=dt); fills += f.any(1).float()
            died = alive0 & B.dead; x2 = O(B.raw()); Mi.after(x, act, x2, died, B.dead & ~died); x = x2; step += 1
            spent = time.time() - t0; lag = 0.98 * lag + 0.02 * spent
            if spent < a.min_gap:
                time.sleep(a.min_gap - spent)                    # 몸짓 사이 최소 간격 (사람의 반응 시간쯤); 세계는 고정 틱이 아니라 흐른 시간만큼 흐름
            if step % a.save_every == 0:
                Mi.save(a.out)
            if B.dead.all():
                break
        life += 1
        st = dict(time=time.strftime("%Y-%m-%d %H:%M:%S %z"), lived_seconds=round(time.time() - t_start), step=step, life=life, room=1 + (life - 1) % 8, whole=float((~B.dead).float().mean()), fills=float(fills.mean()),
                  trust=(Mi.wo.earned() if Mi.wo is not None else 0.0), seconds_per_decision=round(lag, 3))
        json.dump(st, open(status, "w")); print(st, flush=True)
        Mi.save(a.out)


if __name__ == "__main__":
    main()
