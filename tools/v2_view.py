"""Watch the V2 being (latent ReliefWorth value) live in one body of the cued T room, on the room viewer (port 8768). Judge-side display only.
It keeps living and learning in this body; each death is a rebirth at the start. Starts with the mind V2 saved after its 64-body life.
  PYTHONPATH=. python -m tools.v2_view 9101_0"""
import sys, time
import numpy as np, torch
from tools.need_fork import t_room, place, DEV, DEPTH
from tools.a1_channel import V2
from tools import room_live_view as VIEW
from atanor_os.emerge.room_body import RoomBody
from atanor_os.emerge.body_contract import Organs
from atanor_os.emerge.mind import Mind

lay = sys.argv[1] if len(sys.argv) > 1 else "9101_0"; seed, mirror = map(int, lay.split("_"))
V2["on"] = True; RoomBody.default_cues = True; VIEW.start()
rows, info = t_room(seed, bool(mirror)); root = sys.argv[2] if len(sys.argv) > 2 else "a1cache_many_relief"; name = sys.argv[3] if len(sys.argv) > 3 else "V2: 잠재 학습 값 (ReliefWorth)"
ck = f"{root}/{lay}/mind.pt"
B = RoomBody(1, rows, drain=V2["drain"], seed=seed, device=DEV, regrow=V2["regrow"]); O = Organs(B.c, 1, dev=DEV, depth=DEPTH)
Mi = Mind(O, B.c, 1, dev=DEV, value_arm="relief"); Mi.load(ck); Mi.wo.on = True
one = torch.ones(1, dtype=torch.bool, device=DEV); B.w.store[:] = 0.75; place(B, info); O.reset(one); Mi.begin(one); x = O(B.raw())
got = np.zeros(2, int); need = np.zeros(2, int); deaths = 0; t = 0; SRC = ["강한 끌림", "값", "무작위", "유지", "시도"]
while True:
    s0 = B.w.store[0].cpu().numpy().copy(); act = Mi.choose(x); alive0 = ~B.dead.clone(); f = B.step(act); fl = f[0].cpu().numpy().astype(int)
    for c in (0, 1):
        if fl[c]:
            got[c] += 1; need[c] += int(s0[c] < s0[1 - c])
    died = alive0 & B.dead; r = B.raw(); x2 = O(r); Mi.after(x, act, x2, died, B.dead & ~died); x = x2; t += 1
    if t % 2 == 0:
        st = B.w.store[0].cpu().numpy(); sk = B.w.stock[0].cpu().numpy()
        VIEW.push(r[-1][0].cpu().numpy(), rows, B.w.pos[0].cpu().numpy(), float(B.w.head[0]), st, sk,
                  f"{name}을 지닌 존재 - 방 {lay}, 단서 있음, 박자 {t}",
                  f"물 {st[0]:.2f}  먹이 {st[1]:.2f} | 채움 물 {got[0]} (모자랄 때 {need[0]}) / 먹이 {got[1]} (모자랄 때 {need[1]}) | 죽음 {deaths} | 이 박자: {SRC[int(Mi.src[0])]}"
                  + chr(10) + "지도는 판정 쪽만 봄 (파랑 물, 주황 먹이). 존재는 왼쪽의 제 눈만 봄. 64몸 삶을 마친 마음이 한 몸으로 이어 삶.")
    if B.dead.all():
        deaths += 1; B.w.dead[:] = False; B.w.store[:] = 0.75; place(B, info); O.reset(one); Mi.begin(one); x = O(B.raw())
    time.sleep(0.05)
