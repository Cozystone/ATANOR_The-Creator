"""Watch the main line (gate 1 adopted: homing vector + own goal) live in one body of world A2 (food moves between three corners every
`period` beats), on the room viewer (port 8768). Judge-side display only; the being keeps living and learning in this body.
  PYTHONPATH=. python -m tools.a2_view 9104_0 s3_b"""
import sys, time
import numpy as np, torch
from tools.need_fork import t_room, place, DEV
from tools.a1_channel import V2
from tools.l1_life import make
from tools.s4_kin import being, rebirth
from tools import room_live_view as VIEW
from atanor_os.emerge.room_body import RoomBody
from atanor_os.emerge.room3d_gpu import FLOOR, FOOD

lay = sys.argv[1] if len(sys.argv) > 1 else "9104_0"; path = sys.argv[2] if len(sys.argv) > 2 else "s3_b"
period = int(sys.argv[3]) if len(sys.argv) > 3 else 300
EYE_H = float(sys.argv[4]) if len(sys.argv) > 4 else 1.1                 # eye height (founder 2026-11-27: 1.1)
seed, mirror = map(int, lay.split("_")); assert not 7701 <= seed <= 7712
V2["on"] = True; RoomBody.default_cues = True; RoomBody.default_eye = EYE_H; VIEW.start(); n = 1
rows, info = t_room(seed, bool(mirror)); torch.manual_seed(seed * 10 + mirror + 5)
g = np.array([list(r) for r in rows]); H, W = g.shape
f0 = tuple(int(v) for v in np.argwhere(g == "f")[0][::-1]); corners = [f0, (1, 1), (W - 2, 1)]
B = make(rows, n, seed, V2["drain"]); O, Mi = being(B, n, path); w = B.w; G = Mi.going
NAME = ["다리 끝", "왼쪽 위 구석", "오른쪽 위 구석"]


def put(k):
    c = corners[k]; old = w.src[:, 1].clone()
    w.map[torch.arange(n, device=DEV), old[:, 0], old[:, 1]] = FLOOR
    w.map[:, c[0], c[1]] = FOOD; w.src[:, 1] = torch.tensor(c, device=DEV); w.spots[FOOD] = torch.tensor([c], device=DEV)


one = torch.ones(1, dtype=torch.bool, device=DEV); B.w.store[:] = 0.75; place(B, info); O.reset(one); Mi.begin(one); Mi.prev = None
put(0); x = O(B.raw()); k = 0; t = 0; deaths = 0; got = [0, 0]; since = -1; waits = []
while True:
    if t % period == 0:
        k = (t // period) % 3; put(k)
    p_was, h_was = w.pos.clone(), w.head.clone()
    act = Mi.choose(x); al = ~B.dead.clone(); f = B.step(act); r = B.raw(); x2 = O(r); died = al & B.dead
    fl = f[0].cpu().numpy().astype(int); got[0] += fl[0]; got[1] += fl[1]
    st = B.w.store[0].cpu().numpy()
    if st[1] < 0.7 and since < 0:
        since = t
    if fl[1] and since >= 0:
        waits.append(t - since); since = -1
    gp = getattr(Mi, "goal_prev", None); goal = int(gp[0]) if gp is not None else -1
    Mi.after(x, act, x2, died, B.dead & ~died); x = x2; t += 1
    if bool(died.any()):
        deaths += 1; since = -1; rebirth(B, O, Mi, info); B.felt[:] = 0; x = O(B.raw())
    if t % 1000 == 0:
        G.learn()
        with torch.enable_grad():
            Mi.wo.learn(n=max(1, 2500 // max(1, getattr(Mi.wo, "per_call", 1))))
    if True:                                                            # display only: four in-between pictures per beat, at the world's own tempo (a beat = 0.2 s)
        sk = B.w.stock[0].cpu().numpy(); left = period - t % period
        if goal >= 0 and getattr(G, "Pu", None) is not None:
            aim = "제 목표 있음 (스스로 믿는 자리로 귀가 벡터가 걸음)"
        else:
            aim = "목표 없음 (부족 .3 이하 - 평소 고름)"
        med = f"{int(np.median(waits[-20:]))}" if waits else "-"
        p_now, h_now = w.pos.clone(), w.head.clone(); dh = (h_now - h_was + np.pi) % (2 * np.pi) - np.pi; jump = bool((p_now - p_was).norm() > 1.0)
        for k_ in range(1, 5):
            al_ = 1.0 if jump else k_ / 4
            w.pos = p_was + (p_now - p_was) * al_; w.head = (h_was + dh * al_) % (2 * np.pi)
            eye_ = B.render()[0].cpu().numpy().copy(); pos_ = w.pos[0].cpu().numpy(); hd_ = float(w.head[0])   # the being's own picture (its renderer, its wall cues)
            ret_ = O.ret[O.img[0]]                                         # its own eye (canon fovea): where it fixates this beat - centre and the four rings
            for j_, bx in enumerate(ret_.boxes):
                x0_, y0_, hh, ww = int(bx[0][0]), int(bx[1][0]), int(bx[2]), int(bx[3]); Hh, Ww = eye_.shape[:2]
                c_ = np.array([255, 255, 0]) if j_ == 0 else np.array([0, 230, 255]) * (1.0 - 0.18 * j_)
                x1_, y1_ = min(x0_ + ww, Ww) - 1, min(y0_ + hh, Hh) - 1
                eye_[y0_, x0_:x1_ + 1] = c_; eye_[y1_, x0_:x1_ + 1] = c_; eye_[y0_:y1_ + 1, x0_] = c_; eye_[y0_:y1_ + 1, x1_] = c_
            w.pos, w.head = p_now.clone(), h_now.clone()
            fx, fy = corners[k]; rows_now = [("".join("." if ch == "f" else ch for ch in r_)) for r_ in rows]
            rows_now[fy] = rows_now[fy][:fx] + "f" + rows_now[fy][fx + 1:]      # the map shows where food is now (it moves), not where it started
            VIEW.push(eye_, rows_now, pos_, hd_, st, sk,
                  f"관문 1 주선 존재 (귀가 벡터 + 제 목표) - 세계 A2, 방 {lay}, 눈높이 {EYE_H}칸, 박자 {t}",
                  f"물 {st[0]:.2f}  먹이 {st[1]:.2f} | 먹이는 지금 {NAME[k]} ({left}박자 뒤 옮김) | {aim}" + chr(10)
                  + f"채움 물 {got[0]} / 먹이 {got[1]} | 죽음 {deaths} | 배고픔->먹이 대기 (최근 20번 중앙값) {med}박자" + chr(10)
                  + "지도는 판정 쪽만 봄 (파랑 물, 주황 먹이). 존재는 왼쪽의 제 눈만 봄. 64몸 삶을 마친 마음이 한 몸으로 이어 삶." + chr(10)
                  + "화면은 박자 사이를 이어 그림 (보기만 부드럽게; 존재의 몸짓은 그대로 한 박자 0.2초에 하나)." + chr(10)
                  + "왼쪽은 존재 자신의 그림. 노란 네모 = 중심와 (가장 자세히 봄), 하늘색 네모들 = 주변 고리 넷 (멀수록 거칠게). 둘째 길 (전체 그림 80x48, 깊이)은 화면 전체.")
            time.sleep(0.04)
