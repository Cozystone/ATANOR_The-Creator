"""Watch the main line (gate 1 adopted: homing vector + own goal) live in one body of world A2 (food moves between three corners every
`period` beats), on the room viewer (port 8768). Judge-side display only; the being keeps living and learning in this body.
  PYTHONPATH=. python -m tools.a2_view 9104_0 s3_b"""
import os, sys, time
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
S5 = lay.startswith("s5")                                               # "s5_9501" = the strong detour room (born in a pocket that opens away from the food)
A3 = lay.startswith("a3") or S5                                               # gate 2 run 3 (2026-11-27): "a3_9301" = world A3 (two rooms, springs out of view), food does not move
if A3:
    seed, mirror = int(lay.split("_")[1]), 0; period = 10 ** 9
else:
    seed, mirror = map(int, lay.split("_"))
assert not 7701 <= seed <= 7712
V2["on"] = True; RoomBody.default_cues = True; RoomBody.default_eye = EYE_H; VIEW.start(); n = 1
if A3:
    if S5:
        from tools.s5_space import layout_strong
        rows, info = layout_strong()
    else:
        from tools.a3_rooms import layout as a3_layout
        rows, info, _ = a3_layout()
else:
    rows, info = t_room(seed, bool(mirror))
torch.manual_seed(seed * 10 + mirror + 5)
g = np.array([list(r) for r in rows]); H, W = g.shape
f0 = tuple(int(v) for v in np.argwhere(g == "f")[0][::-1]); corners = [f0, (1, 1), (W - 2, 1)]
B = make(rows, n, seed, V2["drain"]); O, Mi = being(B, n, path); w = B.w; G = Mi.going
NAME = (["정면 벽 너머 위쪽 (물은 아래쪽)"] * 3 if S5 else ["오른쪽 방 먼 구석 (물은 왼쪽 방 먼 구석)"] * 3) if A3 else ["다리 끝", "왼쪽 위 구석", "오른쪽 위 구석"]
if A3:
    corners = [f0] * 3


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
    # telemetry (Copilot / founder 2026-11-27: inner state as a few plain axes, not "brain waves"): rolling over recent beats, body 0
    if "TL" not in dir():
        from collections import deque
        TL = dict(pos=deque(maxlen=16), stuck=deque(maxlen=100), again=deque(maxlen=100), cells=deque(maxlen=100), mm=deque(maxlen=300), mm_last=0)
    TL["pos"].append(w.pos[0].cpu().numpy().copy()); P_ = np.array(TL["pos"])
    TL["stuck"].append(1.0 if len(P_) >= 16 and np.linalg.norm(np.diff(P_, axis=0), axis=1).sum() > 3 and np.linalg.norm(P_[-1] - P_[0]) < 1 else 0.0)
    TL["again"].append(float(Mi.again[0])); TL["cells"].append(tuple(w.pos[0].long().tolist()))
    mmn = getattr(Mi, "mm_n", 0); TL["mm"].append(mmn - TL["mm_last"]); TL["mm_last"] = mmn
    SRC_ = ["강한 끌림", "값", "무작위", "하던 것 유지", "시도"][int(Mi.src[0])]
    lw_ = getattr(Mi, "learned_home", None); trust_ = f"{lw_.w:.2f}" if lw_ is not None else "-"
    if "TV" not in dir():
        from tools import tele_view as TV
        TV.start()
    ret0 = O.ret[O.img[0]]; att_ = 0.0
    if getattr(ret0, "attend", False) and getattr(ret0, "phi", None) is not None:
        att_ = float(((ret0.phi[0] - ret0.ab) * ret0.aw).sum(-1).clamp(min=0).sum() > 0)
    TV.push(dict(t=t, trust=float(lw_.w) if lw_ is not None else 0.0, lack=float(1 - min(st)), stuck=float(np.mean(TL["stuck"])), again=float(np.mean(TL["again"])),
                 explore=min(1.0, len(set(TL["cells"])) / 30.0), mismatch=min(1.0, sum(TL["mm"]) / 10.0), attend=att_), label=f"ATANOR 속 상태 - 박자 {t}" + (f" | 장소 다시 배치 (R1) {getattr(G, 'remaps', 0)}번" if os.environ.get("ATANOR_REMAP", "0") == "1" else "") + (f" | 눈으로 본 지도: 제 눈의 틀림 {getattr(G, 'sm_err', 9):.1f}칸 (1칸 아래면 지도에 적음), 지금 앞이 트였다고 보는 거리 {float(G.sm_last[0]) if getattr(G, 'sm_last', None) is not None else 0:.1f}칸, 적은 횟수 {getattr(G, 'sm_writes', 0)} | 눈으로 본 샘: 맞힌 비율 {getattr(G, 'sr_hit', 0):.2f} (.7 넘으면 기억에 적음), 지금 물/먹이가 앞에 있다고 보는 정도 {float(G.sr_last[0, 0]) if getattr(G, 'sr_last', None) is not None else 0:.2f}/{float(G.sr_last[0, 1]) if getattr(G, 'sr_last', None) is not None else 0:.2f}, 적은 횟수 {getattr(G, 'sr_writes', 0)}" if os.environ.get("ATANOR_SEEN", "0") == "1" else "") + (f" | 감: 여정 {getattr(Mi, 'g_ends', 0)}번 끝남, 목표 물음 {getattr(Mi, 'g_asked', 0)}번 중 감이 아는 것 {getattr(Mi, 'g_known', 0)}, 감이 기억과 달리 고름 {getattr(Mi, 'g_diff', 0)}" if os.environ.get("ATANOR_GUT", "0") == "1" else "") + (f" | R2 읽는 세계 묶음 {getattr(G, 'r2_read', -1)} (-1 = 아직 모름, 빈 묶음) | 칸 재활용 {int(G.r2.moves)} | 묶음 갈아씀 {int(G.r2.forgotten)}" if getattr(G, "r2", None) is not None else ""))
    if t % 25 == 0 and getattr(G, "Pu", None) is not None:              # the being's own map (its place units by its own path sense)
        with torch.no_grad():
            Vn_ = G.Pn; seen_ = torch.nonzero(Vn_ > 0)[:, 0].tolist()
            V_ = G.V if getattr(G, "V", None) is not None else torch.zeros(G.KU, 2, device=DEV)
            vm = V_.clamp(min=0); vm = (vm / vm.max(0).values.clamp(min=1e-6)).cpu().numpy()
            units = [[float(G.Pu[i, 0]), float(G.Pu[i, 1]), float(Vn_[i]), float(vm[i, 0]), float(vm[i, 1])] for i in seen_]
            idx = {u: k for k, u in enumerate(seen_)}; S_ = (G.T + G.T.t()).cpu().numpy()
            links = [[idx[a], idx[b]] for a in seen_ for b in seen_ if a < b and S_[a, b] > 0.5]
            me = O.path.p[0].cpu().numpy().tolist(); gk = idx.get(goal, -1)
            wm_ = getattr(G, "wmap", None); wl = []
            if wm_ is not None:
                wl = [[float(i_ - 32), float(j_ - 32)] for i_, j_ in torch.nonzero(wm_ > 1.0).tolist()]   # IM1: the walls in its own map
        TV.push(None, dict(units=units, links=links, me=me, goal=gk, walls=wl))
    tele = (f"[속] 귀가 믿음 {trust_} | 모자람 {float(1 - min(st)):.2f} | 고착(최근 100박자 맴돎) {np.mean(TL['stuck']):.0%} | 다시 생각 {np.mean(TL['again']):.0%}"
            f" | 탐색(최근 100박자 칸 수) {len(set(TL['cells']))} | 어긋남 (최근 300박자) {sum(TL['mm'])}번 | 이 몸짓을 정한 것: {SRC_}")
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
                  ((f"강한 우회 방 (주머니 안에서 태어남: 먹이에서 먼저 멀어져야 감), 시작 {seed}, 눈높이 {EYE_H}칸, 박자 {t}" if S5 else f"관문 2 - 세계 A3 (두 방, 샘은 문 너머 안 보임), 시작 {seed}, 눈높이 {EYE_H}칸, 박자 {t}") if A3 else f"관문 1 주선 존재 (귀가 벡터 + 제 목표) - 세계 A2, 방 {lay}, 눈높이 {EYE_H}칸, 박자 {t}"),
                  (f"물 {st[0]:.2f}  먹이 {st[1]:.2f} | 먹이는 {NAME[k]} | {aim}" if A3 else f"물 {st[0]:.2f}  먹이 {st[1]:.2f} | 먹이는 지금 {NAME[k]} ({left}박자 뒤 옮김) | {aim}") + chr(10)
                  + f"채움 물 {got[0]} / 먹이 {got[1]} | 죽음 {deaths} | 배고픔->먹이 대기 (최근 20번 중앙값) {med}박자" + chr(10)                  + tele + chr(10)
                  + "지도는 판정 쪽만 봄 (파랑 물, 주황 먹이). 존재는 왼쪽의 제 눈만 봄. 64몸 삶을 마친 마음이 한 몸으로 이어 삶." + chr(10)
                  + "화면은 박자 사이를 이어 그림 (보기만 부드럽게; 존재의 몸짓은 그대로 한 박자 0.2초에 하나)." + chr(10)
                  + "왼쪽은 존재 자신의 그림. 노란 네모 = 중심와 (가장 자세히 봄), 하늘색 네모들 = 주변 고리 넷 (멀수록 거칠게). 둘째 길 (전체 그림 80x48, 깊이)은 화면 전체.")
            time.sleep(0.04)
