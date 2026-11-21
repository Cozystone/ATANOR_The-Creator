"""머릿속 지도 기관 (1인칭 3D 세계의 4D 기억: 어디가 트였고 어디가 막혔는가, 그리고 언제 확인했는가). 크기 고정. 세계의 좌표도 이름도 읽지 않는다.

  채우는 길은 몸과 눈뿐이다:
    걸어 본 자리는 트임, 걸으려다 막힌 앞은 막힘 (몸),
    눈앞 그림에서 시선마다 "무언가가 서 있는 데까지"는 트임, 그 자리는 막힘 (눈. 거리는 보이는 높이에서 어림한다 - 크게 보일수록 가깝다).
  자리는 제 걸음과 돈 것을 더해 간 것이다 (태어난 자리가 가운데, 태어날 때 보던 쪽이 앞).
  쓰는 길은 속으로 걸어 보기 하나다 (가장 짧은 길을 셈하지 않는다):
    way(to)   - 떠올린 자리 쪽으로, 막힌 칸을 피해 여러 번 속으로 걸어 본다. 가장 잘 닿은 걸음의 몇 걸음 앞 자리를 돌려준다 ("저리로 가면 된다").
    way(None) - 갈 데를 모를 때. 아는 트인 칸들을 속으로 걸어 보다가 아직 모르는 칸과 맞닿은 데에 먼저 닿은 걸음을 고른다 ("저쪽은 아직 모른다").
"""
import math
import numpy as np

UNKNOWN, OPEN, BLOCKED = 0, 1, 2
MOVES = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
BACK = np.array([1, 0, 3, 2])


class Map:
    def __init__(self, size=65, walks=32, depth=36, seed=0):
        self.size, self.c = size, size // 2
        self.cell = np.zeros((size, size), dtype=np.int8)
        self.when = np.zeros((size, size), dtype=np.int32)
        self.beside = np.zeros((size, size), dtype=np.int16)                  # 그 칸 곁에 서 있었는데도 모르는 채였던 틱 수
        self.walks, self.depth = walks, depth
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.route = None

    def ix(self, p):
        return int(np.clip(round(float(p[0])) + self.c, 0, self.size - 1)), int(np.clip(round(float(p[1])) + self.c, 0, self.size - 1))

    # ---- 겪기 ----
    def stand(self, pos):
        self.t += 1
        i = self.ix(pos); self.cell[i] = OPEN; self.when[i] = self.t
        # 두 칸 안쪽 가까이에 있는데도 끝내 알 수 없는 칸은 들여다볼 수 없는 것이다 (속이 찬 것, 또는 갈 수 없는 데). 나중에 트인 것을 보면 고쳐진다
        x0, x1, y0, y1 = max(i[0] - 2, 0), min(i[0] + 3, self.size), max(i[1] - 2, 0), min(i[1] + 3, self.size)
        un = self.cell[x0:x1, y0:y1] == UNKNOWN
        self.beside[x0:x1, y0:y1][un] += 1
        gone = un & (self.beside[x0:x1, y0:y1] >= 15)
        self.cell[x0:x1, y0:y1][gone] = BLOCKED; self.when[x0:x1, y0:y1][gone] = self.t

    def bump(self, pos, head):
        i = self.ix(pos + 0.8 * np.array([math.cos(head), math.sin(head)]))
        if i != self.ix(pos):
            self.cell[i] = BLOCKED; self.when[i] = self.t

    def see(self, pos, head, offs, far, stops):
        """offs [K] 시선의 쪽, far [K] 그 시선이 트여 있는 데까지 (칸), stops [K] 그 끝에 무언가 서 있는가."""
        offs, far, stops = np.asarray(offs), np.minimum(np.asarray(far), 12.0), np.asarray(stops)
        dx, dy = np.cos(head + offs), np.sin(head + offs)
        ss = np.arange(0.5, 12.0, 0.5)[None]                                    # [1, S]
        on = ss < np.minimum(far[:, None] - 0.6, 6.0)                            # 먼 데는 어림이 거칠다: 여섯 칸 안쪽만 트였다고 적고
        ix = np.clip(np.rint(pos[0] + dx[:, None] * ss).astype(int) + self.c, 0, self.size - 1)[on]
        iy = np.clip(np.rint(pos[1] + dy[:, None] * ss).astype(int) + self.c, 0, self.size - 1)[on]
        keep = (self.cell[ix, iy] == BLOCKED) & (self.when[ix, iy] >= self.t - 5)      # 방금 막힘으로 적은 칸은 스쳐 본 것으로 지우지 않는다
        self.cell[ix[~keep], iy[~keep]] = OPEN; self.when[ix[~keep], iy[~keep]] = self.t
        st = stops & (far < 4.5)                                               # 네 칸 반 안쪽에 선 것만 막혔다고 적는다
        bx = np.clip(np.rint(pos[0] + dx * (far + 0.3)).astype(int) + self.c, 0, self.size - 1)[st]
        by = np.clip(np.rint(pos[1] + dy * (far + 0.3)).astype(int) + self.c, 0, self.size - 1)[st]
        here = self.ix(pos); me = (bx == here[0]) & (by == here[1])
        self.cell[bx[~me], by[~me]] = BLOCKED; self.when[bx[~me], by[~me]] = self.t

    # ---- 속으로 걸어 보기 ----
    def frontier(self):
        u = self.cell == UNKNOWN
        near = np.zeros_like(u)
        near[1:] |= u[:-1]; near[:-1] |= u[1:]; near[:, 1:] |= u[:, :-1]; near[:, :-1] |= u[:, 1:]
        return (self.cell == OPEN) & near

    def way(self, pos, to=None, ahead=3):
        """돌려주는 것: (갈 자리 [2] (제 걸음의 자리), 닿았는가). 갈 데가 없으면 (None, False)."""
        N, D = self.walks, self.depth
        start = np.array(self.ix(pos))
        p = np.tile(start, (N, 1)); last = np.full(N, -1)
        path = np.zeros((N, D, 2), dtype=np.int64); won = np.full(N, D + 1)
        if to is not None:
            goal = np.array(self.ix(to))
            if np.abs(goal - start).max() <= 1:
                return np.asarray(to, dtype=float), True
        else:
            fr = self.frontier()
            if not fr.any():                                                    # 모르는 데가 더 없으면, 확인한 지 가장 오래된 트인 칸들 쪽으로 (오래 안 본 곳을 다시 본다)
                op = self.cell == OPEN
                if op.sum() < 4:
                    return None, False
                fr = op & (self.when <= np.quantile(self.when[op], 0.2))
        best_d = np.full(N, 1e9)
        for k in range(D):
            nxt = p[:, None, :] + MOVES[None]                                   # [N, 4, 2]
            inb = (nxt >= 0).all(-1) & (nxt < self.size).all(-1)
            nc = nxt.clip(0, self.size - 1)
            st = self.cell[nc[..., 0], nc[..., 1]]
            ok = inb & (st != BLOCKED) if to is not None else inb & (st == OPEN)      # 떠올린 자리로 갈 때는 모르는 칸도 지나 본다. 둘러볼 때는 아는 트인 칸만 걷는다
            ok &= ~((np.arange(4)[None] == BACK[last.clip(0)][:, None]) & (last >= 0)[:, None])
            if to is not None:
                pref = -np.abs(nc - goal[None, None]).sum(-1).astype(float) + self.rng.random((N, 4)) * 2.5
            else:
                pref = self.rng.random((N, 4)) + 1.5 * (np.arange(4)[None] == last[:, None])      # 가던 쪽으로 이어 가는 편
            pref = np.where(ok, pref, -1e9)
            a = pref.argmax(1); can = ok[np.arange(N), a]
            p = np.where(can[:, None], nc[np.arange(N), a], p); last = np.where(can, a, -1)
            path[:, k] = p
            if to is not None:
                d = np.abs(p - goal[None]).sum(1); best_d = np.minimum(best_d, d + 0.01 * k)
                hit = d <= 1
            else:
                hit = fr[p[:, 0], p[:, 1]]
            won = np.where(hit & (won > D), k, won)
        if (won <= D).any():
            b = int(won.argmin()); k = int(min(ahead - 1, won[b]))
        elif to is not None:
            b = int(best_d.argmin()); k = ahead - 1
        else:
            return None, False
        self.route = (path[b, :int(min(won[b], D - 1)) + 1] - self.c).astype(float)      # 고른 걸음의 길 (제 걸음의 자리들). 부르는 쪽이 붙들고 따라간다
        return (path[b, k] - self.c).astype(float), bool(won[b] <= D)

    def copy(self):
        m = Map(self.size, self.walks, self.depth); m.cell, m.when, m.t, m.beside = self.cell.copy(), self.when.copy(), self.t, self.beside.copy()
        return m
