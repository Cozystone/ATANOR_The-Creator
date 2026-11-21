"""길의 기억 (빠른 저장소). 세계의 이름이 없다. 크기 고정.

제가 걸어 본 길을 걸음마다 한 번에 적는다: 어디였나 (제 걸음을 더해 간 자리), 그다음 어디로 갔나, 그리고 거기서부터 얼마 뒤에 어느 모자람이 풀렸나.
모자랄 때: 지금 자리 가까이를 지나간 적 있는 길 가운데, 그 모자람이 가장 빨리 풀렸던 길을 떠올려 그 길을 몇 걸음 앞까지 따라간 자리를 돌려준다 ("전에 이리로 갔더니 풀렸다").
신경망을 고치지 않는다 - 적기만 한다. 그래서 살수록 쌓이고, 앞서 배운 것을 잊지 않는다. 가장 짧은 길을 셈하지 않는다 (걸어 본 길을 다시 걸을 뿐이다).
풀리던 자리에 와서 한참 있어도 풀리지 않으면 그 자리로 가는 길들은 한동안 떠올리지 않는다. 얼마나 지나면 다시 떠올릴 만한지는 겪은 것에서 안다
(비었던 자리에서 다시 풀리기까지 걸린 시간들의 가운데값. 겪기 전에는 다시 떠올리지 않는다).
"""
import numpy as np


class Routes:
    def __init__(self, size=6000, kinds=2, stride=0.5):
        self.size, self.kinds, self.stride = size, kinds, stride
        self.pos = np.zeros((size, 2)); self.t = np.zeros(size, dtype=np.int64)
        self.ser = -np.ones(size, dtype=np.int64); self.nxt = -np.ones(size, dtype=np.int64); self.nser = -np.ones(size, dtype=np.int64)
        self.ttr = np.full((size, kinds), np.inf); self.goal = np.zeros((size, kinds, 2)); self.won = np.zeros((size, kinds), dtype=np.int64)
        self.count, self.last = 0, -1
        self.open = [[] for _ in range(kinds)]                     # 이 삶에서 그 모자람이 마지막으로 풀린 뒤에 적은 칸들 (다음 풀림 때 채운다)
        self.empty = [[] for _ in range(kinds)]                    # [풀리던 자리, 비었다고 여긴 때]
        self.back = []                                             # 비었던 자리에서 다시 풀리기까지 걸린 시간들
        self.there = np.zeros(kinds, dtype=np.int64)
        self.used = 0

    def newlife(self):
        self.last = -1; self.open = [[] for _ in range(self.kinds)]; self.there[:] = 0

    # ---- 적기 ----
    def walk(self, pos, t):
        if self.last >= 0 and np.hypot(*(pos - self.pos[self.last])) < self.stride:
            return
        i = self.count % self.size
        self.pos[i], self.t[i], self.ser[i] = pos, t, self.count
        self.nxt[i] = self.nser[i] = -1; self.ttr[i] = np.inf
        if self.last >= 0:
            self.nxt[self.last], self.nser[self.last] = i, self.count
        for j in range(self.kinds):
            self.open[j].append((i, self.count))
        self.last = i; self.count += 1

    def relief(self, j, pos, t):
        for i, ser in self.open[j]:
            if self.ser[i] == ser:
                self.ttr[i, j] = t - self.t[i]; self.goal[i, j] = pos; self.won[i, j] = t
        self.open[j] = []; self.there[j] = 0
        for e in self.empty[j]:                                    # 비었다고 여겼던 자리에서 다시 풀렸다: 그 사이를 든다
            if np.hypot(*(e[0] - pos)) < 1.5:
                self.back = (self.back + [t - e[1]])[-32:]
        self.empty[j] = [e for e in self.empty[j] if np.hypot(*(e[0] - pos)) >= 1.5]

    # ---- 떠올리기 ----
    def _usable(self, j, t):
        wait = float(np.median(self.back)) if len(self.back) >= 3 else np.inf
        ok = np.isfinite(self.ttr[:, j]) & (self.ser >= 0)
        for place, when in self.empty[j]:
            if t - when < wait:
                ok &= ~((np.hypot(self.goal[:, j, 0] - place[0], self.goal[:, j, 1] - place[1]) < 1.5) & (self.won[:, j] <= when))
        return ok

    def recall(self, pos, j, t, ahead=5, near=1.5, linger=25):
        """돌려주는 것: 갈 자리 [2] 또는 None."""
        ok = self._usable(j, t)
        d = np.hypot(self.pos[:, 0] - pos[0], self.pos[:, 1] - pos[1])
        c = np.flatnonzero(ok & (d < near))
        if not len(c):
            return None
        i = int(c[np.argmin(self.ttr[c, j] + 4.0 * d[c])])
        if np.hypot(*(self.goal[i, j] - pos)) < 1.2:               # 풀리던 그 자리에 와 있다
            self.there[j] += 1
            if self.there[j] > linger:                             # 한참 있어도 풀리지 않는다: 이 자리로 가는 길은 한동안 떠올리지 않는다
                self.empty[j].append([self.goal[i, j].copy(), t]); self.there[j] = 0
                return self.recall(pos, j, t, ahead, near, linger)
            return self.goal[i, j].copy()
        for _ in range(ahead):                                     # 그 길을 몇 걸음 앞까지 따라간다
            k = self.nxt[i]
            if k < 0 or self.ser[k] != self.nser[i] or not np.isfinite(self.ttr[k, j]) or self.ttr[k, j] > self.ttr[i, j]:
                break
            i = int(k)
        self.used += 1
        return self.pos[i].copy() if np.hypot(*(self.pos[i] - pos)) > 0.3 else self.goal[i, j].copy()

    def copy(self):
        import copy
        return copy.deepcopy(self)
