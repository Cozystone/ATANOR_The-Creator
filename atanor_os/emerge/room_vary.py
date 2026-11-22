"""여러 조건의 방 (창립자 2026-11-22: 방 환경도 다양한 조건에서 - 중요한 것은 돋아난 기질). 한 방에 맞춘 습관이 아니라 조건을 건너 이어지는 기질을 보려고.
make_room(seed): 크기·칸막이·문·샘 (물 w, 먹이 f) 자리·태어나는 자리 (s)가 씨앗마다 다른 방 (HOME과 같은 꼴의 글줄). drain은 따로 고른다.
"""
import numpy as np


def make_room(seed, size=None):
    r = np.random.default_rng(seed)
    S = int(size or r.integers(14, 26))
    g = np.full((S, S), ".", dtype="<U1"); g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = "#"
    for _ in range(int(r.integers(1, 4))):                       # 칸막이 벽 (가로 또는 세로), 문 1-2개
        if r.random() < 0.5:
            y = int(r.integers(3, S - 3)); g[y, 1:-1] = "#"
            for _ in range(int(r.integers(1, 3))):
                x = int(r.integers(1, S - 3)); g[y, x:x + int(r.integers(2, 4))] = "."
        else:
            x = int(r.integers(3, S - 3)); g[1:-1, x] = "#"
            for _ in range(int(r.integers(1, 3))):
                y = int(r.integers(1, S - 3)); g[y:y + int(r.integers(2, 4)), x] = "."
    for _ in range(int(r.integers(0, 6))):                       # 기둥 몇 개
        x, y = r.integers(2, S - 2, 2); g[y, x] = "#"
    floor = [(x, y) for y in range(1, S - 1) for x in range(1, S - 1) if g[y, x] == "."]
    r.shuffle(floor)
    nw, nf = int(r.integers(2, 6)), int(r.integers(2, 6))
    for (x, y) in floor[:nw]:
        g[y, x] = "w"
    for (x, y) in floor[nw:nw + nf]:
        g[y, x] = "f"
    sx, sy = floor[nw + nf]; g[sy, sx] = "s"
    return ["".join(row) for row in g]


def condition(seed):
    """방 하나의 조건: (글줄, drain). 모자람이 차는 빠르기도 조건마다 다르게 (1/180 ~ 1/360)."""
    r = np.random.default_rng(10_000 + seed)
    return make_room(seed), float(1.0 / r.uniform(180, 360))


if __name__ == "__main__":
    for s in range(3):
        rows, d = condition(s); print(f"seed {s}, drain 1/{1 / d:.0f}"); print("\n".join(rows)); print()
