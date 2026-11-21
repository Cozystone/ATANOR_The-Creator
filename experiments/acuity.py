"""How often is a source that is truly within the field of view caught by the retina at all, by how far it is?  (gaze fixed at the middle; mean vs max in the rings)"""
import math, numpy as np
from atanor_os.emerge import room3d_live as R
from atanor_os.emerge.room3d import Room, WATER, FOOD
R.set_slots(R.CHROMA, 1 + R.CHROMA); R.Retina.OPPONENT = "rgb"
rng = np.random.default_rng(0)
for walls in (0, 5):
    rows = {"mean": [], "max": []}
    for i in range(400):
        w = Room(seed=9000 + i, walls=walls); free = np.argwhere(w.map == 0)
        w.pos = free[rng.integers(len(free))] + 0.5; w.head = float(rng.random() * 2 * math.pi)
        img = R.pictures([w])[0]
        vis = R.pictures([w], PW=256, PH=160, step=0.03)[0]
        for j, kind in ((2, WATER), (0, FOOD)):
            col = R.COLOR[kind]; px = np.abs(vis / np.maximum(vis.max(-1, keepdims=True), 1e-6) - col / col.max()).sum(-1) < 0.05
            if px.sum() < 3: continue                                   # not truly in view
            src = np.argwhere(w.map == kind)[0] + 0.5; d = float(np.hypot(*(src - w.pos)))
            off = (math.atan2(*(src - w.pos)[::-1]) - w.head + math.pi) % (2 * math.pi) - math.pi
            for pool in ("mean", "max"):
                R.Retina.POOL = pool; e = R.Retina(w); e.take(img)
                rows[pool].append((d, abs(off), e.now[j] is not None, abs(e.now[j][0] - off) if e.now[j] is not None else np.nan))
    for pool, r in rows.items():
        r = np.array(r, dtype=float)
        out = []
        for lo, hi in ((0, 3), (3, 6), (6, 10), (10, 30)):
            m = (r[:, 0] >= lo) & (r[:, 0] < hi) & (r[:, 1] > 0.12)       # outside the fovea
            out.append(f"{lo}-{hi} cells: caught {r[m, 2].mean():.2f} (n={int(m.sum())})")
        print(f"walls {walls} | rings by {pool:4s} | " + " | ".join(out) + f" | bearing error when caught (deg, median) {np.degrees(np.nanmedian(r[:, 3])):.1f}")
