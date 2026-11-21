import math, numpy as np
from atanor_os.emerge import room3d_live as R
from atanor_os.emerge.room3d import Room, WATER, FOOD
R.set_slots(R.CHROMA, 1 + R.CHROMA); R.Retina.OPPONENT = "rgb"
rng = np.random.default_rng(1)
for pool in ("mean", "max"):
    for reflex in (False, True):
        R.Retina.POOL, R.Retina.reflex = pool, reflex; rows = []
        for i in range(500):
            w = Room(seed=30000 + i, walls=0); w.map[w.map == FOOD] = 0
            src = np.argwhere(w.map == WATER)[0] + 0.5; free = np.argwhere(w.map == 0); w.pos = free[rng.integers(len(free))] + 0.5
            d = src - w.pos; dist = float(np.hypot(*d))
            if dist < 1.3: continue
            off = (rng.random() - 0.5) * 1.1; w.head = math.atan2(d[1], d[0]) - off
            e = R.Retina(w); img = R.pictures([w])[0]
            for k in range(3): e.take(img)
            if e.now[2] is None: rows.append((dist, np.nan, np.nan)); continue
            rows.append((dist, abs(1 / e.now[2][1] + 0.5 - dist), abs(math.degrees(e.now[2][0] - off))))
        r = np.array(rows)
        print(f"rings by {pool:4s} | {'eye reflex ' if reflex else 'fovea fixed'} | " + " | ".join(f"{lo}-{hi}: dist err {np.nanmedian(r[(r[:,0]>=lo)&(r[:,0]<hi),1]):.2f}, bearing err {np.nanmedian(r[(r[:,0]>=lo)&(r[:,0]<hi),2]):.1f} deg, missed {np.isnan(r[(r[:,0]>=lo)&(r[:,0]<hi),1]).mean():.2f}" for lo, hi in ((1.3, 3), (3, 6), (6, 10), (10, 30))))
