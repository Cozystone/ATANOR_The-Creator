"""VO1 babbling (atanor_os/WORLD_A_PROTOCOL.md, VOICE): the mouth-ear loop alone. A mouth is a physical airway model driven by a few
values; the body's mouth gestures step those values; its ear hears its own voice (16 bands); a forecaster learns from lived babble
"heard now + felt posture + gesture -> heard next". Nothing is named or rewarded. CPU only.
  ATANOR_MOUTH=<path to the mouth program> PYTHONPATH=. python -m tools.voice_babble OUT.json [--bodies 8 --beats 3000]"""
import argparse, itertools, json, os, subprocess, time
import numpy as np, torch, torch.nn as nn

SR, BEAT = 48000, 9600
KEYS = ["tbX", "tbY", "lipAp", "pressure"]
LO = np.array([2.0, 5.8, 0.0, 0.0]); HI = np.array([3.8, 7.6, 1.2, 10.0]); STEP = (HI - LO) / 6.0
COMBOS = np.array(list(itertools.product((-1, 0, 1), repeat=4)), dtype=np.float32)   # 81 mouth gestures
_ED = np.geomspace(80, 8000, 17); _F = np.fft.rfftfreq(BEAT, 1 / SR); _WIN = np.hanning(BEAT)
_BANDS = [(_F >= _ED[i]) & (_F < _ED[i + 1]) for i in range(16)]


def ear(x):
    X = np.abs(np.fft.rfft(x * _WIN)) ** 2
    return np.clip(np.log10(np.array([X[b].sum() for b in _BANDS]) + 1e-10), -6.0, 6.0).astype(np.float32) / 6.0   # silence = -1


class Mouth:
    def __init__(self, exe):
        self.p = subprocess.Popen([exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE); self.v = (LO + HI) / 2
        self.say(gate=1, f0=140)

    def say(self, **kw):
        self.p.stdin.write((" ".join(f"{k}={v}" for k, v in kw.items()) + f" render={BEAT}\n").encode()); self.p.stdin.flush(); b = b""
        while len(b) < 4 * BEAT:
            b += self.p.stdout.read(4 * BEAT - len(b))
        self.last = np.frombuffer(b, dtype="<f4"); return self.last

    def put(self, v):
        self.v = np.clip(v, LO, HI); return ear(self.say(**{k: float(x) for k, x in zip(KEYS, self.v)}))

    def gesture(self, g):
        return self.put(self.v + COMBOS[g] * STEP)

    def felt(self):
        return ((self.v - LO) / (HI - LO) * 2 - 1).astype(np.float32)

    def close(self):
        self.p.stdin.close(); self.p.wait()


class Fore(nn.Module):                                                   # the forecast of its own mouth: what it will hear
    def __init__(self):
        super().__init__(); self.net = nn.Sequential(nn.Linear(16 + 4 + 81, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 16))

    def forward(self, e, f, g):
        return self.net(torch.cat([e, f, nn.functional.one_hot(g, 81).float()], 1))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--bodies", type=int, default=8); ap.add_argument("--beats", type=int, default=3000)
    ap.add_argument("--trials", type=int, default=200); ap.add_argument("--seed", type=int, default=9901); ap.add_argument("--demo", default="")
    a = ap.parse_args(); exe = os.environ["ATANOR_MOUTH"]; rng = np.random.RandomState(a.seed); torch.manual_seed(a.seed); t0 = time.time(); n = a.bodies
    M = [Mouth(exe) for _ in range(n)]; F = Fore(); opt = torch.optim.Adam(F.parameters(), 1e-3)
    e = np.stack([m.put(m.v) for m in M]); g = rng.randint(0, 81, n); hold = np.zeros(n, int); buf = []; test = []; cut = int(a.beats * 0.8); curve = []
    for t in range(a.beats):
        new = hold <= 0; g = np.where(new, rng.randint(0, 81, n), g); hold = np.where(new, rng.randint(1, 5, n), hold) - 1
        f = np.stack([m.felt() for m in M]); e2 = np.stack([m.gesture(int(g[i])) for i, m in enumerate(M)])
        (buf if t < cut else test).append((e.copy(), f, g.copy(), e2.copy())); e = e2
        if t < cut and len(buf) >= 8:                                    # learns as it lives: a few steps a beat on what it has lived so far
            for _ in range(2):
                idx = rng.randint(0, len(buf), 16); E_, F_, G_, Y_ = (np.concatenate([buf[i][k] for i in idx]) for k in range(4))
                loss = ((F(torch.from_numpy(E_), torch.from_numpy(F_), torch.from_numpy(G_).long()) - torch.from_numpy(Y_)) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        if (t + 1) % 500 == 0:
            curve.append(dict(t=t + 1, train_loss=round(float(loss), 4))); print("beat", t + 1, "loss", round(float(loss), 4), "s", round(time.time() - t0), flush=True)
    # test 1: knows its mouth (beats never trained on)
    E_, F_, G_, Y_ = (np.concatenate([q[k] for q in test]) for k in range(4)); mean_e = np.concatenate([q[3] for q in buf]).mean(0)
    with torch.no_grad():
        P = F(torch.from_numpy(E_), torch.from_numpy(F_), torch.from_numpy(G_).long()).numpy()
    mse = lambda A, B: float(((A - B) ** 2).mean()); moved = np.abs(Y_ - E_).mean(1) > 0.02
    dP, dY = P - E_, Y_ - E_; cosm = float(np.mean(((dP * dY).sum(1) / (np.linalg.norm(dP, axis=1) * np.linalg.norm(dY, axis=1) + 1e-9))[moved]))
    t1 = dict(test_beats=len(Y_), forecast=round(mse(P, Y_), 5), same_as_now=round(mse(E_, Y_), 5), average_sound=round(mse(np.broadcast_to(mean_e, Y_.shape), Y_), 5),
              when_the_sound_changed=dict(beats=int(moved.sum()), forecast=round(mse(P[moved], Y_[moved]), 5), same_as_now=round(mse(E_[moved], Y_[moved]), 5), direction_cosine=round(cosm, 3)))
    t1["under_both_by_30pct"] = bool(t1["forecast"] <= 0.7 * min(t1["same_as_now"], t1["average_sound"]))
    # test 2: drives its mouth when told (judge-side diagnostic)
    other = Mouth(exe); body = M[0]; res = {k: [] for k in ("own", "random", "hand")}
    for tr in range(a.trials):
        tv = LO + rng.rand(4) * (HI - LO); tv[3] = 5.0 + rng.rand() * 5.0     # the other's posture: one that sounds
        for _ in range(3):
            tgt = other.put(tv)
        clip = [other.last.copy()] * 3 + [np.zeros(BEAT // 2, np.float32)]
        sv = LO + rng.rand(4) * (HI - LO)
        for arm in res:
            for _ in range(2):
                e0 = body.put(sv)
            d0 = float(np.linalg.norm(e0 - tgt)); ec = e0
            for k in range(12):
                if arm == "own":
                    with torch.no_grad():
                        pr = F(torch.from_numpy(np.repeat(ec[None], 81, 0)), torch.from_numpy(np.repeat(body.felt()[None], 81, 0)), torch.arange(81)).numpy()
                    ec = body.gesture(int(np.argmin(((pr - tgt) ** 2).sum(1)))); clip.append(body.last.copy())
                elif arm == "random":
                    ec = body.gesture(int(rng.randint(0, 81)))
                else:
                    ec = body.put(body.v + np.clip(tv - body.v, -STEP, STEP))
            res[arm].append((d0, float(np.linalg.norm(ec - tgt))))
        if a.demo and tr < 8:                                            # to listen: the other's sound, a pause, then the body's twelve beats toward it
            import wave
            y = np.concatenate(clip); w_ = wave.open(f"{a.demo}/told_{tr}.wav", "wb"); w_.setnchannels(1); w_.setsampwidth(2); w_.setframerate(SR)
            w_.writeframes((np.clip(y / 0.6, -1, 1) * 32767).astype("<i2").tobytes()); w_.close()
    closed = {k: float(np.median([(d0 - d1) / max(d0, 1e-6) for d0, d1 in v])) for k, v in res.items()}
    t2 = dict(trials=a.trials, start_distance_median=round(float(np.median([d0 for d0, _ in res["own"]])), 3), end_distance_median={k: round(float(np.median([d1 for _, d1 in v])), 3) for k, v in res.items()},
              share_of_distance_closed_median={k: round(v, 3) for k, v in closed.items()})
    t2["drives_when_told"] = bool(closed["own"] >= 0.5 * closed["hand"] and closed["random"] < 0.25 * closed["hand"])
    for m in M + [other]:
        m.close()
    r = dict(protocol="atanor_os/WORLD_A_PROTOCOL.md VO1", bodies=n, beats=a.beats, seed=a.seed, gestures=81, values=KEYS, lived_body_beats=n * cut, knows_its_mouth=t1, drives_when_told=t2, curve=curve, seconds=round(time.time() - t0))
    print(json.dumps(r), flush=True); json.dump(r, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
