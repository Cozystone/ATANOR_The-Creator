"""LOKI Memory v4 (github Cozystone/LOKI @ 24e2313) as an ATANOR organ, bound to the universal body contract (2026-11-23).

What the organ is given (contract only - no device names, no judge wiring, no option meanings):
  * per sensor slot: kind, shape, bedrock flag (Contract.sensors), per motor branch: its option count (Contract.motors),
  * the action actually done, as the combo index the body executed -> one-hot per branch (no ordinal meaning, no 'option 0 = rest'),
  * an availability flag per slot from the plumbing (a dead nerve: camera not delivering, mic not delivering) - no names.

Feeling -> LOKI range [0, 1] (LOKI.observe rejects anything else):
  * bedrock dims: the raw bedrock value, already 0..1 (1 = full), clipped.
  * other vector / audio dims: ATANOR's own feeling x (Stat, mean and spread over time and bodies - the 2026-11-23 single-body fix),
    which lies in [-1, 1] (clamp(+-4 sd)/4), mapped by 0.5 + 0.5 x. So LOKI sees exactly what the Mind sees, on the same adapted scale.
  * the (first) image slot: raw 0..255 / 255, area-mean pooled to a small grid built only from the declared shape:
    gh = round(sqrt(32 H / W)) (>= 2), gw = round(32 / gh) (>= 2), 3 colours -> 288x512 and 160x256 both give 4 x 8 x 3 = 96 numbers.
    LOKI's grid must be the suffix of the vector, so the grid goes last whatever the slot order is (slots are shuffled on the laptop).
    Further image slots (none on the laptop or in the room) would be pooled 2 x 4 x 3 into the vector part.
The one persistent LOKI state is LokiMemory.matrix (float32, <= budget bytes, default 262,144).

One beat of life (laptop_live order): raw -> O(raw) -> choose -> B.step(act) -> FORECAST (before the next raw) -> time passes -> raw -> O(raw)
-> ARRIVE (score the forecast against what came, then LOKI.observe with the action really done = learning) -> carry / feel / after.
The bridge never calls O or B.raw() itself: it reuses the raw list and the feeling x already computed for the Mind.
"""
import hashlib, json, math, os, sys
import numpy as np, torch, torch.nn.functional as F

LOKI_PIN = "24e231317a1c8bfb75a4088c66c54b1013eb8dd1"
LOKI_SHA = "fcf5cb06f6183449b04cb5ca24945e2d646e48ff459a138666ad55bd28073212"     # sha256 of organ.py + effects.py + transport.py + __init__.py (cat order)
BRIDGE_VERSION = 1


def loki_module(path=None):
    path = path or os.environ.get("ATANOR_LOKI_PATH", "ext/LOKI")
    src = b""
    for f in ("organ.py", "effects.py", "transport.py", "__init__.py"):
        with open(os.path.join(path, "loki_memory", f), "rb") as fh:
            src += fh.read()
    sha = hashlib.sha256(src).hexdigest()
    if sha != LOKI_SHA:
        raise ValueError(f"LOKI source at {path} is not the pinned v4 ({sha[:12]} != {LOKI_SHA[:12]})")
    if path not in sys.path:
        sys.path.insert(0, path)
    import loki_memory
    return loki_memory


def grid_of(shape):
    H, W = shape[:2]
    gh = max(2, int(round(math.sqrt(32.0 * H / W)))); gw = max(2, int(round(32.0 / gh)))
    return gh, gw, shape[2]


class LokiBridge:
    def __init__(self, contract, organs, budget=262144, path=None, wiring_id=""):
        self._lpath = path; loki_module(path)                               # checks the pinned source once
        self.c, self.O, self.wiring_id = contract, organs, wiring_id
        sens = contract.sensors
        self.bed, self.vec = list(organs.bed), list(organs.vec)                # same slot order as the feeling x (Organs)
        self.sizes = [int(np.prod(sens[i].shape)) for i in self.bed + self.vec]
        self.nb = sum(self.sizes); self.nbed = sum(self.sizes[:len(self.bed)])
        imgs = [i for i, s in enumerate(sens) if s.kind == "image"]
        self.grid_slot = imgs[0] if imgs else None
        self.extra = imgs[1:]
        self.grid = grid_of(sens[self.grid_slot].shape) if imgs else None
        self.gsize = int(np.prod(self.grid)) if imgs else 0
        self.width = self.nb + 24 * len(self.extra) + self.gsize
        self.motors = list(contract.motors); self.combos, _ = contract.codes()
        self.mem = self.L.LokiMemory(self.width, budget, grid_shape=self.grid)
        self.budget = budget
        self.pending = None; self.started = False
        # which public part each LOKI dim belongs to (kind only - finer device groups are the judge's business)
        self.part = np.array(["bedrock"] * self.nbed + ["vector"] * (self.nb - self.nbed) + ["image-extra"] * (24 * len(self.extra)) + ["image"] * self.gsize)

    @property
    def L(self):                                                           # the module itself is not kept in the instance (so the bridge can be copied)
        return loki_module(self._lpath)

    # ---- encoding ----
    def motor(self, act):
        """combo index really executed -> one-hot per branch (concatenated). No ordinal meaning, no assumed rest option."""
        cb = self.combos[int(act)]; v = np.zeros(sum(self.motors), np.float32); o = 0
        for k, c in zip(self.motors, cb):
            v[o + c] = 1.0; o += k
        return v

    def _pool(self, img, gh, gw):
        t = torch.as_tensor(img, dtype=torch.float32)
        t = t.reshape(1, *t.shape[-3:]).permute(0, 3, 1, 2) / 255.0
        return F.adaptive_avg_pool2d(t, (gh, gw))[0].permute(1, 2, 0).clamp(0, 1).numpy().astype(np.float32)

    def features(self, raw, x, available=None, body=0):
        """raw: the contract-ordered raw list already read this beat; x: the feeling already computed from it (Organs, [n, dx]).
        available: per-slot bools from the plumbing (None = all). Returns (values [width] in [0,1], mask [width] bool)."""
        sens = self.c.sensors; av = [True] * len(sens) if available is None else list(available)
        xv = x[body, :self.nb].detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)[body, :self.nb]
        vals = np.empty(self.width, np.float32); mask = np.ones(self.width, bool); o = 0
        for i, sz in zip(self.bed + self.vec, self.sizes):
            seg = xv[o:o + sz]
            vals[o:o + sz] = np.clip(seg, 0, 1) if i in self.bed else np.clip(0.5 + 0.5 * seg, 0, 1)
            mask[o:o + sz] = bool(av[i]); o += sz
        for i in self.extra:
            r = raw[i][body] if raw is not None else None
            vals[o:o + 24] = self._pool(r.cpu().numpy() if torch.is_tensor(r) else r, 2, 4).ravel() if r is not None else 0.5
            mask[o:o + 24] = bool(av[i]) and r is not None; o += 24
        if self.grid_slot is not None:
            r = raw[self.grid_slot][body]
            vals[o:] = self._pool(r.cpu().numpy() if torch.is_tensor(r) else r, *self.grid[:2]).ravel()
            mask[o:] = bool(av[self.grid_slot])
        vals[~mask] = 0.5                                                      # hidden components never enter the encoder (LOKI ignores them via the mask)
        return vals, mask

    # ---- one beat ----
    def begin(self, raw, x, available=None):
        """first observation of a life (or after a cut): no action yet."""
        self.mem.cut(); self.pending = None
        v, m = self.features(raw, x, available)
        if m.any():
            self.mem.observe(v, None, visible=None if m.all() else m)
        self.started = True

    def forecast(self, act):
        """after B.step(act), before the next raw: LOKI's prediction of the next sensory bundle (read-only). act None = no action (sense mode)."""
        u = None if act is None else self.motor(act)
        pred = self.mem.imagine([] if u is None else [u])
        self.pending = (act, pred)
        return pred

    def arrive(self, raw, x, act, available=None, cut=False):
        """the next raw has arrived: score the pending forecast, then learn (observe with the action really done).
        raw None = the whole body went dark (nothing delivered) -> LOKI advances one beat with the action. cut = life boundary.
        Returns (pred, truth, mask) for the judge (None if nothing to score)."""
        if cut or not self.started:
            if raw is not None:
                self.begin(raw, x, available)
            else:
                self.mem.cut(); self.started = False
            return None
        u = None if act is None else self.motor(act)
        pend = self.pending; self.pending = None
        if raw is None:
            if u is not None:
                self.mem.advance(u)
            return None
        v, m = self.features(raw, x, available)
        out = None if pend is None else (pend[1], v, m)
        if not m.any():
            if u is not None:
                self.mem.advance(u)
        else:
            self.mem.observe(v, u, visible=None if m.all() else m)
        return out

    # ---- effects ----
    def effect(self, plan_a, plan_b):
        """signed endpoint difference A - B (combo indices) from the live state; path says which readout answered."""
        A = [self.motor(a) for a in plan_a]; B = [self.motor(b) for b in plan_b]
        trained = "effect_count" in self.mem._layout and bool(self.mem._part("effect_count")[0])
        return self.mem.predict_effect(A, B), ("v4-grid-effect (trained) + absolute difference off-grid" if trained else "absolute-difference (effect readout untrained)")

    def learn_effect(self, plan_a, plan_b, end_a, end_b):
        """only where the judge can really run both futures from one start (a simulator fork). Never on the laptop."""
        self.mem.learn_effect([self.motor(a) for a in plan_a], [self.motor(b) for b in plan_b], end_a, end_b)

    # ---- keeping ----
    def binding(self):
        return json.dumps(dict(sig=self.c.signature(), wiring=self.wiring_id, loki=LOKI_PIN, bridge=BRIDGE_VERSION, grid=self.grid))

    def save(self, path, private=True):
        """private: blank the last-sense anchors (anchor, retina, chart, z, traces, h) before writing - the laptop rule keeps no
        camera/microphone moment on disk. Learned weights stay. The live state is not changed."""
        m = self.mem
        if private:
            import copy
            m = copy.deepcopy(self.mem); m.cut()
        m.save(path, binding=self.binding())

    def restore(self, path):
        """load only if the body wiring, the LOKI version and the bridge version are the same; otherwise start fresh. Returns True if loaded."""
        try:
            loaded = self.L.LokiMemory.load(path, binding=self.binding())
        except (ValueError, KeyError, FileNotFoundError, OSError):
            return False
        if loaded.width != self.width or loaded.grid_shape != tuple(self.grid or ()) or loaded.budget != self.budget:
            return False
        self.mem = loaded; self.started = False; self.pending = None
        return True


class LokiEval:
    """Judge side (the organ never sees this): scores, on the one realized trajectory, LOKI vs the Mind's own core forecast f vs 'last value'.
    1 beat: forecast made after B.step(act) and before the next raw, scored when it arrives.
    k beats (default 256): every `every` beats the judge snapshots LOKI's state, the feeling x and the clock; k beats later it replays the
    actions the being really did on the snapshot (LOKI.imagine, core f rolled) and scores against what really came. Nothing is learned from
    this, and no un-lived second future is ever made up (the laptop cannot be forked).
    Scales: vector dims are scored in ATANOR feeling units (bedrock 0..1, others -1..1: LOKI's [0,1] output mapped back by 2p-1);
    image grid dims in pooled 0..1 colour. groups: {name: [LOKI dim indices]} from the judge's wiring (device names stay here)."""

    def __init__(self, bridge, groups, k=256, every=64):
        self.b, self.groups, self.k, self.every = bridge, groups, k, every
        self.n1 = {}; self.e1 = {}; self.nk = {}; self.ek = {}; self.snaps = []; self.secs = []; self.gsecs = []; self.t = 0; self.lat = []

    def _to_feel(self, p):
        q = p.copy(); q[self.b.nbed:self.b.nb] = 2 * q[self.b.nbed:self.b.nb] - 1
        return q

    def _add(self, store, cnt, name, err, m):
        for g, ix in self.groups.items():
            ix = [i for i in ix if m[i]]
            if ix:
                store.setdefault(g, {}).setdefault(name, 0.0); cnt.setdefault(g, {}).setdefault(name, 0)
                store[g][name] += float(np.mean(err[ix])); cnt[g][name] += 1

    def beat(self, scored, x_before, f_pred, act, x_after, now, gap=None):
        """scored = LokiBridge.arrive(...) result; x_before / x_after: feelings [1, dx] before / after this beat;
        f_pred: the Mind's core forecast of x_after (feeling units, [dx]) or None."""
        self.t += 1
        if scored is not None:
            pred, truth, m = scored
            pl, tr = self._to_feel(pred), self._to_feel(truth)
            lastv = self._to_feel(self._prev_v) if getattr(self, "_prev_v", None) is not None else None
            self._add(self.e1, self.n1, "loki", (pl - tr) ** 2, m)
            if lastv is not None:
                self._add(self.e1, self.n1, "last", (lastv - tr) ** 2, m)
            if f_pred is not None:
                fp = np.full(self.b.width, np.nan, np.float32); fp[:self.b.nb] = np.asarray(f_pred)[:self.b.nb]
                mm = m.copy(); mm[self.b.nb:] = False
                self._add(self.e1, self.n1, "core_f", np.nan_to_num((fp - tr) ** 2), mm)
            self._prev_v = truth
        # k-beat snapshots
        for s in self.snaps:
            s["acts"].append(act); s["gaps"].append(gap)
        done = [s for s in self.snaps if len(s["acts"]) >= self.k]
        if done and scored is not None:
            _, truth, m = scored; tr = self._to_feel(truth)
            for s in done:
                lp = self._to_feel(s["mem"].imagine([self.b.motor(a) for a in s["acts"] if a is not None]))
                self._add(self.ek, self.nk, "loki", (lp - tr) ** 2, m)
                self._add(self.ek, self.nk, "last", (self._to_feel(s["v"]) - tr) ** 2, m)
                if s["f"] is not None:
                    fx = s["f"](s["acts"])
                    fp = np.full(self.b.width, 0.0, np.float32); fp[:self.b.nb] = fx[:self.b.nb]
                    mm = m.copy(); mm[self.b.nb:] = False
                    self._add(self.ek, self.nk, "core_f", (fp - tr) ** 2, mm)
                self.secs.append(now - s["time"])
                if all(g is not None for g in s["gaps"]):
                    self.gsecs.append(float(sum(s["gaps"])))
        self.snaps = [s for s in self.snaps if len(s["acts"]) < self.k]
        return done

    def snapshot(self, now, v, roll_f=None):
        if self.t % self.every == 0:
            import copy
            self.snaps.append(dict(mem=copy.deepcopy(self.b.mem), v=v.copy(), time=now, acts=[], gaps=[], f=roll_f))

    def report(self):
        def avg(store, cnt):
            return {g: {k: round(v / max(1, cnt[g][k]), 6) for k, v in d.items()} for g, d in store.items()}
        return dict(beats=self.t, one_beat=avg(self.e1, self.n1), one_beat_n=self.n1, k_beats=self.k, k_beat=avg(self.ek, self.nk), k_beat_n=self.nk,
                    k_beat_seconds=(dict(mean=round(float(np.mean(self.secs)), 1), min=round(float(np.min(self.secs)), 1), max=round(float(np.max(self.secs)), 1)) if self.secs else None),
                    k_beat_gap_seconds=(dict(mean=round(float(np.mean(self.gsecs)), 1), min=round(float(np.min(self.gsecs)), 1), max=round(float(np.max(self.gsecs)), 1)) if self.gsecs else None),
                    latency_ms=(dict(mean=round(float(np.mean(self.lat)), 2), p95=round(float(np.percentile(self.lat, 95)), 2)) if self.lat else None),
                    matrix_bytes=int(self.b.mem.state_bytes))


def effect_ranks(bridge, true_dims, H=8, S=8, seed=0, nvec=None):
    """Judge side: LOKI's own branch -> channel map from predict_effect on the live state (read-only).
    For branch j: plans of H beats that differ only in branch j's option (every option pair), the other branches drawn at random (same in both arms).
    Mean |predicted difference| per vector dim -> rank of the true device dims (true_dims[j], judge wiring) among the first nvec dims (rank 1 = top)."""
    rng = np.random.default_rng(seed); nvec = nvec or bridge.nb; combos = np.array(bridge.combos)
    index = {tuple(c): i for i, c in enumerate(bridge.combos)}; out = {}
    for j, K in enumerate(bridge.motors):
        acc = np.zeros(bridge.width); cnt = 0
        for _ in range(S):
            base = combos[rng.integers(0, len(combos), H)]
            for o1 in range(K):
                for o2 in range(o1 + 1, K):
                    c1, c2 = base.copy(), base.copy(); c1[:, j] = o1; c2[:, j] = o2
                    d, _ = bridge.effect([index[tuple(c)] for c in c1], [index[tuple(c)] for c in c2]); acc += np.abs(d); cnt += 1
        v = acc[:nvec] / max(1, cnt); order = list(np.argsort(-v))
        ranks = [order.index(i) + 1 for i in true_dims.get(j, [])]
        out[j] = dict(true_rank=(min(ranks) if ranks else None), of=nvec, top=[int(i) for i in order[:4]])
    return out
