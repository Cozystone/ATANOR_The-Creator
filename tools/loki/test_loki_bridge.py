"""LOKI v4 bridge: time alignment, masks, wiring binding, privacy, budget, and the live loop (mock laptop, no hardware).
  PYTHONPATH=. python -m unittest -v tools.loki.test_loki_bridge
needs: ATANOR_LOKI_PATH (pinned LOKI checkout, default ext/LOKI) and the eye models (common_senses.OUT)."""
import copy, json, os, sys, tempfile, unittest
import numpy as np, torch
from atanor_os.emerge.body_contract import Organs
from atanor_os.emerge import loki_bridge as LBR
from tools.loki.mock_laptop import MockLaptop


def body_and_organs(seed=0):
    B = MockLaptop(seed=seed); O = Organs(B.c, 1, dev="cpu", depth=""); return B, O


class Alignment(unittest.TestCase):
    def setUp(self):
        self.B, self.O = body_and_organs(0)
        self.LB = LBR.LokiBridge(self.B.c, self.O, wiring_id=self.B.wiring_id())

    def beat(self, act, available=None, cut=False, dark=False):
        if act is not None:
            self.B.step(torch.tensor([act]))
        self.LB.forecast(act)
        raw = None if dark else self.B.raw()
        x = None if dark else self.O(raw)
        return self.LB.arrive(raw, x, act, available, cut=cut)

    def test_budget_and_grid(self):
        self.assertLessEqual(self.LB.mem.state_bytes, 262144)
        self.assertEqual(self.LB.grid, (4, 8, 3))
        self.assertIn("effect_weights", self.LB.mem._layout)
        self.assertEqual(self.LB.width, 39 + 96)

    def test_first_observation(self):
        raw = self.B.raw(); x = self.O(raw)
        self.assertIsNone(self.LB.arrive(raw, x, 3))                       # nothing to score before the first observation; it begins instead
        self.assertTrue(self.LB.started)
        out = self.beat(5)
        self.assertIsNotNone(out); pred, truth, m = out
        self.assertEqual(pred.shape, (self.LB.width,)); self.assertTrue(m.all())

    def test_forecast_is_read_only_and_before_arrival(self):
        raw = self.B.raw(); self.LB.begin(raw, self.O(raw))
        before = self.LB.mem.matrix.copy()
        p1 = self.LB.forecast(7); p2 = self.LB.forecast(7)
        np.testing.assert_array_equal(before, self.LB.mem.matrix)       # forecasting does not touch the state
        np.testing.assert_array_equal(p1, p2)
        self.B.step(torch.tensor([7])); raw = self.B.raw(); x = self.O(raw)
        pred, truth, m = self.LB.arrive(raw, x, 7)
        np.testing.assert_array_equal(pred, p2)                          # the scored forecast is the one made before the raw arrived
        self.assertFalse(np.array_equal(before, self.LB.mem.matrix))   # ... and learning happened only at arrival

    def test_sense_mode_no_action(self):
        raw = self.B.raw(); self.LB.begin(raw, self.O(raw))
        for _ in range(3):
            self.beat(None)
        coup = self.LB.mem._part("coupling").copy(); traces = self.LB.mem._part("traces").copy()
        out = self.beat(None)
        self.assertIsNotNone(out)
        np.testing.assert_array_equal(coup, self.LB.mem._part("coupling"))   # no action -> no transition learning
        np.testing.assert_array_equal(traces, self.LB.mem._part("traces"))

    def test_partial_missing_camera(self):
        raw = self.B.raw(); self.LB.begin(raw, self.O(raw))
        av = self.B.available(); cam = [i for i, s in enumerate(self.B.c.sensors) if s.kind == "image"][0]; av[cam] = False
        pred, truth, m = self.beat(4, available=av)
        self.assertFalse(m[-96:].any()); self.assertTrue(m[:-96].all())
        self.assertEqual(float(self.LB.mem._part("control")[1]), 0.0)       # a partial frame does not train the transition model

    def test_full_blackout_advances(self):
        raw = self.B.raw(); self.LB.begin(raw, self.O(raw)); self.beat(2)
        anchor = self.LB.mem._part("anchor").copy()
        self.assertIsNone(self.beat(9, dark=True))                         # nothing delivered: LOKI advances one beat with the action
        self.assertFalse(np.array_equal(anchor, self.LB.mem._part("anchor")))
        allmiss = [False] * len(self.B.c.sensors)
        self.B.step(torch.tensor([1])); self.LB.forecast(1); raw = self.B.raw()
        out = self.LB.arrive(raw, self.O(raw), 1, allmiss)                 # delivered but every nerve flagged dead = same as dark
        self.assertFalse(out[2].any())

    def test_life_boundary(self):
        raw = self.B.raw(); self.LB.begin(raw, self.O(raw)); self.beat(2); self.LB.forecast(3)
        self.B.step(torch.tensor([3])); raw = self.B.raw(); x = self.O(raw)
        self.assertIsNone(self.LB.arrive(raw, x, 3, cut=True))           # no forecast crosses a life boundary
        self.assertIsNone(self.LB.pending)
        v, m = self.LB.features(raw, x)
        np.testing.assert_allclose(self.LB.mem._part("anchor"), (v - 0.5).clip(-.5, .5), atol=1e-6)

    def test_motor_is_one_hot_per_branch(self):
        for a in (0, 17, len(self.B.combos) - 1):
            v = self.LB.motor(a); o = 0
            for k, c in zip(self.B.c.motors, self.B.combos[a]):
                self.assertEqual(v[o:o + k].sum(), 1.0); self.assertEqual(v[o + c], 1.0); o += k

    def test_feeling_range(self):
        raw = self.B.raw(); x = self.O(raw); v, m = self.LB.features(raw, x)
        self.assertTrue(((v >= 0) & (v <= 1)).all())
        xv = x[0, :self.LB.nb].numpy()
        np.testing.assert_allclose(v[self.LB.nbed:self.LB.nb], np.clip(0.5 + 0.5 * xv[self.LB.nbed:], 0, 1), atol=1e-6)   # the Mind's own Stat scale

    def test_effect_path_label(self):
        raw = self.B.raw(); self.LB.begin(raw, self.O(raw))
        d, path = self.LB.effect([1] * 5, [2] * 5)
        self.assertIn("untrained", path); self.assertEqual(d.shape, (self.LB.width,))


class Keeping(unittest.TestCase):
    def test_wiring_and_version_binding_and_privacy(self):
        B, O = body_and_organs(0); LB = LBR.LokiBridge(B.c, O, wiring_id=B.wiring_id())
        raw = B.raw(); LB.begin(raw, O(raw))
        for a in range(20):
            B.step(torch.tensor([a])); LB.forecast(a); raw = B.raw(); LB.arrive(raw, O(raw), a)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "l.npz"); LB.save(p)
            self.assertTrue(np.abs(LB.mem._part("anchor")).sum() > 0)       # live state untouched by the private save
            same = LBR.LokiBridge(B.c, O, wiring_id=B.wiring_id()); self.assertTrue(same.restore(p))
            for name in ("anchor", "retina", "chart", "z", "traces", "h"):
                self.assertEqual(float(np.abs(same.mem._part(name)).sum()), 0.0, name)   # no last camera / mic moment on disk
            np.testing.assert_array_equal(same.mem._part("decoder"), LB.mem._part("decoder"))
            B2 = MockLaptop(seed=0)                                        # same shapes and slot kinds, different inner wiring:
            k = sorted(B2.perm)[0]; B2.perm[k] = B2.perm[k][::-1]           # one slot's components in another order
            B2.opt[0] = B2.opt[0][1:] + B2.opt[0][:1]                       # one branch's options in another order
            self.assertEqual(B2.c.signature(), B.c.signature()); self.assertNotEqual(B2.wiring_id(), B.wiring_id())
            O2 = Organs(B2.c, 1, dev="cpu", depth="")
            other = LBR.LokiBridge(B2.c, O2, wiring_id=B2.wiring_id()); self.assertFalse(other.restore(p))
            old = LBR.BRIDGE_VERSION; LBR.BRIDGE_VERSION = old + 1
            try:
                self.assertFalse(LBR.LokiBridge(B.c, O, wiring_id=B.wiring_id()).restore(p))
            finally:
                LBR.BRIDGE_VERSION = old

    def test_pinned_source(self):
        with tempfile.TemporaryDirectory() as d:
            src = LBR.os.environ.get("ATANOR_LOKI_PATH", "ext/LOKI")
            import shutil
            shutil.copytree(os.path.join(src, "loki_memory"), os.path.join(d, "loki_memory"))
            open(os.path.join(d, "loki_memory", "effects.py"), "a").write("\n# changed\n")
            with self.assertRaises(ValueError):
                LBR.loki_module(d)


class LiveLoop(unittest.TestCase):
    def test_one_read_one_organ_call_per_beat_and_being_continues(self):
        from atanor_os.emerge import laptop_live, body_contract
        calls = dict(raw=0, organs=0)
        raw0, call0 = MockLaptop.raw, body_contract.Organs.__call__

        def raw(self):
            calls["raw"] += 1; return raw0(self)

        def call(self, r):
            calls["organs"] += 1; return call0(self, r)
        MockLaptop.raw = raw; body_contract.Organs.__call__ = call
        try:
            with tempfile.TemporaryDirectory() as d:
                for loki in (True, False):
                    calls.update(raw=0, organs=0)
                    argv = ["x", "--body", "mock", "--no_sleep", "--mode", "mind", "--minutes", "0.25", "--being", os.path.join(d, "b.pt"),
                            "--out", os.path.join(d, "judge.json"), "--depth", "mc/mc_depth.pt", "--save_every", "50", "--threads", "2", "--loki_k", "8",
                            "--load", "being/atanor.pt"]
                    if loki:
                        argv += ["--loki", os.environ.get("ATANOR_LOKI_PATH", "ext/LOKI"), "--loki_state", os.path.join(d, "l.npz")]
                    sys.argv = argv; laptop_live.main()
                    self.assertEqual(calls["raw"], calls["organs"])           # every raw read goes through Organs exactly once
                    if loki:
                        r = json.load(open(os.path.join(d, "loki_eval.json")))
                        self.assertEqual(r["beats"] + 1, calls["raw"])         # one first read + one per beat, LOKI reads nothing itself
                        self.assertTrue(os.path.exists(os.path.join(d, "l.npz")))
                    else:
                        self.assertFalse(os.path.exists(os.path.join(d, "loki_eval.json")) and loki)
                    self.assertTrue(os.path.exists(os.path.join(d, "b.pt")))   # second round continues the same being file (load, not wear)
        finally:
            MockLaptop.raw = raw0; body_contract.Organs.__call__ = call0


if __name__ == "__main__":
    unittest.main()
