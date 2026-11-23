# LOKI v4 through ATANOR's body loop - room preregistration (2026-11-23, written and committed before any room is measured)

Question: inside ATANOR's own room body (RoomBody -> raw -> Organs -> feeling x, contract [6, 5, 3] = 90 combos), with LOKI reached only through
`atanor_os/emerge/loki_bridge.LokiBridge` (contract shapes, Stat feelings mapped to [0,1], 4x8x3 pooled grid built from the declared 160x256 image,
one-hot per motor branch), does LOKI's trained effect readout predict how the final view differs when, from one start and with one turn sequence,
some beats are forward instead of a true stand-still?

Organ and data (no change after this file):
- LOKI pinned 24e2313 (source sha256 fcf5cb06...). Fresh LokiMemory(44 vector + 96 grid = 140 wide, budget 262,144 B, grid 4x8x3) - the shipped checkpoint is not used (its
  wiring is 4x8 RGB from world.picture and 5 scalar commands, not this body).
- Judge-side command meanings (the organ only sees combo numbers): stand = (legs 0, neck 0, hand 0), forward = (1,0,0), turn left = (0,1,0),
  turn right = (0,2,0), use = (0,0,2).
- Learning rooms 1-12 only: one lit stream per room, 208 beats, command switching with p 0.42 among the five; then 24 bodies per room for 80 beats,
  forking the simulator (RoomBody + Organs deep copy) at beats 40 and 80 into two 20-beat arms: base commands from {left, right, stand}, k beats
  (k drawn 2..18) forward in one arm and stand in the other (arm order random). learn_effect gets the two final views; both arms are then observed.
- Ablation "LOKI no-effect": identical observations in identical order, learn_effect never called (so predict_effect = difference of two absolute imaginings).
- ATANOR's existing predictor = the Inner organ (atanor_os/emerge/inner.py, h 128) on the same 96 grid numbers with the 90 combos as actions, trained with
  train_inner (2,000 iterations, window 64) on the same lit streams and the 24-body 80-beat streams, then adapted in each test room on its first 129 lit frames
  (64 iterations) as in LOKI's own control. It predicts each arm blind for 20 beats from its hidden state; the difference of its two last predictions is its answer.
  (The Mind's core f is not used here: it forecasts eye summaries, not the pixel grid, so it cannot be scored on this target. It is compared on the laptop.)
- Zero: always predicts no difference.

Test: unseen rooms 6101-6112 (never used for anything before). One body, lit, drain 0. Stream of 248 beats (same switching rule, rng seed + 17000).
At beats 128, 168, 208, 248 a pair: base 20 commands from {left, right, stand}, 4/8/12/16 substitutions (forward vs stand), arm order alternating.
Forecasts are made first from the live state; only then the judge forks the simulator and runs both arms. No test-room pair ever reaches learn_effect.
Unit = room mean (over its 4 pairs) of MSE between predicted and true signed final grid difference (96 numbers, 0..1 colour). Identical endpoints are kept.
Also reported, not gated: the same for the 44 non-image numbers (feeling units), nonzero vs identical pairs, predicted energy.

Pass (each of the three controls separately: zero, Inner, LOKI no-effect): mean MSE at least 10% lower AND strictly lower in at least 10 of 12 rooms.
Report wins exactly; report whether > 10. No reruns with other seeds or settings after seeing the test rooms; any revision needs a new file and new seeds.
