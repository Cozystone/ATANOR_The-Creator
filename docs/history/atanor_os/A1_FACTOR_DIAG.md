# D3: a value learned per motor branch (registered 2026-11-24, before running)

D2 (one-beat vs held target, 5-6 layouts): truth reliable (two clones agree, Spearman .65), but no value ranks the combos (mean Spearman with
recovery -.01 / .04 / .08, with 'held combo enters the needed arm' .00 / .07 / .15). Held is slightly better, below the registered +.2: the target
alone is not the break. In the lived ring an action lasts only ~3.5 beats.

## Theory
The value has one output per combo (M = 90 = legs 6 x neck 5 x hand 3). 12,000 lived beats give each combo only a few tries near any one place,
so each of the 90 heads is learned almost alone. Where the body goes is mostly decided by the legs and the neck (turning); the same experience
of "walk forward while turning left" is shared by every hand option. A value built as a sum of per-branch parts, Q(x, a) = sum_b q_b(x)[option_b(a)]
(6 + 5 + 3 = 14 outputs, combos read through the contract's own branch structure - no meanings, no left/right), shares that experience.
This is a choice of how the value organ reads the body's motor branches (the contract already declares them), not a behaviour rule.

## Design (same as D2 otherwise)
Same cached lives, same 3,000 updates, same truth (two clones per combo; recovery and noise-free hold-entry). Variants: one-beat x {per-combo, per-branch},
held x {per-combo, per-branch}.

## Reading (fixed now)
- per-branch clearly better than per-combo (mean Spearman with hold-entry or recovery higher by >= .2, top-5 hold-entry above the mean) -> the
  break is data sharing across combos; next = give the being a per-branch value head (one change) and rerun the A1 fork and a regression life.
- no gain -> look at what the value's input x carries at the junction (R2), and at whether the lived ring contains the junction choices at all.
