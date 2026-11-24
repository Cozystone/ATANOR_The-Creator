# D4: knowing an action through the body predictor (registered 2026-11-24, before running)

D1-D3 (6 cached layouts, reliable truth .65): no action-value head ranks the 90 combos at the A1 fork - not the lived one, not one-beat or held
targets, not per-branch parts (mean Spearman with 'holding the combo carries the body into the needed arm' between -.06 and .15).

## Theory
An action-value head must learn each (place, action) pair from experience at that place; the body passes the junction only some tens of times in
12,000 beats, spread over 90 combos. But how a held combo moves the body (walk, turn) is the same everywhere, so the being's own body predictor
(core f, trained on every lived beat) can carry that knowledge to the junction; and a **state** value V(x) (one output, trained on every lived
state) can say how good the place it leads to is (A1 v2's left/right mean preference .62-.67 already hints the state part is known).
So: score(a) = V(z_H) with z = x0 rolled H beats through f while holding a (inside dims clipped to [0, 1]), or the predicted wholeness change
plus g^H V(z_H). No new reward, no rule; this is the imagination route the founder set (choose by imagination), with the held action as the unit.

## Design
Same cached lives and truth as D2/D3. State value V: same network size, trained offline on the lived ring (one-beat TD on states, 3,000 updates).
Variants: H in {4, 12} x {V(z_H), wholeness change + g^H V(z_H)}. Also reported: the predictor's own error when rolling a held combo H beats
against the clones' real feelings (does f carry the body there at all).

## Reading (fixed now)
- a rollout variant clearly better than the best head (mean Spearman with hold-entry or recovery >= .2 above, top-5 hold-entry above the mean)
  -> the being should judge actions through its body predictor; next = one change in the Mind (a held-rollout imagination score with a state value)
  and rerun the A1 fork, then a regression life.
- no gain with a small roll error -> the state value does not know which place is good (look at V along the arms).
- no gain with a large roll error -> the body predictor cannot carry a held action (improve f's multi-step use first).
