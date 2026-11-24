# D13: a recalled goal that persists, and learned beats-to-goal (registered 2026-11-24, before running)

## Why D1-D11 could not work
At the junction the combo done in ONE beat does not decide the arm: reaching the left arm needs turning then walking for several beats, and the being
re-picks every ~3.5 beats. Any score of a single combo (value, recall) is then weakly tied to the arm reached. What decides the arm is an intention that
persists over beats (founder: intention = a persisting inner pressure).

## Theory (prior work)
Hindsight goal relabelling (Andrychowicz et al. 2017) and temporal-distance / contrastive goal-conditioned learning (Eysenbach et al. 2022): any later
moment of the being's own life is a goal it reached, so "how many beats from this feeling, doing this, to that scene" is learned from every lived pair,
with no reward - junction visits each give tens of examples. The goal is recalled from memory by the current need: the scenes (non-bedrock feelings) of the
being's own moments of relief of the channel it lacks now. Need picks the goal (bedrock), memory supplies it, the learned beats-to-goal steers each beat,
and the goal persists until relief.

## Design (offline learning on the cached lived rings; clones in the fork)
- d(x, a, g): an MLP on [x (all feelings), the combo's branch code (contract structure, so kin combos share), g (scene dims of x_u)] -> log(u - t),
  trained on lived pairs t < u <= t + 60 inside one joined stretch (no cut between). 3,000 updates of 2,048.
- Goal for a lacking channel c: the mean scene x at the being's lived moments right after channel c rose (its own relief memories).
- Policy in the fork (A1 v2 start: junction, 16 thirsty + 16 hungry clones, T = 150): each beat a = argmin_a d(x, a, g_need) (softmax at the same
  looseness as the Mind), plus the Mind's own 10% try. Compared with: (i) the being's own policy (its lived value) in the same fork; (ii) SWAPPED goal
  (thirsty clones given the food-relief scene and vice versa) - must go to the other arm if the goal steers.
- Measured: first arm entered matches the need, needed-channel recovery share; for the swap: share entering the goal's arm.
## Reading
Goal-directed first match and recovery clearly above the being's own policy (by >= .2 on dev layouts) AND the swap reverses the arm (>= .7 into the
swapped goal's arm) -> the past steers the present need through a recalled, persisting goal; next = this as the Mind's intention (one change, a
learned organ; the goal is chosen by the bedrock lack), then A1 sealed and a regression life. Otherwise: if the swap does not reverse, d does not
carry place; if both go to one arm, the goals are not distinct.
