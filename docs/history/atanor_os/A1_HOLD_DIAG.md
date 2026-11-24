# A1 hold diagnostic (registered 2026-11-23, before running)

A1 v2 is closed as failed (founder 2026-11-23). Sealed seeds 7701-7712 stay untouched. This is a diagnostic, not a new registration with a gate:
it asks where the chain "value -> first choice" breaks.

## Same start as A1 v2
- World, phase A and arms as A1 v2 (`A1_CHANNEL_PROTOCOL_V2.md`): depletion regrow 1/120, drain 1/180, 12,000 fixed beats with rebirth, dev seeds 9101-9112 both mirrors.
- Arms: 1f (Worth), 2 (aux), 3 (bedrock). The shuffled arms and 1L are left out to save time; they are not needed to locate the break.
- The fork is A1 v2's own fork: 2K = 32 clones at the junction, half thirsty, half hungry, no learning, value trust fixed at 1, T = 400 beats.

## Moments recorded per clone (whole population is the denominator)
- **M0, the first beat**: its command (left set / right set / neither), who decided it (strong pull / value / random / hold / try), and whether the mind was free to pick anew (`again`).
- Value left-right margin at M0: mean value over left-going combos minus right-going combos, signed toward the needed side.
- **The first arm actually entered** (centre more than 2.5 cells past the junction, upper corridor), or none.
- **The last channel recovered** within T (the last store that rose), and whether the needed channel recovered at all.
- **M1, the commit beat**: the last beat in the junction zone (|x - x0| < 1.5, upper corridor) before the first entry. Records whether a new pick was possible there (`again`) and what held.

## Three choices from one cloned state at M1
The whole per-body state (body, organs, mind, value organ's running state) and the torch random state are copied at every beat of the original run.
From the copy at each clone's M1:
1. **original**: nothing changed (replayed once per bucket to check that the copy reproduces the run).
2. **hold released**: at that beat only, the mind must pick anew (`again` forced true for that clone). Everything else is unchanged, including the same random draws.
3. **value only (probe)**: at that beat only, the action is the value organ's best combo, which is then held as usual. This is a probe to separate causes, never a candidate for the being.

Outcomes per clone: the first arm entered after M1 and whether the needed channel recovered.

## Counts (all clones that entered an arm form the denominator; those that never entered are counted apart)
- Hold released vs original: **saved** (wrong -> right), **spoiled** (right -> wrong), unchanged right, unchanged wrong.
- The same four counts for value only vs original.
- Split by whether the original M1 was a hold (`again` false) or already a free pick.
- Split by value margin size (quartiles of |margin|) and by the sign of the margin.

## Reading (decided before the results)
- If released holds save much more than they spoil, the hold (keeping what it was doing) is what blocks the value.
- If value-only saves a lot but release does not, the fresh pick itself (strong pull / random) overrides the value.
- If value-only also fails where the margin points the right way, value and body are not lined up (a side-set or motion problem, not the choice rule).
- If most M1 are already free picks, the hold is not the bottleneck.

Tool: `tools/a1_hold.py`. Mind records `again` and the value lean (`Mind.again`, `Mind.lw_last`) and accepts a one-beat forced re-pick (`Mind.force`): recording and judge-only hooks, default off; the being's behaviour is unchanged.
