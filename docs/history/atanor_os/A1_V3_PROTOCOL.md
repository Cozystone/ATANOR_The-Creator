# A1 v3: the past used by the present need, in a room that shows where you are (registered 2026-11-24, before running)

Runs only if D16 shows the cued room makes place visible (raw-frame heading R^2 >= .7, position >= .5). Founder 2026-11-24: value -> choice until it breaks through.

## What changed since A1 v2 (and why)
- World: distal cues on (RoomBody.cues: wall faces tinted by direction). D15: the uncued T room is left-right symmetric and its frames do not carry
  place, so no memoryless value could know which arm it was in. A world property, not a behaviour rule; for founder review.
- Nothing else in the being. Its value as it is (Worth, max target) and the two value fixes found in D7 / D12 are compared as arms, all from the same lives.

## Design (dev 9101-9106 x mirror = 12 layouts; sealed 7701-7712 untouched until the design is chosen)
Lived phase A: the A1 v2 depletion world with cues, 12,000 beats, the being file atanor.pt (max target, as it lives now). Then the A1 v2 fork (32 clones
at the junction, half thirsty, half hungry, trust fixed at 1, T = 400). Arms (values plugged into the forked mind): the life's own value; arm 3 SF with
SARSA futures trained offline on the same lived ring (D12 recipe); arm 1f (Worth, offline) as in A1 v2.
Measured per arm: value preference toward the needed side, first-arm match, needed-channel recovery share, decided-by shares.
## Gate (fixed now; the A1 v2 gate on the best arm, chosen on dev only)
G1 value preference >= .75 and >= .20 above its shuffled-outcome control; G2 first match >= .65 and >= .10 above shuffled; G3 recovery >= .10 above the
being's own value in the uncued A1 v2 dev fork (V1 numbers: .25). Shuffled-outcome controls as in A1 v2 (outcomes shuffled in the ring before training).
If an arm passes on dev: one run on sealed 7701-7712, then a regression life (cued and uncued) before any adoption.
