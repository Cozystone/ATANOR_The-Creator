# V1: the being's value learns from what it really did next (registered 2026-11-24, before running)

Defect (D6, D7): the being's action value bootstraps from the max over 90 noisy heads; in its own lived ring this inflates |Q| to 5-20 (offline 82)
where returns are about +-0.5. An inflated value is noise, its earned trust stays near 0, and it almost never chooses (MineLand <1%, laptop 0).
One change: Worth.learn bootstraps from the value of the combo the body REALLY did next (SARSA; at a cut or the ring's edge the mean over combos),
recorded per body. Reward unchanged (wholeness change only), no rule, nothing else changed. Option `Mind(value_target="sarsa")`, default "max" until adopted.

## A/B (same being file atanor.pt, same rooms, same seeds; only the target differs)
Dev layouts 9101-9106 x mirror (12), the A1 v2 depletion world, 12,000-beat lived phase A each, then the A1 v2 fork with the life's OWN value
(no offline arms). Measured: end-of-life |Q|, earned trust, share of beats decided by the value, fills and deaths in the life; at the fork:
first-arm match to the need, needed-channel recovery share (the A1 v2 G2 / G3 quantities). Plus the regression life (a1_channel `life` stage:
4 varied rooms x 16 bodies x 1,200 beats): mean wholeness, fills, deaths.
## Adoption (fixed now)
Adopt SARSA as the being's default if: |Q| sane (< 1), earned trust higher, and at the fork recovery share not lower and first match not lower
(within 1 se), and the regression life not worse in wholeness or deaths (within 1 se). If trust and value share rise but the fork does not move, adopt
as a defect fix and record that choice is still not steered (the next break). Sealed seeds untouched.
