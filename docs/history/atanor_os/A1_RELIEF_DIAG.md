# D17: latent learning - futures as the share of a lack relieved (registered 2026-11-24, before running)

## Finding behind it
In the lived rings a fill almost never happens when that store is the lower one (water 0-4%, food 0-8% of fills): the being camps at one spring, tops up
an already full store, and dies of the other lack (13-17 deaths per life). Arm 3's per-channel futures psi learn the amount really gained, which is
min(what the spring holds, 1 - store): at a top-up it is tiny, so "water is here" is learned weakly - intake is tied to the need at the time.
Tolman's latent learning needs the need-free quantity: what is available, weighed by the need only at choice time (successor features over
state features, Barreto 2017; homeostatic reward, Keramati & Gutkin 2014).

## One change (designer-given part of the value organ, recorded for founder review; no behaviour rule, bedrock only)
Per channel, two discounted futures learned by SARSA on the lived ring:
- relief share  f+ = max(db, 0) / (1 - b + .02)   (how much of the lack at that moment went away; a top-up from .95 to 1 counts as a whole relief)
- loss          f- = min(db, 0), and -b at death   (absolute, as before)
Score of a combo: b' = b + (1 - b) clip(psi+, 0, 1) + psi-,  S = whole(clip(b', 0, 1)) - whole(b). The current need enters only through b.

## Design
Cued-room lives (A1 v3 world: depletion, cues, 12,000 beats, dev 9101-9103 x mirror), cached. Arms trained offline on the same ring, 2,000 updates:
absolute SARSA psi (D12), relief-share psi (this), and the relief-share arm on a shuffled-outcome ring. Measured: value map along both paths
(needed-path end minus other-path end, by need), value preference at the junction, and the A1 v2 fork (first match, recovery).
## Reading
Relief-share arm: needed-path gap positive in >= 10 of 12 (layout x need) and fork first match >= .65 and recovery >= .10 above the absolute arm
and the shuffled control -> the being should learn futures as relief shares; next = A1 v3b with this arm on all 12 dev layouts, then sealed.
Map right but fork not better -> knowing where is not yet going there (the choosing or the movement units), measured next.

## D17 result (6 cued layouts): registered reading not met
Needed-path gap positive: absolute 6/12, relief 8/12, shuffled 0/12; value preference .58 / .67 / .42; fork first match .51 / .60 / .50; recovery .29 / .40 / .39.
Who decided in the fork: value alone 0%, hold 73-75%, strong pull 15-21%, try 10% - with a lack of .30 the relief-memory pull always fires and the value is
only one term inside its score.

# D18 (registered before running): which organ steers at the junction? (probe)
Same cued caches and arms (absolute, relief, relief-shuf). The fork is run twice per arm: as it is, and with the relief-memory pull and the harm memory
silenced (judge-only probe: the value's lean alone picks whenever the mind re-picks; hold and try unchanged). Never a candidate for the being.
Reading: with the pull silenced the relief arm's first match >= .70 and >= .15 above its shuffled control -> the value knows and the pull overrides it
(next = how the two are weighed, learned not hand-set); no gain -> the value's knowledge itself is still too weak at the junction (more / better experience).
