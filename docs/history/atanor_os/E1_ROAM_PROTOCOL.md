# E1: drawn to where its own place memory is thin (registered 2026-11-24, before running)

## Why (atanor_os/GRAIN_THEORY.md, D25-D30)
The value does not know the way; a stitched place map from the being's own sense would (.67 - .77 per cell) if the bodies travelled their room, and they do not
(86% of the time in the stem in two layouts; no spring-to-spring trip in 192,000 body-beats). In the mind as it is, a beat with no strong pull is decided by a
uniformly random gesture (or the value, by its earned trust): undirected, so the body diffuses where it stands.
Prior work: latent learning needs unrewarded exploring first (Tolman & Honzik 1930); novelty of place as the drive, measured against an episodic memory of
where one has been (Savinov et al. 2019 'Episodic curiosity through reachability'; Badia et al. 2020 'Never give up': distance to the k nearest remembered
states); hunger itself raising exploration (AgRP -> dopamine in the tail of striatum, 2025).

## One change (Mind(roam=...), default 0 = off; the main line is untouched until adopted)
The being's second bedrock - drawn to where its own knowledge is still thin (until now: where its prediction of its insides fails, learning progress) - applied
to place. A bounded memory of its own recently sensed states (4,096, outside senses only, one added per beat, oldest replaced). On a beat with no strong pull,
instead of a uniformly random gesture: each gesture's foreseen state (its own one-step foresight rolled 4 beats holding that gesture) is compared with the
memory (distance to the 4th nearest remembered state); the gesture is drawn with odds rising with that unfamiliarity (softmax, weight roam = 2 over the
standardised unfamiliarity). No target place, no map, no rule about where to go; strong pulls (relief memories, harm), the value, hold and try are as they are.
Open to the founder's review: whether this widening of the second bedrock from the insides to place is acceptable as bedrock.

## Test (dev 9101-9103, not mirrored; one mind in 64 bodies, cued T room, 12,000 beats; V3 recipe = relief value + sleep replay, with and without roam)
Judge-side, last quarter: share of time in the stem; arm leavings and spring-to-spring trips per 1,000 body-beats; deaths per body; need-directed fills per body.
Then the being-side place map (D30's construction: learned one-state code, 64 clusters) on the last 3,000 beats, per true cell.
## Reading (fixed now)
- Travel: spring-to-spring trips >= 1 per 1,000 body-beats (V3 stretches: 0) and arm leavings >= 5 per 1,000 (V3: .2 - 1.6) in all three layouts.
- Not at the cost of the life: deaths per body in the last quarter not above the V3 life of the same layout by more than 5%.
- Map: per-cell Spearman >= .7 in at least 4 of 6 (layout x need).
All three -> roam is adopted into the being (after founder review of the bedrock question) and the next step is the map organ living in the being and
choosing by it. Travel without map -> stricter linking next. No travel -> the foresight cannot tell unfamiliar from familiar a few beats ahead (measured next:
spread of unfamiliarity across gestures against its noise). Deaths up -> wandering costs more than it earns at this drain: report, do not tune quietly.

## E1 result (3 layouts): not met - and the pull never got to act (my design mistake)
Travel in the last 3,000 beats: spring-to-spring .23 / .39 / .17 per 1,000 body-beats (bar >= 1; V3 stretches 0), arm leavings 1.2 / 2.2 / 1.5 (bar >= 5; V3 .7 / .2 / 1.6).
Deaths in the last quarter 248 / 221 / 247 against V3's 181 / 250 / 242 (+37% / -12% / +2%: the 5% bar broken in one layout). Time in the stem in the last quarter
.50 / .40 / .38. Being-side map on the last 3,000 beats, per cell: .22 / .39, .31 / .11, .06 / .23 (bar .7). None of the three readings met.
Why: I put the pull in the slot 'no strong pull this beat'. Measured afterwards: that slot decided 0.00 of the beats in all three lives (hold .66, strong pull .24,
try .10); in a 500-beat check the relief pull was 'strong' on 100% of beats. So E1 as built changed almost nothing, and its numbers say nothing about the idea.
The decision-source counts were already in every life record; I did not look before choosing the slot.
Sizes inside the score all gestures compete on (500-beat check, E1 mind, 64 bodies): relief pull spread across gestures .29, the value's lean 1.00, tiring .07;
softmax looseness .05 (near arg-max); a new pick on 27% of beats. The unfamiliarity itself does separate gestures: its spread across gestures is 44% of its mean.

# E1b (registered before running): the unfamiliarity competes in the score, as the value and harm do
One change from E1: the standardised unfamiliarity of each gesture's foreseen state is added to the score all gestures compete on, weight 1 (the same size as
the value's lean, whose spread is about 1; the relief pull's is about .3), instead of sitting in the never-used slot. Mind.roam_mode = "score" (default "slot" keeps
E1 reproducible; roam itself stays off by default). The weight is designer-given, like the other constants of the score (for founder review).
Same test and the same three readings as E1 (travel: spring-to-spring >= 1 and arm leavings >= 5 per 1,000 body-beats in all three layouts; deaths in the last
quarter not above V3's by more than 5%; being-side map >= .7 per cell in at least 4 of 6), plus the decision-source shares and the share of new picks reported.
Deaths up with travel -> wandering costs more than it earns at this drain, and the question becomes when the being should roam (sated) and when not: report,
do not tune quietly.

## E1b result (3 layouts): not met
Travel in the last 3,000 beats: spring-to-spring .27 / .06 / .01 per 1,000 body-beats (bar >= 1), arm leavings 3.4 / 1.2 / 3.2 (bar >= 5). Deaths in the last
quarter 235 / 251 / 260 against V3's 181 / 250 / 242 (+30% / 0 / +7%: bar broken in two layouts). Need-directed fills fell (9103: 3 against V3's 55).
Time in the stem .57 / .70 / .36. Being-side map per cell: -.04 / .42, .26 / .34, .07 / .25 (bar .7; shuffled as high or higher).
This time the pull did act (new picks 33 - 35% of beats, was 27%), and every cell of the room is now crossed often (least-crossed link 71 - 125; V3 stretches
1 - 17). So the bodies stir more where they are and cover the cells, but do not make trips; the life gets worse and the map is no better.
Two builds of the same idea have failed. I stop varying it. What the two lives do show: more local coverage is not what the map (or the value) lacks -
the judge-driven traveller differed by going somewhere and arriving, not by covering cells.
