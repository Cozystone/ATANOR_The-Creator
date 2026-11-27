# World A - a world that asks for what the being lacks (designed with the ideal LOKI, atanor_os/LOKI_IDEAL_SPEC.md; founder 2026-11-26:
"A랑 B 같이 설계해서 진행해")

# A2-0 (registered before running): the baseline - the current being in a world where food moves in a fixed cycle
World (tools/a2_moving_food.py): the room V2 world; the food spring sits at one of three dead-end corners - its own leg end -> the crossbar's
top-left corner -> its top-right corner - 300 beats each, in that fixed order, for all bodies at once; it does not jump when eaten. Water
stays. Corridors are two cells wide, so no spot blocks a path. Control (--still 1): food stays at its own leg end (the room as before).
Being: the S3-end beings (s3_a in 9101_0, s3_b in 9104_0), 64 bodies, 6,000 beats (20 moves), as they are (no v12, nothing new).
Read: after each move, beats until the first food fill at the new corner (latency; 300 = none); visits to the NEXT corner in the 300 beats
before food comes there; deaths and food fills per 1,000 body-beats vs the still control; where it believes the food is (K1 read-out).
Readings (this is the yardstick v12 must beat, not a pass for the being): latency in the last third of moves <= half the first third AND
visits to the next corner rising -> the being learns the cycle already (then A2 is too easy and A3 comes first); latency falling but no
anticipation -> it re-finds food faster, by place memory of all three corners, without knowing the order (the expected outcome); latency
not falling -> it re-learns each move from scratch. Cost of a moving world = deaths vs still control.

## A2-0 result: the current being pays for a moving world and does not track it - A2 is a fair yardstick for v12
6,000 beats, 64 bodies, 20 moves (moving / still control):
| room | deaths | food fills per 1,000 | need fills | believed food corner = true | believed = NEXT corner just before a move (chance .33) |
|---|---|---|---|---|---|
| 9101 (s3_a) | 6 / 0 | 5.98 / 7.35 (-19%) | 2,432 / 2,739 | .35 / 1.0 | .30 |
| 9104 (s3_b) | 91 / 1 | 3.62 / 5.32 (-32%) | 1,171 / 2,090 | .35 / 1.0 | .30 |
Visits to the next corner before food comes: 9101 first / last third of moves 1,947 -> 1,301; 9104 789 -> 1,358 (mixed; bodies pass the
corners on their way, so this count is confounded - not read as anticipation).
Correction (mine): "beats until the first fill after a move" (9101: 10 -> 1.3) is the fastest of 64 bodies - with many bodies one is always
near some corner; it says nothing about learning and is dropped from the readings. A per-body measure (beats from a body's food lack crossing
.3 to its food fill) replaces it from A2-1 on.
Readings: no anticipation (belief on the next corner at chance), and the being's belief does not even follow where food is now (.35 =
chance): it finds food again by searching, not by knowing. The cost is real - in 9104 deaths 91 vs 1. As registered, this is the yardstick:
v12 (or any planning memory) must lift belief on the food's corner and next corner well above chance and cut these deaths.

# A2-1 (registered before running): the A2 baseline again with the per-body measure
Same four runs as A2-0 (s3_a 9101_0, s3_b 9104_0; moving / still), one change in the read-out: per body, beats from its food store falling
under .7 to its next food fill (deaths while hungry counted apart). Readings: moving vs still, median wait at least 1.5x -> the moving world
costs finding (the being re-searches); last half of the run shorter than the first half by >= 30% in the moving arm -> it learns the moving
food over time (some anticipation or a better search); no change -> it does not.

# A3-0 (registered before running): the baseline in a two-room world (journeys of several turns)
World (tools/a3_rooms.py, 16 x 16): two rooms (5 x 4) joined by a two-cell corridor; each room entered through a two-cell doorway up from the
corridor; a dead-end spur below the corridor on each side between the stem and the doorway; the being is born at the bottom of a stem below
the corridor's middle. Water in the far corner of one room, food in the far corner of the other (mirror swaps them). A spring is out of
view until the body is inside its room. V2 depletion as the room.
Runs: s3_a mirror 0, s3_b mirror 1 (neither has lived here), 64 bodies, 6,000 beats, seed 9301 (not sealed).
Read: deaths per quarter, need fills per quarter, per-body hunger -> fill wait per channel (first / last half), share of lacking beats spent
in the dead-end spurs. Readings (yardstick for v12, not a pass for the being): waits in the last half <= 60% of the first half and spur share
falling -> the current being learns the journeys by itself (then A3 is not a planning test); waits not falling -> the yardstick stands.
Smoke (400 beats): no fill yet, 58% of lacking beats in the spurs.

## A2-1 result: a moving food costs the being 2-3.6x the wait, and it does not learn the moving food
Per body, beats from food store under .7 to its next food fill (median; first half / last half of the run), moving / still:
| room | median wait | first -> last half, moving | first -> last half, still | died while hungry |
|---|---|---|---|---|
| 9101 (s3_a) | 91 / 46 (2.0x) | 142 -> 61 (-57%) | 83 -> 30 (-64%) | 6 / 0 |
| 9104 (s3_b) | 251 / 70 (3.6x) | 235 -> 286 (+22%) | 96 -> 56 (-42%) | 80 / 1 |
Readings as registered: moving vs still >= 1.5x in both -> the moving world costs finding (the being re-searches). The moving arm's last half
is shorter by >= 30% only in 9101, and there the still arm falls as much (settling into the room, not the cycle); in 9104 it grows longer.
Reading: the current being does not learn the moving food. Yardstick for v12 / any planning memory: median wait with moving food near the
still arm's (46 / 70), and deaths while hungry near 0.

## A3-0 result: the two-room world is beyond the being - every body dies about twice per 1,500 beats
| run | deaths per quarter | need fills per quarter | water fills (median wait) | food fills (median wait, first -> last half) | died lacking water / food | lacking beats in spurs |
|---|---|---|---|---|---|---|
| s3_a, water left | 128, 127, 127, 128 | 0, 1, 3, 1 | 11 (416) | 489 (220 -> 95) | 506 / 244 | 21% |
| s3_b, water right | 128, 128, 128, 128 | 0, 0, 1, 0 | 6 (55) | 587 (218 -> 104) | 512 / 326 | 12% |
Reading (mine, not one of the registered two): the bodies find food (waits halving over the run) but almost never water, in either mirror -
so it is not the side; and they die of thirst at the same rate all run long. A3 as built is too big a step: it does not teach step by step
(the founder's rule for world A), so it cannot test planning - a being that never reaches water cannot learn the journey to it. My design
error. Why water and not food, with the rooms mirrored, is open (drain or depletion per channel, or the being's past - its S3 rooms).
Next for A3 (not run): an easier first stage - start in the corridor, rooms nearer - before the full layout; and first the water/food
asymmetry read from the code.

## A3-0 follow-up: each being goes to one side, whatever is there (correcting my first reading)
Where lacking bodies are (s3_a, water left, 1,500 beats): thirsty bodies in the right (food) room 2,581 body-beats, in the left (water) room 0;
hungry bodies in the left room 1,361, right 5. I first read this as the old home's relief map per channel. The two matched runs say otherwise:
| run | left-room spring: fills | right-room spring: fills | deaths per quarter |
|---|---|---|---|
| s3_a, water left | water 11 | food 489 | 128, 127, 127, 128 |
| s3_a, water right | food 106 | water 307 | 128, 127, 120, 110 (need fills 0 -> 67) |
| s3_b, water right | food 587 | water 6 | 128 x 4 |
| s3_b, water left | water 382 | food 67 | 128, 127, 123, 125 |
s3_a finds the right room's spring in both mirrors and seldom the left; s3_b finds the left room's spring in both mirrors and seldom the right.
s3_a's home (9101) had its water on the right, s3_b's home (9104) on the left. So each being carries its home's water SIDE as a direction
habit - it goes there thirsty or hungry alike - not a relief map per channel. My first reading was wrong in that detail.
Reading: A3 first tests the carry-over of an old world's direction, before planning: one side is lived, the other almost never, so the journey
to the other side cannot be learned. In the T rooms this did not show (both legs are one short turn from the stem). An easier first stage
must make the other side lived often enough (e.g. both springs in both rooms at first, then one taken away) - to be registered before it runs.

# A3-1 (registered before running): teach both sides first, then the A3 layout
Stage 1 (beats 0-3,000): each room holds one water corner and one food corner (tools/a3_rooms.py --staged 3000); each body's spring of a kind
sits at one of its two corners and jumps to the other when used (the room engine's own rule), so both journeys are lived. Stage 2 (3,000-6,000):
one spring of each kind at the A3 corners, for every body (the A3-0 layout). The runs where A3-0 failed: s3_a water left (mirror 0), s3_b water
right (mirror 1); seed 9301.
Bars (stage 2 vs A3-0's same runs and window): fills of the A3-0-neglected spring (s3_a: water; s3_b: water) >= 100 in stage 2 (A3-0: 11 and 6
over the whole run), and deaths in stage 2 at most half of A3-0's (A3-0: 255 and 256 in the last two quarters).
Readings: met -> the direction habit gives way once both sides have been lived (a staged world can teach it); then A3-1 is the yardstick for
planning; not met -> the habit outlasts a lived counter-example; the being reverts to its side when the second spring goes.

## A3-1 result: not met - the side habit outlasts both-sides teaching, and A3 kills even with springs in both rooms
| run | stage 2 fills: neglected water / other | deaths per quarter (stage 1 | stage 2) | need fills per quarter |
|---|---|---|---|
| s3_a water left | 5 / 1,188 food | 126, 118 | 130, 130 | 2, 12, 1, 1 |
| s3_b water right | 95 / 356 food | 124, 118 | 128, 115 | 5, 13, 10, 37 |
Bars: neglected spring >= 100 in stage 2 - not met (5; 95 close); stage 2 deaths at most half of A3-0 - not met (260 vs 255; 243 vs 256).
Reading as registered: the habit outlasts a lived counter-example. But also (not registered, seen): with a spring of each kind in each room in
stage 1, deaths stay at ~120 per quarter - nearly every body dies twice per 1,500 beats even when the nearest room has both. So A3's journeys
(stem -> corridor -> doorway -> far corner, ~14 cells) are themselves beyond this being at this drain, whatever the side: it rarely gets
water anywhere in time. A3 needs a shorter first stage (the rooms one turn from the stem, as the T room's legs) before the side question or
planning can be asked. Two A3 runs of five used; next is that shorter stage, registered first.

# A3-2 (registered before running): the springs in sight from the corridor (A3's first stage)
In the T room each spring ends a straight leg, seen from the crossbar; in A3 it sits in a far corner, out of view from the corridor. One
change from A3-0: each spring stands at the back wall in line with its room's doorway (seen from the corridor looking up). Same runs where
A3-0 failed (s3_a water left, s3_b water right), seed 9301, 6,000 beats.
Bars: deaths per quarter in the second half <= 32 (A3-0: ~128), and fills of each kind >= 200. Readings: met -> out of sight was what made A3
lethal; the corner stage then tests knowing what is out of view (stage 1 of the roadmap); not met -> the journey's length or the side habit
is the wall; A3 shrinks further (third of five runs).

## A3-2 result: not met, but the first A3 that improves over the run
| run | deaths per quarter | need fills per quarter | water / food fills (median wait) |
|---|---|---|---|
| s3_a water left | 128, 125, 114, 118 | 0, 4, 25, 57 | 96 (342) / 542 (145) |
| s3_b water right | 128, 123, 110, 87 | 0, 6, 45, 101 | 223 (168) / 422 (331) |
Bars: second-half deaths <= 32 per quarter - not met (116, 99); each kind >= 200 fills - not met for s3_a's water (96). But unlike A3-0/A3-1,
deaths fall and need fills rise quarter by quarter, and the neglected side is now reached (water 96 / 223 vs 11 / 6 in A3-0): springs in
sight let the being learn the far side slowly. As decided in the re-plan (2026-11-26), A3 splitting stops here; A3-2 stays as a learnable
world for later stages.

# B1 (registered before running; the build step of the re-plan): LOKI v12's belief put into choice, in world A2
New in the being (all off by default, main line untouched): Mind.plan_goal - each beat, for a body whose lowest bedrock channel lacks > .3,
the place where its own v12 organ believes that channel's relief is now (resource_at; unnamed channel) is mapped to the nearest going place
unit; Going.lean_to leans the body toward it along the going organ's own lived crossings (hops ahead) with its own walking skill; the lean is
one more part of the score, weighted by its own earned record (Earned: correlation with wholeness change over the next 60 beats, floor .1).
No rule about what to do; the reward stays the bedrock.
Runs: tools/i0_v12.py --plan 1, world A2, s3_a 9101_0 and s3_b 9104_0, 64 bodies, 6,000 beats (A2-1's seeds). Control: A2-1's moving arm
(same world and seeds, no v12): deaths 6 / 91, food wait median 91 / 251.
Big outcomes (bars): deaths at most half of control (<= 3 / <= 45), and median food wait at most .75x control (<= 68 / <= 188).
Watched: the plan's earned weight over time, goals set, the being's own belief and v12's (as I0).
Smoke (16 bodies, 1,200 beats): 415 goals set; the plan's earned weight at its floor (.1) by then.

## B1 result: not met - v12's belief in choice changes nothing (and slows finding in one room)
64 bodies, 6,000 beats, A2 (control: A2-1's moving arm, same world and seeds, no v12):
| room | deaths (control) | food wait median (control) | first -> last half | plan's earned weight at the end | goals set | v12 food now / next correct |
|---|---|---|---|---|---|---|
| 9101 | 7 (6) | 129 (91) | 157 -> 93 | .41 | 62,515 | .56 / .47 |
| 9104 | 90 (91) | 264 (251) | 241 -> 304 | .10 (floor) | 106,316 | .47 / .38 |
Bars: deaths at most half of control - not met (7 vs 6; 90 vs 91); median wait at most .75x - not met (129 vs 91 is worse; 264 vs 251).
Reading: the belief the plan follows is right about half the time (.56 / .47), so a body led by it walks to the wrong corner almost as often
as to the right one; where its record stays low (9104) it is damped to its floor and nothing changes; where it earns some weight (9101)
finding gets slower. Built as one more part of choice, a memory that is right half the time does not help. As the re-plan says: one variant
after a failure, then the next structure. The variant that matters is v12's own belief on a real body (Astra's side, feedback file sent);
on ours, the next structure is not a better weighting but a better knower.

# V1 (registered before running): is the going memory too slow for a moving world
The going organ's relief memory (R, T, Vn) fades by .9995 a beat (half-life ~1,400 beats); A2's food moves every 300 beats, so its memory
averages the three corners - its belief sat at chance (.35). Theory: a memory's rate should match its world's volatility, and animals adapt it
(Behrens, Woolrich, Walton & Rushworth 2007). One variant on our side: fading .99 a beat (half-life ~70). s3_b in 9104 (where A2 costs most),
A2 moving and still, 6,000 beats, A2-1's seeds. Bars: moving - median food wait <= .75x A2-1 (<= 188) and deaths <= half (<= 45); still - deaths
not above A2-1 still + 3 (<= 4) and wait not above 1.25x (<= 88). Readings: moving better and still not hurt -> a faster memory suffices here;
moving better but still hurt -> the rate must adapt to the world's volatility (the structure to build next); no gain -> the time constant is not
the wall.

## V1 result: the being now knows where food is, and does worse - the map was forgotten with it
s3_b, 9104, going fading .99 for everything (R, Vn and the map T):
| | moving (A2-1) | still (A2-1) |
|---|---|---|
| believed food corner = true | .742 (.35) | 1.0 (1.0) |
| deaths | 113 (91) | 3 (1) |
| food wait median | 315 (251) | 82 (70) |
Bars: moving - not met (worse); still - met (deaths <= 4, wait <= 88). Reading: not one of the three registered readings - the knowing
improved strongly (.35 -> .74, the first time the being's own memory follows the moving food) while finding got worse. The fade also took
the map of crossings it walks by (T), which in this world does not change: it knew better where food is and lost its way there. That is the
S3 open problem in a new form: what lasts (the layout) vs what passes (where the food is now).

# V1b (registered before running; the one variant the rule allows): fast relief memory, slow map
Same as V1, one change: only where relief came (R, with its visits Vn) fades .99 a beat; the map of crossings (T) keeps .9995 (Going.rdecay).
s3_b 9104, A2 moving and still, A2-1's seeds. Bars as V1 (moving wait <= 188 and deaths <= 45; still deaths <= 4 and wait <= 88).
Readings: met -> gate 1 is in reach on our side by telling lasting from passing memory - next its rate set by the world's own change
(volatility) rather than by hand; belief stays high but finding does not improve -> knowing where is not enough; the journey there must come
from the memory too (v12.1's line).

## V1b result, moving arm (the still arm and the pair were not started the first time - a shell slip of mine; rerunning)
s3_b 9104, A2 moving, relief memory fading .99 a beat, map .9995: believed food corner .725 (V1 .742, A2-1 .35); deaths 131 (V1 113, A2-1 91);
food wait median 302 (V1 315, A2-1 251); died hungry 123. Not met (moving). Keeping the map did not bring finding back: knowing where the
food is now (.73) with a memory that fades between visits leaves the going's map with nothing to lean toward when a body grows hungry far
away - the fast memory is empty exactly when it is needed. As registered: "belief stays high but finding does not improve -> knowing where
is not enough; the journey there must come from the memory too". Still arm pending.

## World check (judge-side walkers, no being): A2 is solvable - even by "where I last ate + search"
Hand-written walkers on the true map (tools/a2_oracle.py), 64 bodies, 6,000 beats, A2 exactly as built:
| room | knows food now | last place eaten, then search the corners | still food |
|---|---|---|---|
| 9101 | deaths 0, wait 41 | deaths 0, wait 41 | deaths 0, wait 41 |
| 9104 | deaths 0, wait 67 | deaths 0, wait 67 | deaths 0, wait 67 |
The world is sound (no artefact holds scores down) and gate 1's bars are reachable without knowing the cycle: a place memory plus a search
when the place is empty is enough - with competent walking. So the being's gap is not (only) what it knows: V1/V1b's belief .73 did not turn
into arriving. Open: is its walking the gap, or the goal it walks toward?
V1b still arm (rerun): deaths 2, wait 68 (bars met: <= 4, <= 88) - fading only relief does not hurt a still world.

# D44 (registered before running; a diagnostic with the judge's help - never a pass): give the being the right goal
s3_b 9104, A2 moving, A2-1's seeds, 6,000 beats. For bodies whose food is low and below .7, the going goal (Mind.plan_goal, B1's path) is the
going unit of the TRUE food corner; walked by the being's own map and skill. Arm 1: weighed by its earned record (as B1); arm 2: full weight.
Readings: deaths and wait fall to near the walkers' (wait ~67) -> the being can get there once it has the right goal: the gap is the goal
(belief), and a belief that is right ~.75 needs "empty -> look elsewhere" (v12.1's failed-attempt evidence) to become right enough; they do
not fall -> its walking toward a goal is the gap (map, skill or how the lean competes in choice) - the belief-to-action interface itself.

## D44 result (full-weight arm): the right goal does not help - walking to a goal is the wall
s3_b 9104, A2 moving, the true food corner as the going goal at full weight: deaths 126 (A2-1 91), food wait median 280 (251), died hungry
108. Reading as registered: they do not fall -> the being's walking toward a goal (its own map, its skill, or how the lean fares in choice)
is the gap, not the belief. Rerun (same arm, and the earned arm) with two more readings: per beat, bodies holding the goal moving closer /
same / farther on the true map; and how far the going unit chosen for the corner really sits from it (cells).

## D44 rerun (approach read-out): the map is right, the steering is weak
| arm | deaths (A2-1 91) | wait (251) | steps closer / farther / same cell | goal unit from the corner |
|---|---|---|---|---|
| full weight | 126 | 280 | 20,959 / 13,657 / 131,162 | 0.40 cells |
| earned (ended at .1) | 79 | 290 | 21,418 / 14,701 / 124,296 | 0.39 cells |
The going unit for the corner sits 0.4 cells from it (the map holds the corner). Of the steps that change cell, 60% go closer and 40%
farther (a walker on the true map: nearly all closer). Reading: the being's own walking skill turns it toward a known place only weakly.
Gate 1 narrowed: knowing where (V1 .74, v12.1) is partly in hand; arriving there is not.

# D45 (registered before running): the walking skill learns four times as often
Going.skill (goal-conditioned, trained on lived pieces) learns every 250 beats instead of 1,000 (same updates per call); nothing else.
Setting as D44's earned arm (true goal, earned weight; s3_b 9104, A2 moving, A2-1's seeds) - the steering read-out is direct there.
Bars: closer share of cell-changing steps >= .75 (now .59) and deaths below A2-1 (91). Readings: met -> the skill was under-trained for
places it seldom lived; then the same without the judge's goal (the being's own V1-style belief); not met -> learning more is not enough:
the founder's decision on an innate homing vector read from path integration (vector navigation, as insects and mammals).

## D45 result: not met - learning the skill four times as often does not sharpen the steering (and costs lives)
Closer share of cell-changing steps .575 (D44 earned .593); deaths 174 (D44 earned 79, A2-1 91); wait 292. Reading as registered: learning
more is not enough. The being's goal-conditioned walking skill, trained on its own lived pieces, does not become a reliable way to a known place
- more updates on the same lived pieces made it worse, not better. The next step is the founder's decision: an innate homing vector read from
path integration (the direction to a remembered place computed from where the path organ says the body is and where it was at that place -
vector navigation, as insects and mammals). Gate 1 budget: 7 of 10 used.

# Draft, NOT registered (awaiting the founder's decision): an innate homing vector from path integration
What exists: the path organ (body_contract.PathOrgan, innate since 2026-11-24, founder-approved) integrates the body's own felt motion into
a position p and heading h (birth frame) and offers multi-scale phase codes of p to the mind. The going organ remembers places as codes.
What it would add (one innate organ, no want, no rule about when to go): for any remembered place (a going unit), the position the path organ
held when the body was there (kept per unit as a running mean of p, the body's own integrated position - not the judge's); the homing vector
to it = that position minus the current p, turned into the body's frame by -h; offered to the walking skill as its goal input (instead of, or
beside, the goal's code). Animals: desert ant and rodent vector navigation (Wehner; Etienne & Jeffery 2004; grid-cell vector models, Bush,
Barry, Manson & Burgess 2015). Whether to walk there stays the being's (the lack, the value, the going lean); the vector only says which way.
Limits: drift of path integration over long trips (reset at birth only; landmarks do not correct it yet); walls - a straight vector points
through them, so it guides inside open space and the map's chain of units still has to route around walls (the vector to the NEXT unit on the
chain, not to the far goal).
First check if approved (D46, judge's true goal as in D44, earned arm): closer share of cell-changing steps >= .75 (now .59) and deaths below
A2-1's 91; then without the judge's goal. Budget: gate 1 has 3 of 10 left.

# D46 (registered before running; founder 2026-11-26 "Yes, register it"): an innate homing vector from path integration
Built as drafted (atanor_os/emerge/going.py, Going.homing, default off): each place unit keeps the mean position the path organ held there (the
body's own integrated position); toward a goal, the vector to the NEXT unit on its own chain of lived crossings, in the body's frame; each
gesture scored by the body's own forecaster's foreseen felt motion (un-normalised by its own sense statistics): progress along the vector +
heading alignment after the turn (equal weight, each standardised). No gesture is named; whether to go stays with lack, value and choice.
Setting as D44 (s3_b 9104, A2 moving, A2-1's seeds, the true food corner as goal for food-low bodies), two arms: full weight, earned weight.
Bars: closer share of cell-changing steps >= .75 (D44 .59 / .59) and deaths below A2-1's 91, in the earned arm. Readings: met -> the being can
now walk to a known place; next the same without the judge's goal (its own belief: V1b's relief memory or v12.1), which is gate 1 itself;
closer share up but deaths not down -> arriving is no longer the wall; the bodies die of something else (what they do on arrival, water).
Smoke (full weight, 800 beats): closer share .71, deaths 0.

## D46 result: with the right goal, the homing vector lets the being arrive and live - when it is heard
| arm (9104, true goal) | deaths (A2-1 91) | food wait (251) | died hungry | closer share | weight |
|---|---|---|---|---|---|
| full weight | 5 | 131.5 | 3 | .692 | 1.0 |
| earned weight | 126 | 292 | 114 | .570 | .10 (floor) |
Bars (set on the earned arm): not met (.57; 126). But the full-weight arm, the same organ heard at the going lean's own weight: deaths 91 -> 5,
wait 251 -> 132 - the first time the being reaches a known place reliably enough to live in A2. The earned arm failed because the earned
record (correlation of push with wholeness change over the next 60 beats) cannot credit a trip whose payoff comes only on arrival; it floored.
The main line's own going lean is not weighed by that record (it enters at W = 1), so the consistent step is to make the homing vector the
going organ's own way of walking, at its own weight, toward where its own memory believes the relief is.

# D47 (registered before running; gate 1 itself - no judge's help): the being's own memory + the homing vector
The going organ's own lean (its chain toward the unit its relief memory values most for the lacking channel, hops ahead) walked by the homing
vector (Going.homing) instead of the skill; relief memory fading .99 (V1b: belief of the food corner .73), map .9995. No oracle goal, no plan_goal.
Runs: s3_b 9104 and s3_a 9101, A2 moving and still, A2-1's seeds, 6,000 beats (4 runs, two at a time).
Gate 1 bars (moving, vs A2-1 moving): food wait median <= .75x (<= 188 / <= 68) and deaths at most half (<= 45 / <= 3); still: deaths not above
A2-1 still + 3 and wait not above 1.25x. Readings: met in both rooms -> gate 1 passed on our side (then seeds and MineLand); met in one ->
partial, a second start; not met -> what is left is the belief (v12.1 next) or the cost of a fast memory away from food (V1b).
Gate 1 budget: this is 9 of 10.

## D47 result: not met - wired into the going's own chain, the homing vector kills (my wiring, not the vector)
| run | deaths (A2-1) | food wait (A2-1) | believed food corner |
|---|---|---|---|
| 9104 moving | 314 (91) | 419 (251) | .32 |
| 9104 still | 228 (1) | 74 (70) | .64 |
| 9101 moving | 136 (6) | 359 (91) | .34 |
| 9101 still | 0 (0) | 62 (46) | 1.0 |
Reading: worse everywhere, even in a still world (9104). D46 walked a breadth-first path to one chosen goal unit (deaths 91 -> 5); D47 hooked
the vector into the going's own chain (the highest-valued neighbour, hop by hop), whose values under a fast-fading relief memory are near
zero and tied - the chain points almost anywhere, and the vector walks the body there firmly (the skill had been soft enough to forgive it).
The D47 hook is switched off (Going.homing_lean). Gate 1 budget: 9 of 10.

# D48 (registered before running; the last gate-1 run): D46's way with the being's own belief as the goal
Goal (no judge): for a body whose lowest store lacks > .3, the going place unit with the highest relief value for that channel (its own
relief memory, fading .99 as V1b); reached by the breadth-first path over its own lived crossings and the homing vector (D46's path), at full
weight (the going lean's own weight). s3_b 9104 and s3_a 9101, A2 moving and still, A2-1's seeds, 6,000 beats.
Gate 1 bars as D47. Readings: met in both rooms -> gate 1 passed (then seeds, MineLand); not met -> gate 1 is not passed within its budget on
our side: the stop signal is reached for our route, and with Astra's v12.x still below .8 on a real body, the roadmap's rule puts plan B
(atanor_os/PLAN_B_WORLD_MODEL_CHOOSER.md) to the founder.

## D48 result: gate 1 met in 9104, missed narrowly in 9101 - the first time the being keeps up with a moving world by itself
Own belief (relief memory .99) as the goal, breadth-first path over its own lived crossings, homing vector; no judge's help. A2-1 in brackets.
| run | deaths | food wait median | died hungry | believed food corner | gate 1 bar |
|---|---|---|---|---|---|
| 9104 moving | 12 (91) | 154 (251) | 12 | .725 | deaths <= 45, wait <= 188 - met |
| 9104 still | 0 (1) | 53 (70) | 0 | 1.0 | <= 4, <= 88 - met |
| 9101 moving | 9 (6) | 69 (91) | 8 | .708 | deaths <= 3 - not met; wait <= 68 - not met by one beat |
| 9101 still | 0 (0) | 36 (46) | 0 | 1.0 | met |
Reading: in 9104, where the moving food cost most, deaths fall 91 -> 12 and the wait 251 -> 154; still worlds get better, not worse, in both
rooms (wait 70 -> 53, 46 -> 36). In 9101 the wait falls 24% (bar 25%) and deaths are 9 vs 6 (small counts, one start). As registered for the
gate (D47's readings): met in one room -> partial, a second start. The budget of ten is used; whether the second start runs (the eleventh) or
the stop signal is read is the founder's call. This is the strongest result on the line: knowing where (its own fast relief memory) + an
innate homing vector + a path over its own map lets the being keep up with a changing world without any help.

# D48 second start (registered before running; the 11th gate-1 run, one past the budget - run because D47/D48's registered reading
"met in one room -> partial, a second start" calls for it)
Same as D48 with new seeds (+1000), both rooms, moving and still. Control for the moving arms: A2-1's moving arm at the same new seeds
(the current being without homing/own-goal), run alongside, so the bars compare like with like. Gate 1 read over the two starts summed:
moving - wait median (pooled) <= .75x control's and deaths <= half of control's, in both rooms; still - deaths <= control + 3, wait <= 1.25x.

## Gate 1 over two starts: met in 9104, not met in 9101 - the capability is there, the bar in the easy room is not reached
Homing + own goal (D48) vs the current being at the same seeds (start 1: A2-1; start 2: same-seed control):
| room | deaths, two starts summed | wait median (mean of two starts) | ratio | gate 1 bar | still world (deaths; wait ratio) |
|---|---|---|---|---|---|
| 9104 moving | 39 (12 + 27) vs 144 (91 + 53): -73% | 181 vs 266 | .68 | met (<= half, <= .75) | 0 vs 1; .74 - no harm, better |
| 9101 moving | 16 (9 + 7) vs 15 (6 + 9): +7% | 78 vs 100 | .78 | not met | 0 vs 0; .83 - no harm, better |
Reading as registered: gate 1 needs both rooms; it is met in one. In the room where moving food was costly (9104) the being now keeps up:
deaths fall by three quarters and the wait by a third, on both starts. In 9101, where the current being already lives well with moving food
(about 7 deaths per 6,000 beats), the wait falls 22% (bar 25%) and deaths do not fall (16 vs 15 - counts near the floor). And in still worlds
the new way is better, not worse, in both rooms.
My error in the bar: "deaths at most half" in a room whose baseline has ~7 deaths cannot be read (one or two deaths decide it); the wait bar
was the meaningful one there and it was missed by 3 points. Gate 1 budget: 11 used (one past ten).
For the founder: (a) read gate 1 as passed in substance (adopt homing + own goal into the main line; check MineLand; go on to gate 2 with
bars that avoid floor counts), or (b) hold to the letter (not passed in both rooms) and take the stop signal to plan B. My recommendation is
(a); the letter is (b).

# Founder's reading of gate 1 (2026-11-26): passed in substance
The founder read gate 1 as passed in substance: the capability is there (9104: deaths -73%, wait .68x over two starts; still worlds better in
both rooms), and the miss in 9101 sits on a bar near its floor. The bar as written was not met in 9101; that stays on the record.
Adopted into the main line (atanor_os/emerge/going.py Going.innate / Going.own_goal; mind.py uses its own goal at full weight from beat 50):
the innate homing vector (D46) + seeking its own believed relief (D48) + relief memory fading .99 (V1b). Switched on in the room builder
(tools/s4_kin.being) and in MineLand's going organ (tools/ml_live.py --going 1); ATANOR_INNATE_HOMING=0 reproduces every run before this.
ml_live: a recovery now keeps the body's own path organ (its integrated place) as it keeps the going organ, so the homing frame survives.
Check before use (registered): the main-line default in a2_moving_food with no flags reproduces D48 9104 moving (deaths 12, wait 154).

# Gate 2, run 1 (registered before running): A3 with the springs in the far corners, out of view, with the adopted main line
Gate 2 (EMERGE_PLAN, budget 5 registered runs): places out of view and journeys. A3-0's layout (each spring in a far room corner, not seen
from the corridor), s3_a (water left) and s3_b (water right), seed 9301, 6,000 beats, tools/a3_rooms.py, main line as adopted. Control:
A3-0 (same layout, seed and beings, before adoption): ~128 deaths per quarter, the far side almost never reached (water 11 / 6 fills).
Bars (as A3-2's, which were reachable in shape): second-half deaths <= 32 per quarter, and fills of each kind >= 200, in both runs.
Readings: met -> homing + own goal carry the being through journeys out of view; gate 2's next step is MineLand's cake return.
Not met but deaths fall quarter by quarter (as A3-2) -> the far place is learned slowly: the next run is A3-2 (springs in view) with the
main line, to see whether out of view or the journey is what remains. Not met and flat -> its own memory never gets the far relief, so own
goal has nothing to seek (finding first, as G1 in MineLand): report, and the next structure is finding (curiosity toward unknown places,
E1-style second bedrock), not a variant.
Port check (founder 2026-11-26: room principle to MineLand at once): G1's arm A again (wear mineland25/s5000, --going 1, 5,000 steps,
--home -10) with the main line; compared with G1 A (deaths 66, fills 82, cake reach 0 in all windows) and G1 B (85, 124). Reading: deaths
not above G1 A + 25% -> no harm in carry-over; cake reach > 0 in any window -> something new to remember there.

## Check before use: met - the main-line default reproduces D48 exactly
a2_moving_food, 9104 moving, no flags: deaths 12, wait median 154, died hungry 12, believed corner .725, fills [2775, 1852] - identical to D48.

## Gate 2 run 1 result: not met, flat - its own goal holds firmly to places that do not hold relief in the new layout
A3 (springs in far corners, out of view), seed 9301, 6,000 beats, main line (A3-0 before adoption in brackets):
| run | deaths per quarter | need fills per quarter | water / food fills (median wait) | lacking beats in the spurs |
|---|---|---|---|---|
| s3_a water left | 128, 128, 128, 128 (128, 127, 127, 128) | 0, 0, 0, 0 (0, 1, 3, 1) | 0 / 243 (289) (11 / 489 (139)) | .47 (.21) |
| s3_b water right | 128, 128, 128, 128 (128 x 4) | 0, 0, 0, 0 (0, 0, 1, 0) | 302 (106) / 0 (6 / 587 (130)) | .37 (.12) |
Goal changes while held (judge count, Mind.goal_switch / goal_held): s3_a 576 / 356,672 (.16%), s3_b 429 / 356,672 (.12%).
Reading as registered: not met and flat. The far side is never reached (0 fills of the far kind in both), and the near kind is reached LESS
than before (243 vs 489, 302 vs 587 fills) while lacking bodies spend twice the share of their time in the spurs. Its own goal is the place
its relief memory values most, carried from the T room; in the new layout that place (by its own integrated position) is a spur, and the
goal almost never changes (.1-.2% of held beats). So: D47's lesson again, now with its own belief - a firm, skilled walk to a place that no
longer gives. Two things are missing, and they are the same thing seen twice: (1) finding (nothing leads it to where it has never found
relief - G1 in MineLand); (2) giving up a goal when going there brings nothing (the goal is held, not revised; v12.1's failed-attempt
evidence and the founder's / Copilot's "think again only when expectation fails"). As registered, the next is a structure, not a variant.
Gate 2 budget: 1 of 5 used.

# View check (registered before running; judge side only, founder 2026-11-26: "is the viewpoint too high? it sees food from afar and loses
it up close"). Code reading first: the eye sits at half wall height; food is a full-height block (not a floor object), so height cannot hide
it; the picture is 80 degrees wide and a turn is 24 degrees. Record (a2_moving_food --view 1, 9104 moving, main line, A2-1's seeds): share of
beats the food block is in any column of the picture by distance (<1.5, 1.5-3, 3-6, >6 cells), and hungry visits within 2 cells that end
by eating vs by leaving (past 3 cells) or dying, with the share of in-zone beats the food was in view. Readings: in view near (<1.5) far below
far (>6) AND most near misses (>= 60%) had food in view under half the time -> the narrow view is a world-side problem: put a wider view to
the founder; else the near misses are the being's (it sees and still leaves), not the eye's.

# Gate 2 run 1 explained (registered before running; judge side only, explains a built result, does not count against the budget)
a3_rooms --why 1, s3_a water left, seed 9301, 3,000 beats, same main line. Per beat for bodies holding an own goal: share at the goal unit,
share bumping a wall (vs bodies without a goal); for the most-held goal units: where the unit really is (judge's running mean of body
positions there), whether in a spur, its own relief rate (R / visits) and value V per channel. Readings: goals mostly at units whose relief
rate is ~0 but whose V is high -> the goal reads V (discounted value of where it leads) where it should read where relief came: the goal's
source is wrong, and a goal from the relief rate is the one change to try. Goals at units with real relief, bodies seldom there and bumping
more than free bodies -> the homing vector points into walls in a new layout (its own positions carried from the T room); the walk is wrong.
Goals at relief units, bodies reach them -> the memory is right and the world gives nothing there (springs moved): giving up on arrival
without relief, and finding, are what is missing.

## View check result: not met - the eye loses nothing near food; near misses are the being's
9104 moving, main line (deaths 12, wait 154: the same run as D48). Food block in the picture by distance: < 1.5 cells .73, 1.5-3 .40,
3-6 .25, > 6 .09 - near is the MOST visible, not the least. Hungry visits within 2 cells: 574 end by eating, 86 by leaving or dying (13%);
of the 86, the food was in view under half the in-zone beats in 49 (57%, bar 60%); when it ate, food was in view .77 of the in-zone beats.
Reading as registered: the narrow view is not a world-side problem (the first condition fails outright). The founder's sighting ("sees food
from afar, loses it up close, comes back") is real but rare (86 of 660 near visits) and is about half turning away, half leaving with food
in sight. Side finding from the founder's viewer (one body): relief memory fading .99 a beat suits 64 bodies pooling their reliefs; one
body eats about every 400 beats, so its memory of where food was is nearly gone (half-life ~70 beats) when it is hungry again. To register:
a one-body check (fading by beats vs by its own experience).

# D46b (registered before running; founder 2026-11-26 on the viewer: "it knows where to go but cannot move its body skilfully")
The viewer shows hexagon loops around the food corner: 0.4 cell forward + 24 degrees a beat traces a polygon ~2 cells across. Cause in my
D46 formula: a gesture scores by progress ALONG the homing vector (projection), so overshooting still scores, and forward+turn beats turning
in place even inside the turning circle. Change (one): the progress part becomes minus the distance left after the gesture, by its own
foreseen felt motion (an overshoot counts against it); alignment part unchanged. Not a behaviour rule: the same forecaster, the right measure
of "closer". Run: A2 9104 moving, main line + D46b (--home-rem 1 --view 1), A2-1's seeds, 6,000 beats; control = the view-check run (main
line, same seeds: deaths 12, wait 154). Also recorded: beats spent within 2 cells while hungry before eating (docking). Bars: docking beats
median down by >= 25% and food wait not worse (<= 154 x 1.1), deaths not above 12 + 5. Met -> adopt into the main line; then the same in 9101
and A3 (the walk out of view). Not met -> the loops are not the formula's; next is the body's motor range (graded speed and turn).

## Gate 2 run 1 explained: most goals are place units it has never been to in this world, walked to its own birth point - my bug
s3_a, 3,000 beats (judge side). Held-goal beats 177,024; at the goal unit 259 (.15%). Most-held goal units:
| unit | share of held goals | judge-seen in A3 | its own homing position | relief rate (water, food) | V |
|---|---|---|---|---|---|
| 39 | .71 | 0 (never there) | (0, 0) = never set = its own birth point | 0, 0 | 0, .006 |
| 36 | .16 | 492, in a spur | (2.4, 4.0) | 0, 0 | 0, .009 |
| 40 | .13 | 0 | (0, 0) | 0, 0 | 0, .010 |
(Wall bumps read 0 for every body - the bump reading is not informative here and is not used.)
Reading: none of the three registered explanations exactly; nearest is the first (the goal's source is wrong), and the cause is a bug in my
own goal (Going.own_goal): (1) it takes the argmax of V even when every value is ~0 - after the relief memory fades (.99 a beat) and with
no far relief ever found, the "goal" is the top of noise; (2) units never visited in this layout have no homing position (Pu stays at its
zero start), which is the body's own birth point - so the homing vector walks it back toward where it was born, and it never reaches
the unit, so the goal never changes. In A2 every unit was visited, so this did not show. Fix (a correction, not a gate-2 run): a goal only
where its own memory holds relief for that channel (relief rate > 0) and only at a unit whose place it has been (homing position set);
otherwise no goal (its ordinary choice). Check: A2 9104 unchanged within noise (deaths <= 17, wait <= 170). Expected in A3: back to about
A3-0 (no harm, no gain) - the far spring still has to be found first. Then gate 2 run 2 is the finding structure.
Order amended before any D46b data (the own-goal fix landed in the code first): (1) the fix check (main line + fix, --view 1) is run and
read against the view-check run; (2) D46b's control becomes that fix-check run (same code but --home-rem), so D46b still changes one thing.
D46b reading amended before its data (founder 2026-11-26, "명심해"): distance and direction must be content offered to choice, never the
decider; "least distance left" choosing the gesture is a hidden objective function. D46b is therefore a diagnostic only (does the loop come
from the measure of closeness?) and is not adopted into the main line whatever it shows. Said plainly about what is already adopted: D48's
own goal enters choice at full weight (not weighed by its own record), so in practice the walker decides once a goal is set - the same
borderline. The principled form, to design next: the homing organ offers, per gesture, what its own forecaster says that gesture does to the
way left (content); how much that counts is learned from its own wholeness change (the one allowed value), as the other parts of the score.

## Fix check: not met - the fix costs A2 (deaths 19 vs 12, wait 212 vs 154; bars <= 17, <= 170)
Requiring a relief rate > 1e-4 also drops faint memories: with relief fading .99 a beat, where food was an hour of beats ago is faint but
still points the right way (its relative values carry it), and in A2 that faint pointer helped. So the faint values in A3 were faint
memory from the T room, not noise; the real bug is only (2): units never visited in this layout have no homing position (zero start =
birth point). Fix 2 (one amendment, before any further run): keep only the homing-position condition (Pn > 0); no relief threshold.
Check as before (A2 9104: deaths <= 17, wait <= 170), then A3 s3_a 3,000 beats --why to see where its goals go.

## D46b result (diagnostic only, not adopted): the loops come largely from my measure of closeness
Against the fix-check run (same code but the measure): docking beats (hungry, within 2 cells, until eating) median 10 -> 6 (-40%, bar
-25%), mean 17.8 -> 12.7; food wait 212 -> 180; deaths 19 -> 8. So "progress along the vector" makes the body circle near the goal; a
measure that counts overshoot docks faster. As the founder ruled, this stays a diagnostic: the main-line form must offer this as content
(per gesture, what its forecaster says the gesture does to the way left) and let choice learn how much it counts.

## Fix 2 check: not met, and the same numbers as fix 1 (A2 9104: deaths 19, wait 212, docking 10) - my reading of fix 1 was wrong
The relief threshold changed nothing in A2; what costs A2 is the homing-position condition itself. In A3 (s3_a, 3,000 beats) fix 2 sends
goals to units it has been to (36, 38: in the spurs, .75 / .25), bodies are at the goal 12.6% of held beats (was .15%), food fills 26 (11),
deaths the same (64 a quarter), water never found. Implication to check before anything else: in 9104 the birth point (path-organ zero) is
the stem's top, the hub of the T room, equally near all three food corners. So D48's walk "to a never-visited unit" was a walk to the hub
- a return-to-nest move that nobody registered, and it may be part of gate 1's result.
# Check (registered before running; judge side): how much of D48's A2 result came from goals at units with no homing position
A2 9104 moving, ATANOR_GOAL_FIX=0 (D48 as it ran; must reproduce deaths 12, wait 154), counting the share of goal-beats at units with no
homing position. Reading: share >= 25% -> gate 1's result rests in part on an unregistered return-to-birth-point move; say so in the gate-1
record, and treat "return to nest when the goal is unknown" (path integration's home vector, as desert ants) as a candidate innate trait to
put to the founder, not as found by the being. Share < 25% -> the fix's cost has another cause; look before changing anything.

## Speed (founder 2026-11-26, roadmap step 1): the path search copies the map to the host once and searches once per starting unit
Profiled (400 beats): choice 19.6 s -> 7.2 s; the search was 40% of a beat (per-body GPU reads and writes). Equivalence over a full run:
A2 9104 moving, 6,000 beats, every number identical to the run before the change (deaths, fills, waits, view check); 456 s -> 394 s with
other runs sharing the GPU. Next costs: the world's step (12 sub-steps a beat, 6.5 s / 400 beats) and learning in after() (5 s).

## D48 birth-point check: < 25% - the result does not rest on it; and my "bug" reading was mostly wrong
ATANOR_GOAL_FIX=0 reproduces D48 exactly (deaths 12, wait 154). Goal-beats at a unit with no homing position: 6,459 of 161,334 (4.0%);
goal changes 1,026. Reading as registered: the fix's cost has another cause. Looking (code, no run): the homing vector aims at the unit
`hops` (4) ahead on the breadth-first path over its own lived crossings, not at the goal itself; the goal's own position matters only in
the last hops. So a never-visited goal is still walked toward correctly through units it knows - "walks to its birth point" happens only
near the end of such a way, rarely. Excluding those goals (fix 1 / fix 2) throws away correct goals (where it remembers food from its
earlier life, not yet visited in this one) and costs A2 (12 -> 19 deaths, 154 -> 212 wait). In A3 the goals were the T room's remembered
places, walked along T-room crossings - the old map, not a bug. Decision: the main line goes back to D48's goal (ATANOR_GOAL_FIX default
0) once LW1's arms (all run with fix 2, so they compare among themselves) are done. Gate 1 stands as recorded.

# W1 wall pattern (registered before building; founder 2026-11-27, a quick world check: "walls look all alike - put patterned wallpaper?")
Now: each wall face has one colour, tinted by which way it faces (the 2026-11-24 far cues, N/E/S/W) and shaded by distance - so two places
facing the same way look nearly the same, and moving along a wall shows no flow. Motion itself is known from the body (felt motion -> path
organ); the question is whether the EYE can tell places apart and follow motion. Change (world, one): RoomBody.pattern - every wall cell
face carries vertical stripes whose spacing and phase come from a fixed hash of the cell and face (a fixed property of the world, like
real wallpaper; nothing about food or water). The being's eye is unchanged (its encoder never saw stripes).
Run: A2 9104 moving, main line, A2-1's seeds, 6,000 beats, with and without pattern (judge records identical). Judge-side probe: ridge
regression from the being's own eye code (Organs.z) to the judge's (x, y), fitted on beats 0-3,000, tested on 3,000-6,000.
Bars: probe error falls >= 25% with pattern, and the being is not worse (deaths <= control + 5, wait <= 1.1x control). Readings: both ->
put patterned walls to the founder as the room's default; probe better but behaviour worse -> the eye needs to grow into it first (pattern
during growth, a later step); probe not better -> the eye code does not carry place even when the world does (an eye question).
Runs after LW1 (one GPU run at a time).

## Correction (2026-11-27): the view check measured the wrong picture - the founder was right; the being cannot see food when it can eat
The view check counted food in the ray-cast columns (room3d_gpu.picture's geometry: eye at half a 1-cell wall, springs full height). The
being's own eye is RoomBody.render: eye 1.6 cells high, walls 2.0, springs 1.0 high, horizon at row 102 of 160 (58 rows below it). A
spring's top is below eye level, so near it the spring falls under the bottom of the picture. Rendered (body facing the food block's face
in the leg): food pixels at distance .6 / .9 / 1.2 / 1.5 / 2.0 cells = 0 / 0 / 0 / 96 / 1,072 at eye 1.6, and 8,142 / 6,692 / 5,612 /
4,662 / 3,672 at eye 1.1. Reach is 1.2 cells: at eye 1.6 the food is out of sight exactly when it can be eaten - the "sees it from afar,
loses it up close, comes back" the founder saw, and a world-side cause of the docking loops. My answer that the eye was fine was wrong
(I read the wrong renderer). The founder decided on 2026-11-24 to lower the eye (1.1) "from the next world, if it checks out"; it was
never applied. The viewer had the same error (it drew the columns' picture); it now draws the being's own picture with its fovea and rings.

# E-height (registered before running; founder 2026-11-27 "Yes, go to 1.1", applying his 2026-11-24 decision)
The room's eye height goes from 1.6 to 1.1 cells (RoomBody.default_eye), so a spring stays in the picture up to touching it. The being's
eye grew at 1.6; nothing else changes. Run after LW1: A2 9104 moving, main line with D48's goal (ATANOR_GOAL_FIX default back to 0 first),
A2-1's seeds, 6,000 beats, --eye 1.1 --view 1. Control: the D48 reproduction (goalfix0_9104: deaths 12, wait 154, docking beats median
in its record). Bars: docking beats median down >= 25%, deaths <= 17, wait <= 170. Readings: met -> 1.1 becomes the room's default
(all later room runs; earlier runs keep 1.6 in their records); docking not better -> the loops are the walk's (D46b), not the eye's;
worse elsewhere -> its eye needs to grow at the new height (a growth stage), reported before any default change. 9101 follows if met.

## E-height result: not met as registered (docking -17%, bar -25%); deaths and wait better
A2 9104 moving, eye 1.1 vs the D48 reproduction (eye 1.6): deaths 7 vs 12, wait 161 vs 154 (bar <= 170), docking beats median 5 vs 6
(mean 10.9 vs 12.6), near visits ending in eating 571 / leaving 114 (574 / 86). Reading as registered: docking not better by the bar ->
the loops are mostly the walk's (D46b), not the eye's; 1.1 is not made the default on this run. (The eye grew at 1.6; one start.)

# Anticipation check (registered before running; founder 2026-11-27 saw it wait in a corner before food came; Copilot: log, don't read
screenshots). Judge side: in the last 100 beats of each food period, body-beats within 1.6 cells of each corner. Each corner is compared
with itself: its count when it is the NEXT corner vs when it is the OTHER one (neither current nor next) - this removes "it likes that
corner anyway" (e.g. top-left is on the way to water). Run: the D48 setting (main line), A2 9104, A2-1's seeds, 6,000 beats, after LW1b.
Reading: next > other by >= 1.5x for at least two of three corners -> an early expectation of where food goes next (never seen before on
this line: A2-0/A2-1 found none); else -> staying at good places and food arriving (Copilot's hypothesis A).

# World bug (found by the founder on the viewer, 2026-11-27): food moved onto a body traps it inside the block
A2's put() turns the new corner's cell into food for every body, wherever the body stands. A body in that cell is then inside a block,
and the world lets a body move only into floor cells - so it cannot leave until food moves again (300 beats). It can eat (the reach ray
starts inside food) but cannot reach water: seen on the viewer with food .99, water .07. Present in every A2 run (A2-0 on, D44-D48, LW1,
E-height); both arms of each comparison had it, but a being that waits in corners (homing + own goal) is trapped more often.
Count added (judge side): body-beats inside a block, and deaths inside. Smoke: 20 body-beats in the first 320 beats (one move).
Plan (registered): the anticipation run (D48 setting) records it. If deaths inside >= 5% of deaths or trapped body-beats >= 1% of all,
the world is fixed (a body may step out of the cell it stands in, whatever that cell has become - the least change; nothing else) and
gate 1's pair (A2-1's moving arm and D48, 9104 and 9101) is rerun in the fixed world before anything else builds on A2.

## Anticipation check: not met - it lingers where food WAS, it does not wait where food WILL be
D48 setting, 9104, 6,000 beats (reproduces D48: deaths 12, wait 154). Body-beats near each corner in the last 100 beats of a period,
each corner compared with itself (next vs other = the corner food just left):
corner 0 (leg end): 127 vs 61 (2.1x); corner 1 (top-left): 705 vs 1,420 (.50x); corner 2 (top-right): 91 vs 219 (.41x).
Reading as registered: one of three >= 1.5x -> no expectation of the next place. Two corners show the opposite: bodies stay more at the
corner food has just left (its relief memory lags the world), as A2-1 found. The founder's "waits in the corner" scenes are lingering at a
remembered place and, in part, being trapped (below). Copilot's hypothesis A stands.

## World bug size: over the bar -> world fixed
Same run: trapped body-beats 7,200 of 384,000 (1.9%; bar 1%); deaths inside 0 (deaths 12). Fix (one change): a body standing in a cell
that became a block may move within it and step out (atanor_os/emerge/room_body.py RoomBody.step, under ATANOR_WORLD_FIX=1; my first try
put it in room3d_gpu.World.step, which the room body does not use - caught by a smoke check). Smoke, 700 beats: trapped body-beats 300 -> 2.
# Gate 1 in the fixed world (registered before running): A2-1's moving arm (ATANOR_INNATE_HOMING=0) and D48 (main line), 9104 and 9101,
A2-1's seeds, 6,000 beats, ATANOR_WORLD_FIX=1. Bars as gate 1's (deaths <= half of the control, wait <= .75x the control) read per room.
Reading: met as before (9104) -> gate 1's result is not the bug's; the fixed world becomes the default (ATANOR_WORLD_FIX default 1) and
later A2 work uses it. Changed -> report the size before anything builds on A2. Runs after LW2 (LW2 compares with controls in the old world).

## Gate 1 in the fixed world: the effect shrinks - the bug hurt the control more than the homing being
A2, A2-1's seeds, 6,000 beats, ATANOR_WORLD_FIX=1 (old world in brackets):
| room | A2-1 control deaths / wait | D48 deaths / wait | deaths change | wait ratio | gate 1 bars (<= half, <= .75x) |
|---|---|---|---|---|---|
| 9104 | 46 / 247 (91 / 251) | 25 / 174 (12 / 154) | -46% (was -87%) | .70 (.61) | deaths not met (25 vs <= 23), wait met |
| 9101 | 5 / 93.5 (6 / 91) | 11 / 82 (9 / 69) | +6 bodies (small counts) | .88 (.76) | not met |
Bodies still in a block for a while: 92-449 body-beats (they can leave now; standing in the food cell they can eat from inside); none die
there. Reading as registered ("changed -> report the size before anything builds on A2"): in the true world the innate homing + own goal
still shortens the hungry wait in both rooms (.70x, .88x) and roughly halves deaths where the moving food was costly (9104), but the
"deaths -73% over two starts" figure was inflated by the bug, which trapped the control's bodies more (91 -> 46 deaths when fixed). Gate
1 by its letter is not met in the fixed world in either room. Said plainly to the founder: the capability is real but about half the
size first reported. Decision taken while the founder sleeps (he asked me to carry on): the fixed world is the true world, so every A2
run registered from now uses ATANOR_WORLD_FIX=1; LW3's runs already queued stay in the old world only because their controls are there.

# G2a (registered before running; diagnostic, does not count against gate 2's budget): can unfamiliarity reach the far side at all?
Theory first: gate 2 run 1 never reached the far kind (0 fills); LW3 in A3 (s3_a, 3,000 beats) distrusts the old map (trust .02-.15) and
deaths fall 64 -> 52 in the last quarter, but nothing leads it to where it has never found relief. Before building a learned trust for an
unfamiliarity content, check cheaply whether such content, at a fixed designer weight, gets bodies to the far spring at all.
Runs: a3_rooms, s3_a (water left), seed 9301, 3,000 beats, true world, main line (learned trust in homing) - arm M; the same + E1b's
unfamiliarity in the score at weight 1 (--roam 1) - arm U. One start each (diagnostic: the answer sought is 0 vs not 0).
Readings: U's far-kind (water) fills > 0 and M's ~0 -> the content can carry bodies there; next = its trust learned in the LW3 frame
(promise kept = a relief event after following it), registered as gate 2 run 2 with >= 3 starts. Both ~0 -> unfamiliarity of the
foreseen state does not reach through the doorway in 4 beats of foresight: report; the next is the reach of the content (longer
foresight or the place organ's thin units as content), not a weight. U far fills > 0 but deaths up by > 25% -> say so; same next step.

## G2a result: the content reaches the far side, but at a fixed weight it trades the near kind away
s3_a, seed 9301, 3,000 beats, true world. (Lacking -> fill events; deaths per quarter.)
| arm | deaths per quarter | far kind (water) fills | near kind (food) fills | homing trust end |
|---|---|---|---|---|
| M main line (learned homing trust) | 64, 64, 64, 52 | 29 | 182 | .15 |
| U main line + unfamiliarity, weight 1 | 64, 64, 64, 63 | 79 | 7 | .42 |
Reading: no registered reading fits exactly. M is not ~0 on the far kind: with the homing trust learned (and falling to .15) the main line
alone already reaches the far spring 29 times (gate 2 run 1, full weight, 6,000 beats: 0) - distrusting the old map frees the bodies.
U reaches it almost three times as often (79), so the content does carry bodies through the doorways - but at weight 1 it drowns the rest
of the score: the near kind falls 182 -> 7, last-quarter deaths 52 -> 63 (+21%, under the 25% line). That is exactly what a learned trust
should settle (how much to follow the unfamiliar, from its own record). So the next is as the first reading: gate 2 run 2.

# Gate 2, run 2 (registered before running): the unfamiliarity's weight learned from its own kept promises (LW3 frame)
One change from G2a's U: Mind.learned_roam = LearnedWeight(mode="reli", prior .5) replaces the fixed weight 1. A promise = a beat on which the
unfamiliarity swung the choice (the gesture done is the unfamiliarity's top one and differs from the top of the rest of the score); kept =
a relief event (wholeness rise beyond its own running mean + 3 sd) within 30 beats; one open promise per body; counts fade x.995 per
closed promise - the same rule as the homing trust, nothing new. Content, never decider; reward only its own wholeness change.
Runs: a3_rooms, s3_a, 6,000 beats, true world, seeds 9301, 9302, 9303; arms M (main line), UF (--roam 1), UL (--roam 1 --roam-trust 1). 9 runs.
Bars, pooled over the three starts: (1) gate 2's own (second-half deaths <= 32 per quarter, fills of each kind >= 200 per start) -
not expected; (2) the frame: UL's total deaths below both M's and UF's, and UL's far-kind fills above M's. Readings: (2) met -> the
learned unfamiliarity trust goes into the main line (finding by its own record); then s3_b and 9104/9101 A2 checks for harm (>= 3 starts).
(2) not met with UL ~ M -> the promise is kept too rarely to earn weight (report its trust path); UL ~ UF -> the trust does not fall where
it should. Gate 2 budget: this is run 2 of 5. The L1 comparison (full homing weight) is gate 2 run 1 (seed 9301: 0 far fills).

## Gate 2 run 2 result: not met - the unfamiliarity's learned trust sinks to ~.06 in every start (the promise rule is unfair to it)
s3_a, 6,000 beats, true world; per start deaths by quarter, far (water) / near (food) lacking->fill events, homing trust end, unfamiliarity trust end:
| seed | M main line | UF unfamiliarity weight 1 | UL unfamiliarity trust learned |
|---|---|---|---|
| 9301 | 128 116 124 128; 181 / 369; .43 | 128 127 99 147; 258 / 65; .24 | 128 128 128 128; 0 / 538; .02; .063 |
| 9302 | 128 113 119 115; 346 / 224; .61 | 128 128 128 128; 298 / 0; .43 | 128 127 120 120; 386 / 105; .66; .057 |
| 9303 | 128 105 100 81; 238 / 446; .59 | 128 127 128 126; 357 / 9; .56 | 128 107 123 112; 354 / 128; .58; .066 |
| pooled | deaths 1385; 765 / 1039 | 1522; 913 / 74 | 1477; 740 / 771 |
Bars: (1) gate 2's own - not met (as expected). (2) UL's deaths below both M and UF (1477 vs 1385 / 1522) and far fills above M's (740 vs
765) - not met. Reading as registered: UL ~ M and its trust is kept too rarely to earn weight. Its path (9301): .5 -> .009 within 600 beats,
then .04-.07 for the rest of the life, in all three starts. Why, by reading the rule (my design flaw, said plainly): the homing promise is
judged at arrival (the moment the content says "here"), the unfamiliarity's at the first step it swings - and most first steps of a journey
give nothing within 30 beats. Judged by the absolute rate of relief within 30 beats, an explorer always loses; ~.06 is likely near the base
rate of a relief within any 30 beats. Side findings: the main line alone reaches the far kind in every start (765 pooled; gate 2 run 1: 0)
and in 9303 its deaths fall quarter by quarter (128 -> 81) - distrusting the old map lets it learn the new room slowly; at weight 1 the
unfamiliarity starves the near kind (74 pooled). Gate 2 budget: 2 of 5 used.
Next (to register after a cheap judge-side check): judge each promise against the base rate - P(relief within 30 | followed) vs
P(relief within 30 | not followed), trust = their contingency (ANCCR, Jeong 2022; .5 = no difference). Check first: the base rate of a
relief within 30 beats on ordinary beats in these lives; if it is ~.05, the reading above holds.

# C1 (registered before running; judge side only, founder 2026-11-27 on the one-body viewer: "circling like a stereotypy - does it fail to
# find because it circles, or circle because it cannot find?")
Circling (judge's definition): over the last 15 beats the body moved > 3 cells yet ends < 1 cell from where it was. Per living beat, circling
vs not: share; decision source (strong pull / value / random / hold / try); own goal held; hungry; for hungry beats the distance to the food
(< 2, 2-5, > 5 cells); gestures done. Runs: a2_moving_food --circ 1, 9104_0, s3_b, one body (the viewer's case) and 64 bodies, main line,
3,000 beats, reps 0-2. (64-body 400-beat smoke: circling 10% of beats; hold decides 73% of circling beats vs 77% otherwise.)
Readings: hungry circling mostly far from food (> 5) with no own goal -> it circles because it has nowhere to go (nothing pulls; the held
gesture turns): the fix is in what pulls, not the walk. Hungry circling mostly near food (< 2) or with a goal held -> it has a place and the
walk fails there (the D46 loop): circling stops finding; the walk's measure of closeness is what to look at, as content (D46b stays a
diagnostic). One gesture class dominating circling beats with hold deciding -> a held curving gesture traces the circle (persistence).
C1 amended before the extra runs: one more judge-side measure - share of goal-holding beats at which the body is in its own goal unit
(its own place code nearest that unit), circling vs not. Reps 0 for one body and 64 bodies, same settings.

## C1 result: it circles with a goal held, far from food and not at its goal unit, by re-choosing a strafe-and-turn gesture
9104, s3_b, 3,000 beats, main line. Circling share of living beats: one body .15 / .18 / .15 (reps 0-2), 64 bodies .12 / .13 / (r2 in file).
One body: goal held while circling .97-1.00 (else .91-.93); hungry .97-1.00; hungry circling > 5 cells from food .67-.88 (else .67-.82);
at its own goal unit while circling .11 (else .08; 64 bodies .05 vs .08). Decision source while circling: hold .73-.75 (else .78-.81),
strong pull .14-.17 (else .10-.12) - not a held gesture; the circling gesture is re-chosen. Gestures: a few dominate circling beats (one
body r2: gesture 80 = forward + turn left, 40% of circling beats vs 6% otherwise; r0: 66 = side-step right + turn right 15%, 48 = side-step
left + turn left 14%, vs 3% each otherwise; the same few in 64 bodies).
Reading against the registered three: not "nowhere to go" (a goal is held almost always), not "orbiting at the goal" (only ~11% at the goal
unit). Nearest is the second: it has a place to go and the walk fails on the way - and the third in a new form: the strong pull (the homing
content) keeps choosing a step-and-turn-the-same-way gesture, which traces a circle. Mechanism suspected from the code (D46's score):
progress (step toward the vector) and alignment (turn toward it) are each standardised and added, so the gesture that does a bit of both
wins every beat; step sideways toward a point while turning toward it = orbiting it. Also possible: its own path-integrated heading drifts
from the true one, so the target always reads as "to the side". The founder's "it knows where to go but cannot move its body" was right.
# C2 (registered before running; judge side only): which of the two - the score's mix, or the heading's drift
During circling vs not: the homing vector's own distance d and angle to the target in the body frame (|th| near 90 degrees steadily ->
the target always reads as to the side); the path organ's heading minus the true heading, its spread over each circling bout vs outside
(drift -> the vector itself is wrong). One body and 64 bodies, rep 0, 3,000 beats.
Readings: heading error spread within circling bouts >= 2x outside -> drift (the path organ, not the score); |th| steady near 90 with small
heading error -> the score's mix orbits a correct target (what to fix is how the homing content is offered: content, with its trust
learned; D46b stays a diagnostic).

## C2 result: no heading drift; the target reads as to the side (|angle| ~90-104 degrees median) whether circling or not
One body: |th| median 104 circling vs 92 otherwise, d 3.7 vs 4.8 cells; 64 bodies: 84 vs 82, 4.7 vs 4.6. Heading error per beat: 0 (the path
organ integrates the true felt turn exactly). Reading: not drift. Neither registered reading cleanly: the target is "to the side" all
the time, not only in circles - the walk seldom lines up with its own vector at all. Turning 30 degrees a beat should line it up in 3
beats, so the suspect is what the homing score believes each gesture does: its own forecaster's foreseen felt motion (prog and algn
both read it).
# C3 (registered before running; judge side only): does its own forecaster foresee the felt motion of each gesture?
For the gesture done: foreseen (forward, side, turn) from the same forecaster the homing score reads vs the felt motion after the step;
correlation per component, turn-sign agreement on turning beats. One body and 64 bodies, rep 0, 3,000 beats. Reading: turn correlation
< .5 or sign agreement < .8 -> the homing score orbits because its forecaster cannot tell which way a gesture turns (fix: the forecaster's
learning of its own motion, not the score); both high -> the score's mix itself (z(prog) + z(algn)) is what orbits.

## C3 result: its forecaster foresees each gesture's motion almost exactly - the circling is the homing score with held gestures
Foreseen vs felt (forward, side, turn) correlation: one body .92 / .94 / .99, 64 bodies .96 / .96 / 1.00; turn sign agreement 1.00 in both.
Reading as registered: the score's mix. Mechanism, from the mind's code: a gesture is re-chosen only on surprise or when its score falls
more than 2 sd below the best (Mind.choose: novel | pre); otherwise it is held (~75% of beats). The homing content scores each gesture by
ONE beat of motion - a step sideways toward the target plus a 30-degree turn toward it is the best single beat when the target is to the
side - and that gesture is then held. Held, it orbits; and the orbit keeps the target at the side, so the held gesture keeps scoring
near-best and is never dropped: a self-sustaining loop. Not a broken eye, heading or forecaster.
# D49 (registered before running): the homing content says what a gesture does when held, as gestures are held
One change (ATANOR_HOME_HOLD=H, default 1 = as now): the homing score reads each gesture's motion held H = 4 beats (its own one-beat
foresight integrated over 4 beats, the same horizon as E1's foresight), with the same two measures (progress along its own vector,
alignment after the turn). The content becomes truer to how the body acts; no target, no rule; its trust stays learned (LW3).
Runs: a2_moving_food --circ 1, 9104_0, s3_b, 3,000 beats, reps 0-2; one body and 64 bodies; arms H=1 and H=4 (12 runs).
Bars (pooled over the three starts): circling share down >= 30% in both body counts; 64 bodies: deaths and median food wait not worse
than H=1 by > 10%. Met -> adopt into the main line (then 9101 with >= 3 starts, and the viewer). Circling down but life worse -> report.
Circling not down -> the loop is not the one-beat score; next = the hold rule itself (when to think again), registered then.

## D49 result: not met - circling is not down (one body .159 -> .178, 64 bodies .128 -> .125); life about the same
Pooled over reps 0-2 (deaths; median of the three food waits; hungry->food events; mean circling share):
one body H=1 8; 269; 7; .159 | H=4 6; 151; 6; .178. 64 bodies H=1 47; 206.5; 899; .128 | H=4 43; 200.5; 896; .125.
Reading as registered: the loop is not the homing content's one-beat view. My C3 mechanism was a reading of the code, not a measurement:
C1 already showed circling beats are not held more than others (.73-.75 vs .78-.81), and the strong pull decides more of them (.14-.17 vs
.10-.12). Not adopted. Before touching the hold rule, measure what decides a circling gesture.
# C4 (registered before running; judge side only): which part of the score chooses the circling gesture
Mind.record_parts on: at each beat a new gesture is chosen (again), which score part's own top gesture equals the chosen one (pull = relief
memory pull, harm, futile, value, going, plan = homing content), circling vs not, and the parts' spreads across gestures. One body and
64 bodies, rep 0, 3,000 beats, H=1. Readings: one part's top matches circling choices far more than other choices -> that part drives the
loop (fix there, as content); none -> the loop is the held gesture after all and the hold rule is next.

## C4 result: the going organ's homing lean picks ~70-83% of all new choices, circling or not; circling is a state of constant re-choosing
Share of new choices whose part-top equals the gesture chosen (circling vs else): one body going .83 / .80, plan .01 / .03, pull .01 / .03;
64 bodies going .68 / .69, value .05 / .04, pull .03 / .03, plan .02 / .02. Spread across gestures: going 4.7 / 4.2 (one body), 1.9 (64);
plan (own-goal homing) .18; pull 1.2 / .4. New choices: one body 288 in ~450 circling beats (~64%) vs 69 in ~2,550 other beats (~3%).
Reading as registered: no part drives circling more than other choosing - the going lean (which itself reads the homing vector, D46)
decides almost everything. But the second half of the registered reading ("the held gesture") is refuted: circling is where the body
re-chooses nearly every beat. So the loop is the homing lean re-choosing each beat with its target to the side. Suspect: the target
itself moves - the lean aims 4 hops ahead along its own chain of place units, and as the body moves its unit (and so the target) changes.
# C5 (registered before running; judge side only): does the homing target jump while circling?
Per beat: the going organ's target unit changed since last beat; the change of the target's angle in the body frame (|d theta|), circling
vs not. One body and 64 bodies, rep 0, 3,000 beats. Readings: target changes far more often in circling (>= 2x) or |d theta| large
(> 30 degrees median) -> it chases a jumping target (fix: what target the content offers, e.g. its own place units' aliasing); else ->
the vector is steady and the lean's score itself turns it (look at prog/algn with real turns).

## C5 result: the target does not jump (changes 11% vs 7.5% of beats, one body; 14% vs 10%, 64 bodies - under 2x)
|d theta| median 26 vs 2 degrees circling vs else - that is the body's own 30-degree turns, not the target moving. Reading as registered:
the vector is steady; the lean's score itself turns the body. Next suspect: walls. The homing vector is a straight line to a point
4 units ahead; if a wall lies across it, the forecaster says forward gives nothing, so turns and side-steps win, and the body turns and
slides along the wall around a point it cannot go straight to.
# C6 (registered before running; judge side only): is the straight line to its own target blocked by a wall while circling?
Judge's ray from the body along its own target direction (heading + theta): wall nearer than the target's own distance d = blocked; and
the chosen gesture's foreseen forward motion. Circling vs not, one body and 64 bodies, rep 0, 3,000 beats.
Readings: blocked >= 2x more often while circling (or > 60% of circling beats) -> the homing vector aims through walls: what the content
lacks is the way around (the unit chain's next unit when the line is blocked - a content change, registered next); else -> report.

## C6 result: the homing line runs into a wall most of the time - circling or not
Straight line from the body to its own target blocked by a wall: one body .64 circling vs .68 else; 64 bodies .72 vs .62. The chosen
gesture's foreseen forward motion: one body .02 vs .04, 64 bodies .09 vs .09 (cells a beat; a full step is ~.4).
Reading: the registered "> 60% of circling beats" is met, but not as a circling-specific cause: the content aims through walls about
two-thirds of the time everywhere, and the body hardly walks forward at all. The homing vector aims 4 hops ahead along its own chain
(D34c chose 4 before the vector existed, when the going skill walked by place code, not by a straight line). With a straight-line vector,
4 hops ahead is often around a corner. That is a general flaw of the content, and circling is one face of it.
# D50 (registered before running): how far ahead the homing vector aims, now that it is a straight line
One change: the going organ's hops (L1d: 4) - arms 1, 2, 4. a2_moving_food --circ 1 (judge-only measures incl. C7: each circling bout's
own homing distance at start and end), 9104_0, s3_b, 3,000 beats. Fast probe first: rep 0 for one body and 64 bodies (6 runs); if one k
stands out (circling down >= 30% and deaths/wait not worse on 64 bodies), confirm it on reps 1-2 against 4 before adopting.
Readings: smaller k cuts circling and blocking with life no worse -> adopt that k (a designer constant, as D34c's); no k helps -> the
straight-line vector itself lacks the way around: next is the content offering the direction of the next lived unit, not a far point.

## Correction to C4 (found while moving the walk diagnostics into tools/walk_diag.py, 2026-11-27)
C4's "choices" list was written [else, circling], not in the order of the other fields. I read it the wrong way round. Correct: one body
288 new choices in ~2,550 other beats (~11%) vs 69 in ~450 circling beats (~15%); 64 bodies 21,502 else vs 4,007 circling (similar
shares). So circling is NOT a state of constant re-choosing - it is mostly held, as C1's decision sources already said (hold .73-.75).
What stands from C4: the going lean picks ~70-83% of new choices in both. What falls: "the held-gesture reading is refuted". It is not
refuted; D49 (scoring the held gesture over 4 beats) still did not reduce circling, so the hold is not fixed by a longer view of the
homing content alone. C5-C6 are unaffected. tools/walk_diag.py now writes every list as [circling, else]; outputs checked identical to the
old inline code on a 300-beat run (all fields equal).
