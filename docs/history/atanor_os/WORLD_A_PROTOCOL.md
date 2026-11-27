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

## D50 result (fast probe, rep 0): no nearer aim helps - 4 hops stays; and C7: circling bouts are real traps
64 bodies (deaths; median food wait; hungry->food events; circling share; line blocked circling/else; C7 bouts: own distance start ->
end, share ending within 1 cell of where it began): k1 67; 291; 221; .152; .36/.28; 1.35 -> 1.34, .90 | k2 31; 239; 279; .147;
.52/.43; 2.47 -> 2.43, .79 | k4 14; 187.5; 320; .124; .72/.62; 4.70 -> 4.66, .86.
One body (1-3 food events a life - too few to read): k1 4 deaths, k2 2, k4 3; circling .23 / .12 / .15.
Reading: a nearer aim blocks less and lives worse (k1 deaths 67 vs 14), so the blocked line is not what traps it (C6's reading falls);
no k cuts circling by 30% with life no worse: not adopted, hops stays 4. The registered "next = direction of the next lived unit" is
answered by k1 itself (aiming at the next unit) - worse. C7 (Copilot's measure): 79-90% of circling bouts end at the same distance -
true traps, not slow detours. Eliminated so far: eye, heading, forecaster, target jumps, held one-beat view (D49), walls / aim distance
(D50). Open (Copilot's suspicion): the target's representation - states the being takes as different that lead nowhere new.
Decision (founder 2026-11-27: focus on what is built): the circling hunt pauses here with this list; the walk is re-examined when the
place memory is rebuilt in M2 (place units are the target's representation).

# R1 (registered before running; founder 2026-11-27: "one operation at a time, like a robot vacuum"): a body that does one thing at a time
One change (ATANOR_ONE_OP=1; RoomBody.one_op, Mind.allowed): the body offers only gestures that step, or turn, or use the hand - never two
at once (26 of 90 combos). A body's make-up, not a behaviour rule: nothing says when to turn or walk. Rationale from C1: the circling
gestures are step-and-turn combos (side-step + turn, forward + turn); held, they orbit.
Runs: a2_moving_food --circ 1, 9104_0, s3_b, 3,000 beats, reps 0-2, 64 bodies and one body; control = D49's H=1 arm (same code path for
the main line, same seeds: 64 bodies 47 deaths, waits 187.5 / 224 / 206.5, circling .128). Bars (pooled): circling down >= 30% and 64-body
deaths and median wait not worse by > 10%. Met -> into the main line and the viewer. Shown in the viewer at once, labelled as a trial.

## R1 result (early verdict, rep 0): fails badly - circling gone but the body stops turning and cannot eat
64 bodies: deaths 251 (control 14), median food wait 392 (187.5), 41 hungry->food events (320); circling .005 (.124). One body: 4 deaths,
no food event. Gestures done: turns 1-3%, steps 83-90% (most of them side-steps), hand 8-9%. Reading: the founder saw it in the viewer
("it does not seem to know how to turn its head"): with step-and-turn combos gone, the grown being strafes toward its targets and almost
never turns - so it rarely faces food, and eating needs the food in front of the hand. Stopped after rep 0 (fast-probe rule); not adopted.
# R1b (registered before running): the robot vacuum proper - one operation at a time and no side-steps (drive or rotate)
One change from R1 (ATANOR_ONE_OP=2, RoomBody.vacuum, 18 of 90 combos): side-steps removed too, so the only way to face a target is to
turn. Fast probe: 64 bodies, rep 0, 3,000 beats, --circ; same control. Readings: deaths <= control x 1.5 with circling down -> run reps
1-2 and the one-body case; else -> a body change needs the being regrown on it (s3_b grew with combos), registered next.

## R1b result (rep 0): fails as R1 - deaths 253 (control 14), 41 food events (320); turns 3.7% of gestures even with no side-steps
The being grown with step-and-turn combos does not take up turning in place when the combos are gone: its value, going skill and
habits were all learned on the old body. Not adopted.
# R1c (registered before running): grow the being in the robot-vacuum body from the start (same recipe as s3_b)
s3_worlds with ATANOR_ONE_OP=2 (drive or rotate, one at a time): 9104_0 9105_1 9106_0 (6,000 beats each, childhood 3,000 at the start,
then 9104_0 again), tag _v -> probe-work/s3_v. Then the fast probe: a2_moving_food --circ 1, 9104_0, s3_v with ATANOR_ONE_OP=2, rep 0,
64 bodies, 3,000 beats, against s3_b's main line (14 deaths, wait 187.5, circling .124). Readings: deaths <= 1.5x and circling down ->
reps 1-2, one body, the viewer; else -> report: the one-operation body is not the way out of the orbit for this mind.

# S1 (registered before running; founder 2026-11-27 on the viewer: "why does it keep going round? if circling gets nothing it should stop")
Found in the code: the homing trust (LW3) is judged only on arrival (relief within 30 beats of reaching the goal unit). A circling body
never arrives (C1: 11% of circling beats at the goal unit), so no broken promise is ever recorded and the trust never falls; and C7 shows
79-90% of circling bouts end no closer. One change (ATANOR_STALL=1, LearnedWeight.stall): the homing content promises that following it
brings the body closer; while a goal is held, every 30 beats the body's own distance to its goal's own place (path organ to the goal unit's kept position - not the 4-hop
target, which moves ahead as it walks; caught in the smoke before any run) is compared with 30 beats before - not closer
by 1 cell -> one broken promise in the same record (nothing new counted as kept). Its own distance only; no judge, no rule of when to stop.
Runs: a2_moving_food --circ 1, 9104_0, s3_b, 3,000 beats, reps 0-2, 64 bodies and one body; control = D49's H=1 arm (64 bodies 47 deaths,
waits 187.5 / 224 / 206.5, circling .128; one body 8 deaths, circling .159). Bars (pooled): circling down >= 30% and 64-body deaths and
median wait not worse by > 10%. Met -> main line and viewer. Shown in the viewer at once as a trial.

# R2 (registered before running; founder 2026-11-27: "grow it again at eye height 1.1 too")
Why: at eye 1.6 (walls 2.0, springs 1.0, 35 degrees up / 22 down) a spring vanishes from the picture within ~1.2 cells - it cannot see what
it is about to eat; E-height (eye 1.1 on the 1.6-grown being) cut deaths 12 -> 7 but missed its docking bar, and a grown being does not
take a changed body well (R1, R1b). One change: the being grown at eye 1.1 from the start, same recipe as s3_b (s3_worlds 9104_0 9105_1
9106_0, tag _e, ATANOR_EYE=1.1; ordinary body, main line). Queued after R1c's growth (GPU kept to two heavy runs).
Then: a2_moving_food --eye 1.1 --circ 1, 9104_0, s3_e, 3,000 beats, reps 0-2, 64 bodies and one body; control = s3_b main line (64 bodies
47 deaths, waits 187.5 / 224 / 206.5, circling .128). Bars (pooled): deaths and median wait better than control by >= 10%, circling not
up. Met -> eye 1.1 becomes the room body's eye (main line, viewer).

## R1c result: the growth itself failed - the being grown in the robot-vacuum body never learned to eat or drink
Growth (s3_worlds, ATANOR_ONE_OP=2): need-directed fills 0 in every stay (s3_b's growth: 2.8-7.6 per 1,000 body-beats), deaths a flat
1.33 per 1,000 (every body dying of drain). Probe: 253 deaths, 35 food events, turns 14% of gestures. Reading: inconclusive about the
body - the growth recipe (childhood, practice, the base being it starts from) does not take in this body at all. The one-operation line
stops here (R1, R1b, R1c); recorded.
## S1, rep 0 (early look): circling not down although the homing trust falls to .14 - and the reason is in the code
64 bodies: 49 deaths (control 14), wait 257.5 (187.5), circling .130 (.124), trust .14 after 2,158 stall-breaks. One body: 1 death (3),
wait 191 (266.5), circling .137 (.152). The learned trust weighs only the own-goal channel (Mind: plan, spread ~.2), while the going
organ's lean - which reads the same homing vector inside going.lean, spread 2-4.7, and picks 70-83% of choices (C4) - runs at full weight
(going_earned is not set in the main line). So since LW3 the being's trust in the homing content has governed the minor channel only.
# S1b (registered before running): the same learned trust weighs the going lean's homing content too
One change (ATANOR_TRUST_LEAN=1): gl x learned_home.w when the going organ walks by the homing vector. Arms after S1 finishes: S1b (trust on
lean, no stall) and S1c (trust on lean + stall), fast probe rep 0, 64 bodies, 3,000 beats, --circ; control s3_b main line (14 deaths,
wait 187.5, circling .124). Bars as S1. If one arm meets them -> reps 1-2 and one body, then main line and viewer.

## S1b / S1c result (rep 0): circling rises when the homing content is weighed down - circling is the default, homing reduces it
64 bodies (deaths; wait; food events; circling; trust end): S1b trust on lean 43; 248.5; 208; .163; .56 | S1c trust on lean + stall
203; 359.5; 134; .225; .00 (control 14; 187.5; 320; .124). Not adopted. Reading: with the homing content at weight 0 (S1c) the body
circles MORE (.225) and dies far more - the circle is not made by the homing vector; it is what the body does when nothing pulls it
well, and the homing content cuts it. Why a circle: ~75% of beats keep the current gesture; 2 of 5 neck options turn (30 degrees) and
5 of 6 leg options move, so a held gesture is often a step-and-turn - and a held turn turns again every beat: an arc, then a circle.
# T1 (registered before running): a turn is one act - a held gesture keeps its walking and hand, not its turn
One change (ATANOR_TURN_ONCE=1; RoomBody.no_turn, Mind.turn_once): a 30-degree neck turn happens on the beat the gesture is chosen (as a head
turn or a saccade is one act); on held beats the same gesture runs with the neck still. The body's motor make-up, not a rule of when to
turn; all 90 gestures stay choosable. Fast probe: rep 0, 64 bodies, 3,000 beats, --circ, s3_b main line; control 14 deaths, wait 187.5,
circling .124. Met (circling down >= 30%, deaths and wait not worse by > 10%) -> reps 1-2, one body, viewer.
## T1 result (rep 0): circling cut by 69% (.124 -> .039) - but the old-grown being lives worse (deaths 44 vs 14, wait 228 vs 187.5)
Bars: circling met; deaths / wait not (worse by > 10%). Not adopted on s3_b. As with R1, a being grown on the old motor make-up does not
live as well in the changed one. The circling itself is gone - the body change does what it was for.
# T1g (registered before running): grow the being with turns as single acts (same recipe as s3_b, tag _t, ATANOR_TURN_ONCE=1)
Queued after R2's growth. Then the same probe with s3_t; bars as T1. Met -> main line (T1 body + s3_t being) and viewer.
Meanwhile the viewer shows T1 on s3_b as a trial (founder wants to see the change).

## R2 result: not met - grown at eye 1.1 it dies less (33 vs 47) but waits a little longer (226 vs 206.5); circling the same
Growth healthy (need-directed fills rise 0.6 -> 8 per 1,000 body-beats across stays, deaths fall to 0, like s3_b). A2, 3,000 beats,
reps 0-2: 64 bodies deaths 2 / 20 / 11 = 33 (control 47), waits 204 / 226 / 234 -> 226 (206.5), circling .121 (.128); one body deaths
5 (8), waits 464 / 503.5 / 355 on 1-3 food events (too few to read). Bars: deaths better by >= 10% - met (-30%); wait better by >= 10% -
not met (+9%). Not adopted as registered; the eye stays 1.6 on the main line. Said plainly: fewer deaths with a lower eye is the second
time (E-height 12 -> 7); the eye height is worth carrying into T1g's successor if T1g is adopted (one change at a time).

# J1 (registered before running; founder 2026-11-27: "its judgement is slow - turn the corner, no food, at once go where it likely is")
Found: when the body reaches the place its own memory says holds relief and nothing comes, nothing happens at once - the relief memory
only fades (.99 a beat), the place values are recomputed every 100 beats, the goal is re-taken from those stale values, and "expected
relief did not come" is not a surprise to the chooser (only foreseen motion errors are). One change (ATANOR_MISMATCH=1): a body that has
been 5 beats at its own goal place, lacking, with no relief of that need -> that place's relief memory for that need is set to the
present (empty), the place values are recomputed at once (64 units, milliseconds), and the body chooses again this beat; its next goal is
whatever its own memory now values most. Its own expectation and its own memory only; no judge, no rule of where to go.
Fast probe: newest viewer arm (s3_e eye 1.1, T1) with and without J1, rep 0, 64 bodies, 3,000 beats, --circ; and the main line (s3_b) with
and without J1. Bars: deaths and median food wait better by >= 10% in both pairs. Shown in the viewer at once (newest arms together).

# P2 (registered before running; founder 2026-11-27 on the viewer: "right up against it the box is just hidden - loosen the head up / down
# a little; the eyes catch first and the head follows")
One change (ATANOR_PITCH=1; RoomBody.pitch): the neck's up / down options, which did nothing in the room, tilt the view 10 degrees a
gesture, between 15 degrees up and 30 degrees down (start level; reset at birth). The eye's fovea and rings already move by the reflex to
picture change inside the frame (eyes first); the head now can follow down. No rule of when to look down. Shown in the viewer at once with
the newest arms; measured on the same fast probe as J1 (rep 0, 64 bodies, s3_e eye 1.1 + T1 + J1, with and without P2), and on a
being grown with it later if it is adopted.

## World fix 2 (founder 2026-11-27 on the viewer: "is the wall behind the box opened on purpose?") - no: a render bug
RoomBody.render stopped each ray at the first non-floor cell, springs included; a spring is 1 cell high, so above it the picture showed
open sky where the wall behind should stand (checked: standing 0.6 cells from water, facing it, eye 1.1 - 1,632 sky pixels in the centre
columns; with the fix 362, the sky above the wall's top). Fix (ATANOR_RENDER_FIX=1; room3d_gpu.World.cast(through=...)): a second ray
passes springs and draws the wall behind them. A world-side correction (the world looked wrong), not a change to the being. On in the
viewer at once; default off until a check that the main line is not worse (as world fix 1).
P2 amended before any run: the body's head-tilt sense (contract slot 4, zero in the room until now) carries its own pitch, so the being
feels where its head points (as the Minecraft body's tilt does).

# M1d (registered before running; founder 2026-11-27 on the viewer: "it hardly leaves home")
Found: the place units' own path positions (Going.Pu, the homing positions) are not part of the saved being (going.pt predates the homing
vector) - every life starts with none, and a goal unit with no position is walked toward the birth point (gate 2 run 1's bug). With 64
bodies the positions fill within beats; a lone body fills them slowly, so its homing walks it back home. One change: train many, live as
one, for place too - a 64-body life writes Pu / Pn (a2_moving_food --save-place), the next life starts with them (ATANOR_PLACE_FROM), and
keeps learning them. With M1a's carried trust (ATANOR_TRUST_FROM) for the same being. Viewer at once; measured on one body, reps 0-2,
against the same arm without carry.

# A1 (registered; founder 2026-11-27 + Copilot: "vision is there, attention is not - build gaze attention now")
The gaze has moved only by a reflex to picture change (bottom-up: when it turns, the sky-wall edge moves most, so it looks at the sky).
One change (ATANOR_ATTEND=1; eye_gpu.RetinaGPU): the eye keeps a short trace of the colours in view (12 x 12 places); when the body's own
insides rise (the bedrock, read in Organs), colours that were in view before gain relevance, and slowly lose it when in view without
relief. The gaze goes to the centre of (picture change + learned relevance of what is where), each normalised. No colour, object or target
named; the only teacher is its own relief. Viewer at once; measured with the newest arms, with and without A1 (rep 0, 64 bodies).

## J1 result (fast probe, rep 0): met in both pairs - judging again at once when expected relief fails
64 bodies, 3,000 beats (deaths; median food wait; food events; circling; mismatch events):
- s3_b main line: without J1 14; 187.5; 320; .124 | with J1 8; 163.5; 336; .134; 207 -> deaths -43%, wait -13%.
- s3_e (eye 1.1) + T1: without J1 51; 249; 281; .034 | with J1 11; 179; 332; .036; 312 -> deaths -78%, wait -28%.
Bars (deaths and wait better by >= 10% in both pairs) met on rep 0. Next: reps 1-2 for both pairs, then into the main line.
Side: T1 with the eye-1.1 being keeps circling at .034-.036 (T1's -70%), and J1 removes most of T1's cost in deaths (51 -> 11).
## T1g, rep 0 (being grown with single-act turns): 75 deaths, wait 289, circling .040 - grown with T1 it lives worse than s3_b's main line
(14) and worse than T1 on the eye-1.1 being; reps 1-2 running. Reading after all three.

## T1g result: not met - the being grown with single-act turns lives worse (deaths 75 / 47 / 59 = 181 vs s3_b main line 47)
Circling .037-.040 (T1's cut holds), waits 289 / 215 / 259.5. Growth healthy (fills 4-8 per 1,000). The best arm so far remains the eye-1.1
being + T1 + J1 (rep 0: 11 deaths, wait 179, circling .036); its reps 1-2 come with J1's confirmation.
# IM1 (registered before running; founder 2026-11-27: "attach the wall map and the imagination organ now")
Checked first: the being's learned depth sense cannot carry a wall map - against true wall distance it reads close walls as far (median
estimate 25 at 0-1 cells, corr -.51); not usable. Built instead from what it knows for sure: Going.wall_map_observe - a fixed 64 x 64 map in
its own path frame: free where it walked; a wall where its own forecaster foresaw a step and its own felt motion stayed ~0 (pushed, did
not move). Judge check after 400 beats, 64 bodies: 32 of 38 wall cells on true walls, 56 of 70 free cells on true floor (rounding at
cell borders). Imagination (ATANOR_IMAGINE_MAP=1): the homing content walks its own map in its head - the distance from each place to its
target around the walls it knows (unknown cells passable) - and offers per gesture how much its own foresight (held 3 beats) shortens that
imagined way, and which way the imagined route leaves from here; the same two measures as before, now around walls instead of through
them (C6: the straight line was blocked 2/3 of the time). Content only; its trust learned as before. Fast probe: newest arm (s3_e eye
1.1 + T1 + J1) with and without IM1, rep 0, 64 bodies, 3,000 beats; bars: deaths and wait better by >= 10%. Viewer and dashboard at once
(the dashboard map now shows its own walls).
## IM1 result (fast probe, rep 0): deaths 8 vs 11 (-27%), wait 178 vs 179 (same), food events 308 vs 332 - not met (wait bar)
The imagined route around its own walls does not shorten the food wait in A2 on rep 0; deaths fewer. Not adopted; kept on the viewer as the
newest arm. A2's corridors are short and the map fills only as walls are bumped; the place where routes around walls should matter more is
A3 (gate 2), registered after J1's confirmation.
# J1 confirmation (registered): reps 1-2 for both pairs (s3_b main line +-J1; s3_e eye 1.1 + T1 +-J1), 64 bodies, 3,000 beats.
Pooled over reps 0-2, the same bars. Met -> J1 into the main line (ATANOR_MISMATCH default 1).

## J1 confirmation result (reps 0-2 pooled): met in one pair, not the other - not adopted alone
64 bodies, 3,000 beats (deaths per rep; total; median of the three waits):
- s3_b main line: without J1 14 2 31 = 47, 206.5 | with J1 8 14 25 = 47, 198 -> deaths 0%, wait -4%: not met.
- s3_e (eye 1.1) + T1: without J1 51 15 26 = 92, 235 | with J1 11 18 18 = 47, 199 -> deaths -49%, wait -15%: met.
Reading: rep 0's large gain on the main line was a lucky start; on the plain main line J1 is within noise. Where the body turns once (T1),
J1 halves the deaths: judging again at once matters when the body no longer circles in place. Not adopted alone (bar: both pairs).
## The newest combination against the main line (same reps, pooled): meets T1's registered bars
s3_e eye 1.1 + T1 + J1 vs s3_b main line: deaths 47 vs 47, median wait 199 vs 206.5, circling .036 / .034 / .036 vs .124-.134 (-72%).
T1's bars (circling down >= 30%; deaths and wait not worse by > 10%) are met by the combination over three starts. Decision: the
combination becomes the room main-line recipe (being s3_e, eye 1.1, ATANOR_TURN_ONCE=1, ATANOR_MISMATCH=1; env defaults left off so older
runs stay reproducible; CURRENT_STATE records the recipe). Said plainly: three changes were adopted together because each was tested in
this order and the combination is what met the bar; their separate shares are T1 (circling), J1 (deaths, given T1), R2 (deaths -30%).

# Gate 2, run 3 (registered before running; founder 2026-11-27 "Yes, run gate 2"): the imagined route around its own walls, in A3
Theory: in A2 the corridors are short, so routes around walls barely differ from straight lines (IM1 rep 0: wait same). In A3 every
spring is out of view behind a doorway and the straight line from the stem to a far corner is blocked by walls most of the way; if the
imagined route (Going.imagine_route on its own wall map) carries homing content around walls, journeys should end sooner here.
Runs: a3_rooms, being s3_e at eye 1.1 (--eye 1.1), with the adopted room recipe (ATANOR_TURN_ONCE=1, ATANOR_MISMATCH=1), 6,000 beats,
64 bodies, true world, seeds 9301, 9302, 9303; arms R (recipe) and RI (recipe + ATANOR_IMAGINE_MAP=1). 6 runs, serial.
Bars, pooled over the three starts: (1) gate 2's own (second-half deaths <= 32 per quarter, fills of each kind >= 200 per start);
(2) the frame: RI's total deaths and the median waits of both kinds better than R's by >= 10%. Readings: (2) met -> IM1 into the room
recipe and MineLand check; (1) met -> gate 2 passed. (2) not met -> the imagined route does not move journeys; report how much of its own
map is filled and how often the imagined first step differs from the straight one. Context, not a bar: gate 2 run 2's M (s3_a, 1385 deaths).
Gate 2 budget: this is run 3 of 5.

## Gate 2 run 3 result: not met - the imagined route does not move journeys in A3 (deaths -3%)
s3_e eye 1.1 + T1 + J1, 64 bodies, 6,000 beats (deaths by quarter; total; water fills / median wait; food fills / median wait; trust end):
| seed | R recipe | RI recipe + IM1 |
|---|---|---|
| 9301 | 128 124 116 115; 483; 195 / 152; 228 / 236; .34 | 128 121 113 110; 472; 199 / 140; 238 / 320.5; .41 |
| 9302 | 128 121 117 124; 490; 184 / 81; 219 / 201; .42 | 128 125 116 103; 472; 227 / 137; 194 / 262; .45 |
| 9303 | 128 122 110 116; 476; 172 / 220.5; 269 / 170; .40 | 127 122 115 99; 463; 196 / 236; 320 / 143.5; .45 |
| pooled | 1449 | 1407 (-3%) |
Bars: (1) gate 2's own - not met (second-half deaths 99-124 a quarter vs <= 32; fills near 200). (2) RI vs R by >= 10% in deaths and both
waits - not met (deaths -3%, waits mixed). Context: gate 2 run 2's M (s3_a) 1385 - the new recipe does not change A3. Gate 2 budget: 3 of 5.
Reading as registered: the imagined route does not move journeys. What it was (measured after, judge side - see G2L below): the place units
were not this world's.

# G2L ladder (judge-side diagnostics, never a pass; registered as diagnostics 2026-11-27 night, founder: "re-examine the whole")
Rung 1 - a scripted walker (turn toward the given vector, else forward and reach) with the judge's vector, 64 bodies, 3,000 beats, A3 9301:
way around the walls (next cell on the true shortest way) - deaths 0, fills water 1,408 / food 1,344; the straight vector to the spring -
deaths 256, food 0. The world and body are sound; in A3 going around walls is everything.
Rung 2 - the being's own choice with the judge's way-around vector as its homing content (Going.oracle; the rest of the being as is), 3,000
beats: deaths 230 vs 252 without (-9%), trust .17. The being does not follow correct content. Choice re-decides only on surprise or a big
gap (dashboard: ~8% of beats) and otherwise holds its gesture.
Place units in A3 (judge side, s3_e's saved unit centres over every A3 cell): all 64 were used in growth (used = 64, no birth possible);
34 cover A3, 30 lie nowhere in A3; the far rooms are 2-3 units each (unit 55 = the water room + its doorway + the corridor's left end).
Dying bodies' goals (first 900 beats, 64 deaths): all at a unit no body has stood in in this world. Full 3,000 beats: 189-192 of 230-252.

# R1 (registered before running; EMERGE_PLAN "전체 재정비"): place remapping with a fixed number of units
One change (ATANOR_REMAP=1; Going._remap): when a body stands farther than one cell's worth of code (rad, the birth rule) from every unit and
all KU units are used, the unit longest unvisited - if longer than its own map's memory span 1 / (1 - decay) = 2,000 beats - is cleared and
placed here (code, relief memory, crossings, homing position afresh). At a life's first beat in a world every unit counts as last visited one
span ago (it comes from elsewhere). Nothing named; which units are used is its own walking. Known cost: the old world is forgotten (R2's job).
Gate 2 run 4: a3_rooms, s3_e eye 1.1 + T1 + J1, 6,000 beats, 64 bodies, seeds 9301-9303; arms R (= run 3's R, reused) and RM (+ ATANOR_REMAP=1).
Bars as run 3: (1) gate 2's own; (2) RM's total deaths and both median waits better than R's by >= 10%. Also report: remaps, share of dying
bodies' goals at units never stood in (run 3's R: ~80%). Readings: (2) met -> R1 into the room recipe, then R2 (world sets, sleep) and
MineLand; (2) not met with the never-stood-in share near 0 -> place was not the bottleneck alone - rung 2b (choice) decides the next build.
Rung 2b (diagnostic, with run 4): the judge's way-around vector + every body chooses afresh every beat (Mi.force), with learned trust and at
full weight - does the being follow correct content when it re-decides?
## R1 first look (one start, 3,000 beats, 64 bodies; not the registered run): "where" is fixed, "getting there" is not
RM (ATANOR_REMAP=1) vs the same arm without (g2l o0): deaths 240 vs 252, fills water 74 vs 39, food 66 vs 48, remaps 38. Dying bodies' goals:
at the right spring's unit 144 of 240 (without: 0), wrong 32, never stood in 64 (all in the first life, before remapping took hold).
## Rung 2b result: the being lives on correct content only when it re-decides AND the content keeps its full weight
Judge's way-around vector as homing content, 3,000 beats: every body chooses afresh every beat + learned trust - deaths 64 64 64 64 = 256,
trust .14 (worse than holding: the trust has sunk, so the content barely counts and fresh choices go elsewhere); every beat + full weight
(ATANOR_LEARNED_TRUST=0) - deaths 64 64 0 2 = 130, water 254 fills / wait 70, food 230 / 96: second half 2 deaths (gate 2's own bar is
<= 32 a quarter). Two holes on the choice side: (a) the learned trust sinks in A3 (.14-.17; promises are judged at arrival at a goal unit,
and the goals were old-world units never reached), (b) holding. Next diagnostics (running): oracle + full weight without forcing (is (b) a
hole on its own?); R1 + full weight with and without forcing (rung 3: its own content with both fixes).
## Rung 2b/3 diagnostics (one start, 3,000 beats): full weight matters most; forcing is not the fix; R1 at full weight -24% deaths
- judge's way-around vector, full weight, no forcing: deaths 64 30 22 15 = 131, water 255 / 193, food 185 / 260 - lives (holding costs
  waits, not lives: forcing gave waits 70 / 96).
- R1 + full weight + forcing every beat: 256 deaths, remaps 21, every dying goal never stood in - forcing with its own content breaks it.
- R1 + full weight, no forcing: deaths 64 40 39 49 = 192 (R1 with learned trust 240; neither 252), water 211 / 150, food 111 / 341; every
  dying body aimed at the right spring's unit (192 / 192). It knows where and still dies on the way. Rung 1 says what kills on the way in
  A3: the straight vector (the homing vector aims at a unit `hops` ahead on its lived chain, then straight - and R1's new units have few
  lived crossings yet). Next (running): R1 + IM1 (its own way around its own walls) at full weight and with learned trust.
## R1 + IM1 (one start, 3,000 beats): IM1 still does not help even with the right goals
R1 + IM1 full weight: deaths 64 63 49 35 = 211 (R1 full weight without IM1: 192), water 175 / 271, food 61 / 350; dying goals right 161 / 211.
R1 + IM1 learned trust: 253, trust .27. IM1's own wall map + BFS does not carry what rung 1's true way around carries (131); left for later.
Fetched (founder approved): Astra's R2 prototype, LOKI c7c304c (loki_memory/place_bundles.py) into ext/LOKI-v13 -
kept off; R1 vs R2 on the same being and starts after gate 2 run 4.

# Gate 2, run 4 (registered before running): R1 place remapping, homing content at full weight
Arms on s3_e eye 1.1 + T1 + J1, a3_rooms, 64 bodies, 6,000 beats, seeds 9301-9303: R (= run 3's R, reused, learned trust on) and RW
(ATANOR_REMAP=1 + ATANOR_LEARNED_TRUST=0). Two changes against R, said plainly: R1 makes the goals this world's; the learned trust was
adopted (LW3) to distrust the old map, which R1 now replaces - and in A3 that trust sinks (.14-.39) because promises are judged by arrival at
goals that were unreachable. Rung 2b and the one-start look show full weight is what lets correct content work (130-192 vs 240-256).
The trust question is not closed: if RW meets its bars, the next registered step is a fairer trust rule (contingency, followed vs not
followed) that must match RW without fixing the weight by hand.
Bars, pooled: (1) gate 2's own (second-half deaths <= 32 per quarter, fills of each kind >= 200 per start); (2) RW's total deaths and both
median waits better than R's by >= 10%. Report: remaps, dying bodies' goals (right / wrong / never stood in). Budget: run 4 of 5.
## Confound found (my error, said plainly): with ATANOR_LEARNED_TRUST=0, J1 (and S1) were silently off
In Mind.after, J1's block sat inside the learned-trust block, so every run without learned trust also ran without J1: the R1 full-weight
diagnostics (r1w 192, r1iw 211, o2w, o2fw) and gate 2 run 4's RW arm (launched before the fix; its processes run the old code). RW is
therefore "R1 + full weight - J1" against R "learned trust + J1". Fixed (mind.py): J1 and S1 run whenever a goal is held, trust or not;
learned-trust runs are unchanged (same code path). Run 4 is judged as registered, with this stated next to its result.

# TC1 (registered before running): the homing trust by contingency, not by arrival
LearnedWeight mode "cont" (ATANOR_TRUST_MODE=cont): on every beat the homing content is held, did the gesture go its way (its content for the
done gesture > 0)? Each body keeps two traces fading over ~30 beats (followed / not followed); a relief event (as LW3) credits both; each
credit is divided by its own lived opportunity; w = p(relief | followed) / (p(relief | followed) + p(relief | not followed)); .5 = no
difference. Rescorla 1968; ANCCR. Reward only its own wholeness; no fixed weight.
Runs (after gate 2 run 4, J1 fixed): a3_rooms, s3_e eye 1.1 + T1 + J1 + R1, 64 bodies, 6,000 beats, seeds 9301-9303; arms RW2 (full weight)
and RT (TC1). Bar (non-inferiority, pooled): RT's deaths and both median waits within 10% of RW2's, and RT's trust above .5 at the end in
every start. Met -> TC1 replaces LW3 as the homing trust (the weight learned, not fixed); not met -> report its path and what it learned.

## Gate 2 run 4 result: not met, narrowly - deaths -13%, water wait -32%, food wait -5% (bar: all three >= 10%)
RW (R1 + full weight; ran without J1 - see the confound above), 64 bodies, 6,000 beats (deaths by quarter; total; water fills / median wait;
food fills / median wait; remaps; dying goals right / wrong / never stood in):
| seed | RW | R (run 3) |
|---|---|---|
| 9301 | 104 88 102 40; 334; 488 / 103.5; 376 / 198; 39; 334 / 0 / 0 | 483; 195 / 152; 228 / 236 |
| 9302 | 125 90 117 95; 427; 416 / 116.5; 242 / 190.5; 34; 361 / 2 / 64 | 490; 184 / 81; 219 / 201 |
| 9303 | 128 121 125 128; 502; 281 / 76; 92 / 167.5; 34; 357 / 81 / 64 | 476; 172 / 220.5; 269 / 170 |
| pooled | 1263 (-13%); waits (median of starts) water 103.5 vs 152, food 190.5 vs 201 | 1449 |
Bars: (1) gate 2's own - not met (9301's last quarter 40, nearest yet to <= 32; 9303 food fills 92). (2) deaths -13% met, water wait -32% met,
food wait -5% not met -> not met as registered. Gate 2 budget: 4 of 5 used.
Reading: R1 makes dying bodies aim at the right spring (1,052 of 1,263; run 3's R mostly at units never stood in) and cuts deaths and the
water wait; it is uneven by start (9301 learns the room - deaths fall to 40 in the last quarter; 9303 does not improve, 81 wrong goals).
RW ran without J1, so the registered TC1 comparison (RW2 full weight + J1 vs RT contingency trust + J1) also shows what J1 adds back.

# R2 wired (founder 2026-11-27: "attach R2 now and show it on the viewer") - Astra's place bundles, LOKI c7c304c, behind ATANOR_R2=1
Going._r2_observe: Astra's PlaceBundles (64 units shared by all worlds; relief and walked crossings per context, 3 bundles) takes over place
learning; Going reads its snapshot (C, R, Vn, T, Pu) every beat. Inputs are the being's own: path code, inside rises per channel, and a
blocked move (its forecaster foresaw a step, its path sense did not move). The grown being's map is imported as context 0; each unit's
place is read from its own path code (odometry). Consolidation (sleep) every 100 beats with the map's refresh - an interim stand-in.
J1's erasing of a place's relief is overwritten by the snapshot (R lives in the bundles) - J1 is weaker under R2.
First look (64 bodies, 1,200 beats, A3 9301, TC1 + J1 + T1; not a registered run): fills water 6 / food 19, deaths 64 in the third quarter
(the first lives), dying goals: none at units never stood in (the old world's relief is no longer read; R1's first lives: all 64), all 64
"wrong". Churn: 2,283 unit recycles in 1,200 beats (64 bodies bumping walls -> blocked moves count as informative and lower the recruit
threshold) and 9 bundle overwrites - Astra's tests were one body. Registered comparison R1 vs R2 to follow (same being, starts, recipe).
## P2b (founder 2026-11-27: "the head goes down on purpose and comes back to facing ahead, as a person's does - is that in?") - it was not
P2 kept the pitch where the last neck gesture left it, so the being (grown without pitch; neck 3/4 did nothing in its growth) random-walked
its head to the limits and looked at the floor for long stretches (the all-green picture on the viewer). Body change (ATANOR_PITCH_RETURN=1,
room_body.step): a head not held up or down this beat drifts back toward level by x .7 a beat (~1 s), the neck's own tone - body mechanics,
not a choice. Viewer at once; P2 + P2b stay trial flags until measured (pitch off vs P2 vs P2 + P2b, rep 0, 64 bodies, A2).

## TC1 comparison result (J1 decoupled, R1, 6,000 beats; RW2 9302 crashed without a trace and is being rerun): not met - and two findings
RW2 (R1 + full weight + J1): 9301 126 106 102 103 = 437 (RW without J1: 334), water 372 / 126.5, food 220 / 239, J1 fired 803;
9303 128 x 4 = 512, no fills, J1 fired 3,248 (RW: 502). RT (R1 + TC1 + J1): 453 / 456 / 469; trust ends .26 / .28 / .35 (path .5 -> ~.27).
(1) J1 hurts under R1: it erases a place's relief after 5 lacking beats at its goal unit, and with new, coarse-to-fine remapped units the
body is "at" its goal unit while the spring is still a cell or two away - it erases the right memory (9303: 3,248 erasures, no fills).
(2) TC1 as built is unfair to the homing content (my design flaw): drinking / eating gestures are not "going the homing way", so the
relief that comes while reaching at the spring credits "not followed" - p(relief | not followed) > p(relief | followed), trust < .5.
Not adopted. A fair version must judge the journey (the beats that brought the body to the spring), not the reaching beat.
Found in the code (Going.lean_to): when there is no lived chain of crossings from where it is to its goal, the homing content is zero -
R1's new units have few crossings, so a body knowing where to go gets no direction there. IM2 (ATANOR_IMAGINE_GOAL=1, with IMAGINE_MAP):
then the way is imagined on its own wall map to the goal itself. Diagnostic (one start, 3,000 beats): RW's recipe with and without IM2,
counting goal-held beats with no lived way.

# S5 spatial tests (registered before running; founder 2026-11-27: "build the shortcut and detour tests now") - judge side, tools/s5_space.py
World: 16 x 16 room, a central block (rows 4-11, columns 3-12); food top middle, water bottom middle, birth below the block; two ways round.
After 3,000 beats of living, for every body and its rebirths: shortcut - a 2-wide corridor opens straight through the block (way to food
~22 -> ~11 cells); detour - the side used more for food is walled (2 cells across the side corridor). 2,000 beats after. 64 bodies, seed 9501.
Arms: RW's recipe (R1, full weight, J1 off, T1, eye 1.1, s3_e) with and without IM2. Read-outs: which way food journeys came (left / right /
gap), beats to the first food journey through the gap, deaths and food wait before / after.
Readings (Tolman 1948): shortcut - a share of food journeys through the gap >= 50% within the 2,000 beats = it uses space, not only a lived
chain; detour - food fills after >= 50% of before and deaths after not above before's rate = it goes round. Expected for now (said before
running): the shortcut fails - its wall map keeps the block as wall until it walks there, and nothing it sees corrects the map.
## IM2 diagnostic (one start, 3,000 beats): the "no lived way" case is only 10% of goal-held beats; IM2 makes it worse
RW recipe without IM2: deaths 64 40 39 49 = 192 (reproduces r1w exactly), goal-held beats 172,951, with no lived way 17,711 (10%);
with IM2: 64 64 55 40 = 223, water 117 / 263, food 75 / 330. Reading as set beforehand: the path layer's "no way" is not the main gap -
90% of the time a body with a goal has a lived way and still dies on it. What remains on the way (rung 1: the straight vector is lethal in
A3): the lean aims `hops` = 4 units ahead on its chain and then straight, which cuts corners through walls in A3's doorways. Diagnostic
(running): hops 1 and 2 on the RW recipe. Then S5 (shortcut / detour) as registered.

# B0 (Plan B first step; founder 2026-11-27 "Yes, build B") - work in progress, NOT working yet, behind ATANOR_B0 (default off)
Going.b0_foresight + a hook at the end of Mind.choose: while a goal is held, each gesture is scored by imagined wholeness (1 - mean squared
deficit) at a horizon - its own forecaster's direction for the gesture, its own free pace and felt drain (learned in observe), then the
best of 12 imagined headings walked `hold` beats, then the imagined way left on its map (ATANOR_B0=own: its wall map; =oracle: the judge's
true field, a ladder rung, never a pass). State at stop (4-body debug, oracle map): bodies leave the stem at first and then stand still
choosing gesture 0 (ties between candidates near walls; argmax takes the null gesture), and the GPU ran out of memory with the viewer, the
hops diagnostic and S5 running. Next: break ties toward moving gestures / check the wall stop, rerun the ladder (oracle map -> own map)
when the GPU is free; then S5. Nothing measured yet.

# W0 (registered before running; founder 2026-11-27 "run the smallest world-model test now"; Copilot's three powers: prophet / judge / executor)
Prophet: the being's own learned forecaster (core.foresee: next senses per gesture), rolled 20 beats for 9 futures made of its own gestures
(walk on; turn 1, 2, 3 or 6 times left or right, then walk). Judge: its own wholeness in the imagined end, plus lack x what its place
memory holds at the imagined place for the lacking channel. Executor: the first 4 beats of the best future. Called only while it lacks (a
goal is held) and its last plan has run out; habit (the main line's choice) otherwise. ATANOR_W0=1, default off. Nothing from the judge.
Runs: RW's recipe (R1, full weight, J1 off, T1, eye 1.1, s3_e) with W0 - S5 shortcut and detour (seed 9501, as registered) and A3 9301
3,000 beats (without W0: 192 deaths).
Pass (the only one that counts, set by the founder's side before running): the shortcut is taken - >= 50% of food journeys through the gap
after it opens, where the arm without W0 does not. Fold: survival numbers move a little and the shortcut is not taken.
Expected (said before running): fail - a 20-beat roll of a one-beat forecaster blurs, and nothing tells it the block has opened.

# E1 control arm (registered before running): Astra's one-step prototype (LOKI 27f33dd, loki_memory/embodied_choice.py; fetched, approved)
Astra's own synthetic result: oracle ladder 24/24, first-opened shortcut 24/24, familiar detour (first step away from the spring) 0/24 - and
Astra's point: a one-step predictor passes the shortcut, so the shortcut is not evidence of multi-beat foresight; the detour is.
Wired as ATANOR_E1=1 (Mind): EmbodiedChoice with the being's own path position (2 numbers) as "path" and its own heading (cos, sin) as the
sense; one instance for the mind, a transition per living body per beat. Same calling rule as W0 (a goal held and a remembered relief
place; habit otherwise). Runs: as W0 (S5 shortcut, S5 detour seed 9501; A3 9301 3,000 beats), RW's recipe.
Amended yardstick (after Astra's report, before any W0 result was read): W0 is judged by the shortcut as registered, but multi-beat foresight
counts only if W0 beats E1 on the detour (food fills after the block and deaths). W0 ~ E1 on the detour -> its 20-beat roll does nothing.

## S5 baseline results (RW's recipe, seed 9501; before / after the change: deaths, food fills, median wait, ways left / right / gap)
shortcut: without IM2 256, 0 | 80, 192, 152.5, 42 / 36 / 114 (never found food before the gap - not a shortcut test); with IM2 105, 227, 157,
91 / 136 / 0 | 0, 217, 55, 73 / 54 / 90 (41% through the gap, first at 36 beats; bar 50% - not met, near). detour: without IM2 no food at
all; with IM2 (right side walled) 105, 227 | 40, 195, 177, 176 / 18 - goes round (86% of fills kept, death rate not up): met.
Caveats: the gap lies on the straight line birth -> food (path integration alone can take it); the detour's two ways are equal, so no first
step away from the goal is needed. IM2 helps in this open room and hurt in A3.

# S5 strong detour (registered before running; Copilot / Astra: "the first step must go away from the goal")
tools/s5_space.py --test strong: the body is born inside a pocket (walls above, left, right; open only at the bottom), the food straight
ahead beyond the pocket's top wall, water below. Every life must first walk away from the food to leave the pocket, then go round a side.
No world change; 5,000 beats, 64 bodies, seed 9501, RW's recipe. Arms: plain, IM2, W0, E1 (Astra's one-step control).
Read-outs: food fills and deaths by half, beats from birth to leaving the pocket (median, by half), share of hungry beats at the pocket's
top. The claim "it gives up a near gain for a later one" needs: an arm with food fills >= 200 in the second half and deaths in the second
half below E1's by >= 30%. Expected (said before): E1 fails as in Astra's synthetic test; W0 fails (blur); IM2 may pass once its wall map
holds the pocket (walls are learned by bumping).
Amended before any run (founder 2026-11-27: "verify the likely failures fast and move on"): W0's and E1's queued long runs stopped unread;
the strong detour goes first as a fast probe - 2,000 beats (1,000 + 1,000), four arms at once. A clear failure here closes an arm; only an
arm that lives gets the registered 5,000-beat run.

## Strong detour, fast probe (2,000 beats = 1,000 + 1,000, 64 bodies, seed 9501; deaths, food fills, median wait by half)
plain 64, 1 | 44, 49, 332 (hungry beats at the pocket's top 28%); IM2 64, 0 | 64, 0 (pocket top 4%, leaves the pocket in 10 beats - and
never finds the food at all); W0 63, 12 | 49, 53, 113 (pocket top 41%); E1 (Astra's one-step control) not finished - its per-body loop is
slow with 64 bodies. No arm lives: in 1,000 beats a body has ~1.5 lives, and the food has to be found first. Expected "IM2 may pass" was
wrong: with no food yet found there is no food goal for it to imagine a way to. W0 does not differ from plain enough to say anything.
Reading: the strong detour as built mixes two things - finding the food the first time and going round to it. Nothing closed, nothing passed.

# SM1 seen map (registered; founder 2026-11-27: "does what it sees enter its map? build it now and show it")
Going._seen (ATANOR_SEEN=1, with the wall map): a small net from a 12 x 20 picture to "how far ahead is free", grown from scratch, taught
only by its own feet - pictures of the last 32 beats facing the way it is going are paired with how far it then got (exactly at a stop, at
least that far while walking). Once its own error on new stops is under a cell (and >= 300 stops), looking writes its map: free as far as it
expects, a faint wall where it expects a stop (weaker than touched; looking through a touched wall clears it). No geometry given.
Smoke (64 bodies, 600 beats): 232 stops, its error 3.3 cells - not yet writing. Viewer at once. To measure: its error over a 6,000-beat
life; then strong detour and the S5 shortcut with the gap off the straight line.

# SM2 seen relief (registered; founder 2026-11-27 on the viewer: "the water is in view and it does not look at it")
Going._seen_relief (with ATANOR_SEEN=1): when an inside rises, the pictures of the last 32 beats that were facing the place where it rose
learn "relief of this kind lies ahead, this far"; pictures kept 32 beats with no relief learn "none". Once it has 200 such pictures and
calls > 70% of new ones right, looking alone writes an expected relief (fainter than lived) into its place memory at the place it expects,
with that place's position - a spring it sees can become its goal before it has drunk there in this world. Teacher: its own insides and
path sense. Smoke: runs; nothing learned in 1,500 beats of A3 without a fill. To measure with SM1 (long life, then S5).

# F100 (registered; founder 2026-11-28: "decide the field of view by generality - glasses, ordinary cameras" -> "go ahead")
Looked up: Rokid Glasses camera 109 degrees, Ray-Ban Meta (Gen 2) 100 (one regional page), HP webcams 66-92 diagonal; Minecraft's default
~100 across. The room eye was 82.6 degrees across (256 px at the Minecraft focal length; I had said 80). Change: ATANOR_FOV=100 sets the
room eye's horizontal field of view (room_body.FY); unset = as before, so every earlier run reproduces.
Growth: s3_worlds 9104_0 9105_1 9106_0, tag _f, ATANOR_EYE=1.1 ATANOR_FOV=100 (as s3_e's growth but for the field of view; ordinary body).
Then the seen-map / seen-spring eyes grown on it in a 64-body life (a2_moving_food 9104_0, 6,000 beats, --save-seen) and carried.
Check before use: s3_f against s3_e on A2 9104 (64 bodies, 3,000 beats, T1): deaths and food wait not worse by > 10%; share of beats with
the whole picture one wall colour (judge side) lower. The 82.6-degree seen-eye training (running) only answers "can it be learned at all".

# Fetched (founder approved, 2026-11-28): Astra's gut-feeling prototype, LOKI de50f9f (loki_memory/somatic_readout.py) - kept off
Astra's result (synthetic, 24 wirings x 2 needs): on a fork whose first 20 beats are alike and whose relief differs at beat 24, the fast
readout alone 48/48; erased 23/48, pairs shuffled 22/48, random 25/48; in the familiar fork the slow readout was never called; with the look
kept and the effect flipped the first choice was wrong 24/24, recovering 1% -> 4% -> 27% -> 96% over four blocks of four.
How it is to be wired (registered before building; G1): not per gesture - a held gesture's change of the insides is drain alike for all
gestures except at a spring, where the pull already works. The unit of a gut feeling is a journey: cue = where it is (its place unit's own
position) and what it lacks, action = which remembered place it sets out for, outcome = how its insides changed from setting out to the
journey's end (arrival, giving up, or death), drain on the way included. Goal choice then asks the gut, not "the highest relief memory for
the lacking channel": among its few best-remembered places per channel, the one whose remembered journeys leave it most whole. Where the
gut has no record (confidence low) the choice falls back to the relief memory. Tests: ties between two remembered places (near small, far
large) - choice follows past journeys >= 80%; erased / shuffled gut -> as the relief memory alone. After F100's check.
## SM1 / SM2 grown in a 64-body life (A2 9104, 6,000 beats, s3_e, eye 1.1, field of view 82.6): both learn
Seen map: its own error on new stops .68 cells (bar < 1) over 2,062 stops; it wrote its map 243,061 times. Seen spring: 88% of new relief
pictures called right (bar > 70%) over 12,355; 28,876 writes into its place memory. Saved to probe-work/s3_e_seen.pt and carried into the
viewer's one body (ATANOR_SEEN_FROM). Not yet measured: whether they hold at a place never seen (Astra's second condition), and what they do
to a life (deaths, waits) - both with the strong detour's three conditions. Head pitch is off on the viewer for now: these eyes were grown
with a level head.

# TF1 touch feel (registered; founder 2026-11-28: "it backs into something and does not know what is behind it" -> "grow it with direction
# and feel"). Correction of what I told the founder: the body's skin already feels which side is touched (front / right / behind / left,
# within .8 cells; skin 0-3, and a copy in 4-7) - I had said there was one bump bit only (that is the felt-motion channel). What it lacked
# is what the touched thing feels like. ATANOR_TOUCH_FEEL=1: skin 4-7 = closeness x feel per side (wall .3 hard, food .65 soft, water 1.0
# wet) - a feel on the skin, not a name; the sense vector's size is unchanged. Unset = as before.
# Growth G (replaces F100's growth, stopped 14 minutes in, unread): s3_worlds 9104_0 9105_1 9106_0, tag _g, ATANOR_EYE=1.1 ATANOR_FOV=100
# ATANOR_TOUCH_FEEL=1; then the seen eyes (with bearing) grown on it. Two body changes at once (field of view, touch feel) - said plainly;
# the check against s3_e is of the pair. Separate later only if the pair is worse.
