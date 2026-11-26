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
