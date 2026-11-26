# K1 (registered before running): where the being believes its springs are, against where they are

Roadmap 2026-11-26, stage 1 ("hold what is out of view"). D39-D43 closed the choice line: the knowing is the wall. Before building anything,
read the being's knowing directly.
Theory: animals learn a food place in one or a few visits (one-shot place learning, hippocampus; Morris water maze, Tse et al. 2007 schema
learning). The going organ counts reliefs per place unit every beat (online), so one fill should already mark a place. But its place units
were formed in the being's first room, and an old unit barely moves (k-means rate = 1 / count); in a new room the old relief memory sits on
old places (S3). If the belief moves to the new spring within a few fills, the knowing forms fast and the new-room cost is elsewhere; if it
stays on old places for hundreds of beats, slow or blocked knowing is the cost.
Read-out (judge side only, tools/k1_knowing.py): each place unit's true position = slow mean of the true positions of bodies in it (fading
.99 per entry). Every 50 beats: believed water place = position of the unit with the largest water value V[:, 0] (Going.refresh), believed
food place likewise; error = cell distance to the nearest true spring cell of that kind. Also: first beat any body fills water / food; the
fills by then. S3-end beings: s3_a in 9101 (home) and 9105 (never lived), s3_b in 9104 (home) and 9102 (never lived); 3,000 beats, 64 bodies.
Readings: in the new rooms, error falls to within 1.5 cells within 300 beats of the first fill of that kind -> knowing forms fast (one-shot
like); the new-room cost is in getting there (skill, or the hold-before-learning of S3). Error stays above 3 cells for more than 1,000 beats
after the first fill -> the knowing is slow or held by old memory; that is the first thing stage 1 must change. Home rooms: error should
stay within 1.5 cells throughout (if not, the read-out is wrong).

## K1 result: the room being knows where its springs are, and learns a new room's in 100-700 beats
Believed spring place (unit with the largest value) vs the nearest true spring cell, 3,000 beats, 64 bodies:
| room | first fill water / food | error <= 1.5 cells first at (beats after first fill) | last 1,000 beats mean error | largest error ever | deaths |
|---|---|---|---|---|---|
| 9105 new (s3_a) | 33 / 127 | water 367, food 673 | .73 / .42 | 2.0 | 1 |
| 9102 new (s3_b) | 77 / 199 | water 23, food 1 | .50 / .60 | 1.4 | 1 |
| 9101 home | 89 / 71 | from the start | 1.05 / 1.05 | 1.4 | 0 |
| 9104 home | 35 / 149 | from the start | 1.05 / 1.16 | 1.3 | 0 |
Readings as registered: 9102 fast (within 300 beats of the first fill, met); 9105 in between (367 and 673 beats, never above 3 cells) -
neither "one-shot" nor "held by old memory". The error never exceeded 2 cells in any room. Deaths were 1 in each new room in these 3,000
beats: for springs that stay put, the room being's knowing of out-of-view places is good and comes within hundreds of beats.
So in the room, stage 1's gate is nearly met for still things. What stage 1 still lacks is elsewhere:
- MineLand: the being has no going organ (found today) - it cannot know where a cake was. And the going organ cannot simply be carried
  over: it reads the room's inside order by position (ins[1] = food, ins[2] = water, "the world's store order"), while the universal body
  contract gives unnamed, shuffled bedrock channels (MineLand: life, food, oxygen in a per-body order). That is a room assumption in a
  part meant to be universal.
- Moving things (the guardian, S4k): place units hold places, not things that move.
Next (registered below): G1 - the going organ made channel-agnostic (one relief memory per bedrock channel, read from the contract's
bedrock slot, no names), checked to give the same room numbers, then given to the MineLand being.

# G1 (registered before building): a going organ that reads the bedrock by contract, not by room order
Change: Going takes the bedrock channel list from the body contract (Organs.inside) and keeps R, V, nxt per channel (k channels, not 2);
lack and "which channel" come from the bedrock as it is (lowest channel, as now); no channel is named. A channel that never lacks (the
room's constant) simply never pulls. Nothing else changes (place units, map, skill, hops, W).
Checks, in order: (1) equivalence - the S3-end beings in 9101 / 9104, 3,000 beats, same seeds: need fills and deaths within 5% of the
current organ, believed-spring error (K1) within .3 cells; (2) smoke_main 6 PASS; only then (3) MineLand: the being continues life 25's
file with the organ added (path organ on: MineLandBody(path=True)), 5,000 steps, judge-side D35c record. Bar for (3): hungry body-steps
in reach of a cake and eating in at least 3 of the 10 windows (life 25: 1 of 10), with no rise in deaths over life 25.

G1 amended before any MineLand run (found while wiring it): with the path organ on, the being's sense grows by the path code, so life 25's
mind file cannot be continued as it is. The MineLand step therefore wears life 25's last file (Mind.wear: eye parts carried, the rest new -
the founder's "find this body by living"), and a control arm does the same with the path organ on and the going organ off, so the only
difference is the going. Arms: A wear + path + going (chan contract); B wear + path, no going. 5,000 steps each, one at a time, judge-side
D35c record. Bar (3) restated: hungry body-steps in reach of a cake with eating in at least 3 of 10 windows in A, and more such windows than
B; deaths in A not above B. A recovery reloads the mind but keeps the body's going organ (it is the body's place memory).

## G1 MineLand result: not met - no body came within reach of a cake in either arm
5,000 steps each, 10 healthy windows each, no faults. Cake reach [in reach, hungry, eats] = [0, 0, 0] in all 20 windows.
Deaths A (wear + path + going) 66 vs B (wear + path) 85 (-22%, one start each); fills ("relief") A 82, B 124.
Reading: bar (3) not met. The going organ can only remember a relief that happened at a place; in 10,000 body-steps the bodies never came to
the cakes, so it had nothing to remember there. Finding the food in the first place is the gap in the valley, before remembering it -
which belongs to the world question (A: a world where finding, returning and planning are needed and can be learned step by step) and to
the planning organ (B). Deaths fewer with the going organ (it may remember other reliefs - life, oxygen - by place); not claimed on one start.
