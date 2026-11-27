# M2a: sleep as memory consolidation (theory first, 2026-11-27)

Founder chose sleep and dreams as the first side track; Copilot's review (same day) put it as M2's first step, sleep before dreams:
sleep = gather, replay, tidy statistics, compress; dreams (recombination, imagined situations, the imagination organ) only after.
Order kept on the main line: D50, M1a -> gate 2 (A3) -> M2a sleep -> MineLand round trip -> dreams, M3, M4.

## Why the room being needs it (read from the code, to be measured in S0)
- Place memory is fixed: 64 place units (Going.KU). Units are born only while fewer than 64 exist (_recruit); after that a new world
  can only be held by dragging the old units' centres toward the new codes (online k-means). So the first world fills the memory, and a
  second world overwrites it: the M2 round trip (room -> MineLand -> room) is forgetting by construction.
- Gate 2 run 1 showed the same thing from the other side: in A3, 71% of its goals were units it had never been to in that world.
- All learning happens awake: the world model every 20 beats, the place value every 100, the going skill every 1,000 beats (in after()
  ~12 ms of the ~39 ms a beat the being costs). Sleep would move the big learning off the awake beat (efficiency rule, 2026-11-26).

## What sleep does (M2a, no dreams yet)
While the body sleeps (does not act; the world's clock runs): replay the day's stored beats (the bounded experience ring) to
(1) consolidate: merge place units that stand closer than the birth radius and carry the same transitions and reliefs (frees units),
(2) protect: units that hold relief or many crossings are kept and their statistics renormalised, (3) learn the big things offline
(going skill, place value by reverse replay from the reliefs - Foster & Wilson 2006, Ambrose 2016), then (4) free units for new places.
When to sleep: an evolved bedrock (sleep pressure rising while awake, falling in sleep, as hunger does) - for founder review, as every
bedrock. First version (for S1 only): fixed sleep at the end of each period, to test the consolidation before the pressure.

# S0 (registered before running; judge side only): is the place memory full and redundant, and does a second world overwrite it?
After an A2 life (9104_0, s3_b, 64 bodies, 6,000 beats): units used; pairs of units closer than the birth radius; units never crossed
or with no relief. Then the same mind (saved) lives 3,000 beats in A3, and again 3,000 beats in A2: mean drift of the A2 units' centres;
A2 deaths and food wait after the A3 excursion vs before (same seed for both A2 checks).
Readings: units all used and >= 20% redundant or empty -> room to consolidate; A2 worse after A3 by >= 25% with drift -> the second world
overwrites the first (M2's problem is real): build S1 (sleep consolidation) and test it on the same round trip. Neither -> report, and
M2's first block is something else.

## Context calibration for the LOKI place store (judge side, 2026-11-27): the two rooms do not look different
LOKI v13 (3a64b96, fetched to ext/LOKI-v13 with the founder's approval; its 8 store tests pass here) tells worlds apart by a slow
context of appearance. Read from the code first: the going organ's place code is the innate path code (position from the birth point),
not appearance - so any two worlds born at the same point share units by construction (gate 2 run 1's goals at A2 units).
Measured (600 beats each, 64 bodies, s3_b): the being's 104 picture features, standardised, slow mean (fading .98 / .995 / .999),
all or half of them: A2 and A3 told apart by nearest mean 60-65% (chance 50%). The two rooms are built of the same materials; their
look carries almost nothing of which world it is. So in A2 <-> A3, LOKI's context cannot keep the worlds apart by appearance; between the
room and MineLand (a different look entirely) it should. What would tell A2 from A3 is layout and where relief comes (latent cause, the
going organ's S3b contexts). Decision deferred to S0's result.
S0 as run (deviation from the registration, said before reading): each phase 3,000 beats (the first A2 too, not 6,000), the loaded
being's units already all used at birth (s3_b grew elsewhere); reps 0-2.

## S0 result: A2 after A3 dies far less and waits ~24% longer - but phase order is also learning time; a control is needed
Reps 0-2 (deaths; median food wait), A2 before -> A3 -> A2 after: r0 92; 245 -> 256 deaths -> 11; 309 | r1 110; 294.5 -> 256 -> 54; 366 |
r2 92; 295 -> 256 -> 39; 364.5. Units: all 64 used; 20 pairs closer than the birth radius; 14 of 64 quiet (no visits, no relief: 22%).
A2 unit centres moved 0.07-0.12 (birth radius 2.55; none beyond it); homing positions moved 1.1-1.16 cells.
Reading against the bars: units >= 20% redundant or empty - met (room to consolidate). A2 worse after A3 by >= 25% - not met: deaths fell
(294 -> 104 summed), waits rose 19-26% (median 294.5 -> 364.5, +24%). Unit centres did not move (as read from the code: the units are
positions); what moved are the homing positions kept per unit (1.1 cells) and, by construction, the reliefs and crossings - the same units
were relived as A3. But the first A2 is also the being's first 3,000 beats in this run (trust learning from .5, the world model young),
so "after" carries 6,000 more beats of learning. Without a control the excursion's own cost cannot be read.
# S0b (registered before running): the control - A2 -> A2 -> A2 (the middle stretch in A2 with another seed), same reps and beats
Reading: S0's final A2 vs S0b's final A2 - deaths and median wait; the excursion's cost = S0 minus S0b. Excursion worse by >= 25% on
either -> the shared units cost the first world (identity confusion, Copilot's Case B), the M2 block is the context layer; else ->
the room pair does not show the M2 problem, and M2 is tested room <-> MineLand where it can show.

## S0b result: the excursion costs the first world - met (deaths ~4x, wait +28%); identity confusion is the M2 block
Final A2 phase, same reps and seeds (deaths; median food wait): after A3 (S0) 11; 309 | 54; 366 | 39; 364.5 -> 104 deaths, median 364.5.
After A2 only (S0b) 2; 224.5 | 13; 284.5 | 12; 284.5 -> 27 deaths, median 284.5. Excursion worse in every start: deaths 104 vs 27,
wait +28%. Reading as registered: the shared units cost the first world (Copilot's Case B: A3 is lived on A2's units from its first beat,
since a unit is a path position) - the M2 block is the context layer that tells worlds apart, not memory size.

# P1 (registered before running; read-only probe, Astra's recommended next step, founder approved fetching LOKI f65d767)
Astra's RouteWitness (loki_memory/walls.py, f65d767; its tests pass here) read beside the going organ in the S0 round trip (A2 -> A3 -> A2,
reps 0-2), one witness for each of the first 8 bodies, fed only the being's own senses (tools/witness_probe.py: picture features ->
fixed projection to 12, own path position / 30 + .5 (the witness takes 0-1), own inside rises). Nothing reaches choice, map, value or homing. Counted per phase,
before vs after the body's first relief in it: right / wrong / withhold (a bank's world = the world it was first written in); which
world each 64-beat consolidation wrote into; the going organ's A3 beats on units it lived in A2.
Readings: the witness right on the return to A2 (after first relief) >= 80% and wrong <= 10%, while the going organ lives A3 on A2's units
-> the place is readable from its own senses, the going organ just does not read it: next = a context layer in the going organ fed by
this evidence (as content, its trust learned). The witness mostly wrong or withholding on the being's own senses (Astra's 96/96 was on
judge-made wall distances) -> the being's senses do not carry what tells the rooms apart: the representation must be grown, not wired.

## P1 result: on the being's own picture features the witness withholds ~99.5% - nothing to read; A3 never written
Reps 0-2, 8 watched bodies: return to A2 after first relief - right 0 / 102 / 0, wrong 0, withhold 20,632 / 20,868 / 21,277 beats.
First A2: right 1-2%, the rest withheld. A3: 0-10 wrong, never right; no 64-beat consolidation in A3 was written into any bank
(only A2's: 65-68 per run). Going organ: 83-87% of A3 beats on units it lived in A2.
Reading as registered: the second - on these inputs the being's senses do not give the witness what tells places apart. Said plainly
about my wiring: the picture features change with every turn of the head (the eye is ego-centred), while Astra's 96/96 used world-fixed
wall distances in four directions - a viewpoint-free view. So P1 shows that the ego-centred picture is the wrong key for a place, not
that the being lacks the information: its depth sense and its innate heading together hold a viewpoint-free code.
# P1b (registered before running): the same probe fed a boundary code grown from its own depth + innate heading
One change: view = its own depth sense (5 nearness sectors) placed by its innate heading into 12 world-fixed directions, each keeping
what was last seen there (head direction x egocentric boundary -> boundary-vector cells; Hartley et al. 2000, Bicanski & Burgess 2018;
tools/witness_probe.py view="boundary"). Same runs and readings as P1. Met (return to A2 right >= 80%, wrong <= 10%) -> this boundary code
is the place key the going organ lacks: build it into the being (as an organ of its own senses) and feed the context layer; not met ->
report, and the representation is grown (learned) rather than computed.

## P1b result: not met - the boundary code reads early, then withholds; A3 mostly written into A2's memory
Shares of watched beats (right / wrong / withhold): first A2 before first relief .73 / 0 / .27, .54 / 0 / .46, .29 / 0 / .71; after it
.10-.14 right, rest withheld. Return to A2 after first relief: right .03 / .01 / .01, wrong 0 (bar >= .80 right). A3: never right,
wrong .03-.14. 64-beat consolidations in A3 written into the A2 bank 10 / 23 / 16 times (into an A3 bank 0 / 6 / 0).
Reading as registered: not met. The viewpoint-free code is better than the picture (P1: ~0 right) and never calls A2 wrong, but over a
3,000-beat life the witness falls silent and the second world is written into the first. The place key is not solved by wiring a
probe; per the registration the representation has to be grown (learned) in the being. This line pauses here (founder: build what
changes the life); the boundary code stays as a candidate sense for that growth.
