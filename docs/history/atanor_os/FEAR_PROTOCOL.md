# Fear of death and the will to live - theory note (founder 2026-11-25, after the pit experiment)

Founder's question: pain is a moment and the end is not felt, yet people fear death - something far ahead - blindly. Find the research,
and find how a fear of death / a "blind will to live" can emerge in the being.

## What the research says (no one learns to fear death from having died)
1. Fear is of death's precursors, prepared by evolution, then tuned by experience.
   - Heights: infants avoid the visual cliff (Gibson & Walk 1960), but the wariness appears only after the infant has crawled on its own
     (Campos, Bertenthal & Kermoian 1992): locomotor experience makes it, on a prepared base. Looming objects make infants and many animals
     flinch from birth (Schiff 1965).
   - Prepared learning: fear of snakes, heights, falling is learned in one or a few pairings and hard to unlearn, where fear of flowers is
     not (Seligman 1971; Ohman & Mineka 2001).
2. Fear looks ahead: defensive behaviour follows how near the threat is in space and time - pre-encounter wariness, post-encounter freezing,
   circa-strike flight (Fanselow & Lester 1988; Mobbs et al. 2015). It is a prediction, not a feeling of the harm itself.
3. In learning terms (homeostatic reinforcement learning, Keramati & Gutkin 2014): a good is the reduction of an inner deficit. Death is the
   state after which no deficit is ever reduced again and the deficit is maximal - an absorbing state of zero wholeness. A learner that looks
   ahead values life simply because the future holds reliefs; any act that risks the end loses all of them. This is the plain computational
   form of a "will to live" (Spinoza's conatus; Schopenhauer's Wille zum Leben; Friston's free-energy account: a living thing persists by
   keeping itself inside the states it expects).
4. Others' deaths teach: monkeys reared without snake fear acquire it from watching another monkey's fear (Mineka & Cook 1984/1993); people
   learn fear from watching another's distress as strongly as from their own (Olsson & Phelps 2007).
5. Humans add a concept: children grasp that death is irreversible and universal at about 5-7 years (Speece & Brent 1984); knowing "I will
   die" beside the evolved drive to live gives the anxiety Terror Management Theory studies (Becker 1973; Greenberg, Pyszczynski & Solomon 1986).
   That layer needs the inner model of oneself through time - our stages 5-7 - and is not the first step.

## Why the being does not fear death now (measured, not assumed)
- Death restores it. In MineLand life 21 (e/f), 264 of 449 deaths (59%) left the body with as much or more life after the rebirth as before.
  The rebirth refills food and life; the room's rebirth sets the store to .75. Nothing lost is felt.
- The value learner does treat a death as an absorbing end (loses the inside it holds, no reliefs after) - so the end is in principle
  foreseeable; but see the next two lines.
- The harm memory keeps only the last beat before death (`Core.keep_harm`): in a fall that is the body in mid-air, not the step at the edge.
- Learned value hardly steers: in life 21 value decided 73 of ~17,000 steps (0.4%). A look-ahead fear would have no road into choice.
- One mind lives in many bodies and a body's death never ends the mind - more like a species than one creature. A true will to live needs a
  Self that is at stake when its body dies (our original goal: one persistent Self).

## How it could emerge (in this order; one change at a time; each registered before running)
- F1 (no new bedrock) - a diagnostic first, because the value learner already treats death as an end: `ReliefWorth.learn` (channel_worth.py)
  scores a death as losing all the inside held now, with no future reliefs after it (absorbing). So the open question is where the fear is lost:
  (a) the value does not foresee it - at the edge, stepping off is valued no worse than staying; or (b) it foresees it but does not steer
  (value decides 0.4% of choices). D37 reads both at edge states in a world with a deadly drop. (a) -> credit over the seconds of a fall /
  longer horizon; (b) -> it is the value-to-choice problem (founder 2026-11-24 focus) and fear will come with it, not as its own organ.
- F2: felt precursors - the body's own vertical motion / loss of support felt in the body organ (vestibular), with the fear grown from its own
  falls as Campos found (experience on a prepared base). A bedrock decision for the founder only if F1 shows the signal is missing.
- F3: others' deaths - harm resonance (the empathy bedrock's other half: another's death seen is felt as harm where the observer stands).
- F4: generations (roadmap 10): bodies that die early pass on less; a prepared fear can then be selected, not written.
- Later: a concept of its own end, when the Self models itself through time.
First step: D37 in the room (fast): a ledge cell kind that kills on entry, beside the corridor; after a life there, read at edge states the
value of the step off vs the step along (a) and how often value decides there (b). Registered with numbers before it runs, after S4c/S4p/S4n
free the GPU.

# D37 (registered before running): a deadly drop beside the path - foresight and choice
World (room, a world property, not a rule): two drops cut into the walls beside the stem, one each side (atanor_os/emerge/room3d_gpu.py PIT,
seen as a dark low rim; walked into, the body dies). The S3-end being (one mind, 64 bodies) lives 8,000 beats in 9101 / 9104 with drops,
depletion world V2, rebirth at the start (near the drops). Read each beat for bodies at an edge (within 1.4 of the cell's centre, facing it
within 35 degrees): the value's vote for forward gestures (Mind.lw_last, forward minus others), and whether forward was chosen; the same at plain
wall cells beside the stem (a wall blocks, a drop kills). Also deaths by drop and by need per 1,000 beats, and who decided (Mind.src).
Bars (both layouts): (a) foresight - in the last 2,000 beats the vote at drop edges is below the vote at wall edges by at least .25 of the
lean's spread (lw_sd); (b) choice - drop deaths in the last 2,000 beats at most half of the first 2,000, and forward chosen at drop edges at
least .05 less often than at wall edges in the last 2,000.
Readings: (a)+(b) -> fear of the drop grew from its own pooled deaths through the value it already has (F1 enough); (a) without (b) -> it
foresees but does not act on it: the bottleneck is the value's authority over choice (founder's review: "action authority"); (b) without (a)
-> another path (harm memory) does it; neither -> credit does not reach the edge.
Smoke (1,000 beats, 9101, drops placed one row higher - since moved onto the stem; not a result): 249 drop deaths; vote at drop edges -1.45
vs walls -.07 (spread .95); forward at drop edges .23 vs walls .39; who decided: strong pull 12%, value 0, random 0, hold 78%, try 10%.

## D37 result (9101, 9104; 8,000 beats each): (a) foresight met in both; (b) choice met in 9104, not in 9101
Per 1,000 beats (64 bodies). Drop deaths: 9101 155, 17, 7, 0, 11, 11, 19, 16; 9104 158, 59, 62, 19, 12, 13, 45, 25.
Value vote at drop edges vs wall edges (last 2,000 beats): 9101 -5.0 / -7.4 vs -.05 / -.11 (lean spread .32-.44); 9104 -2.4 / -4.2 vs -.02 / -.01.
Forward chosen at the edge, last 2,000 beats: 9101 drop .40 vs wall .30; 9104 drop .39 vs wall .51. Edge visits fell: 9101 783 -> 70-90, 9104
846 -> 170-230 per 1,000 beats. Who decided (whole run): hold 78%, strong pull 11%, try 10%, value-alone 0.
- (a) met in both, by far: the value sees the drop as the end (votes 10-60x the spread below the wall's).
- (b) drop deaths in the last 2,000 fell to 20% (9101) and 32% (9104) of the first 2,000 - met; forward at the edge less than at a wall by .12
  in 9104 - met; in 9101 forward at a drop edge was MORE often than at a wall (.40 vs .30) - not met. So (b) is not met as registered.
- Reading: the deaths fell mostly because the bodies came near the drops far less (edge visits down 4-10x), not because they chose differently
  at the edge. At the edge, where the value says "this is the end" most strongly, the body still steps forward 34-47% of the time in 9101;
  most beats are not choices at all (holding the last gesture 78%). Drop deaths never reach zero (8-45 per 1,000 beats at the end). This is the
  registered "(a) without (b)" reading at the moment that matters: it foresees, but foresight has little authority over the act once in motion.
- Next (theory, not registered yet): what breaks a held act. Habit vs goal-directed control (Dickinson; Balleine & Dickinson 1998); conflict
  monitoring (Botvinick et al. 2001: anterior cingulate signals when the act under way conflicts with what is valued, which calls for control);
  fear freezing an act under way. Candidate: re-choosing is called whenever the value's vote against the gesture held is large - a general
  structure of choosing, no rule about drops. Holding is a fixed trait the founder set (2026-11-20), so this needs the founder's word.

# D37b (registered before running; founder 2026-11-25: allow conflict monitoring as a general structure, do not weaken holding)
One change to D37: Mind.conflict = 1.0 - when the value's lean against the gesture being held (mean lean minus the held gesture's) exceeds 1.0
(lean units), that beat becomes a new choice made by the usual choosing. No rule about drops, holding itself untouched. Same rooms, seeds,
beats as D37 (D37 is the control).
Smoke (1,000 beats, 9101): 434 extra re-choices called (0.7% of body-beats); first-1,000 drop deaths 150 (D37 155); value's earned trust .09.
Bars (both layouts): forward at drop edges at least .05 below wall edges in the last 2,000 beats; drop deaths in the last 2,000 at most half of
D37's last 2,000 (9101: 35, 9104: 70); conflict calls at most 5% of body-beats (the habit stays mostly intact).
Readings: met -> giving a strong value-act conflict the right to call a new choice is what let foresight reach the act (action authority).
Not met with calls well used -> the re-choice itself does not follow the value (the choosing at that beat, not the calling, is the gap).
The felt conflict signal (the being sensing its own conflict) is the next, separate step.

## D37b result: not met - the conflict monitor seldom fired and changed nothing beyond noise
Calls: 1,152 (9101) and 1,574 (9104) of 512,000 body-beats (.2-.3%). Drop deaths last 2,000 beats: 9101 48 (D37 35), 9104 36 (D37 70) - one
up, one down. Forward at drop vs wall edges, last 2,000: 9101 .48 vs .33 (not met), 9104 .46 vs .52 (met). Value's earned trust .25 / .40.
Caution found in reading: "forward at an edge" is confounded - bodies reach a drop edge mostly while walking (holding forward), bodies at a wall
edge are often standing; so the per-beat forward rate at edges cannot carry the verdict alone. The death counts are the clean outcome.
Next diagnostic (running): who decided the fatal step itself (strong pull, value, random, hold, try) - if most falls come from the 10% random
tries or from holds, the gap is there, not in what the value knows.

## Fatal-step diagnostic (D37 settings, 3,000 beats per room): who decided the step into the drop
| room | falls | strong pull | value alone | random | hold | try | share of all beats: strong / hold / try |
|---|---|---|---|---|---|---|---|
| 9101 | 179 | 40 (22%) | 0 | 0 | 124 (69%) | 15 (8%) | 11% / 79% / 10% |
| 9104 | 279 | 47 (17%) | 0 | 0 | 190 (68%) | 42 (15%) | 11% / 78% / 10% |
- Both scenarios the review named are there. Most falls are a held gesture carried into the drop (scenario A, 2/3 of falls) - but holds are
  slightly fewer among falls than among beats (69% vs 79%). Fresh choices by the strong pull - where the value is in the score - are twice as
  likely to be fatal as their share of beats (17-22% vs 11%): when it does choose at the edge, it still chooses the drop too often (scenario B).
  Random tries are about their share (8-15% vs 10%): exploration is not the main killer (scenario C small).
- Reading: the gap is not one switch. Holding carries the body in; and the moment of choosing near the drop is not ruled by the value enough
  (the score mixes relief memory, harm memory, futility, value; the value's part did not win there). Next to read before any change: at fresh
  choices one beat before a fall, the size of the value's lean against the fatal gesture vs the other parts of the score that chose it.

## D37c: the score at each fatal step, by part (fatal gesture minus that part's mean over gestures; 3,000 beats per room)
| room | who chose (falls) | going lean | value | relief pull | futility | harm memory | total |
|---|---|---|---|---|---|---|---|
| 9101 | hold (124) | +2.45 | +1.80 | -.08 | -.31 | -.02 | +3.32 |
| 9101 | fresh choice (40) | +5.21 | +2.34 | +.17 | -.03 | -.03 | +6.67 |
| 9101 | try (15) | +.05 | -7.96 | +.12 | .01 | -.03 | -4.63 |
| 9104 | hold (190) | +3.80 | +1.45 | +.42 | -.53 | -.02 | +4.93 |
| 9104 | fresh choice (47) | +5.95 | +2.28 | +.79 | .01 | -.01 | +8.74 |
| 9104 | try (42) | +.38 | -1.26 | +.28 | 0 | -.02 | -.56 |
(value present on 71-93% of these beats; spreads are wide, about the size of the means.)
- Correction of my own reading of D37: I wrote "it foresees but does not act". At the fatal beats the value is NOT against the fatal gesture:
  on held and freshly chosen falls it leans FOR it (+1.5 to +2.3). Only on random tries is it against (-8, -1.3). The strong vote at edges
  (D37 a) was an average over edge states facing the drop; the step that kills is usually another one - a gesture that leads on toward a spring.
- The part that carries the body in is the going organ's lean (+2.5 to +6, the largest): its map knows where relief was and how to get there,
  and nothing about where bodies died. The path to a spring runs beside the drop; the lean pulls along it and steps in. The harm memory is
  about zero at those beats.
- So the gap at the fatal moment is not authority (the value is not overruled - it agrees), it is that the strongest pull is blind to death:
  the place memory remembers reliefs by place but not deaths by place. D37b's conflict monitor could not help - there was no conflict.
- Next (to register): the going organ remembers where bodies died, as it remembers reliefs (r of a place lowered by its death rate) - a
  general place memory of harm, no rule about drops.

# D37d (registered before running): the going organ remembers where bodies died
One change to D37: Going.dread = 1 - the organ keeps, per place unit, a fading count of deaths there (as it keeps reliefs); a unit's worth on its
map is its relief rate minus its death rate, for both insides. Grounded in place-avoidance learning (rats avoid the place of a shock; the
hippocampus codes where harm happened as well as where food was). No rule about drops: any death anywhere is remembered by place.
Same rooms, seeds, beats as D37 (control). Bars (both layouts): drop deaths in the last 2,000 beats at most half of D37's (9101 <= 17,
9104 <= 35); need fills in the last 2,000 at least 80% of D37's (it still travels to its springs - no freezing). Also read D37c's parts.
Caution before running: place units are about one cell wide; a unit may span a stem cell and the drop beside it, then the stem itself is
dreaded - the map may then avoid its only road; the need-fill bar catches that.

## D37d result: not met - and harmful in 9104
Drop deaths per 1,000 beats: 9101 182, 22, 3, 5, 1, 9, 17, 24 (last 2,000: 41; D37 35); 9104 149, 111, 83, 132, 114, 93, 87, 63 (last 2,000:
150; D37 70). Need fills 9101 near D37 (242-285 vs 269-305); 9104 collapsed to 0-20 (D37 88-167): the bodies stopped reaching their springs.
At the fatal steps the going lean stayed the largest part (9101 +2.6 / +5.1, 9104 +4.0 / +7.7).
- Reading: a death remembered by place unit is the wrong grain. A unit is about a cell wide; a drop beside the stem maps onto the stem's own
  unit, so the road itself is dreaded (9104: the road to both springs) - the map breaks while the step into the drop, a matter of how the
  skill steers along the road, is untouched. The death is in the last half-cell of a step, not in a place on the map.
- What D37c-d show together: the going lean decides the step (softmax of the score at temperature .05, the lean's parts reach +5 to +8 while
  harm memory sits near -.02), and it carries no record of what following it has cost. The value earns its say by its own record (earned);
  the going lean does not.
- Next (registered below): the going lean earns its weight by its record, as the value does.

# D37e (registered before running): the going lean earns its weight
One change to D37 (dread off): the lean of the going organ is multiplied by its own earned trust - the correlation, over its own beats,
between how hard it pushed and the body's wholeness change in the next 60 beats (a death counts as losing all wholeness), the same tracker
the mind uses for its strong pull (Daw, Niv & Dayan 2005: who decides is set by each controller's reliability). Weight 1 until 60 beats are
tracked, then clamp(trust, .1, 1). Bars as D37d (drop deaths last 2,000 at most half of D37's; need fills at least 80% of D37's), both rooms.

## D37e result: not met - and the reason all of D37 failed is in the world I built, not in the being
Going lean's earned weight fell to its floor (.1) in both rooms. 9101: drop deaths last 2,000 25 (D37 35; bar <= 17), need fills higher than
D37 (341-401 vs 269-305); 9104: need fills 0 from beat 1,000 on, drop deaths 35-223 - without the lean the bodies there could not reach
their springs. At fatal steps in 9101 the value now carried the step (+4.1 hold, +6.7 fresh): the value too favours the fatal gesture.
- Found when asking why every part of the score favours the step that kills: the body cannot sense the drop at that moment. Touch
  (room_body.raw) casts four short rays and reports any non-floor cell as touch, whatever it is - a drop beside the body feels exactly like a
  wall. The eye sees 80 degrees ahead; a drop at the side, where bodies walking along the stem step in, is out of view. A step sideways
  toward a "wall" is harmless everywhere else (it is blocked), so value, lean and memory all learned that it is fine. D37's strong vote at
  drop edges was only where the drop was in front and seen.
- So D37-D37e tested whether the being avoids a danger it cannot perceive when it acts. Of course it fails. This is my design error in the
  drop world; the chain of readings (authority, place dread, earned lean) were all reading a blind being. Not a finding about the mind.
- What a real body has: the ground under and ahead of the feet is felt (feet, whiskers, the inner ear as the floor gives way). A drop must
  feel different from a wall to touch. That is a change to the body's senses (founder's decision: sense additions were closed after touch).

# D37f (registered before running; founder 2026-11-25: add ground touch - a sense, no meaning)
Body change (atanor_os/emerge/room_body.py raw): touch no longer reports a drop as a touched thing (a drop is not touched); four unused skin
cells (9-12) feel "no ground just ahead / right / behind / left" within the same short reach - like feet or whiskers. Nothing says it is
dangerous, nothing steers away; what it means the being must learn by living. Everything else as D37 (same rooms, seeds, beats, being).
The real D37 question can now be asked: does a being that can feel the danger choose it anyway?
Bars (both layouts): drop deaths in the last 2,000 beats at most half of D37's (9101 <= 17, 9104 <= 35); need fills at least 80% of D37's.
Also read: D37c's parts at fatal steps (does any part now lean against the fatal gesture), who decided the fatal steps, and the value's vote
on beats where the ground cell is lit (the being feels the missing ground) vs not.

## D37f result: met in 9104, not in 9101 - the sense helps where it fires; it misses 39% of fatal steps
Drop deaths per 1,000 beats: 9101 197, 22, 10, 1, 1, 6, 16, 15 (last 2,000: 31; D37 35; bar <= 17 - not met); 9104 129, 41, 34, 29, 13, 27,
8, 16 (last 2,000: 24; D37 70; bar <= 35 - met). Need fills last 2,000: 9101 593 (D37 544), 9104 445 (D37 317) - both met (it lives better).
Beats with missing ground felt fell in 9101 from 794 to 0-30 per 1,000 (it stays away from the edge); in 9104 they stayed 83-181.
Check (9104, 2,000 beats): of 170 falls, the missing ground was felt on the beat of choosing in 103 (61%) - just ahead in 47 (28%);
39% of falls were not felt at all: the four ground rays point ahead / right / behind / left, and bodies fall in on the diagonal while walking
along the stem. At fatal steps the going lean (+2.3..+6.9) and the value (+1.1..+5.3) still favour the step.
- Reading: with the sense, the being lives better and in 9104 halves its drop deaths; the being did learn something from feeling the ground.
  Two gaps remain: (1) the sense still misses the diagonal (a coverage gap in my body design, same kind as before); (2) on felt falls the
  value has not learned the felt drop's meaning strongly enough to outweigh the lean.
- Next (registered below): close the coverage gap first (one change), then read the felt-but-stepped falls.

# D37g (registered before running): ground touch on six rays
Same as D37f, the ground rays at 0, +-45, +-90 and 180 degrees (skin cells 9-14) instead of four. Bars as D37f; plus falls not felt on the
beat of choosing below 15% of falls.

## D37g result: not met - six rays did not close the gap; the falls are a matter of fine footing
Drop deaths last 2,000 beats: 9101 58 (D37 35, D37f 31), 9104 26 (D37 70, D37f 24); need fills 502 (544) and 355 (317). Missing ground felt
(cell > .5) on the beat of choosing: 145 of 271 falls (54%) and 192 of 295 (65%); bar (unfelt < 15%) not met.
At fatal steps the going lean (+2.4..+6.3) and the value (+.8..+3.4) still favour the step.
- Reading of the whole D37 line now: once the drop can be felt, the being does live better (9104: 70 -> 24-26 deaths, more fills), and it
  learned to stay off the edge in 9101 for long stretches. What remains is not foresight and not authority: a body walking along an edge
  falls in by a fraction of a cell, the same gesture in the same felt situation is safe most of the time, so the value of "forward here" is
  rightly positive on average. The sense is graded (closer = stronger), but the step that kills differs from the safe one by less than what
  the being's choosing can resolve. Animals have fine footing for this (edge reflexes of feet and whiskers - innate or very early motor
  learning); the being has none. That is a motor-precision question, not the founder's question about fearing death.
- Decision: the D37 line pauses here (the review's priority is S3: telling worlds apart). Kept: ground touch (six rays) as a body sense.
