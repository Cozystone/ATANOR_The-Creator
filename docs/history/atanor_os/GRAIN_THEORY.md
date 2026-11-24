# The grain of choice: why a value that knows does not steer (theory + D25, registered 2026-11-24 before running)

Founder, 2026-11-24: stop the small local steps on V4; find the fundamental breakthrough, with prior research.

## What every result so far has in common
- The value reads the right side AT the junction (D22 greedy probe: enters the needed arm about .9) but is 'a signpost, not a route', and knows nothing from afar.
- Making the value better (SARSA, relief share, longer horizon, sleep replay) lifted the fork (V3 first .73, recovery .69) and need-directed fills x6.6, yet deaths
  moved only -12%. Giving the value more of the wheel (V4, track-record arbitration) made the life WORSE in all three layouts run so far.
- Deaths are two pictures of one thing (D24): staying where it is (the other spring, 62%) and circling where it starts (36%). 92% never reach the lacked store.

## Theory: the being chooses at the wrong grain
The mind picks one of ~90 motor combos about every 3.5 beats. A trip to the lacked spring takes 50+ beats directed and hundreds wandering.
1. The gap between combos vanishes. With a horizon of ~100 beats, what one combo changes is about 1/100 of the value; the value is ~.05, so the differences
   between combos at one beat are ~.0005 - below the approximation noise of any learned value. This is a known law, not a tuning problem: the action gap is
   O(step) (Baird 1994 'Advantage updating'; Tallec, Blier & Ollivier 2019 'Making deep Q-learning robust to time discretization': Q-learning does not exist in
   continuous time; Bellemare et al. 2016 'Increasing the action gap'). At the junction one choice commits to an arm, so there the gap is large: a signpost only.
2. For far goals the value's signal-to-noise per step is poor, but the same value compared ACROSS A SPAN of beats is clear: this is the stated ground of
   hierarchical goal reaching (Park et al. 2023 HIQL: pick a state k steps ahead by the value, reach it by a learned low level; Savinov et al. 2018
   semi-parametric topological memory; Eysenbach et al. 2019 search on the replay buffer).
3. Animals choose where to go, not which muscle: place and goal coding (Pfeiffer & Foster 2013: hippocampal sequences depict the path to a remembered goal before
   moving), and replay stitches paths never run in one piece (Gupta et al. 2010: A->B and A->C run, B->A->C replayed) - which answers the data problem
   (spring-to-spring trips are almost absent from the lived ring, D23). The predictive map (Stachenfeld, Botvinick & Gershman 2017) is the same knowledge
   as the relief futures, held over places. Hunger itself drives leaving and exploring (AgRP -> dopamine in the tail of striatum, 2025; AgRP activity
   falls on approach to food and rises on leaving).
So the suspected root is not the value's content alone, and not who decides (V4), but that a place-sized question is asked of a muscle-sized choice.
If right, the breakthrough is one organ change: the being's value chooses WHERE (a state of its own lived experience some tens of beats ahead, in its own
code - no symbols, no search), that choice persists as the intention until reached or given up (inner pressure, not a plan), and a reach-skill learned from
its own steps in hindsight (every lived stretch teaches how to get from its start to its end) moves the body. Wants still come only from the bedrock.

## D25: three cheap measurements that can break this theory (no new being; V3's saved lives, dev 9101-9103 not mirrored)
Material: V3 mind and its 300k ring per layout; two relief values trained offline on the same ring with different seeds (A, B: gamma .99, 10,000 updates,
sync 25), one on shuffled outcomes (control), and the life's own value. A fresh logged stretch: the V3 mind in 64 bodies, 3,000 beats, frozen (no learning),
judge-side position kept beside each sensed state. All values are read with the insides overwritten to thirsty (.30 / .95) and to hungry (.95 / .30), so only
what the value knows about PLACE is compared.
- (a) Reliability of the per-beat choice signal: per state, the correlation across the ~90 combos between A's and B's scores (and the gap best - median over
  the state value), by zone (stem, junction, arm of the needed spring, arm of the other spring).
- (b) Reliability across a span: correlation between A and B of V(x[t+k]) - V(x[t]) along each body's own lived steps, k = 1, 3, 10, 30, 100.
- (c) Does the value know the way: Spearman correlation between the state value and the true path distance to the needed spring (judge-side), overall and
  inside each zone; same for the shuffled control and the life's own value.
- (d) The pieces of the map in the lived stretch: per 1,000 body-beats, how often a body leaves an arm, and how often it goes spring to spring.
## Reading (fixed now)
- Grain confirmed: (a) per-beat reliability < .3 outside the junction, (b) span reliability rises to >= .6 by k = 30, and (c) Spearman >= .5 overall with
  >= .3 inside the other arm (control near 0). -> the knowledge is there at place grain; build place-level choice + hindsight reach (registered separately).
- Map missing: (c) < .3 inside the other arm or the stem even though (b) is reliable -> the value does not hold the way from afar; the place-level choice is
  still needed but must stand on a stitched map (temporal distance / successor over a place code, replay stitching), with (d) saying whether the pieces exist.
- Theory wrong: (a) per-beat reliability already >= .6 everywhere and (c) high -> the value is fine at muscle grain and the break is in the choosing
  machinery (pull, hold, try); then measured there, not here.

## D25 result (3 layouts x 2 needs): my grain prediction did not hold; the registered reading is 'map missing'
(a) per-beat agreement of two seeds across the combos: .17 - .54 overall (needed arm .37 - .78, elsewhere mostly .2 - .5): weak in places, not uniformly below .3.
(b) span agreement is already .70 - .90 at ONE beat and barely rises (.75 - .93 at 100): the prediction 'reliability appears only across a span' was wrong.
(c) the value does not know the way: Spearman with the true path distance to the needed spring .13 - .38 overall (shuffled control -.05 - -.48; the life's own
value .14 - .40), inside the arm of the other spring -.71 - +.08, inside the needed arm mostly negative. It knows the side near the junction and little else.
(d) pieces: spring-to-spring trips 0 in 192,000 body-beats per layout; leaving an arm .2 - 1.6 times per 1,000 body-beats.
So before who decides (V4) or at what grain, the value has no map to stand on, and the lived experience holds almost no trips to learn one from by TD.
V4 closed by its registered criteria (see V2_RELIEF_PROTOCOL.md).

# D26 (registered before running): does a stitched map made from the same lived steps know the way?
Prior work: the hippocampus as a predictive map (Stachenfeld et al. 2017); replay stitching paths never run in one piece (Gupta et al. 2010); topological
memory from own experience (Savinov et al. 2018); the default representation, a map not tied to the current habit (Piray & Daw 2021).
Material: the same frozen V3 stretches (64 bodies x 3,000 beats, dev 9101-9103 not mirrored), saved with the judge's position.
Being-side construction (no position used): the sensed state without the insides -> a place code learned by time contrast on its own steps (states a few
beats apart in one life are near; InfoNCE, small net) -> 64 place units (k-means; a fixed-size memory) -> unit-to-unit transitions counted from lived steps
-> per unit and channel, how often that store rose there (need-free: what is where) -> value over places V_c = (I - .99 T)^-1 r_c.
Variants: T as lived / symmetrised (reachability, not habit); place units from the learned code / from the raw sensed state (no learning) / from the judge's
true cells (ceiling, judge-side only); control = unit reliefs shuffled.
Measured as D25 (c): Spearman of V_c at each lived state with the true path distance to spring c, overall and inside the arm of the other spring.
## Reading (fixed now)
- Learned-code map >= .5 overall and >= .3 inside the other arm in at least 4 of 6 (layout x need), control near 0 -> the map organ is the missing piece:
  build it into the being (place units as bounded memory, value over places, replay), then the choosing.
- Ceiling (true cells) >= .8 but learned code below the bar -> the place code is the gap (reading place from the eye), worked on first.
- Ceiling itself below .5 -> the lived steps do not hold the pieces even for a perfect place sense: leaving / exploring comes first (what makes a body with
  a growing lack leave where it is).

## D26 result (3 layouts x 2 needs): the lived steps DO hold the map; the being's place sense is the gap
Ceiling (judge's true cells as place units, same lived steps, same counting): Spearman with the true path distance .81 - 1.00 overall (6 of 6), inside the arm of
the other spring .90 - .97 with symmetrised transitions (lived: .60 - .97) - although not one spring-to-spring trip was lived. Stitching works; the experience
suffices. Learned place code (time contrast on single sensed states, 64 units): .24 / -.07 / .20 / .59 / .33 / -.15 overall, other arm negative in all 6 - bar
not met (raw sensed state: .49 / .23 / .36 / .64 / .31 / -.14). Unit purity: true position spreads 1.05 - 1.45 cells inside a learned unit (true cells .41, raw 1.3 - 2.7).
Registered reading: the place code is the gap. Caveat (mine): with true cells the shuffled-relief control is also high inside one arm (.1 - .97) - along a
corridor any source gives a monotone gradient, so the 'other arm' number is weak evidence on its own; the overall number (shuffled .07 - .96, mixed) is
the one that separates, and only partly. The clean statement: with a perfect place sense the map reads the way; with the being's sense it does not.

# D27 (registered before running): which place sense makes the map work
Prior work: place needs more than the present view - path integration and sequence context resolve views that look alike (McNaughton et al. 2006;
clone-structured cognitive graphs, George et al. 2021: aliased observations split by their sequence context). Same stretches (re-run to keep each body's own
actions), same 64 units, same map, same measure as D26. Place codes compared, all by time contrast on the being's own steps:
- one sensed state (D26);
- a short past: the sensed states at t, t-4, t-8, t-12 together;
- a carried state: a small recurrent net over the last 16 sensed states and the body's own actions (can integrate its own movement).
Also reported: how often a unit mixes the two arms (judge-side), and the spread of true position inside a unit.
Reading (fixed now): a code reaching >= .5 overall and >= .3 inside the other arm in at least 4 of 6 -> that place sense + the map organ go into the being.
None -> report what is being confused (arms with each other, headings) and whether the room gives the eye enough to tell places apart at all.

## D27 result (3 layouts x 2 needs): none of the three place codes passes
Overall Spearman: one state -.17 - .65, short past -.11 - .63, carried state .02 - .54 (bar .5 in 4 of 6: 1 / 1 / 1 of 6); inside the other arm mostly negative.
Follow-up description (tools/d27_confuse.py) - what is NOT the cause:
- the eye does tell places apart: nearest-neighbour read-out of true position from the raw sensed state errs by a median .11 - .15 cells (learned code .23 - .30),
  more than 3 cells off in only 1 - 5% of states;
- coarse units are fine: true-position units as coarse as 6 per room (spread ~1 cell) still read the way at .86 - .94;
- my guess of a leaking relief record was wrong: 0 - 16% of a channel's record sits at the wrong spring, and the map's best unit is 1 - 2 cells from the needed spring.
What is left: 2 - 6% of a unit's states lie more than 3 path-cells from its commonest cell. Suspected: these rare far members join far places inside one unit
(a false shortcut); with a horizon of 100 beats a few % of false shortcuts outweigh a 20-cell corridor. Known in topological memory (Savinov et al. 2018
needed several consecutive frames to agree before linking, to avoid false shortcuts).

# D28 (registered before running): are false shortcuts what breaks the being's map, and does the body's own continuity remove them?
Same stretches, the 'one state' learned code, same map and measure. Variants:
- judge-purged (causal check, judge-side only): states more than 3 path-cells from their unit's commonest cell are taken out of the unit.
- continuity (being-side): a body cannot jump, so each state takes the commonest unit over its own +-4 beats (within one life).
- continuity + rare links dropped: unit-to-unit links seen in fewer than 1% of a unit's departures are not kept.
- each at 64 and 256 units.
Reading (fixed now): judge-purged >= .8 overall in 5 of 6 -> false shortcuts are the cause. A being-side variant >= .5 overall in at least 4 of 6 with the
shuffled control clearly lower -> that map goes into the being (next: build the organ and live with it). Being-side below the bar while judge-purged passes ->
the memory needs stricter linking (several consecutive beats must agree; node = a lived moment rather than a cluster), tested next. Judge-purged fails ->
false shortcuts are not it; look at the transition structure itself (headings).

## D28 result: false shortcuts are part of it, not the whole; continuity does not repair it
Judge-purged units: overall .42/.42, .33/.56, .79/.82 at 64 units (as is .19/-.04, .15/.58, .30/-.22); >= .8 in only 1 of 6 (256 units: 2 of 6): registered
reading 'false shortcuts are the cause' not met. Continuity (+ rare links dropped): .24/-.01, .23/.62, .37/-.10 - no repair.
Looking inside one failing map (description): the bodies of the frozen V3 mind spend 86 - 87% of their time in the stem in two layouts (31% in the third), and
about 50 of the 64 units sit in the stem; the arms and springs get a handful of large mixed units. My measure (every lived state counted once) therefore
asked mostly about the order of cells inside the stem. Re-measured with every true cell counted once: true cells .97 - 1.00; learned units .11 - .53
(256 units .20 - .48). So the map from clustered units fails at room scale too - the measure was lopsided, but the verdict stands.
Cause now suspected: clustering gives units to where time is spent, not to where places differ. A place seen rarely (the way out, the other spring) gets
no unit of its own. Brains do not lay down place cells in proportion to dwell time; novelty lays them down (a new place gets a new field at once).

# D29 (registered before running): landmarks laid down by novelty, not by dwell time
Same stretches. The being keeps a fixed number of landmarks (512, and 128): a lived moment becomes a landmark when it is unlike every landmark kept so far
(farthest-first over its own sensed states; no position used). Every lived state belongs to its nearest landmark; links between landmarks are counted from
lived successions (symmetrised); what rose where is kept per landmark; value over landmarks as before. Code: the raw sensed state and the learned
'one state' code. Control: reliefs shuffled over landmarks.
Measures: every true cell counted once (main, since D28 showed the dwell-time bias) and every lived state counted once (as registered in D26).
Reading (fixed now): per-cell Spearman >= .7 in at least 4 of 6 (layout x need) with the control clearly lower (<= .3 on average) -> the landmark map goes
into the being as its place memory (bounded, novelty-kept), and the next step is choosing by it. Below -> report the share of landmarks per zone and the
false links (judge-side), and go to moments linked only when several consecutive beats agree.

## D29 result: landmarks kept by novelty do not make the map work either
Per-cell Spearman .06 - .48 (bar .7 in 4 of 6: 0 of 6), although far members fall to 1 - 3% with the learned code at 512 landmarks. My dwell-time guess was wrong
as the main cause. Third wrong guess in a row (leak, shortcuts-only, dwell time): so the ceiling itself was broken on purpose to see what it is fragile to.
Sensitivity of the ceiling (description, judge-side): true cells, with a share of lived states given a random wrong cell -
  clean .97 - 1.00;  0.2% wrong .58 - .95;  1% wrong .49 - .74;  3% wrong .39 - .60 (a shorter horizon, gamma .9, does not help).
And how thin the lived map is: the least-crossed cell on the map is crossed 1 / 11 / 17 times in 192,000 body-beats (median link 110 - 458 crossings).
So the D26 statement 'the lived steps hold the map' is true only for a perfect place sense: the way between the springs hangs on a handful of crossings, and any
real place sense (the being's is wrong by more than 3 cells in 1 - 5% of states) lays hundreds of false links over it. The stitched map and the place sense are
not broken in themselves; the being has hardly travelled its room (86% of the time in the stem in two layouts, camping in the third), so there is nothing
robust to stitch. What makes a body go through its world before it needs to is the open question (latent learning needs the unrewarded exploring first:
Tolman & Honzik 1930; novelty-driven exploration; hunger itself raising exploration, AgRP -> tail of striatum 2025).

# D30 (registered before running): if the bodies had travelled, would the being's own place sense and map know the way?
Judge-driven traveller (a probe, not a being; no claim about the being's behaviour): 64 bodies in the same rooms walk toward cycling targets (water spring,
food spring, bottom of the stem, a random cell) by the judge's path, 20% of beats a random gesture, the hand tried at random; 3,000 beats; sensed through the V3
being's own organs. Then exactly the being-side map of D29 (learned 'one state' code and raw sense; 512 and 128 novelty landmarks) and the 64-cluster map of D26,
measured per true cell; plus the ceiling, its noise sensitivity and the least-crossed link.
Reading (fixed now): being-side map >= .7 per cell in at least 4 of 6 (control <= .3 on average) -> travel is the missing ingredient; the place sense and the
stitched map are good enough as they are, and the work moves to what makes the being travel (a drive from its own bedrock: progress of its own place memory),
then living with the map. Below the bar even with travel -> the place sense / linking must also improve (links only when several consecutive beats agree).

## D30 result (3 layouts x 2 needs): travel lifts the being's own map from .11 - .53 to .67 - .77; the registered bar is missed by a hair
Traveller stretches: both stores rise about equally (660 - 820 each), ceiling .98 - .99, and the ceiling is sturdier against wrong places (0.2% wrong .88 - .96,
1% wrong .68 - .79; was .58 - .95 and .49 - .74 on the being's own stretches).
Being-side map, per true cell: learned code, 64 clusters .77 / .67, .75 / .67, .73 / .67 (the same construction on the being's own stretches: .11 - .53);
learned code, novelty landmarks 512: .50 - .60, 128: .47 - .59; raw sensed state: -.19 - .26 (no map). Shuffled control for the 64 clusters:
-.12 / -.66, -.38 / .29, .69 / -.71 (mean -.15; one case as high as the map).
Registered bar (>= .7 in at least 4 of 6): 3 of 6, all six >= .67 -> not met as written; I do not round it up. What is established: with the same place sense
and the same stitched map, travelling the room is what makes the difference between no map and most of a map; the learned time-contrast code is needed
(raw sense gives nothing); novelty landmarks were worse than plain clusters once the travel is even.
Still open: the control is noisy with so few cells (one shuffled case .69), so per-cell rank correlation over ~45 cells is a blunt yardstick; the last
stretch to a reliable map (.67 -> clearly above .7) may need the stricter linking named in the reading.
Next (to be registered before running): the being itself must do the travelling. Candidate, from its existing second bedrock (drawn to where its own
prediction fails - until now only about its insides): the same pull applied to its place code, i.e. drawn to where its own place memory is still thin.
Novelty is local (the unfamiliar is always next to the familiar), so unlike the far spring it can steer a muscle-sized choice.
