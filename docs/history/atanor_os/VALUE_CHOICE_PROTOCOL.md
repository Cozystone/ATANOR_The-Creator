# Value -> choice: why what it values does not become what it does

Founder 2026-11-26 (after a review of the logs): the same shape came back in four places - D37 (ground missing is felt, steps change little),
S4k (beside the guardian is valued 2-5x, no approach; choice followed value on 11% of beats), MineLand life 25 (cakes eaten once, value decided
0-0.6% of steps), D18 (value alone decided 0% of beats). "It is not that it does not understand the world; what it understands is not
translated into action." Order: no new part, diagnosis only; S3, D37, S4, D38 and MineLand rest until this is read. "확실히 뚫어봐."

## How a beat is decided now (read from atanor_os/emerge/mind.py Mind.choose, nothing changed)
1. Gate. A new gesture is picked only when the gate opens: the last beat was not foreseen (novel), or a strong pull's best score beats the
   held gesture by more than 2 standard deviations (pre), or the body was cut (rebirth). Otherwise the held gesture goes on ("hold"); the
   score is computed and not used.
2. Gate open, strong pull: one draw from softmax(score / .05), score = pull + harm + futility + value lean + going lean (each a vector over
   gestures). Gate open, no strong pull: with probability "earned trust" the value alone is drawn from (softmax(value lean / .05)), else a
   uniform random gesture.
3. Try: on 10% of beats a uniform random gesture is done whatever was picked.
So value can reach the act through two doors only - as one term among five on strong beats, and by its earned share on weak beats - and
through neither while the gate is shut.

# D39 (registered before running; a readout only - the being lives and learns as always, nothing in choice is touched)
The founder's question: on what share of beats is the gesture done the one its value rates best? And where is it lost?
Run: tools/d39_choice.py, the S3-end beings (s3_a in 9101_0, s3_b in 9104_0), one mind with 64 bodies, 10,000 beats, V2 world as in S3/S4.
Measured every body-beat (value's best = argmax of the value lean; chance = 1 / number of gestures):
- M1 (the number asked for): share of beats the gesture done is the value's best; also value's top 3; by thirds of lack; by who decided.
- M2 gate: share of beats the gate is open; why (pre / cut / novel); mean length of a hold; on hold beats, is the held gesture the value's
  best, and where it ranks in the value (0 = best).
- M3 strong beats: softmax mass the score gives the value's best; score's best = value's best; share where taking the value out changes
  the score's best (value decisive); spread over gestures of each term (pull, harm, futility, value, going); the score's top margin in
  units of the temperature (how nearly the draw is already decided).
- M4 weak beats: earned trust; share the value was drawn from; mass the value's own softmax puts on its best (does it single one out).
- M5 value ready at all (share of beats it gives a lean).
Readings, stated before the run (they can hold together; the arithmetic below says which is largest):
 (a) gate: hold > 70% of beats and the held gesture is the value's best no more often than chance -> the gesture is decided before the
     value is asked (the founder's third suspicion: "hold decides before choice").
 (b) outvoted: on strong beats the value's spread is under 20% of the largest term's and value is decisive on under 5% -> it loses on size
     every time (first suspicion); if the score's top margin is over 3 temperatures on most strong beats the draw is near-greedy, so a
     small term cannot matter at all (second suspicion: temperature).
 (c) seldom asked: on weak beats the value is drawn from on under 30% -> its earned share shuts its own door.
 (d) flat: the value's own softmax puts under 2x chance on its best gesture, or its best is done no more than chance even when it alone
     decides -> the value rates states, not gestures (evaluation without direction, as S4k); then the wall is in the value, not in choice.
Arithmetic reported with the result: M1 = P(hold) x P(held = best | hold) + P(strong) x P(pick = best | strong) + P(weak) x P(pick = best |
weak), each factor measured, so the largest loss is named by numbers, and what M1 would be if one factor alone were perfect.
No bar: this decides what the next registered step is. Nothing is fixed in this run.

## D39 result (two rooms, 10,000 beats x 64 bodies each; deaths 0 and 1 - these beings live well at home)
| | 9101 (s3_a) | 9104 (s3_b) |
|---|---|---|
| M1: gesture done = value's best (chance 1.1%, 90 gestures) | 13.5% | 12.3% |
| done in value's top 3 | 25.6% | 24.1% |
| beats held (gate shut), of pick-level beats | 87.7% | 88.3% |
| mean hold | 8.1 beats | 8.5 beats |
| held gesture = value's best / its mean rank in the value | 12.6% / 18 of 90 | 11.9% / 19 of 90 |
| gate open, strong: pick = value's best | 31.2% | 26.5% |
| ... value decisive (taking it out changes the score's best) | 66% | 61% |
| ... spread of terms: value / going / pull / harm / futility | 1.00 / .96 / .11 / .02 / .02 | 1.10 / 1.22 / .17 / .03 / .03 |
| ... score's top margin, in temperatures (share over 3) | 12.7 (80%) | 15.4 (83%) |
| weak beats (value's own door) | 0% of beats | 0% of beats |
| value's own softmax mass on its best / earned trust | .88 / .59 | .89 / .62 |
| M1 by lack: under .3 / .3-.6 / over .6 | 14.0 / 8.5 / 6.2% | 13.1 / 10.1 / 10.0% |
| gate open by lack | 12.3 / 12.5 / 13.3% | 11.7 / 11.8 / 12.5% |
| value's spread by lack | .93 / 1.54 / 6.82 | .96 / 1.48 / 4.27 |
Arithmetic (pick level): 9101 .877 x .126 + .123 x .312 = .149; if holds were perfect .916, if strong picks were perfect .233.
9104 .883 x .119 + .117 x .265 = .136; holds perfect .914, strong picks perfect .222.

Readings against what was registered:
- The founder's number: about 13% (bin 10-20%), 12x chance. Value is not shut out of choice.
- (a) gate - the largest loss by far: 88% of beats are holds, and on them the held gesture is the value's best only 12% of the time (above
  chance, so not "decided with no regard to value" - the pick it came from had value in it - but the value's best moves on while the
  gesture stays). A perfect hold would give .92; a perfect pick only .23.
- (b) outvoted - no: the value's spread equals the going's and is 6-9x the pull's; it is decisive on 61-66% of picks. Temperature: the draw
  is near-greedy (margin 13-15 temperatures), but the value is large enough to matter in it.
- (c) seldom asked - in an extreme form: the "strong pull" flag is on at every beat, so the value's own door (weak beats) never opens.
- (d) flat - no: the value singles out one gesture (.88 of its own softmax) and its earned trust is .6.
- Not registered, found: urgency does not reach the gate. When the lack is high the value's spread grows 4-7x, yet the gate opens no more
  often (12-13% at every lack) and the value's best is done LESS (6-10%). The gate's yardstick is this beat's own score spread (best beats
  held by 2 standard deviations of the score), so it is scale-free: however loud the value gets, the bar rises with it.
Correction to earlier reports (mine): "value decided 0%" (D18, MineLand lives, life 25's 0-0.6%) read the decided-by label, which counts
only the value's own door on weak beats. That door is shut whenever the strong flag is on (here: always), while the value still weighs in
the strong score. So those numbers did not show that value has no say; MineLand needs this same readout before that is said there.
What D39 says in one line: the value speaks at picks and is heard there; picks are rare (every ~8 beats), and when they happen is decided
by a rule that cannot hear how urgent the value is.

# D40 (registered before running): does the gate hear urgency if its bar stops rising with the score (a probe, off by default)
Theory: in action selection an ongoing act is kept until a competitor's salience beats it by a margin (hysteresis: Redgrave, Prescott &
Gurney 1999), and the margin that must be beaten falls as urgency grows (urgency gating: Cisek, Puskas & El-Murr 2009; Thura & Cisek 2014).
The being's gate has the hysteresis but not the urgency: its bar is 2 standard deviations of THIS beat's score, so a score made louder by
a high lack raises its own bar (D39: value spread 4-7x at high lack, gate open 12-13% at every lack, value's best done less).
One change (Mind.gate_usual, default off; the main line is untouched): the bar is 2x the being's own USUAL score spread - a slow mean of the
spread over its lived beats (fading .999) - instead of this beat's. No new constant, no new term in the score, no rule about what to do:
only the yardstick of an existing comparison becomes the being's own baseline. At its usual state nothing changes; when the score is louder
than usual (it lacks more), the same gap opens the gate.
Runs (tools/d39_choice.py readout, 3,000 beats, 64 bodies, same seeds in both arms): each S3-end being at home (s3_a 9101_0, s3_b 9104_0)
and in a room it has never lived in (s3_a 9105_0, s3_b 9102_0); arms: gate as now / gate_usual. 8 runs, two at a time.
Bars: (1) mechanism: with gate_usual the gate opens more at lack >= .3 than under .3 (now: equal), and M1 at lack >= .3 is at least 1.5x
the baseline arm's; (2) life where learning matters: deaths per 1,000 body-beats in the first 1,500 beats of the never-seen room at least
25% below the baseline arm, in both rooms (a room whose baseline has under .1 cannot tell and is reported as such); (3) no harm at home:
deaths not above baseline + .05, need fills at least 90% of baseline.
Readings: all three -> the gate's deafness to urgency is a real part of the wall; adopting it in the main line is put to the founder.
(1) only -> the gate now hears urgency but lives do not change: holds were not what cost lives; next is what the picks themselves choose
(value against going). (1) fails -> the mechanism is not what I think; back to the readout.

## D40 result: not met as registered - the gate now hears urgency, the value's best is not done more, new-room lives get better
3,000 beats, 64 bodies, one start, same seeds per pair (baseline / gate_usual):
| room | gate open at lack <.3 / .3-.6 / >.6 | value's best done at lack >= .3 | deaths per 1,000, first 1,500 | need fills first 1,500 | M1 all |
|---|---|---|---|---|---|
| 9105 new (s3_a) | .12/.14/.17 -> .09/.18/.35 | .038 -> .051 (1.35x) | .156 -> .042 (-73%) | 2.54 -> 3.19 | .054 -> .056 |
| 9102 new (s3_b) | .12/.13/.17 -> .10/.20/.46 | .046 -> .048 (1.03x) | .073 -> .010 (baseline < .1) | 3.39 -> 4.11 | .063 -> .059 |
| 9101 home (s3_a) | .13/.14/.18 -> .11/.18/.21 | .044 -> .069 (1.57x) | 0 -> 0 | 5.48 -> 4.91 (all 6.54 -> 5.62, 86%) | .086 -> .086 |
| 9104 home (s3_b) | .12/.13/.14 -> .10/.17/.36 | .059 -> .055 (.92x) | 0 -> 0 | 4.24 -> 4.41 (all 5.03 -> 4.82, 96%) | .076 -> .067 |
- Bar (1) mechanism: half. The gate opens with urgency in all four (at high lack 2-3x as often, at low lack less); but the value's best is
  done 1.5x more at lack >= .3 in 1 of 4 (1.03-1.57x).
- Bar (2) new rooms: 9105 met (-73%); 9102 cannot tell (baseline .073 < .1), same direction (.073 -> .010). Counts are small (about 15 vs 4
  deaths in 9105) and one start.
- Bar (3) home: deaths 0 in both arms; need fills 86% in 9101 (bar 90%) - not met there; 96% in 9104.
Reading (not one of the registered three, reported as measured): opening the gate when lacking more helps lives in rooms the being does not
know - more fills, fewer deaths - but not by doing what the value rates best: the extra picks go mostly where the score as a whole points,
and there the going lean (its place memory toward the lacking inside) is as large as the value. At home, extra re-picks cost a little.
So the gate is part of the wall (urgency now reaches it) but not the part that turns value into acts. That part is inside the pick: the
value and the going speak at equal size and often point to different gestures (value decisive 61-66% of picks, value's best picked 27-31%).
Not adopted; the probe stays off. Next (theory first): when value and going disagree at a pick, which of them was right - read from what
followed (wholeness change over the next beats), not from the value's own rating.
Also corrected here: D39 counted M1 over 10,000 beats of the S3-end beings at home (13%); over 3,000 beats from a fresh start it is 5-9% -
the share grows with time lived (D39 halves 9.8-11.9% -> 14.9-15.1%).

# D41 (registered before running; readout only, nothing in choice touched): when value and going disagree at a pick, which was right
At every pick (gate open, strong) where the value's best gesture differs from the going's best, record what was picked: the value's best,
the going's best, or another; then read what followed for that body: wholeness change over the next 20 beats (Worth.whole, the bedrock),
and whether it died in them. Same for picks where the two agree (reference). Lack at the pick is kept, so the comparison is made within
lack thirds (high lack is where the outcome tells most).
Runs: tools/d39_choice.py --d41, baseline gate, 5,000 beats, 64 bodies: s3_a in 9101_0 (home) and 9105_0 (new), s3_b in 9104_0 (home) and
9102_0 (new); four runs, two at a time.
Readings: within the same lack third, picks of the value's best are followed by a larger wholeness rise than picks of the going's best, in
3 of 4 rooms -> the value knows better and the going covers it: the wall is the arbitration between them (next: weigh each by its own record,
as the going already is by Earned); going's picks better in 3 of 4 -> the value is not yet right enough to lead; the wall is the value's
learning, not choice; mixed -> each is right in different states; read where.

## D41 result: mixed - the value's picks fare better at middle lack (4 of 4 rooms), the going's at high lack at home
Picks 36,000-40,000 per room; value and going disagree on 97.8-98.0% of them (agree on 2%). Wholeness change over the next 20 beats after a
pick (deaths counted apart), value's best picked vs going's best picked [count]:
| room | lack < .3 | lack .3-.6 | lack > .6 |
|---|---|---|---|
| 9105 new | -.0015 [4,747] vs -.0020 [9,128] | +.0097 [763] vs +.0002 [6,397] | +.023 [148] vs +.020 [1,891]; died 3.4% vs 1.4% |
| 9102 new | -.0017 vs -.0016 | +.0104 [734] vs +.0032 [4,465] | +.075 [134] vs +.027 [1,054]; died 1.5% vs .7% |
| 9101 home | -.0008 vs -.0004 | +.0142 [490] vs +.0086 [2,603] | +.014 [7] vs +.045 [141] |
| 9104 home | -.0021 vs -.0026 | +.0088 [1,536] vs +.0061 [5,199] | +.019 [56] vs +.054 [200] |
- Middle lack: value's picks followed by more wholeness in 4 of 4 rooms (1.4x to 40x); "other" gestures also beat the going's there in 4 of 4.
- High lack: value better in the two new rooms (but more deaths after it), going better at home (few value picks: 7 and 56).
- Low lack: no difference (both small and negative - nothing to gain).
- Choice follows the going: going's best picked 5-9x as often as value's at middle and high lack.
Reading as registered: mixed - each is right in different states: when the lack is middling, the value knows better and the going covers it;
when the lack is high at home, the going's place memory takes the body to its springs and the value's rare picks do worse.
Caution (mine): this is not causal. The value's best is picked when the value is loudest in the score - perhaps where it is most sure - so its
better outcomes may be selection. D37e already showed the naive fix fails: weighing the going by its own record took it to its floor (.1)
and in 9104 the bodies could no longer reach their springs. So nothing is changed on D41 alone.

# D42 (registered before running): the same question, randomised
At a pick where value and going disagree, with probability .2 the judge replaces the gesture by the value's best or the going's best (a coin,
50 / 50); the being holds it as its own (the hold carries on as usual). Everything else as D41 (baseline gate, 5,000 beats, the same four
rooms, two runs at a time). Read only the replaced picks: wholeness change over the next 20 beats and deaths in them, by lack third.
Readings: value's arm better than going's at middle lack in 3 of 4 rooms (and not worse in deaths) -> D41 holds causally: at middle lack
the going should yield to the value; going's arm better or equal -> D41 was selection: the going is not covering a better value; then the
value's knowing is the wall, not the arbitration.

## D42 result: D41 was mostly selection - randomised, the value's best and the going's best fare about the same
Disagreeing picks replaced by a coin (value's best / going's best), 6,700-7,200 per room; wholeness change over the next 20 beats
[count], deaths in them:
| room | lack < .3 | lack .3-.6 (the registered test) | lack > .6 |
|---|---|---|---|
| 9105 new | -.0020 vs -.0023 | +.0036 [932] vs +.0035 [979] | +.010 vs +.013; died 3.6% vs 1.5% |
| 9102 new | -.0014 vs -.0012 | +.0086 [772] vs +.0064 [743] | +.038 vs +.037; died 0 vs .6% |
| 9101 home | -.0010 vs -.0011 | +.0086 [545] vs +.0087 [548] | +.060 [15] vs +.053 [19] |
| 9104 home | -.0025 vs -.0025 | +.0067 [1,166] vs +.0070 [1,139] | +.038 vs +.041; died 2.7% vs 2.0% |
- Middle lack: value's arm better in 1 of 4 rooms (9102, +33%); the other three equal within 5%. Bar (3 of 4) not met.
- D41's 1.4-40x advantage at middle lack shrinks to ~0 when the pick is assigned at random: it came from WHEN the value's best was picked
  (moments the value was loud and sure), not from the value's best being the better gesture.
- High lack in the new room 9105: more deaths after the value's best (3.6% vs 1.5%, both studies) - the value is worse where it matters most.
Reading as registered: going's arm equal -> D41 was selection; the going is not covering a better value. The wall is not the arbitration
between value and going: at a single pick, which of the two gestures is done changes the next 20 beats hardly at all. What the value knows
about gestures is no better than what the going knows.
Where this leaves the line (mine, for the founder): D39-D42 together say the choice machinery is not hiding a better value. The value's best
is done 13% of beats; making it done more would not help, because its best gesture is not better. So "value -> choice" as a wiring problem
is closed for now; the open problem is the knowing itself - what the being knows about what its gestures will bring (in new rooms, at high
lack, out of view, toward moving things).

# D43 (registered before running; readout over the D42 randomisation, nothing in choice touched): does each part know when it knows
D42 on average: the value's best and the going's best fare the same. An average can hide a useful signal: perhaps each is right when it is
sure and wrong when it is not. If so, the being already holds the cue the founder's arbitration needs ("when to trust whom"), in its own
margins, with no new part. Theory: reliability-based arbitration (Daw, Niv & Dayan 2005; Lee, Shimojo & O'Doherty 2014) needs a per-state
signal of each controller's reliability; a controller's own margin is the cheapest such signal, if it is calibrated.
Same randomised swap as D42 (p .2, coin 50/50), 5,000 beats, the same four rooms. Each swap event keeps: arm (value / going), the value's
margin (its rating of its best minus its rating of the going's best), the going's margin (the same from the going's side), lack, and the
next 20 beats (wholeness change, death).
Measure: split events by the value's margin into thirds (cut points over all four rooms' events), and in each third the value arm minus
the going arm (wholeness change; deaths); the same by the going's margin.
Readings: in the top third of the value's margin the value arm beats the going arm in 3 of 4 rooms and in the bottom third it does not ->
the value knows when it knows; a confidence-weighted arbitration is the next registered step. Flat across thirds -> the margins carry no
reliability; arbitration has no cue to use, and the knowing itself (what gestures bring) is the next line. The same reading for the going.

## D43 result: flat - neither part knows when it knows
About 3,300-3,700 randomised swaps per room. Value arm minus going arm, wholeness change over 20 beats (standard error), by thirds of the
value's own margin (cuts 1.10, 2.18 over all rooms):
| room | value's margin low | middle | high |
|---|---|---|---|
| 9105 | -.0021 (.0009) | +.0019 (.0009) | +.0001 (.0015); died .7% vs .2% |
| 9102 | -.0000 (.0007) | +.0001 (.0008) | -.0003 (.0014) |
| 9101 | +.0007 (.0007) | -.0005 (.0006) | -.0002 (.0006) |
| 9104 | +.0000 (.0008) | +.0012 (.0008) | -.0015 (.0009) |
By the going's own margin (cuts 1.67, 3.33), high third, going arm better in 3 of 4 (-.0007, -.0011, -.0003, +.0006) - all within about
one standard error. Middle lack only: no trend over the value's margin either.
Reading as registered: flat across thirds -> the margins carry no reliability. When the value is most sure its best gesture is no better
(0 of 4 rooms); when the going is most sure, not clearly better either. An arbitration has no cue in them to use. The knowing itself -
what the being's gestures will bring - is the next line.
D39-D43 in one paragraph: the value's best is done on 13% of beats (D39); the gate is deaf to urgency (D39), and hearing it helps new rooms a
little without doing the value's best more (D40); value and going disagree on 98% of picks, and what looked like the value being right was
selection (D41 -> D42); neither knows when it is right (D43). Choice is not hiding a better knower. Nothing in the main line was changed.
