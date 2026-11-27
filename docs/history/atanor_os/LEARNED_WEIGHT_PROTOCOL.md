# Learned weights over organ content (the core step after gate 1; founder 2026-11-26, plan A only)
This experiment does not build a new ability. It tests whether the being can learn by itself how much to trust a feature already shown to
be useful (homing; later unfamiliarity and the elder's direction). Principle: organs give content, the being learns the weight
(atanor_os/WORLD_A_PROTOCOL.md D46b ruling: content that decides the gesture directly is a hidden objective function).

## Why this can work now when Earned (D37e, D46's earned arm) did not
Earned weighed a part by the correlation of its push with the wholeness change over the next 60 beats, read as a slow running moment; it
sat at its floor (.1) in D46 and B1. Here the credit is per choice: each beat, the content of the gesture the body actually chose, minus
what the choice expected on average (its own softmax), goes into an eligibility trace (lam .98, ~50 beats); each beat's own wholeness
change (a death = losing all the wholeness it had), minus its running mean, moves the weight along that trace. That is a linear policy
gradient with the bedrock as the only reward - the form allowed (founder 2026-11-22: value learning only with its own wholeness change).
Both signals are standardised by their running spread, so the step (lr 1e-3 a beat) has no scale to tune.

## LW1 (registered before running): the homing content at a learned weight, starting from zero trust
Code: atanor_os/emerge/mind.py LearnedWeight; Mind.learned_home multiplies the homing content (Going.lean_to with the homing vector: per
gesture, progress + alignment to its own goal, scaled by the lack) by w instead of the fixed full weight. Everything else is the main line
as it stands after the gate-2 run-1 fix (goals only at units with a homing position). tools/a2_moving_food.py --lw W0 [--lw-lr].
Arms, A2 9104 moving, A2-1's seeds, 6,000 beats, 64 bodies:
- L0: --lw 0 --lw-lr 0 (homing content present, weight held at 0: the being without trust in it)
- L1: the main line as is (weight fixed 1) - the fix-2 run already gives it: deaths 19, wait 212
- LW: --lw 0 (weight learned from 0)
Bars: (1) LW's weight at the end >= .5 and rising over the last third; (2) LW's deaths and food wait each at least halfway from L0 to L1
(or better than L1). Readings: both met -> the being learns to trust a useful feature from its own life; then 9101, then a second feature
(unfamiliarity, gate 2) in the same frame. (1) met, (2) not -> it learns the trust too slowly for this life length; one variant: a longer
life (12,000 beats). (1) not met -> the credit does not reach the feature (trace too short for the beats between choosing and eating, or
the bedrock change too noisy): measure the lag from goal-set to eating before any variant.
Smoke (600 beats, LW): runs; w 0 -> .25.

## LW1 result: bar (2) met, bar (1) not - the being learns from zero to trust the homing content, but its weight settles near .2
A2 9104 moving, 6,000 beats (all arms with fix 2's goal):
| arm | weight | deaths | food wait median | docking beats median |
|---|---|---|---|---|
| L0 weight held at 0 | 0 | 131 | 302 | - |
| LW weight learned from 0 | 0 -> .19 (beat 500) -> .26 (1,000) -> .20-.24 to the end | 34 | 198.5 | 8 |
| L1 weight fixed 1 (main line, fix-2 run) | 1 | 19 | 212 | 10 |
Bar (1) (end >= .5 and rising over the last third): not met - .24 at the end, flat or slightly falling after beat 1,000.
Bar (2) (at least halfway from L0 to L1: deaths <= 75, wait <= 257): met - deaths 34, wait 198.5 (better than full weight).
Reading. Not one of the registered branches: the credit does reach the feature (the weight rises from 0 within ~500 beats, from the bedrock
alone), and with its own learned trust the being gets most of what full trust gives (deaths 131 -> 34 vs 19; its wait is the best of the
three). Where it stops (~.2) the content already decides most choices (the score's softmax is sharp, temperature .05), so more weight
buys little more wholeness; my bar (1) assumed "trust" means a weight near 1, which this scale does not need. Bar (1) stays not met as
written. What this shows: for the first time a weight on an organ's content, learned only from the being's own wholeness change, moves its
choice to a large real effect - the value -> choice wall cracked for one feature. Not shown: replication (one start, one room).
# LW1b (registered before running): a second start (rep 1, seeds +1000) in 9104 and the first in 9101, arms L0 and LW (L1 from the same
seeds run alongside), same bars as LW1 unchanged. Read over all three: bar (2) in all -> the frame is the core of plan A from here (next:
a second feature, unfamiliarity, in the same frame, gate 2); bar (2) in 9104 only -> room-dependent, look at 9101's lag before any variant.

# LWD (registered before running; diagnostic, explains LW1/LW1b - founder: "배우는 무게 확실히 뚫어보자")
LW1 rose to +.26; LW1b 9104 rep 1 fell to -.13 first and crept back to ~0. Copilot's reading: the learner learns something, not the
right thing - credit assignment. Recorded (judge side): the weight's step split by what each body's beat was (death / a fill / else), and
the lag from a goal being set to food (per body). Smoke (300 beats, rep 0): goal -> food median 177 beats (p90 223) against a trace of
~50 beats; the step came almost all from ordinary beats (+.105) and not from fills (-.007), no deaths yet.
Runs: --lw 0, 1,500 beats, 9104 rep 0 and rep 1 (the two starts that went opposite ways). Readings: ordinary beats carry most of the
step in both, with opposite signs -> the learner follows the drain's noise (the wholeness falls faster when lacking, whatever the choice),
not the outcome of following the content: the frame needs credit per goal (from a goal set to its end), not per beat. Deaths carry the
sign in rep 1 -> rare large losses decide it: the same remedy plus a critic. Fills carry it -> the beat-level frame is right and only the
trace is short (lam to cover the measured lag).
# LW2 candidate (to register after LWD): credit per goal episode, given back when it ends (reverse replay, Foster & Wilson 2006; options,
Sutton 1999). A goal episode runs from a goal set to food, death, or the goal dropped; at its end its outcome (the wholeness change over
the episode, a death = all lost) minus the outcome expected for such an episode (running mean of episodes by lack at the start) moves the
weight by how much the choices in it followed the content (mean of f[a] - E_pi f over the episode). The bedrock stays the only reward.

## LW1b result: not met - learning the weight is not reliable (bar 2 met in one of three starts)
| start | L0 (w 0) deaths / wait | LW deaths / wait, w at end | L1 (w 1) deaths / wait | bar 2 |
|---|---|---|---|---|
| 9104 rep 0 (LW1) | 131 / 302 | 34 / 198.5, .20 | 19 / 212 | met |
| 9104 rep 1 | 142 / 337 | 96 / 277, -.01 (fell to -.13 first) | 7 / 157.5 | not met (<= 74.5, <= 247) |
| 9101 rep 0 | 14 / 120.5 | 5 / 122, .05 | 2 / 83 | not met on wait (<= 101.75) |
Reading: not the registered "9104 only" branch - it fails in 9104's second start too. The weight goes where its first few hundred beats
push it; the rule as built does not learn the trust reliably. Meanwhile full weight (L1) is good in every start (7 / 157.5 and 2 / 83 on
new seeds). Next as registered: LWD (where the step comes from), then LW2 (credit per goal episode).

# LW2 (registered before its runs; founder 2026-11-27 "직접 더 찾아서 연구해봐" - from a literature search, replacing the earlier candidate)
What the search gave (sources in the session report):
- Jeong et al. 2022 (Science), ANCCR: animals learn causes RETROSPECTIVELY - learning is triggered by a meaningful event and looks back at
  what preceded it more than usual (predecessor contingency = trace at the event minus the trace's base rate); the trace's time constant
  scales with the interval between rewards. Burke et al. (2023 preprint; Nat. Neurosci.): the time between rewards, not the number of
  trials, sets the learning rate - a reward-triggered retrospective rule explains few-shot learning.
- Foster & Wilson 2006; Ambrose et al. 2016; Whelan et al. 2023: reverse replay at reward gives credit along the path just taken; with
  replay, long per-beat traces are not needed and do not reinforce unrelated earlier behaviour.
- Lee, Shimojo & O'Doherty 2014: control is divided by each system's reliability (its own prediction error) - kept as the next candidate
  (weight = how often the organ's promise of relief was kept on arrival) if LW2 fails.
- RUDDER (Arjona-Medina 2019), Hindsight Credit Assignment (Harutyunyan 2019): redistribute a delayed return to its causes - same aim,
  heavier machinery; not used. IDBD / TIDBD (Sutton 1992; Kearney 2018): per-feature step sizes show which features matter - for later,
  when there are several features.
Rule (mind.py LearnedWeight mode="retro"): the per-beat step of LW1 is gone. An event is a beat whose wholeness change is far outside the
being's own usual (|change| > running mean + 3 sd: fills and deaths; nothing is named). Each body keeps a trace of how much its choices
followed the content (f[a] - E_pi f, standardised), decaying over 2 x its own mean interval between events. At an event the weight moves by
lr x (the event's wholeness change, standardised over events) x (the body's trace minus the trace's running mean, standardised). Ordinary
beats move nothing. Reward: the bedrock only. One fixed number (the trace reaches back two event intervals), lr .05.
Smoke (600 beats, 9104 rep 0): 158 events, trace time 208 beats (goal -> food median 174), the whole step from fills (+.109), none from
ordinary beats; w 0 -> .11.
Runs: the three starts of LW1/LW1b (9104 rep 0, 9104 rep 1, 9101 rep 0), --lw 0 --lw-mode retro --lw-lr 0.05, same goal code as their
L0 / L1 controls (already run). Bars: (1') the weight ends above +.1 in all three and is never below -.05 after beat 500 (reliable sign);
(2) as LW1: deaths and food wait each at least halfway from L0 to L1 in all three. Readings: both in all three -> the being learns to
trust a useful content from its own life, reliably: the frame is plan A's core; next a content that is wrong (A3's old map) must be
UNlearned by the same rule, then a second content (unfamiliarity). (1') met, (2) not -> right sign, too slow or too weak: lr x2 once.
(1') not met -> the retrospective rule does not fix the sign either: next is the reliability rule (Lee 2014), registered then.
LWD still runs first (it explains LW1; its result does not change LW2's runs).

## LWD result: the ordinary beats decide the sign (the drain's noise), deaths push down in both, fills are weak
1,500 beats, --lw 0 (LW1's rule): weight step split by the body's beat.
| start | death | fill | ordinary | end | goal -> food median (p90) |
|---|---|---|---|---|---|
| 9104 rep 0 | -.059 | +.065 | +.059 | +.065 | 129 (480) |
| 9104 rep 1 | -.070 | -.008 | -.162 | -.240 | 96.5 (465) |
Reading as registered: the ordinary beats carry the difference between the starts (+.06 vs -.16, opposite signs) - the learner follows
the noise of how fast wholeness falls, not the outcome of following the content; deaths pull down in both (rare large losses on bodies
whose recent choices happened to follow it); fills, the outcome that matters, move it little. This is what LW2 changes (only events move
it; the trace reaches back over the being's own event interval). LW2's runs follow as registered.

## LW2 result: not met - only events move it now (as designed), but the sign is still set by the first few hundred beats
| start | w path (every 500 beats) | split (death / fill / ordinary) | deaths / wait (L0; L1) | bar 2 |
|---|---|---|---|---|
| 9104 rep 0 | 0, -.13, -.13 ... -.12 | -.008 / -.120 / 0 | 92 / 277 (131/302; 19/212) | not met (<= 75, <= 257) |
| 9104 rep 1 | 0, .07, .08 ... .12 | +.017 / +.125 / 0 | 30 / 236 (142/337; 7/157.5) | met |
| 9101 rep 0 | 0, .12, .12 ... .11 (end .096) | +.003 / +.093 / 0 | 10 / 128.5 (14/120.5; 2/83) | not met |
Events 1,107 / 1,228 / 1,478; trace time 307 / 280 / 170 beats. Bar (1'): not met (-.127, .142, .096).
Reading: the noise of ordinary beats is gone (they move nothing); now the FILLS decide, and in 9104 rep 0 they decide negative: bodies that
ate had followed the homing content less than usual. Then the weight stops moving in all three after ~500 beats (contingency settles once
the content's use is set). Likely why rep 0 goes negative (not tested): early in a life the remembered goal can be a place food has left
(it lingers at the old place - anticipation check), so following the content then does lead away from food; the rule learns that
correctly, and never revisits it. So the content is good only when its goal is right - what has to be learned is how far to trust
THIS organ's promise, which is the reliability rule registered as the next step.
# LW3 (registered before running; Lee, Shimojo & O'Doherty 2014 - control by each system's reliability)
The weight on the homing content = how often its promise was kept: per goal, when a body arrives at its goal unit, the promise is kept if
its wholeness rises by more than .1 within 30 beats of arriving (relief came there), broken if not; kept and broken counts fade (x .995 per
counted arrival, so the recent ~200 arrivals matter); weight = kept / (kept + broken), with one kept and one broken at birth (w starts .5).
The scale (w = 1 when every promise is kept) is the organ's own unit, the full weight gate 1 used - said plainly, a fixed choice.
Nothing about which way to go is learned here; only how far this organ's promise is believed, from the bedrock (relief) on arrival.
Runs: the same three starts, --lw-mode reli. Bars (1'') the weight ends >= .4 in all three, and in A3 (s3_a, 3,000 beats; old-map goals)
it falls below .3 by the end - the same rule must trust where the organ is right and distrust where it is wrong; (2) as before in A2.
Readings: all met -> the frame stands (trust learned from its own life, both ways); A2 met but A3 not falling -> it cannot unlearn a
wrong organ; (2) not met -> the reliability is right but the organ needs more than its reliability to steer (report, then founder).
LW3 amended after its smoke, before any registered run: "relief within 30 beats" was a wholeness rise > .1, but one fill from half to
full raises wholeness by ~.08 (mean squared deficit over three channels), so almost no promise could count as kept (smoke, 600 beats:
118 arrivals, weight .5 -> .09). Relief is now LW2's event, rises only: a rise far outside the being's own usual (> running mean + 3 sd)
- no fixed size. Smoke again: 118 arrivals, weight .5 -> .52.

## LW3 result: the trust is learned the right way in every start, both ways - bar (1'') met; bar (2) met in two of three
| start | w path (every 500 beats) | arrivals | deaths / wait (L0; L1) | bar 2 (halfway) |
|---|---|---|---|---|
| 9104 rep 0 | .5 .67 .48 .36 .42 .45 .42 .48 .51 .49 .49 .46 -> .52 | 1,224 | 14 / 129 (131/302; 19/212) | met (better than full weight) |
| 9104 rep 1 | .5 .5 .42 .39 .49 .46 .49 .51 .42 .41 .39 .47 -> .50 | 1,136 | 22 / 253 (142/337; 7/157.5) | deaths met, wait not (253 vs <= 247) |
| 9101 rep 0 | .5 .5 .35 .37 .40 ... .49 -> .51 | 694 | 3 / 94 (14/120.5; 2/83) | met |
| A3 s3_a (old-map goals), 3,000 beats | .5 .04 .04 .05 .02 -> .15 | 252 | deaths per quarter 64, 64, 64, 52 (main line: 64 x 4) | - |
Bar (1''): met - A2 ends >= .4 in all three (.52, .50, .51), and in A3, where its goals are the old room's places, the trust falls to .02-
.05 within 500 beats and ends .15 (< .3). Bar (2): met in 9104 rep 0 and 9101; missed in 9104 rep 1 on the wait by 6 beats (deaths met).
Reading: the first rule on this line that learns how far to trust an organ's content from the being's own life, reliably, in the right
direction where the organ is right AND where it is wrong (and in A3 the last quarter's deaths fall 64 -> 52 once it stops trusting the old
map). Not fully met by the letter (one wait 6 beats over). Said plainly: one fixed choice remains (trust 1 = the organ's full unit).
# LW3b (registered before running, while the founder sleeps): the same in the true (fixed) world, with the main line's goal
ATANOR_WORLD_FIX=1, D48's goal (ATANOR_GOAL_FIX=0, the main line), A2 9104 rep 0, 9104 rep 1, 9101 rep 0; arms L0 (--lw 0 --lw-lr 0,
content present at weight 0), LW3 (--lw 0 --lw-mode reli), L1 (main line; 9104 rep 0 and 9101 rep 0 are the fixed-world gate-1 runs
already done: 25/174, 11/82). Bars as LW3: (1'') w ends >= .4 in all three; (2) halfway from L0 to L1 on deaths and wait in all three.
Readings: met -> LW3 replaces the fixed full weight in the main line (one hand-set number removed: the trust is learned), then gate 2
in the same frame (unfamiliarity content; A3); (1'') met, (2) missed by little in one -> adopt, and say so; (1'') not met -> report.

## LW3b result (true world, main line's goal): trust learned the same way everywhere (~.5); the outcome at that trust is mixed - not adopted
| start | L0 (w 0) | LW3 (learned; w end) | L1 (w 1) | bar 2 (halfway on deaths and wait) |
|---|---|---|---|---|
| 9104 rep 0 | 74 / 244 | 18 / 168.5 (.53) | 25 / 174 | met (better than full weight on both) |
| 9104 rep 1 | 90 / 231 | 8 / 244 (.51) | 15 / 172 | deaths met (better than full), wait not (244 vs <= 201.5) |
| 9101 rep 0 | 17 / 107 | 24 / 108.5 (.50) | 11 / 82 | not met (deaths worse than both) |
Bar (1''): met (.53, .51, .50). Bar (2): met in one of three. Reading as registered: (2) is missed in two, not by little in 9101 ->
not adopted; reported. What stands: the reliability rule finds the organ's trust the same way in every start of both worlds and both rooms
(and distrusts the old map in A3) - the learning is no longer at the mercy of its first beats. What does not: whether trusting at the
kept-promise rate (~.5) is the right amount - in 9104 it beats full trust on deaths, in 9101 it is worse than no trust on deaths.
# LW3c (registered before running; diagnostic - explains LW3b, not a variant): is it the learning or the level?
The weight held fixed at .5 (--lw .5 --lw-lr 0, mode beat: no learning), same three starts, true world. Reading: LW3c ~ LW3b (deaths and
wait within ~15% in each start) -> the learning is fine and the open question is the level a trust maps to (what "trust .5" should mean
in choice); the next step is then competition between contents' reliabilities (Lee 2014's arbitration), not another learning rule.
LW3c differs a lot from LW3b -> the moving trust itself matters (when it rises and falls in a life): look at its path against the food
moves before anything else.

## LW3c result: fixed .5 is far from learned ~.5 - the moving trust itself matters
| start | fixed .5 (LW3c) | learned (LW3b) | fixed 1 (L1) | fixed 0 (L0) |
|---|---|---|---|---|
| 9104 rep 0 | 61 / 197 | 18 / 168.5 | 25 / 174 | 74 / 244 |
| 9104 rep 1 | 32 / 252 | 8 / 244 | 15 / 172 | 90 / 231 |
| 9101 rep 0 | 8 / 97.5 | 24 / 108.5 | 11 / 82 | 17 / 107 |
Reading as registered: differs a lot -> how the trust moves within a life matters. In 9104 the moving trust has the fewest deaths of all
four (18 and 8 vs 25 and 15 at full trust; 3-4x fewer than the same mean held still); in 9101 it is the worst on deaths. Next as
registered: its path against the food moves.
# LW3d (registered; diagnostic, judge side): mean trust by beat within the 300-beat food period (bins of 50), from the per-beat path, in
LW3b's 9104 rep 0 and 9101 rep 0 (deterministic reruns, same numbers expected). Reading: trust low just after food moves and recovering
later in 9104 -> the trust works as "stop believing when the promise breaks" (the expectation-failure signal the founder and Copilot
asked for, found by the rule, not built in); if 9101 shows the same shape, its cost there is something else (look at deaths' timing).

## LW3d result: not the registered shape - the trust hardly moves within a food period
Deterministic reruns reproduce LW3b exactly (9104: 18 / 168.5; 9101: 24 / 108.5). Mean trust by beat within the 300-beat period (bins of
50 from the move): 9104 .480 .466 .455 .460 .470 .480; 9101 .392 .384 .379 .388 .390 .395. A dip of ~.02 after a move, nothing more.
Reading: the trust does not work as "stop believing when the promise breaks" within a period. So the large gap between learned (~.47)
and fixed .5 (9104: 18 vs 61 deaths) is not explained by when it moves. Two candidates: the level is sharp (a few hundredths change much),
or one run per arm is too noisy to compare (deaths in 9104 already range 8-90 across starts and arms). The second would weaken several
single-start comparisons on this line, so it is checked first.
# LW3e (registered before running; noise check): three more starts (rep 2, 3, 4) of 9104 in the true world for fixed .5 and learned
(LW3), 6 runs. Reading: learned beats fixed .5 on deaths in >= 4 of the 5 paired starts (reps 0-4) -> the moving trust's gain is real;
otherwise -> single-start comparisons here are within noise: future bars on A2 use >= 3 starts per arm (and LW3's adoption question is
read on pooled starts).

## LW3e result: within noise - learned beats fixed .5 in 3 of 5 paired starts (bar 4)
9104, true world, deaths / wait, fixed .5 vs learned: rep 0 61/197 vs 18/168.5; rep 1 32/252 vs 8/244; rep 2 22/163 vs 42/197;
rep 3 22/225 vs 33/202; rep 4 27/200 vs 12/180. Pooled deaths 164 vs 113; median of the waits 200 vs 197.
Reading as registered: single starts on A2 are within noise (one arm's deaths over five starts range 8-42). From here A2 bars use >= 3
starts per arm, and LW3's adoption is read on pooled starts. Said plainly: several earlier one-start readings on this line (LW1's
"crack", D46b's docking, E-height) were single starts and carry this noise too; gate 1 in the true world rests on one start per room.
# LW3f (registered before running): LW3's adoption on pooled starts, 9104 true world, reps 0-4 for every arm
Arms: L0 (content at 0), LW3 (learned; reps 0-4 done), L1 (main line, full weight). New runs: L0 and L1 reps 2-4 (6). Bar (pooled over
five starts): LW3's total deaths and median wait each at least halfway from L0's to L1's (or better than L1). Met -> LW3 replaces the
fixed full weight in the main line (one hand-set number gone); 9101 gets the same five-start check next. Not met -> the full weight stays;
report. Gate 1 (true world) is also read pooled from L0 vs L1 here: L1 vs L0 is the homing content's effect over five starts.

## LW3f result: met on pooled starts - adopted. The trust in the homing content is learned, not set by hand
9104, true world, five starts each (deaths per start; total; median of the five waits):
- L0 (content at 0): 74 90 83 76 81 = 404; waits 244 231 324 408 330 -> 324
- LW3 (learned trust): 18 8 42 33 12 = 113; waits 168.5 244 197 202 180 -> 197
- L1 (full weight, gate 1's main line): 25 15 19 18 27 = 104; waits 174 172 158 169 186 -> 172
Bar: halfway from L0 to L1 - deaths <= 254, wait <= 248: met (113, 197). LW3 reaches about 90% of full trust's death reduction and
80% of its wait reduction, with the trust found from the being's own kept promises (and the same rule distrusts the old map in A3).
Adopted (founder asleep, as registered): tools/s4_kin.being and tools/ml_live.py now give the main line Mind.learned_home =
LearnedWeight(mode="reli"); ATANOR_LEARNED_TRUST=0 restores the fixed full weight (all runs before 2026-11-27 04:xx).
Gate 1 read pooled (true world, 9104): the homing content at full weight vs present at weight 0 - deaths 104 vs 404 (-74%), wait 172 vs
324 (.53x): gate 1's bars met over five starts. (The control here is the homing organ at weight 0 with the fast relief memory, which is
worse than A2-1's being; against A2-1's single true-world start, 46 / 247, the effect is smaller. 9101 still to be read pooled.)
# LW3g (registered): 9101, true world, five starts for L0, LW3 (now the main line), L1 (ATANOR_LEARNED_TRUST=0), same bar. Then gate 2.
World fix is now the default (ATANOR_WORLD_FIX=0 reproduces runs before 2026-11-27).

## LW3g result: not met in 9101 - the room barely separates its own arms; learned trust's deaths trend worse than both
9101, true world, five starts each (deaths per start; total; median of the five waits):
- L0 (content at 0): 17 15 9 16 11 = 68; waits 107 122 109.5 126 115 -> 115
- LW3 (learned trust, main line): 24 29 18 15 10 = 96; waits 108.5 102 97 122 101.5 -> 102
- L1 (full weight, ATANOR_LEARNED_TRUST=0): 11 8 14 10 10 = 53; waits 82 79 119 91.5 84 -> 84
Bar (halfway from L0 to L1: deaths <= 60.5, wait <= 99.5): not met on either (96; 102 is 42% of the way).
Reading, plainly: in 9101 the homing content itself does little (L0 vs L1: 68 vs 53 deaths, within about one standard error of
five-start sums, per-start spread ~ +-6); the waits order L0 > LW3 > L1 as expected, but LW3's deaths are higher than both
(96 vs 68: about 1.6 standard errors - a trend, not settled). Trust ended ~.39 in 9101 (vs ~.5 in 9104): the promises are kept less
often here, as they should be when the content helps less. Learned trust was active in every LW start (600-750 arrivals each).
Decision: the adoption rule registered in LW3f was the 9104 pooled bar, which was met; LW3 stays in the main line, flagged.
The flag: in a room where the content helps little, a middling trust may cost deaths (mixing two pulls). Where the trust matters is
where the homing content is wrong (A3: trust fell to .02-.15) - gate 2 run 2 reads that with >= 3 starts, and L1 rides along there as
the hand-set comparison, so a harm from learned trust would show. If gate 2 also shows learned trust worse than full weight on deaths
pooled, the flag becomes a finding and the next step on this line is why middling trust hurts (registered then, not now).

# B1 (registered before running; founder 2026-11-27 on the one-body viewer: "why worse than yesterday?"): learned trust in ONE body
LW3 was adopted on 64 bodies sharing one mind (600-750 kept/broken promises per 6,000 beats). The viewer runs one body; a 600-beat smoke
had 1 promise (trust .33). Runs: a2_moving_food --bodies 1, 9104_0, being s3_b (the viewer's), 6,000 beats, true world, reps 0-4;
arms LW (main line) and L1 (ATANOR_LEARNED_TRUST=0, yesterday's viewer). Recorded: deaths, food waits, trust end, promises.
Readings (pooled over five): L1 better than LW on deaths or wait by > 25% -> experience density: the learned trust cannot settle from one
body's few promises; next = how a lone body's trust can stand (prior from its kind, or slower fading), registered then. Both alike and poor
(waits >> the 64-body 172-197) -> the one-body problem is the fading relief memory (half-life ~70 beats vs ~400 beats between meals),
not the trust. Both alike and fine -> the viewer scene is within one body's noise.
B1 amended before any result (founder/Copilot 2026-11-27: an hour is too much for this narrow question): early read after reps 0-1 by
direct measures (promises per start, trust path, food waits, deaths). If decisive (L1 vs LW waits differ by > 25% the same way in both, or
both alike and both far above the 64-body waits), stop there; else run on to rep 4.

## B1 result: one body lives poorly with either trust; learned trust dies a little more (small numbers)
9104, s3_b, one body, 6,000 beats, reps 0-4 (deaths; median food wait; hungry->food events; trust end / promises):
- LW: 6 4 7 2 7 = 26; waits 422 120 369.5 407.5 428.5 -> 407.5; 27 events; trust .17-.37 from 12-28 promises per life
- L1: 4 3 3 2 6 = 18; waits 417.5 246.5 303.5 427 488 -> 417.5; 26 events
Reading: both alike and poor on waits (~410 vs the 64-body 172-197) -> the one-body problem is mostly not the trust. Deaths favour full
weight (18 vs 26, > 25%), but on a handful of deaths. A lone body makes 12-28 promises a life: the learned trust sits at .17-.37 -
experience density is real and is a second, smaller cost. The first cost was found by C1-C3 (WORLD_A_PROTOCOL.md): the homing walk
orbits with a held step-and-turn gesture; D49 tests the fix. A lone body's trust (prior from its kind, slower fading) is left for after D49.

# M1a (registered before running; build roadmap M1): train many, live as one - the trust learned in training carried into a one-body life
Found: being() makes a fresh trust each time; the 64-body training's record (kept / broken) was never kept with the being, so a lone body
relearned from 12-28 promises a life (B1). Built (default off until checked): LearnedWeight.state() / carry(); a2_moving_food --save-trust
writes the record at the end of a life; ATANOR_TRUST_FROM=<file> gives it to the next life (then it keeps learning as before).
Run: train = a2_moving_food, 64 bodies, 9104_0, s3_b, 6,000 beats, rep 9 (not a test rep), --save-trust; then one body, reps 0-4,
6,000 beats, with the carried trust (arm LC) against B1's LW (fresh trust) and L1 (full weight) on the same reps.
Bars (pooled): LC's deaths <= L1's (18) and LC's median food wait not worse than LW's (407.5). Met -> carry becomes the default for the
being file (train many, live as one). Not met -> report (the lone body's poor life is the walk, M1b).
