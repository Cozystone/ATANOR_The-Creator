# D21: latent learning with enough experience, and the test that matches the life (registered 2026-11-24, before running)

## What D20 showed (first 3 layouts) and why more bodies alone cannot do it
With one mind in 64 bodies the fork is no longer at chance but leans hard to one arm, and the lean is not tied to the need (9101 right for both needs,
9102 wrong for both, 9103 one side for both). In the life every body starts at the stem with both stores at .75, passes the junction about once per
life with BALANCED stores, camps at the spring it finds and dies of the other lack (about 4 deaths per body per 3,000 beats, unchanged over the life).
So the fork's state - at the junction with one store low and the other full - never occurs in the life, however many bodies live it.
A value that must have lived a state to rate it cannot answer; a value built as "what is where" x "what I lack now" can (latent learning:
Tolman & Honzik 1930; successor features; homeostatic reward). D17's relief-share arm is that structure; with one body it had too little data
(map right 8 / 12).

## Design
Lives: D20's (one mind, 64 bodies, cued full-size T room, 12,000 beats, the being as it is), dev 9101-9103, ring of the last 300k lived steps saved with
each body's own next action. Arms trained offline on that ring (2,000 updates): relief-share (D17), its shuffled-outcome control; and the life's own value.
Two tests, the being's choosing machinery unchanged (pull, hold, try as they are; value trust 1 as in A1 v2):
- junction fork (A1 v2): 16 thirsty + 16 hungry clones at the junction, T = 400.
- far fork (matches how the life fails): 16 thirsty clones standing at the FOOD spring and 16 hungry clones at the WATER spring (stores .30 / .95), T = 400,
  no drain: do they leave and reach the other spring (needed-channel recovery)?
## Reading (fixed now)
Relief arm: junction first match >= .70 and recovery >= .50, both >= .15 above its shuffled control, in at least 5 of 6 (layout x need) cases the needed arm is
preferred; far fork recovery >= .40 and >= .15 above shuffled and above the life's own value -> the latent structure with enough experience answers A1;
next = the being lives with this value (A/B over a 64-body life: deaths must fall), then sealed seeds.
Junction passes but far fork does not -> knowing where at the junction, not from afar (the horizon of the futures or the trip itself), measured next.
Neither -> the structure does not learn 'what is where' even with ample data (look at psi along the paths with the organ state loaded).

## D21 result (6 layouts, 12 cases): registered reading not met, a real partial effect
Needed side preferred: relief 9/12, life 6/12, shuffled 5/12. Junction first match .59 / .52 / .51; recovery .58 / .37 / .33 (relief - shuffled +.26 +-.10, 9/12);
far fork recovery .17 / .18 / .08. The latent structure reads 'what is where' from the lived experience, but the body does not go promptly, and hardly
leaves the wrong spring.

# D22 (probe, registered before running): is the knowledge good enough to get there?
Same six saved lives and relief arm (and its shuffled control). Judge-only policy: every beat the relief score's best combo (no pull, hold, try, softmax),
and the same with 10% random combos; from the junction and from the wrong spring, T = 400.
Reading: greedy relief recovery >= .80 from the junction and >= .60 from the wrong spring, both far above shuffled -> the knowledge suffices and the being's
noisy choosing (re-picks, softmax over 90 combos, tries) is the break: next = how intention persists (learned, not a rule). Greedy fails too -> the
score is right about the side at the junction but not a usable gradient along the way (the futures' horizon / place resolution), measured next.

## D22 result (probe)
Greedy relief score + 10% random from the junction: enters the needed arm .91 / .90 / .96 / .88 / .53 / .47 (shuffled about .5), but recovery is low (about .24) and
pure greedy gets stuck (0); from the wrong spring 0. The score is a signpost at the junction, not a route, and knows nothing from afar.

# D23 (registered before running): the horizon of the futures
The futures discount at .97 (about 33 beats); a spring-to-spring trip takes 50+ beats directed and hundreds wandering, so distant relief is worth ~0.
Brains carry several horizons at once (Masset et al. 2023; Tano, Dayan & Pouget 2020). Same saved lives; relief arm at gamma .97 / .99 / .997, each trained
10,000 updates with target sync 25 (so the futures can propagate), with shuffled controls. Measured: junction fork and far fork under the being's own choosing;
the greedy + 10% probe from the junction and from the wrong spring.
Reading: a longer horizon lifts far-fork recovery (being's choosing) to >= .40 and junction first match to >= .70, above its shuffled control by >= .15 ->
adopt several horizons in the relief value (then the being lives with it). No lift from afar -> the lived ring lacks spring-to-spring trips to learn from
(count them), and exploration is the open problem.
