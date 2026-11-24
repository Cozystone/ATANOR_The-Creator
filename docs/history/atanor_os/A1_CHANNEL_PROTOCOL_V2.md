# A1 channel-outcome test v2 - preregistration (2026-11-23, written and committed before any v2 run)

v1 (A1_CHANNEL_PROTOCOL.md, a28faab4) is recorded as failed (founder). Its dev diagnosis: the experience held exactly one successful trip to the rarer
spring per layout (phase A ended as soon as both springs had been filled once), so no learner could split "that side -> that channel".
v2 changes **only the world and the length of phase A** (the founder's choice: let the individual meet the junction many times by itself), measured first
on the judge side (EMERGE_PLAN 2026-11-23: 12,000 beats + spring depletion gave >= 5 successful trips on both arms in 5 of 6 dev layouts).

Unchanged from v1: the question and hypothesis H; arms 1L, 1f, 2, 3 and their shuffled controls; the channel predictor P and the arm-3 bedrock formula
(designer-given conversion, exact difference primary, gradient form reported only); the fork (32 clones at the junction centre, half thirsty half hungry,
drain 0, no learning, value weight lam 2 with trust 1 for every arm); the recorded chain (channel prediction -> value side preference -> first choice ->
recovery of the needed channel) and "value knew, habit won"; the gate G1-G3 on arm 3; adoption rules (efficiency, regression life).

Changed:
- World in phase A: spring depletion on (room3d_gpu.World / RoomBody regrow = 1/120 per second: a spring gives only what it holds and refills over ~2 minutes),
  drain 1/180 per second (twice v1), rebirth with memory on death. Phase A lasts a fixed 12,000 beats (not "until both filled").
- Experience given to every arm in sleep = all lived steps of phase A, taken from the value organ's own ring (Worth ring: every lived transition, cuts
  excluded), not from the core's 4,000-step buffer (which would keep only the last third).
- Layout flag: besides "both recoveries lived", each layout records the successful trips per arm (judge count as tools/a1_data.py); primary numbers use
  layouts with >= 5 successful trips on both arms, whole-population numbers beside them.
- Fork world: the same layout with depletion on and both springs full; drain 0 as in v1.

Seeds: dev 9101-9112 (never used). Sealed 7701-7712 (never used). On dev I may change only the sleep updates (500-4,000) and P's width (64-256).
No reruns of the sealed seeds with other settings; a revision needs a new file and new seeds.
