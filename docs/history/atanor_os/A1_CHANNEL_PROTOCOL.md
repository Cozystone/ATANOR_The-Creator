# A1 channel-outcome test - preregistration (2026-11-23, written and committed before any dev or sealed run)

Question (founder, 2026-11-23): A1 = "does the being use its past for the need it has now". The replay diagnostic D0 (EMERGE_PLAN 2026-11-23) showed that even with
perfect memory the existing value (Worth: one number, reward = change of wholeness) does not prefer the side of the spring that the current lack needs.
Hypothesis H: what is missing is learning the outcome per body channel. If the being learns "after this act, how will each inside channel change from here on"
(learned from its own life only), then the same view with a different lack can be valued differently at once, because what that change is worth now
depends on the present body. H predicts arm 3 > arm 1, and the comparison 2 vs 3 says whether the gain comes from learning split outcomes (2 ~ 3)
or from turning them into value through the wholeness formula (3 > 2).

## Arms (the same experience for all; nothing in any arm names water, food, left or right)
1. **Worth (control, kept unchanged)** - atanor_os/emerge/core.py Worth, action values, reward = whole(x2) - whole(x) (death = -whole(x)), TD with replay.
   Two versions: 1f = fresh weights (matched experience), 1L = the being file's own Worth (as loaded; reference only, it has 18,000 earlier steps).
2. **Split outcome as an auxiliary task** - a shared channel predictor P (below) is trained; an action-value head Q2 on [x, P's frozen trunk features]
   is trained exactly like Worth (same reward, same TD, same replay count). The split outcome only shapes the features Q2 sees.
3. **Split outcome turned into value by the bedrock formula** - the same trained P (the very same weights as arm 2). Score of act a in feeling x with inside
   channels b: S(x, a) = whole(clip(b + psi(x, a), 0, 1)) - whole(b), whole(b) = 1 - mean((1 - b)^2) (the founder's first bedrock, ALIVE_WHOLE=4).
   **Designer-given part (recorded as such): this conversion of predicted channel change into an act score.** The weights of the channels come from the
   bedrock's own definition, not from me; the split into "learned channel futures x the bedrock now" is my structural choice.

Channel predictor P (new, atanor_os/emerge/channel_worth.py): trunk (dx -> 128 -> 128, ELU) + head (128 -> M x ni), psi(x, a) = expected discounted future
change of each inside channel after doing a here (gamma 0.97 as Worth). Target from a lived step (x, a, x2): db = b(x2) - b(x) (death: db = -b(x)),
psi_target = db + gamma (1 - died) psi_old(x2, a*), a* = argmax_a S(x2, a) with the target network (successor-feature style, Barreto et al. 2017).
Inside channels and the wholeness formula are what the body contract already marks (bedrock=True); no world meaning is given.

## Primary scoring decision (made now, before results)
Arm 3 uses the **exact difference** whole(clip(b + psi, 0, 1)) - whole(b) for the main verdict. The gradient form sum_k 2 (1 - b_k)/n psi_k
(small-change approximation; differs near a spring when a channel fills a lot) is computed and reported beside it, never used for the gate.

## World, experience, sleep (same as need fork v2 unless written)
T rooms from tools/need_fork.t_room, each seed drawn and mirrored left-right (the spring sides swap: a side habit cannot pass).
Phase A: the room being file (being/atanor.pt) lives one body, both stores 0.75, drain 1/360, rebirth with memory on death,
until it has filled both channels at least once (cap 3,000 beats). Its lived steps (Mind.buf) are the one experience every arm learns from.
Sleep (matched): each arm starts from the same seeded init (1L excepted) and replays the same steps: the steps are put in its ring until >= 4,096 + steps,
then 2,000 updates of batch 2,048 for each trained network (P for arms 2/3 together, Q2 for arm 2, Worth for 1f and 1L). Same seeds for all arms.
**Shuffled control (memory-outcome pairing broken):** the same steps, with the body outcome (the inside channels of x2 and the death flag) permuted
across steps before training; arms 1f, 2, 3 are rebuilt from it.

## Fork (no learning)
For each layout and arm: 2K = 32 clones of the being after phase A (core, memories, organs as saved) with that arm as its value, put by the judge at the
junction centre (x0, y = 2.0) facing the crossbar wall. Clones 0-15: water 0.30 / food 0.95; 16-31: water 0.95 / food 0.30. drain 0. The value's weight
in choosing is the same for every arm: lam = 2 and trust = 1 (judge setting, so arms differ only in what they value). 400 beats or until every clone chose.
Side sets (judge only): each of the 90 combos is played 6 beats from the junction pose in a judge copy; L = combos ending > 0.5 cell left of x0, R = > 0.5 right.

Recorded per layout, per lack (thirsty half / hungry half), in this order:
1. **Channel prediction** (arms 2/3, P): for the lacking channel k, mean psi_k over L minus over R has the sign of the needed side.
2. **Value side preference** (every arm): mean score over L minus over R (1f/1L/2: action values; 3: S) has the sign of the needed side.
3. **First actual choice**: share of clones whose first arm (centre > 2.5 cells past x0 in the upper corridor) holds the spring of their lack.
4. **Recovery of the needed channel**: share of clones whose lacking store rose by > 0.05 within the 400 beats, and the mean rise of the needed and the other channel.
"Value knew, habit won": value side preference right but first-choice match < 0.5. Counted per arm.
Flag per layout: both recoveries lived in phase A (fills of both >= 1) with the count of each. Primary numbers use flagged layouts; whole-population numbers
(failed phase A = 0.5 for 2-3, 0 for 4) are reported beside them.

## Dev and sealed seeds
Dev: 9001-9012 (24 layouts). On dev I may change only the number of sleep updates (500-4,000) and P's width (64-256); any other change needs a new file.
Sealed: 7501-7512 (24 layouts, never used), one run with the dev-chosen setting.

## Gate (sealed, arm 3)
- G1 value side preference right in >= 0.75 of (layout x lack) cells, and arm 3 minus arm 3-shuffled >= 0.20.
- G2 first-choice match >= 0.65 and >= arm 3-shuffled + 0.10.
- G3 share of clones that recover their needed channel >= arm 1f + 0.10.
A1 through arm 3 passes if G1, G2 and G3 hold. Arms 1f, 1L, 2 get the same numbers and are reported against the same thresholds (not gated).

## Adoption as the default value (separate from the A1 verdict)
Efficiency at matched experience, replay count and seeds: report parameters per arm, multiply-adds per choose for the 90 combos, sleep seconds, choose latency.
Regression life (unseen rooms room_vary.make_room 7601-7604, 16 bodies, 1,200 beats, drain 1/360, all learning on, the loaded being with its own value
replaced by the arm - arm 3 learns P online from its ring; main line = 1L): mean wholeness, fills per body, deaths. Arm 3 is adopted as default only if
A1 passes AND mean wholeness >= main line - 0.02 AND fills per body >= 0.9 x main line AND deaths not more than main line + 5 points. Otherwise
it stays an option (off by default) whatever A1 says.
No reruns of the sealed seeds with other settings after seeing them; a revision needs a new file and new seeds.
