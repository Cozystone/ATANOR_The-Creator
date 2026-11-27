# Plan B: a being whose own learned world model chooses (designed 2026-11-26, kept ready; founder "대안 B도 설계해둬")

When it is used: the stop signal of the roadmap (atanor_os/EMERGE_PLAN.md, 2026-11-26) - gate 1 ("keep up with a changing world") not passed
on both routes within its budget of ten registered experiments. Until then the main line stays as it is; parts of B may be built and checked
judge-side beforehand (as I0), but nothing replaces the main line before the stop signal or the founder's decision.

## Why (what the evidence says the main line cannot do)
- Choice is a hand-weighted sum of learned parts (pull, harm, futility, value, going lean) behind a hand-set gate. D39-D43 and B1: changing
  how the parts are summed or weighed changes nothing - no part knows what a sequence of gestures brings later (D42-D43, LK1).
- What each part knows is separate and partial: the core forecasts one beat (badly on a new body), the value rates states (not gestures),
  the going organ knows places and a walking skill (not time, not a changing world), LOKI knows the body one beat ahead and, in v12/v12.1,
  some of where resources are.
- V1: a faster-fading place memory follows moving food (.35 -> .74) but also forgets the layout - "what lasts vs what passes" is decided by
  one hand-set number per organ.

## The shape of B (one model, one learning signal, choice by imagination)
1. One latent world model per body (shared weights across a mind's bodies, as the core now): from the senses (eye magno + parvo, skin, ear,
   own motion, path code, bedrock) a latent state z_t; a recurrent transition z_{t+1} = f(z_t, gesture_t) with a deterministic and a
   stochastic part (as RSSM, Hafner et al. 2019-2023 "Dreamer"); decoders for the next senses and the next bedrock. Trained only on the
   being's own lived stream, predicting what comes (no labels, no reward but the bedrock).
2. Two time scales inside it, not two hand-set numbers: slow weights for what lasts (layout, body, how gestures move it), fast state /
   fast weights for what passes (where food is now). The rate of the fast part set by the model's own surprise (prediction error drives the
   learning rate - Pearce-Hall, Behrens 2007; the "dopamine as prediction error" idea): a world that keeps surprising it is learned and
   forgotten faster, a steady world slowly.
3. Choice by imagination over movement units: candidate sequences (gestures held a few beats each, or the going skill's journeys toward
   remembered places) are rolled in the latent model for H beats (H ~ 30-60); each scored by foreseen wholeness change (bedrock only);
   the first step of the best is done; replanned every few beats. The score is the model's own foresight, not a hand-weighted sum.
4. Knowing when it is wrong: an ensemble (or stochastic rolls) gives the spread of foresight; disagreement lowers a plan's weight and raises
   the urge to look (information gain toward the founder's second bedrock: drawn to where its own prediction fails). Calibrated against
   outcomes as it lives (as Earned), never a command.
5. What stays from the main line: the bedrock (wholeness = 1 - mean squared deficit), the universal body contract, path integration (an input),
   the going organ's walking skill (as a movement unit), babbling/protected childhood, the guard and judge-side records. What goes: the
   hand-weighted score sum and the hand-set gate (replanning is the model's own: when foresight and what came part).
6. LOKI's place: B's model is the role LOKI v12 was specified for (R1-R8). If Astra's line reaches it first (real-body R5 >= .8, R6
   calibrated), LOKI becomes B's world model through the bridge; if not, B is built in ATANOR's own core. One model either way - no second
   memory beside it.

## Costs and risks (honest)
- Compute: a latent model with 30-60-beat imagination for 64 bodies is heavier than the main line. Budget: one roll batch per body every 4
  beats, H 32, 16 candidates -> ~32k latent steps per decision round; on this PC (16 GB GPU) at a 128-wide latent that is feasible at room
  scale; MineLand (2 bodies) is cheap; the laptop needs a distilled small model (later).
- It can fail the same way: a model that predicts the next beat well can still be blind to long consequences (LK1). Dreamer-like models
  learn long horizons only with enough lived variety - world A's staged worlds are its curriculum.
- It loses what the main line earned by hand-tuning (home deaths ~0). B must first match the main line at home before any new test.

## How it would be checked (gates, registered when started)
- B0 (judge side, no choice): the latent model's 32-beat foresight of the bedrock under randomised sequences, corr >= .3 (LK1's test).
- B1: B chooses at home (T rooms): deaths and need fills within 10% of the main line.
- B2: B in A2 (moving food): gate 1's bars (wait near the still arm, deaths at most half).
- Then gates 2 and 3 as the roadmap.

## First steps if the stop signal comes
1. Core: replace Core.f (one-beat forecast) by a small RSSM trained on the main line's lived rings (existing data); check B0 judge-side.
2. Add the surprise-set fast part; check on A2 judge-side (does its belief follow the moving food without forgetting the layout).
3. Choice by imagination at home (B1), then A2 (B2).
