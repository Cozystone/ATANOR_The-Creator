# D2: one-beat target vs held-action target (registered 2026-11-24, before running)

Follows D1 (`A1_OPTION_DIAG.md`). Same cached phase-A lives (`probe-work/a1cache/`), dev seeds only.

## Question
If the value were trained on the being's own unit of choice - "keep doing this combo until I would re-pick" - would its ranking of combos at the
A1 fork track what holding them really does?

## Fixed design
- One lived Worth ring per layout (the being's own phase-A experience, chronological, one body). Same network (Worth, m = combos, d = 128),
  same optimiser, same 3,000 updates of 2,048, same gamma .97, same seed. Only the target differs:
  - **one-beat** (control, today's rule): r_t + g max Q'(x_t+1)
  - **held**: the discounted bedrock change over the L beats left in the run of the same action from t (a run ends where the action changes,
    at a death or at a gap in the ring), then g^L max Q'(x at the run's end). Reward is still only the wholeness change (bedrock rule kept).
- Also reported: the being's own lived value as it was ("life").
- Truth (independent of the variants): at the fork start every combo is held 12 beats by one clone per need (thirsty / hungry), then the being
  continues with its lived value until beat 150; recovery = the needed channel rose by more than .05.
- Amended before running (after D1, whose truth used one clone per combo): two clones per combo, and the truth's own split-half reliability
  (Spearman between the two clones' recovery vectors) is reported - a value cannot track a truth that does not repeat. A second, noise-free truth:
  did holding the combo for the 12 beats alone carry the body into the needed arm (deterministic physics, no choosing involved).
- Metrics per layout x need: Spearman(Q at the start, truth over combos), the truth recovery share of the value's top-5 combos vs the mean.

## Reading (fixed now)
- **held** clearly better than **one-beat** (mean Spearman higher by >= .2 and top-5 recovery above the truth mean in most layouts) ->
  the target is the break; next = give the being the held target (one change) and rerun the A1 fork and a regression life.
- both near 0 -> the target alone is not the break (look at state: what x carries at the junction, R2).
- both high but choices still bad -> the choosing machinery overrides (pull, hold, try).
