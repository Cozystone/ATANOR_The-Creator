# D1: does the value know what sticking to an action does? (registered 2026-11-24, before running)

Founder 2026-11-24: one focus, the value -> choice link, until it breaks through.

## Theory (written before any D1 number)
- The being chooses in **held actions**: once picked, a combo is kept for many beats until surprise, a much stronger pull, or futility re-picks it
  (Mind.choose, `again`). Hold shares in the A1 v2 forks and MineLand are 33-80% of beats.
- The value (Worth with m = combos, and ChannelWorth arms) learns **one-beat** Q: Q(x, a) = r + gamma max_a' Q(x', a') (core.py Worth.learn).
  In one beat almost every combo leads to nearly the same next feeling, so the differences between combos are small next to the network's noise,
  and the max over 90 noisy estimates inflates them. A ranking of single combos is then mostly noise, even when the mean over many left-going
  combos minus right-going combos leans the right way (A1 v2: value_pref_right .62-.67, first choice at chance; hold diagnostic interim:
  choosing the value's best combo spoils as much as it saves, with no trend in margin size).
- Prediction R1 (target mismatch): the value's per-combo ranking at the fork does not track what **holding** that combo really leads to.
  Prediction R2 (state blindness): even the true option outcome is not predictable from x (then a better target would not help).

## Measurement
- Same phase A, arms and room as A1 v2 / the hold diagnostic (depletion world, 12,000 beats, dev seeds), arms 1f, 2, 3.
  Phase-A minds, the lived Worth ring and the three arms are cached to `probe-work/a1cache/` for later diagnostics.
- At the A1 fork start (junction, half thirsty / half hungry, value trust 1, no learning), every one of the M combos is **held** by P = 1 clone per need (amended before running: the GPU is shared, 2 x 90 clones fit)
  for H = 12 beats (enough to enter an arm), then the mind chooses as usual (a cut first) until T = 150 beats.
- Truth per combo and need: did the needed channel rise by more than .05 (recovery), and which arm was entered first.
- The value's scores at the start: Q(x0, a) for every combo (lean before trust), per need.
- Reported per layout x need x arm: Spearman(Q, truth recovery rate over combos); the share of combos whose held outcome differs by leg option only;
  the truth's best-vs-worst spread (is there anything to know at all); where the value's top-5 combos lead (recovery share) vs a random 5.
- Reading fixed now: Spearman near 0 with a large truth spread = R1 (the target, not the knowledge of left / right, is wrong).
  A high Spearman yet bad choices = the choosing machinery (pull, hold, try) overrides. A small truth spread = the fork does not test the value at all.

Dev seeds 9101-9106 only (both mirrors). Sealed seeds 7701-7712 untouched. Tool: `tools/a1_option.py`.
