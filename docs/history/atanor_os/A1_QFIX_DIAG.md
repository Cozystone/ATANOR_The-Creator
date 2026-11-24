# D7: value targets without the max bias (registered 2026-11-24, before running)

D6: the lived action value Q sits at 5-20 where returns are of order +-0.5 - the textbook maximisation bias of a max over 90 noisy heads compounded by
gamma .97. Arm 3's score is stuck at .164 everywhere. The value -> choice link is broken at the value's learning target.

Offline, same cached lived rings, same network and updates, only the target differs (bedrock reward unchanged, no behaviour rule):
- max (control, today): r + g max_a' Q'(x', a')
- double: r + g Q'(x', argmax_a' Q(x', a'))
- sarsa: r + g Q'(x', a'_done) - the action the being really did next (it mostly keeps holding: on-policy, no max)
- held+sarsa: the discounted bedrock change over the rest of the run of the same action, then g^L Q'(x_end, a_done at the run's end)
Metrics: value scale (mean |Q| at the fork), Spearman with the noise-free truth (holding the combo carries the body into the needed arm) and with recovery,
top-5 hold-entry, and the value map gap (needed path end minus other path end, same need) along the arms.
Reading: a target with sane scale (|Q| < 1) and Spearman with hold-entry >= .3 and a positive needed-path gap in most layouts -> give the being that target
(one change: Worth.learn), then rerun the A1 fork (dev) and a regression life; if none reaches it, the max bias was real but not the only break.
