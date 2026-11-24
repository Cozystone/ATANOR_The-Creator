# D12: arm 3 (successor features x the current need) without the max bias (registered 2026-11-24, before running)

Prior work: successor features let a value re-weigh the futures of each channel by the current need at once (Dayan 1993; Barreto 2017), and homeostatic
reward (Keramati & Gutkin 2014) - our wholeness - makes that need-dependent. That is A1's question. Arm 3 has this form but D6 found its score stuck at .164
everywhere. Its per-channel futures psi bootstrap from argmax_a' S(x', a') - the same max bias that inflated Q - so psi grows until every channel is
predicted to fill, clip(b + psi, 0, 1) = 1, and S = 1 - whole(b) is the same for every place and combo.

One change: psi bootstraps from the combo the body really did next (SARSA; mean over combos where not joined). Offline on the cached lived rings,
same network, same 2,000 updates, same seed; arm 3 'max' (as before) vs arm 3 'sarsa'.
Measured: psi scale (mean psi), how many channels the score saturates, the score's spread along both paths and its needed-minus-other gap (value map, D6),
and the A1 v2 fork with the arm (value preference, first match, recovery share; trust fixed at 1 as in A1 v2).
Reading: sarsa psi not saturated AND a positive needed-path gap in most layouts AND fork first match / recovery above 'max' -> adopt the SARSA psi in
ChannelWorth and test arm 3 as the being's value (V2 A/B like V1). Not saturated but no gap -> the futures are learned but not where; look at the junction data.
