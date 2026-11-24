# D14: does a sense of place emerge from predicting what comes next? (registered 2026-11-24, before running)

D13: the eye scene does not carry place identity (relief scenes at one spring vary as much as all scenes; heading changes what the eye sees).
The being has no sense of where it is. Prior work: path integration and grid-like codes emerge in recurrent networks trained to predict
(Banino et al. 2018; Cueva & Wei 2018); the Tolman-Eichenbaum machine learns place from predicting observations along paths (Whittington et al. 2020).

## Design
A lived stream (the being's own life, A1 v2 depletion room, 12,000 beats, dev layouts 9101-9103): feelings x_t, done combo a_t; the judge records
position and heading (never given to the learner).
Inner state h_t = GRU([input_t, code(a_t)], h_t-1), trained self-supervised only: predict the next scene (non-bedrock feelings) x_t+1 and x_t+4
from h_t (and the codes of the actions in between). Inputs: (A) all feelings; (B) only the controlled dims (felt motion; D11 shuffle test) - pure
path integration that must predict the eye. Truncated windows of 64 beats, cut at deaths / gaps.
Measured on held-out stretches: linear (ridge) decoding of position (R^2) and heading (cos/sin R^2) from h_t, against the same decoder from x_t alone
and from a random (untrained) GRU of the same size.
## Reading
h (A or B) decodes position R^2 >= .6 where x alone is <= .3 -> a place sense emerges from prediction; next = this inner state as the being's working
memory (Mind inner, one change), goals as inner states of relief (D13 redone in h-space), then the A1 fork.
Otherwise: the prediction target or window is wrong (try longer horizons) before anything else.

## Result (first run) and a post-hoc check
Registered reading not met: trained h decodes position R^2 -.56 to .27 (random GRU -.42 to .37), x alone -.11 to .33, heading about 0 everywhere.
Even heading near 0 from the feelings is implausible (the eye sees the wall directions), so the contiguous last-25% hold-out is suspect (the being
ranges differently late in life). Post-hoc robustness check, labelled as such and not changing the verdict: same runs, held-out = every 4th block of
500 beats (spanning the whole life).

### Post-hoc check result
With spread hold-out blocks the feelings alone decode position R^2 .19 / .30 / .45; the predict-next GRU (A) .00 / .14 / .37 and a random GRU .17 / .31 / .46;
motion-only (B) -.45 / -.13 / .27. A place sense does not emerge from 1- and 4-beat prediction: the next eye summary is well predicted by 'about the same',
so nothing pushes the inner state to know where it is.

# D14b (registered before running): far, action-conditioned prediction
Same streams. The inner state must predict the scene 16 and 32 beats ahead given the actions in between (summed branch codes over the interval -
which way the body was driven), besides 1 and 4 beats. Only knowing where it is (and where it heads) makes the far scene predictable.
5,000 updates, windows of 96 beats. Primary metric now the spread hold-out (every 4th block of 500 beats), justified by the first run.
Reading: trained h decodes position R^2 >= .6 and at least .2 above both the feelings alone and a random GRU on most layouts -> place emerges from
far prediction; next = this inner state as the being's working memory (one change), then D13 (goal) in h-space and the A1 fork.
Else: the eye summary does not hold enough of the scene for place (next suspect: what the eye organ keeps), recorded before anything else.

# D14c (registered before running): is place in the feelings at all (nonlinear read-out)?
Saved D14b streams (12,000 beats each). An MLP (2 x 256) reads position and heading from the current feelings x_t alone, spread hold-out blocks as
D14b, compared with the linear read-out. Also from the last 8 feelings stacked (a short window, no learned memory).
Reading: nonlinear R^2 of position >= .6 -> the eye summary holds place and the failure is in what the inner state learns; <= .4 -> the eye summary has
lost most of it (next = what the eye organ keeps for place: e.g. the canon fovea's coarse layout), before any memory design.
