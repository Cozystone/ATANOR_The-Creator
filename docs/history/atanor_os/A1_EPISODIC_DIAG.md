# D8: choosing from recalled episodes (registered 2026-11-24, before running)

D7: SARSA fixes the max bias (|Q| 82 -> .04) yet no slow value ranks the 90 combos at the fork (Spearman about 0). In 12,000 lived beats the body went
from the junction into a needed arm and filled only 5-12 times: too few for a slow TD value. Animals use fast episodic recall for this (complementary
learning systems; episodic control, Lengyel & Dayan 2007): "in a moment like this, what did I do, and how much of my wholeness came back afterwards".

## Episodic score (bedrock only, no rule)
For every lived moment t in the being's own ring: its feeling x_t, the combo done a_t, and its outcome G_t = the discounted wholeness change over the
next K beats (K = 60, gamma .97, stopping at a death or a gap; death = losing all wholeness, as in Worth). At the fork, with the current feeling x0:
similarity w_t = exp(-d(x0, x_t)^2 / s^2) with d = distance over the non-bedrock dims (where am I, what do I see) and the bedrock dims scored by the
current need through the wholeness of the outcome only - the recalled outcome is the wholeness change the being would get with its CURRENT stores:
G_t is recomputed with the current stores replaced in the fill pattern, i.e. the recalled per-channel rises (x_t+k inside - x_t inside) applied to x0's
stores. Score(a) = sum_t w_t [a_t = a] G_t(x0) / (sum_t w_t [a_t = a] + 1) (a pseudo-count keeps rarely done combos near 0).
s^2 = the median of the closest 1% squared distances from x0 to the ring (set from the ring itself, no tuning on the truth). The ring and its size are the being's (bounded).
Also a per-branch version (combos sharing the leg option pooled), since the legs move the body.

## Measure and reading
Same truth as D2-D7 (noise-free hold-entry into the needed arm, and recovery). Reading: Spearman with hold-entry >= .3 or top-5 hold-entry at least
twice the mean in most layouts -> episodic recall can steer choice; next = an episodic value read in the Mind beside Worth (one change), A1 fork, regression life.
No gain -> what the ring holds at the junction (count its junction visits and their outcomes) before anything else.
