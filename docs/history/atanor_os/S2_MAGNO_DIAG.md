# D15: a whole-frame (magno) code grown from the being's own frames (registered 2026-11-24, before running)

D14c: even a nonlinear read-out of the current feelings recovers position R^2 .18-.55 and heading <= .26. The eye organ (atanor_os/emerge/body_contract.py)
keeps a fovea code (z 64, zf 32) around the gaze, gaze and a 'bug' flag; the whole frame (48 x 80) feeds only the depth learner (5 numbers). Place and heading
live in the whole frame's layout. The founder's official vision is two paths (magno whole frame + parvo fovea); the magno path exists only as depth.

Prior work: predictive coding on an agent's own frames in Minecraft yields spatial maps (Gornet & Thomson 2024); predicting the next frame given the
action is the self-supervised route; no pretrained vision (founder rule).

## Design
Record lived streams (A1 v2 depletion room, 12,000 beats, dev 9101-9103) with the whole frame downsized to 48 x 80 (as the depth path sees it) and the
judge's position / heading. (a) Raw ceiling: an MLP reads position / heading from the pooled raw frame (12 x 20 x 3). (b) Grown code: a small conv encoder
(48 x 80 -> 32 numbers) trained ONLY to predict the next frame's code and pixels given the combo code (action-conditioned next-frame prediction, from
scratch, 3,000 updates); position / heading read from the 32-number code (linear and MLP), against the same encoder untrained.
Spread hold-out blocks as D14b.
## Reading
Grown code: heading R^2 >= .6 and position R^2 >= .5 (MLP), clearly above the untrained encoder and above the current feelings (D14c) -> grow this magno
code as a new part of the being's eye (one change, learned in life, no pretrained weights), then D14b/D13 on top of it. If the raw frame itself reads
place poorly, the room's look is the limit (record and widen the room's visual cues - a world property, founder to approve).

## D15 result
Even the raw whole frame (pooled 12 x 20 x 3, MLP) reads position R^2 .05 / -.03 / .31 and heading .23 / .14 / .37; the grown code no better. The room's
look is the limit: the T room is left-right symmetric and every wall looks the same; only the two springs break the symmetry. Rats in T mazes use distal
room cues; our rooms had none.

# D16 (registered before running): the same with distal cues
World property (my design choice under the founder's 'set up the environment well', for founder review; not a behaviour rule): each wall face direction
(north / east / south / west) has its own tint (RoomBody.cues, default off). Same recording and read-outs as D15 on 9101-9103 with cues on.
Reading: raw-frame heading R^2 >= .7 and position >= .5 -> cues make place visible; then A1 v3 (cued world, registered separately) for the value -> choice
link with the being as it is, and the grown magno code as the next eye part if the being's current eye summary still misses it.

## D16 result (registered reading met)
Cued room, raw frame (MLP): heading R^2 .86 / .82 / .81, position .68 / .60 / .51 (uncued: heading .14-.37, position -.03-.31). Grown and untrained codes:
heading .75-.82, position .37-.55. Distal cues make place visible -> A1 v3 runs.
