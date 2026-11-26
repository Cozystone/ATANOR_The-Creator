# Carrying the going to MineLand (founder 2026-11-25: "MineLand로 옮겨서 진행해")

## What the valley is (judge words; nothing here is given to the being)
Stone trench open to the sky. Place A: lit floor (glowstone, no monster appears on it). Place B: dark. Three cakes at fixed spots on the dark side
near B (dx -2..2, dz 15 from the centre), grown back every 600 steps (the room's springs, in Minecraft). Hunger II on every body, more when it
works. At night the world summons two zombies 30 blocks off both open ends every 300 steps. Dead bodies come back at the centre.

## What kills (life 21, first 10,000 steps, server's own messages)
121 deaths: slain by Zombie 89, Zombie Villager 6, Creeper 5, Skeleton 1 (mobs 101); starved 16; drowned 2; fell 1.
Food just before death: 0 in 18, above 6 in 103. Life just before death <= 6 in 114. Bodies are worn down by mobs over several steps, not starved.

## Theory (before any build)
The room's going takes the lack to where that inside was relieved, by place. In the valley:
- food is place-bound (cakes) -> the going can carry the food lack to the cakes; this addresses the starvation share (~13% of deaths), not more.
- life comes back by Minecraft's own healing when fed and not being hit. If the places where bodies are not hit (A lit, or away from the open
  ends) are where life rises more, then the same organ with the life lack takes a hurt body there - shelter emerging from the bedrock and its own
  place memory, no "danger" rule. If life rises equally everywhere (mobs follow bodies), place memory cannot help with mobs, and the zombie deaths
  need something else (flight from what hurts, or a closed shelter) - not the going.
- Prediction (mine, before the data): food: yes; life: weak - zombies are summoned at the ends and walk in, A only stops spawning on it.

## D35 (registered before running): where in the valley is life given back and where is it taken (judge-side record only)
Every step, each body's place (x, z from the valley centre, 2-block cells), whether its life rose, its food rose, it was hurt, and night or day.
Read after 10 windows (5,000 steps). Readings: life rises concentrated in some cells (top 10% of visited cells hold >= 30% of night life rises,
and those cells see hurt at <= half the valley's rate) -> register the going in the MineLand being with the life lack; else -> the going goes in
for food only, and zombie deaths are a separate problem (report).
Also in the same restart (judge loop, not the being): the senses re-read after a rebirth or a lift no longer re-feel every body's last motion
(the room's drift bug, P1_PATH_PROTOCOL L1d), and the lift puts the body back facing the same way as a rebirth does.

## D35 result (10 windows, life 21d; confounded from step ~1,000 by the founder-asked shafts)
- Night: 153 life rises over 2,751 body-steps; the top 10% of cells by life-rise rate hold 14% of them (bar 30%), and are hit 2.6x MORE often than
  the valley (bar: half). Day: 11 life rises in all. Food rises: 0 in 8,236 body-steps - no body ate a cake in these 5,000 steps.
- Reading as registered: not met -> life is not given back by place; the going cannot shelter a hurt body by place memory here. And food, the one
  place-bound relief, was not eaten at all in this stretch, so its relief memory would stay empty: the going has nothing to lean toward yet.
  Before any build: why cakes went uneaten (the bodies starve 16 times in 10,000 steps with three cakes in the valley).

## D35b (judge side, 2026-11-26, no being): can a body eat a valley cake at all
tools/mineland/cake_check.py: body 0 put 1.3 blocks before a cake, facing it, made hungry (food <= 14), 20 steps of each hand option.
- 'use' eats: head tilted down 15 deg food 8 -> 16, 30 deg 13 -> 20, 45 deg 12 -> 20; level (0) and 60 deg: no eating. A full body cannot eat
  (Minecraft's rule: pitch 30 at food 20 gave nothing in the first check). 'attack' and 'dig' never eat.
Reading: no world fault. Eating asks for four things at once: hungry, within reach, head nodded down one to three units (the neck's resting
tone pulls the head back to level every 4 still steps), and 'use'. D35's bodies were rarely hungry (food > 6 before 103 of 121 deaths) and
the cakes sit on the dark side where zombies come - so "no cake eaten" is the being's life, not the valley's rules. Next on this line
(roadmap stage 4, one being many bodies): how often bodies are within reach of a cake while hungry, in a life.

# D35c (registered before running; judge-side record only, roadmap stage 4)
Life 25: the being continues from mineland18b/s1000 (with its LOKI memory), as life 24b (home -10, window 500), 5,000 steps. Every window,
body-steps within 2 blocks of a cake, of those how many hungry (food < 19.5), and food rises there (eating).
Readings: hungry body-steps in reach >= 50 in the life and eating 0 -> the being comes hungry to food and does not eat (the hand + head
combination is the gap, a learning question); hungry in reach < 50 -> it seldom comes hungry to food at all (where it lives is the gap:
zombies on the cake side, food mostly full); eating > 0 -> first eating in the valley, measure whether it rises over windows.

## D35c result (life 25, 5,000 steps, 10 healthy windows, no faults)
Cake reach [body-steps within 2 blocks, of those hungry, eats]: window 1 [51, 51, 3], window 4 [1, 0, 0], all others 0. Deaths per window
0, 2, 9, 25, 37, 10, 2, 2, 0, 2. Value decided 0-3 of 500 steps per window (strong pull and hold decide the rest).
Reading as registered: eating > 0 (first cakes eaten in the valley, 3 in the first window), but it does not rise - after window 1 the
bodies never come back to the cakes. Hungry in reach summed 51 (bar 50, met only by window 1). The relief there did not bring them back:
the being's value, which would carry "food is there", chooses almost nothing (0-0.6% of steps) - the same wall as in the room (S4k: choice
follows value 11%). Line rests here; next is the core: why value does not move choice.
