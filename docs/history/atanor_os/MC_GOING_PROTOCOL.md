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
