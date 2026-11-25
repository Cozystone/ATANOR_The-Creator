# Innate developmental scaffolds in newborns: a map for an artificial developing creature

Survey date: 2026-11-25. Scope: human infants and newly hatched/newborn animals (chicks, rat pups, monkeys), plus computational readings from cognitive science, AI and developmental robotics.

Conventions:
- **ANSWER** = the scaffold behaves like built-in content (it works correctly before relevant experience and resists contrary evidence).
- **BIAS** = the scaffold is an attention or learning bias; the content is filled in by experience.
- **[CONTESTED]** marks findings with failed replications, strong methodological critiques, or genuine open debate.
- **[UNSURE]** marks a detail (year, sample, exact result) I could not verify in this pass. Citations without this mark are ones I am confident of; the 2018-2026 items marked "(checked)" were verified by web search during this survey.

---

## 0. The big frame: core knowledge vs neuroconstructivism vs "priors"

- **Core knowledge** (Spelke & Kinzler 2007, Dev Sci; Spelke 1990s-2000s; Carey 2009 *The Origin of Concepts*): a small set of domain-specific systems (objects, agents, number, places/geometry, possibly social partners) present early, shared with other animals, limited in scope, and later combined by language and learning.
- **Counter-view**: Elman, Bates, Johnson, Karmiloff-Smith, Parisi & Plunkett 1996 *Rethinking Innateness*; Karmiloff-Smith 1998 (TICS, "Development itself is the key"): very little is innate as *representation*; what is innate are architectural and timing constraints (learning rates, connectivity, what gets attended to), and domain specificity emerges through development (neuroconstructivism). Thelen & Smith 1994 dynamic systems is a related view.
- **A middle position that fits the evidence best for agents like ours**: *predispositions + learning*. Vallortigara and colleagues (Versace, Martinho-Truswell, Kacelnik & Vallortigara 2018 TICS "Priors in animal and artificial intelligence: where does learning begin?"; Vallortigara 2021 book *Born Knowing*, MIT Press) argue newborn animals have a few strong, cheap, often low-level attentional predispositions (to faces, biological motion, self-propelled motion, speed changes) that steer imprinting and later learning. Johnson & Morton 1991 (CONSPEC/CONLERN) is the classic two-process version for faces.
- **AI framing**: Lake, Ullman, Tenenbaum & Gershman 2017 (BBS) "Building machines that learn and think like people": "start-up software" = intuitive physics + intuitive psychology, plus compositional, causal model building and learning-to-learn. Zador 2019 (Nature Communications, "A critique of pure learning") argues the genome is a bottleneck that can only encode compact wiring rules / priors, not weights — an argument for small biases over rich answers.
- **2018-2026 shift**: large self-supervised models trained on child-perspective video (Orhan, Gupta & Lake 2020 NeurIPS on SAYCam; Orhan & Lake 2024 Nature Machine Intelligence; Vong, Wang, Orhan & Lake 2024 Science word learning from one child's headcam) show that a lot can be learned from a single child's sensory stream with weak inductive biases. Meanwhile the Wood lab's controlled-rearing chick work shows some abilities are present with *almost no* relevant experience (see sections 2, 5). The two lines together suggest: generic predictive learning gets far, but specific precocial abilities need specific priors or prenatal "training" (spontaneous activity, own movement in ovo/utero).

---

## TIER 1

### 1. Place / geometry / spatial navigation

**What is present early**
- **Rat pups (neural)**: Wills, Cacucci, Burgess & O'Keefe 2010 (Science) and Langston, Ainge, Couey, Canto, Bjerknes, Witter, Moser & Moser 2010 (Science) recorded the first exploration outside the nest (~postnatal day 15-16, around eye opening). Head-direction cells are adult-like almost immediately; place cells are present but immature and sharpen over weeks; grid cells mature last (~3-4 weeks). Head-direction signals exist even *before* eye opening (Tan, Bassett, O'Keefe, Cacucci & Wills 2015 Current Biology; Bjerknes, Langston, Kruge, Moser & Moser 2015 Current Biology [UNSURE exact author list]). Border/boundary-responsive cells appear early, before grid cells (Bjerknes, Moser & Moser 2014 Neuron). Muessig, Hauser, Wills & Cacucci 2015 (Neuron): place-cell stability in open fields improves when grid cells mature; early place cells rely on boundaries. Bjerknes et al. 2018 (PNAS) report early path integration in place cells is poor and improves with age [UNSURE details].
- **Chicks and fish**: dark-reared or visually inexperienced chicks reorient by room geometry (Chiandetti & Vallortigara 2008, and related work by Vallortigara, Sovrano, Lee). Geometry use is broadly shared across species (Cheng 1986 rats: the "geometric module"; Cheng & Newcombe 2005 review).
- **Human children**: Hermer & Spelke 1994 (Nature) — disoriented toddlers reorient by room shape and ignore a colored wall; Lee & Spelke 2010 and Lee, Sovrano & Spelke 2012 — children use 3D boundary distances, not 2D markings or freestanding objects. [CONTESTED] The "geometric module" is strongly qualified: Learmonth, Newcombe & Huttenlocher 2001 showed children use landmarks in larger rooms; adaptive-combination accounts (Newcombe & Huttenlocher; Twyman & Newcombe 2010) treat geometry as a heavily weighted cue, not an encapsulated module.

**ANSWER or BIAS?** Mixed. *Head direction and the boundary-driven skeleton* look like ANSWERS (they work before vision and before navigation experience). *Metric maps* (grid code calibration, place-cell stability, cue combination) are completed by experience. Geometry-preference looks like a strong BIAS toward boundaries in reorientation.

**Minimal computational form**
- A ring (1-D) continuous attractor for heading integrated from self-motion (Skaggs et al. 1995; Zhang 1996), and continuous attractor models of grid cells (Burak & Fiete 2009). Boundary vector cell model (Hartley, Burgess, Lever, Cacucci & O'Keefe 2000).
- Learned versions: grid-like units emerge in recurrent nets trained on path integration (Cueva & Wei 2018 ICLR; Banino et al. 2018 Nature; Sorscher, Mel, Ganguli & Ocko 2019/2023). [CONTESTED] Schaeffer, Khona & Fiete 2022 (NeurIPS, "No free lunch from deep learning in neuroscience") show the grid emergence depends on specific hand-chosen read-out/encoding choices — so "emergence" here partly smuggles in the prior.
- Takeaway for an agent: innate self-motion-to-heading integration plus attention to boundaries is the minimal well-supported scaffold; let the metric map be learned.

**Recent picture change**: neural recordings increasingly show HD signals before eye opening and spontaneous activity shaping circuits pre-experience; the deep-learning "grids emerge" story is now viewed as partly prior-dependent.

### 2. Object permanence and intuitive physics (solidity, continuity, cohesion)

**What is present early**
- **Looking-time studies**: Baillargeon, Spelke & Wasserman 1985 (drawbridge; 5-month-olds); Baillargeon 1987 (3.5-4.5 months); Spelke, Breinlinger, Macomber & Jacobson 1992 (Psych Review; continuity and solidity at 2.5-4 months); Spelke 1990 (cohesion, object segmentation by common motion — Kellman & Spelke 1983: 4-month-olds perceive unity of a center-occluded rod by common motion). Baillargeon's later work: an "explanation-based learning" account where infants acquire event-specific variables (e.g., height of occluder) month by month (Baillargeon 2002; Baillargeon et al. 2009) — i.e., her own model is principles + incremental learning.
- **Action measures lag**: Piaget's search failures until ~8-9 months (A-not-B). Hood & Willatts 1986: 5-month-olds reach in the dark for objects. Munakata, McClelland, Johnson & Siegler 1997 (Psych Review): a PDP model where graded, strengthening representations explain why looking succeeds before reaching.
- [CONTESTED] **Looking-time critiques**: Bogartz, Shinskey & Speaker 1997; Haith 1998 ("Who put the cog in infant cognition?"); Rivera, Wakeley & Langer 1999 (drawbridge result reproduced as a familiarity/novelty preference for more motion); Cashon & Cohen; Schöner & Thelen 2006 (dynamic field model explains habituation results without object concepts). Core debate unresolved but the existence of *some* early expectation of persistence is broadly accepted; the strength and nature ("knowledge" vs perceptual expectation) are not.
- **Chicks**: newly hatched chicks retrieve imprinting objects that disappear behind occluders (Regolin, Vallortigara & Zanforlin 1995; Chiandetti & Vallortigara 2011 Proc R Soc B, "Intuitive physical reasoning about occluded objects by inexperienced chicks"). Wood lab controlled rearing: object permanence develops in chicks reared without any occlusion events (Prasad, Wood & Wood 2019 Dev Sci [UNSURE exact year, 2018/2019]); and **(checked)** Wood, Ullman, Wood, Spelke & Wood 2024 (arXiv 2402.14641, "Object permanence in newborn chicks is robust against opposing evidence"): chicks reared in a world where objects *violated* permanence thousands of times still behaved as if permanence held. Preprint status at time of check; treat as strong but provisional.

**ANSWER or BIAS?** The chick data are the strongest case in this whole survey for an innate ANSWER (works without experience, resists contrary experience). For human infants the data are consistent with an early expectation that experience refines (Baillargeon's own model). Practically: persistence-through-occlusion looks like a genuine prior.

**Minimal computational form**
- Object files / tracking with persistence and spatiotemporal continuity (Kahneman, Treisman & Gibbs 1992; Pylyshyn FINST; Carey & Xu 2001).
- "Physics engine in the head" (Battaglia, Hamrick & Tenenbaum 2013 PNAS; Ullman, Spelke, Battaglia & Tenenbaum 2017 TICS "Mind games"); ADEPT (Smith et al. 2019 NeurIPS).
- Learned with object-centric bias: PLATO (Piloto, Weinstein, Battaglia & Botvinick 2022 Nature Human Behaviour) — learns violation-of-expectation physics only if representations are object-centric. **(checked)** Garrido et al. 2025 (arXiv 2502.11831, Meta FAIR): V-JEPA trained on natural video shows above-chance VoE surprise on IntPhys (high), GRASP and InfLevel (~62%), while pixel-prediction and text-video models are near chance — suggesting prediction in a learned latent space is the key, not a hand-built object prior. A follow-up (checked, authors not recorded here) trained V-JEPA on SAYCam child video and found poor intuitive-physics performance — i.e., the amount/kind of data a single child sees may not suffice without stronger priors.
- Developmental-field view: Schöner & Thelen 2006 dynamic neural fields — persistence as sustained activation.

**Recent**: (a) chick robustness study (prior is rigid); (b) latent-prediction models get VoE "for free" with lots of data, but not from child-scale data. Together: persistence is cheap to give as a prior, expensive to learn.

### 3. Agent detection: self-propelled, contingent, goal-directed

This is really several separable scaffolds. Evidence strength differs sharply among them.

**3a. Self-propulsion / motion-onset animacy cues (perceptual bias)**
- Premack 1990 (Cognition, "The infant's theory of self-propelled objects"): self-propulsion as the primitive cue of intentionality. Leslie 1984; Leslie & Keeble 1987 (6-month-olds perceive Michotte launching causality — reversal of launch is more interesting than reversal of delayed launch). Michotte 1946/1963 launching/entraining.
- Chicks: Mascalzoni, Regolin & Vallortigara 2010 (PNAS) — naive chicks prefer to approach an object that starts moving on its own over one set in motion by collision. Rosa-Salva, Grassi, Lemaire, Vallortigara and colleagues: chicks prefer objects that change speed (sudden acceleration), and the preference disappears if the moment of speed change is occluded (checked, via Vallortigara's summaries) [UNSURE exact paper: Rosa-Salva et al. 2016 PLoS One?]. Also preference for objects moving with the main axis aligned with motion direction.
- Human newborns: **(checked)** Di Giorgio, Lunghi, Vallortigara & Simion 2021 (Scientific Reports, "Newborns' sensitivity to speed changes as a building block for animacy perception"): no preference for a single speed change; preference for abrupt speed-up-then-slow-down but not the reverse. Earlier Di Giorgio et al. 2017 (Dev Sci?) [UNSURE venue]: newborns prefer starting-from-rest self-propulsion over launched motion. Vallortigara notes this could derive from in-utero experience of own movement (not ruled out).

**3b. Contingency between self and other (social contingency)**
- Watson 1972 (contingency awareness: infants smile at a mobile they control); Watson 1979/1985 "contingency detection module"; Gergely & Watson 1999. Bahrick & Watson 1985: 5-month-olds discriminate live (perfectly contingent) video of own legs from a non-contingent recording; Watson's hypothesis — infants first prefer perfect contingency (self), then from ~3 months imperfect/high-but-not-perfect contingency (others, social partners). [CONTESTED in its precise timing; the age-shift claim rests on few studies.]
- Contingency as agent cue: Johnson, Slaughter & Carey 1998 (Dev Sci) — 12-month-olds follow the "gaze" of a faceless furry blob only if it previously behaved contingently with them (beeped/lit in response). Movellan & Watson 2002 (robot learning of social contingency) [UNSURE exact year/venue]. Still-face paradigm (Tronick et al. 1978) shows infants expect contingent responses from caregivers by ~2-3 months. Murray & Trevarthen 1985 double video (delayed mother replay disrupts infant).

**3c. Goal attribution / teleological stance**
- Woodward 1998 (Cognition): 6- and 9-month-olds encode a reaching hand's *goal object* not its path; not for a rod or claw (agent-specificity). Sommerville, Woodward & Needham 2005: 3-month-olds' "sticky mittens" grasping experience produces goal encoding — **action experience causally helps**, a key finding for learning accounts. Skerry, Carey & Spelke 2013 (PNAS) similar with efficient-action expectations.
- Gergely, Nádasdy, Csibra & Bíró 1995 (Cognition); Csibra, Gergely, Bíró, Koós & Brockbank 1999; Csibra 2008 (Cognition): 12-month-olds (and some 6-9-month studies) expect agents to take the efficient path to goals given constraints; Csibra 2008 showed goal attribution to a featureless box. Luo & Baillargeon 2005 (Psych Sci): 5-month-olds attribute goals to a self-propelled box. Liu, Ullman, Tenenbaum & Spelke 2017 (Science): 10-month-olds infer value of goals from cost an agent pays.
- Saxe, Tenenbaum & Carey 2005 (Psych Sci) "Secret agents": 10-12-month-olds infer a hidden agent as the cause when an inert object (beanbag) flies out from behind a screen — inanimate things need a pushing cause; agents need none.
- [CONTESTED] Social evaluation (Hamlin, Wynn & Bloom 2007, Nature: preference for helpers): **(checked)** ManyBabies 4 (Lucca et al., Developmental Science, 2025, 37 labs, 567 infants 5.5-10.5 months): ~49% chose the helper; no preference — effect weaker, absent, or later than thought. Goal attribution itself (Woodward 1998) has replicated reasonably in many labs, though the youngest-age claims (3-5 months) are thinner. A large-scale ManyBabies-style replication of Woodward-type goal attribution was under way [UNSURE status].

**ANSWER or BIAS?**
- 3a self-propulsion/speed change: BIAS (attention/approach predisposition), present at birth in chicks and newborns; very cheap.
- 3b contingency: BIAS / learning mechanism (a detector of conditional probabilities between own action and world events) that *produces* the self-vs-other and agent categories.
- 3c teleological expectations: debated. Gergely & Csibra argue for an early non-mentalistic teleological "stance" (an inference schema = more ANSWER-like); Woodward and Sommerville argue it is bootstrapped from the infant's own goal-directed action (BIAS + experience). Evidence (sticky mittens) supports at least a strong experiential contribution.

**Minimal computational form**
- Inverse planning: Baker, Saxe & Tenenbaum 2009 (Cognition, "Action understanding as inverse planning"); Baker, Jara-Ettinger, Saxe & Tenenbaum 2017 (Nature Human Behaviour, BToM); Jara-Ettinger, Gweon, Schulz & Tenenbaum 2016 (TICS, naive utility calculus): assume agents act approximately efficiently to maximize reward minus cost; invert to infer goals.
- Learned theory of mind: Rabinowitz, Perbet, Song, Zhang, Eslami & Botvinick 2018 (ICML, "Machine Theory of Mind"): a meta-learned "ToMnet" predicts other agents' next actions from a few observed trajectories, forms an agent-character embedding, and passes a gridworld Sally-Anne test. Shows goal-prediction can be learned from *watching* agents with a generic architecture. Benchmarks: Gandhi, Stojnic, Lake & Dillon 2021 (NeurIPS, Baby Intuitions Benchmark, BIB) and Stojnić et al. 2023 (Cognition) — learned models fail several infant-level goal/efficiency tests that infants pass.
- **Mover detection as a bootstrap**: Ullman, Harari & Dorfman 2012 (PNAS, "From simple innate biases to complex visual concepts"): a single innate bias — detect "mover" events (an image region moves and causes another region to start moving) — lets a learner discover hands from natural video, then gaze direction (hands → faces → where people look). One of the best existence proofs that a tiny agent/causal-motion bias bootstraps a lot.
- Curiosity-driven animate attention: Haber, Mrowca, Wang, Fei-Fei & Yamins 2018 (NeurIPS, "Learning to play with intrinsically-motivated self-aware agents"); **(checked)** Kim, Sano, De Freitas, Haber & Yamins 2020 (ICML, "Active world model learning with progress curiosity"): a learning-progress curiosity signal (γ-Progress) makes an agent attend to other agents (animate, complex-but-learnable dynamics) over both random noise and simple inanimate dynamics, emulating infant preference for animate entities. Sano et al. 2020 CogSci extends with peekaboo, chasing, mimicry agents.
- Contingency detection in robots: Movellan & Watson; Gold & Scassellati 2009 (robot self-recognition by kinesthetic-visual contingency/timing) [UNSURE year].

**Recent**: ManyBabies 4 null for helper/hinderer; Di Giorgio et al. 2021 refined newborn speed-change bias; Wood lab and Yamins lab work on emergence of animacy preferences in learners; BIB shows ML gaps on goal/efficiency inference.

---

## TIER 2

### 4. Gaze following and joint attention

**What is present early**
- Newborns prefer faces with direct (mutual) gaze (Farroni, Csibra, Simion & Johnson 2002 PNAS) and show gaze-cued attention shifts with a moving pupil in a schematic face (Farroni, Massaccesi, Pividori & Johnson 2004) — but only when the pupils visibly *move*, suggesting motion cueing, not gaze understanding.
- Real gaze following emerges ~3-6 months for nearby targets, improving through 12-18 months (Scaife & Bruner 1975; Butterworth & Jarrett 1991; D'Entremont et al. 1997). Brooks & Meltzoff 2005: 10-month-olds follow open-eyed vs closed-eyed heads differently; Meltzoff & Brooks 2008: own blindfold experience changes infants' gaze following (experience shapes it).
- Senju & Csibra 2008 (Current Biology): 6-month-olds follow gaze more after ostensive signals (eye contact, infant-directed speech) — part of natural pedagogy.
- Chicks and other animals follow gaze in various forms; ravens, goats, tortoises reported [UNSURE strength].

**ANSWER or BIAS?** BIAS (eye-contact preference + motion-cue orienting) completed by learning. Strong computational and behavioral evidence that gaze following is learned from the contingency between where a caregiver looks and where interesting things appear.

**Minimal computational form**
- Triesch, Teuscher, Deák & Carlson 2006 (Dev Sci, "Gaze following: why (not) learn it?"): a reward-driven learner that finds looking at caregiver's face mildly rewarding and interesting objects rewarding learns gaze following from the regularity that caregivers look at interesting things; the model predicts deficits when caregiver-gaze reliability or face preference is altered. Nagai, Hosoda, Morita & Asada 2003 (Connection Science) robot learns joint attention from self-evaluated success. Ullman et al. 2012 learns gaze direction from hand-object contact events.
- So: minimal = face/eye attention bias + predictive/reward learning on "where interesting things appear after the other's head turns."

### 5. Imitation readiness

**What is present early**
- Meltzoff & Moore 1977 (Science), 1983: newborns imitate tongue protrusion, mouth opening, etc. Meltzoff & Moore's AIM (active intermodal mapping) hypothesis = an innate supramodal body map.
- [CONTESTED, strongly] Jones 2009 (Phil Trans): tongue protrusion is a general arousal response (also to flashing lights, music). Oostenbroek et al. 2016 (Current Biology): longitudinal study of 106 infants at 1, 3, 6, 9 weeks, 9 gestures — no evidence of imitation beyond matching base rates. Meltzoff et al. 2018 (Dev Sci) re-analysis and rebuttal arguing the design diluted the tongue effect; Oostenbroek et al. 2018 reply. **(checked)** Davis, Redshaw, Suddendorf, Nielsen, Kennedy-Costantini, Oostenbroek & Slaughter 2021 (Perspectives on Psychological Science): meta-analysis of 336 effect sizes from 33 samples / 26 articles: pooled d = 0.68 but huge heterogeneity, no methodological moderator explained it, *researcher affiliation did*. Authors leave open "real but fragile" vs "does not exist, researcher degrees of freedom". The safest reading: neonatal imitation is not established.
- Clear imitation emerges later: deferred imitation by 6-9 months (Meltzoff 1988; Barr, Dowden & Hayne 1996); selective/rational imitation at 14 months (Gergely, Bekkering & Király 2002 Nature, head-touch lamp) — [CONTESTED] Paulus, Hunnius, Vissers & Bekkering 2011 and Beisert et al. 2012 argue motor-resonance/action-feasibility explanations. Over-imitation in children vs chimpanzees (Horner & Whiten 2005; Lyons, Young & Keil 2007).
- Heyes 2016 / 2018 *Cognitive Gadgets*; Ray & Heyes 2011 (associative sequence learning, ASL): imitation is learned from correlated experience of doing and seeing (mirrors, caregivers imitating the infant — parents imitate infants far more than vice versa), mirror neurons are products of this learning (Cook, Bird, Catmur, Press & Heyes 2014 BBS).

**ANSWER or BIAS?** Best-supported: BIAS (attention to others' actions + strong contingency/correlation learning), with matching learned. The innate-supramodal-map ANSWER view is weakened by the replication record.

**Minimal computational form**: ASL = Hebbian association between own-motor and seen-motion codes when they co-occur (from self-observation and being imitated). In robots: Asada and colleagues' work on learning self-other mapping via caregiver imitation (Asada et al. 2009 IEEE TAMD "Cognitive developmental robotics: a survey"); Demiris & Hayes; Cangelosi & Schlesinger 2015 *Developmental Robotics* covers this. Inverse-model approaches: imitation as using one's own forward/inverse models to explain others (Wolpert, Doya & Kawato 2003 Phil Trans).

### 6. Social learning, natural pedagogy, attachment/imprinting

**Natural pedagogy** (Csibra & Gergely 2006, 2009 TICS, 2011 Phil Trans): humans have a dedicated system to receive generic knowledge from ostensive communication (eye contact, infant-directed speech, contingent response, name), followed by referential cues; Yoon, Johnson & Csibra 2008 (PNAS): 9-month-olds encode object identity when shown ostensively, location when not. Topál et al. 2008 (Science): A-not-B errors partly induced by the experimenter's communicative cues. [CONTESTED] Heyes 2016 (Developmental Science?, "Born pupils? Natural pedagogy and cultural pedagagy") [UNSURE venue]: the sensitivity to ostensive cues is learned and culturally variable; replication of some pedagogy effects is uneven. Status: the cues clearly modulate infant attention; whether a dedicated innate system exists is debated.

**Attachment and filial imprinting**
- Bowlby 1969; Harlow & Zimmermann 1959 (contact comfort over food). Imprinting (Lorenz; Bolhuis & Honey 1998; Horn 2004): chicks have a predisposition (Johnson & Horn 1988 — preference for head/neck region of hens) separate from an imprinting learning system — a direct template for "bias + learning" (and the inspiration for CONSPEC/CONLERN).
- Rosa-Salva, Mayer, Versace, Hébert, Lemaire & Vallortigara 2021 (Cognition) [UNSURE exact title]: sensitive periods for social development arise from the interaction of predisposed and learned mechanisms.
- Wood lab: newborn chicks' imprinting + view-invariant object recognition from a single imprinted object (Wood 2013 PNAS); **(checked)** Pandey, Lee, Wood & Wood 2024 (PLOS Computational Biology, "Parallel development of object recognition in newborn chicks and deep neural networks"): CNNs and transformers trained on first-person images from digital twins of the chick rearing chambers develop the same view-invariant recognition as chicks, even from one object over a 60° range — i.e., generic learners + embodied temporal-contiguity data can match chicks *for this ability*. Lee, Gujarathi & Wood 2021 (arXiv 2112.06106 [UNSURE exact author list]) built a "newborn embodied Turing test"; a 2023 preprint found RL agents with intrinsic motivation imprint-like but fall far short of chick object recognition; the ICLR 2024 "Newborn Embodied Turing Test" for object segmentation (Garimella et al.? [UNSURE]) reports gaps too. Overall the Wood program finds: some chick abilities are reproducible with generic learners on the chick's data (object recognition), others look like strong priors (object permanence robustness).

**Social learning by observation**: observational fear learning (see 7), local/stimulus enhancement (attention drawn to where conspecifics act) is the cheapest and phylogenetically widest form (Heyes 1994; Galef). Rabinowitz 2018 and Ndousse, Eck, Levine & Jaques 2021 (ICML, "Emergent social learning via multi-agent reinforcement learning") show RL agents learn from experts only if given an auxiliary predictive (model-based) loss — a generic prediction pressure rather than a dedicated imitation module.

**ANSWER or BIAS?** BIAS throughout (attention to conspecifics/ostensive cues + generic learning). Imprinting predispositions are innate biases with a sensitive period.

---

## TIER 3

### 7. Prepared fear learning (snakes, spiders, heights, strangers)

- Seligman 1971 (preparedness); Öhman & Mineka 2001 (Psych Review, "Fears, phobias, and preparedness: toward an evolved module of fear and fear learning"). Cook & Mineka 1989/1990: lab-reared rhesus monkeys acquire fear of (toy) snakes by watching a fearful conspecific video, but not of flowers or toy rabbits — selective associability (strong evidence for prepared *learning*, not innate fear). Mineka et al. 1984: lab-reared monkeys do not initially fear snakes.
- Infants: LoBue & DeLoache 2008/2010: preschoolers and infants detect snakes faster / match fear voices to snakes. Hoehl, Hellmer, Johansson & Gredebäck 2017 (Frontiers in Psychology): 6-month-olds' pupils dilate more to snakes/spiders than flowers/fish. [CONTESTED] LoBue & Adolph 2019 (Developmental Psychology, "Fear in infancy: lessons from snakes, spiders, heights, and strangers"): argue there is no evidence of innate *fear* of any of these in infants — there is enhanced attention/detection, and fear is learned; reviews of stranger anxiety similarly show large individual and contextual variation.
- Heights: Gibson & Walk 1960 visual cliff. Campos, Bertenthal & Kermoian 1992 (Psych Science): wariness of heights (heart-rate acceleration, avoidance) follows locomotor experience, not age; infants given walker experience show it earlier. Adolph 2000 (Psych Science) and Adolph's later work: infants' avoidance of drop-offs is learned *per posture* (sitting vs crawling vs walking) and does not transfer — they are not afraid, they learn affordances; Dahl et al. 2013 (Psych Science): visual proprioception from self-produced locomotion mediates wariness [UNSURE detail]. Many species (chicks, goats) avoid the cliff on first locomotion, so precocial animals may have more of an ANSWER here than altricial humans.

**ANSWER or BIAS?** For humans: BIAS (attention/fast detection + prepared associability), fear itself is learned, often socially (social referencing: Sorce, Emde, Campos & Klinnert 1985 visual-cliff mothers' faces). For precocial animals some avoidance may be innate.

**Minimal form**: weighted associability — certain stimulus classes (sudden looming, specific shapes) have a higher learning rate when paired with aversive outcomes or a conspecific's fear signal; looming avoidance is a genuine innate reflex across species (Ball & Tronick 1971 infants; Schiff 1965; mouse superior colliculus looming circuits, Yilmaz & Meister 2013 Current Biology).

### 8. Face bias

- Goren, Sarty & Wu 1975; Johnson, Dziurawiec, Ellis & Morton 1991 (Cognition): newborns track a schematic face further than scrambled. Morton & Johnson 1991 (Psych Review): CONSPEC (subcortical, innate, coarse three-blob template, guides orienting) + CONLERN (cortical, learns specific faces from the input CONSPEC supplies). Johnson, Senju & Tomalski 2015 (Neurosci Biobehav Rev) updated version.
- [CONTESTED mechanism] Simion, Valenza, Macchi Cassia, Turati & Umiltà 2002; Turati 2004; Macchi Cassia, Turati & Simion 2004: the newborn preference is a general "top-heavy" bias (more elements in the upper half) + congruence with contour, not a face template.
- Fetus: Reid, Dunn, Young, Amu, Donovan & Reissland 2017 (Current Biology): third-trimester fetuses turned heads more toward upright face-like light patterns. **(checked)** A correction was later issued (tabular error in reported means/SDs), and a critique paper argued three substantial methodological problems [UNSURE authors]. **(checked)** Ronga et al. 2025 (Developmental Science, "At first sight: fetal eye movements reveal a preference for face-like configurations"): reports a fetal face-like preference from about 26 weeks using ultrasound-tracked eye movements — a partial independent support. Treat fetal evidence as suggestive, methodologically fragile.
- Newborn cortex: Buiatti et al. 2019 (PNAS): face-like patterns evoke a specific cortical (EEG frequency-tagged) response in newborns [UNSURE exact details].
- Chicks: Rosa-Salva, Regolin & Vallortigara 2010 (Dev Sci): visually naive chicks prefer face-like configurations; also in tortoise hatchlings (Versace et al. 2020). 
- Monkeys: Sugita 2008 (PNAS): monkeys reared 6-24 months without seeing faces still preferred faces (and could discriminate them). Arcaro, Schade, Vincent, Ponce & Livingstone 2017 (Nature Neuroscience): monkeys raised without face exposure *did not develop face-selective cortical patches* (but developed hand-selective ones) — face domains need experience; a proto-organization (retinotopy, eccentricity) is innate. So: an orienting bias may be innate; face-specialized cortex is built by experience (consistent with CONSPEC/CONLERN and with neuroconstructivism).

**ANSWER or BIAS?** Clearly BIAS (coarse orienting template) + learning.

**Minimal form**: a coarse low-spatial-frequency "top-heavy / two-blobs-over-one" orienting prior, or simply attention to self-propelled, contingent things that happen to have faces (faces then become learned predictors of agency). Computationally, face-selectivity also emerges in deep nets trained on generic object recognition (various 2020-2022 studies) [UNSURE specific citation], supporting the "learned given exposure" reading.

---

## OUTSIDE THE LIST (evidence says these matter)

### 9. Contingency detection and the self-other distinction

- Watson's contingency detection (above). Rovee & Rovee 1969 / Rovee-Collier conjugate reinforcement: 2-3-month-olds learn to kick to move a mobile tied to the leg and remember for days/weeks; extinction can cause distress — infants care about control.
- Rochat & Hespos 1997: newborns root more to an external finger than to self-touch ("double touch") — an early self/non-self discrimination [replication base small]. Rochat & Morgan 1995: 3-5-month-olds prefer viewing a non-ego-centric (mirror-reversed) view of their legs — they detect self-produced spatial congruence. Rochat 2003 (Consciousness & Cognition, "Five levels of self-awareness"). Filippetti, Johnson, Lloyd-Fox, Dragovic & Farroni 2013 (Current Biology): newborns prefer synchronous visuo-tactile stimulation of their own face (body ownership cue). Mirror self-recognition ~18-24 months (Amsterdam 1972; Gallup 1970 chimps); [CONTESTED cross-culturally: Broesch et al. 2011 found much lower pass rates in some non-Western samples].
- **ANSWER or BIAS?** A learning mechanism (contingency/synchrony detector) + preference for self-caused effects. Self-other is constructed from it.
- **Minimal form**: compare predicted sensory consequences of own motor commands (efference copy / forward model) with observed changes; perfectly predicted = self, imperfectly contingent = other agent, uncorrelated = inanimate/background (Blakemore, Wolpert & Frith 1998-2000 on self-tickle; Gold & Scassellati; Lanillos & Cheng 2018 "Adaptive robot body learning and estimation through predictive coding" [UNSURE exact title]).

### 10. Biological motion preference

- Simion, Regolin & Bulf 2008 (PNAS): 2-day-old newborns prefer a point-light hen walking over random motion, and upright over inverted — even with a non-human (hen) display, suggesting a generic, non-species-specific predisposition. Vallortigara, Regolin & Marconato 2005 (PLoS Biology): naive chicks prefer biological motion, including of a cat (non-conspecific). Bardi, Regolin & Simion 2011 (Dev Sci) [UNSURE]: newborns prefer point-light displays with a coherent center of mass; Bidet-Ildei et al. 2014 [UNSURE]. Some sensitivity is carried by local foot-motion cues (Troje & Westhoff 2006, "life detector").
- **ANSWER or BIAS?** BIAS (attention to "life-like" local motion with gravity-consistent acceleration).
- **Minimal form**: a "life detector" keyed to local ballistic/pendular limb motion that is asymmetric under inversion (gravity), as proposed by Troje & Westhoff 2006.

### 11. Statistical learning and generic prediction

- Saffran, Aslin & Newport 1996 (Science): 8-month-olds segment speech by transitional probabilities after 2 minutes. Kirkham, Slemmer & Johnson 2002 (Cognition): visual sequences at 2 months. Fiser & Aslin 2002 (PNAS) spatial co-occurrence. Teinonen et al. 2009 (BMC Neuroscience): newborns segment speech streams (ERP). Bulf, Johnson & Valenza 2011 (Cognition): newborns learn visual statistics if sequences are short.
- Kidd, Piantadosi & Aslin 2012 (PLoS One, "Goldilocks effect"): infants look away from sequences that are too predictable or too unpredictable — attention is governed by information/predictability, a ready-made curiosity signal. Stahl & Feigenson 2015 (Science): 11-month-olds learn more about, and explore more, objects that violated expectations (checked; I found no published failed replication in this pass, but it has not been put through a ManyBabies-scale test to my knowledge [UNSURE]).
- **Reading**: perhaps the single most general and best-replicated "scaffold" — a domain-general predictive learner whose attention is drawn to intermediate predictability and to violations. Strongly BIAS/mechanism, not ANSWER.
- **Minimal form**: online prediction + attention allocated to learning progress (Oudeyer, Kaplan & Hafner 2007 IEEE Trans. Evol. Comp. "Intrinsic motivation systems for autonomous mental development"; Gottlieb, Oudeyer, Lopes & Baranes 2013 TICS; Schmidhuber 1991/2010 compression progress; Kim et al. 2020 γ-Progress). Oudeyer & Smith 2016 (Topics in Cog Sci) show how such curiosity yields developmental stages (Playground experiment).

### 12. Number / magnitude

- Izard, Sann, Spelke & Streri 2009 (PNAS): newborns match approximate numerosity across audition and vision (4 vs 12). Xu & Spelke 2000 (Cognition) 6-month-olds 8 vs 16 at 1:2 ratio. Wynn 1992 (Nature) addition/subtraction — [CONTESTED] (Wakeley, Rivera & Langer 2000 failed to replicate; continuous-extent confounds, Clearfield & Mix 1999). Chicks track small numbers (Rugani, Vallortigara et al. 2009). Probably a magnitude BIAS; not top priority for the creature but relevant to "more food".

### 13. Play

- Play is widespread in mammals and birds (Burghardt 2005 *The Genesis of Animal Play*). Špinka, Newberry & Bekoff 2001 (Quarterly Review of Biology, "Mammalian play: training for the unexpected"): play deliberately creates loss of control and recovery. Chu & Schulz 2020 (Annual Review of Developmental Psychology, "Play, curiosity, and cognition"). Rat play has dedicated circuits (Panksepp; tickling, Burgdorf & Panksepp). Computational: play ≈ intrinsically motivated exploration in safe states (Haber et al. 2018 "learning to play"; Oudeyer). BIAS, gated by safety/satiation.

### 14. Infant false belief ("implicit theory of mind") — mostly not a newborn scaffold, but often cited

- Onishi & Baillargeon 2005 (Science): 15-month-olds look longer when an agent searches where the toy is but where she should not believe it to be. Southgate, Senju & Csibra 2007 (Psych Sci) anticipatory looking at 25 months; Kovács, Téglás & Endress 2010 (Science) 7-month-olds; Surian, Caldi & Sperber 2007.
- [CONTESTED, strongly] Many non-replications: Kulke, Reiß, Krist & Rakoczy 2018 and Kulke & Rakoczy 2018 review (Cognitive Development); Dörrenberg, Rakoczy & Liszkowski 2018; Powell, Hobbs, Bardis, Carey & Saxe 2018 (Cognitive Development, replications of violation-of-expectation false belief: partial); Phillips et al. 2015 (Kovács effect explained by timing artifact). Heyes 2014 (Perspectives on Psych Science, "False belief in infancy: a fresh look"): low-level novelty explanations. **(checked)** ManyBabies 2 (Schuwerk et al., multi-lab, anticipatory looking, knowledge vs ignorance): adults (n ≈ 703) showed clear epistemic-state-based anticipation; toddlers (18-27 months, n ≈ 521) showed goal-based anticipation but did *not* differentiate knowledge vs ignorance — inconclusive / no evidence for the infant implicit ToM claim at that age [UNSURE final journal citation; found as Lancaster eprint]. 
- Reading for the creature: goal-based prediction of others is solid; belief attribution in infancy is not a safe target to copy.

---

## Summary table

| Scaffold | Earliest age / precocial evidence | ANSWER vs BIAS | Robustness of evidence |
|---|---|---|---|
| Head direction / boundary use | Rat pups before eye opening; dark-reared chicks | ANSWER-like skeleton; map learned | Strong (neural) |
| Geometry reorientation | Chicks naive; children ~18-24 mo | Strong cue BIAS | Moderate; "module" contested |
| Object permanence | Chicks naive and robust to contrary data; human looking 2.5-5 mo | ANSWER (chicks); early expectation + learning (humans) | Chicks strong; infant looking-time contested |
| Self-propulsion / speed change | Chicks naive; newborns | BIAS | Moderate-strong |
| Social contingency | 2-3 mo (still-face, live video) | Learning mechanism | Strong for effect, weaker for ages |
| Goal attribution / efficiency | 3-12 mo; helped by own action | Debated; experience contributes | Moderate; social evaluation (helper) failed ManyBabies 4 |
| Gaze following | Eye-contact pref at birth; following 3-6 mo+ | BIAS + learning | Strong |
| Neonatal imitation | Claimed at birth | Not established | Weak, contested (Oostenbroek 2016; Davis 2021) |
| Natural pedagogy | ~6-9 mo | Debated | Moderate, contested |
| Imprinting / attachment | Chicks hours; human months | BIAS with sensitive period | Strong |
| Prepared fear | Attention to snakes ~6 mo; fear learned | BIAS (associability) | Strong in monkeys; innate fear in infants not supported |
| Fear of heights | After locomotion | Learned per posture | Strong (Campos 1992, Adolph) |
| Face bias | Newborns, chicks, maybe fetus | BIAS (coarse template / top-heavy) | Strong for orienting; fetal fragile; cortex needs experience (Arcaro 2017) |
| Biological motion | Newborns, chicks | BIAS | Moderate-strong |
| Statistical learning / Goldilocks | Newborns | Domain-general mechanism | Strong |
| Self-other via contingency | Newborn double-touch, visuo-tactile synchrony | Learning mechanism | Moderate |
| Infant false belief | 7-15 mo claimed | — | Weak; ManyBabies 2 toddler null |

---

## What would an agent-detection scaffold minimally be, for a creature with only an eye (first-person image), a felt-motion sense, and inside levels?

Not a design; candidate *biases* that the literature supports as small, cheap, and generative:

1. **Attention to self-initiated motion onset and abrupt speed change in the visual field, after subtracting own-motion.** Chicks (Mascalzoni et al. 2010; speed-change studies) and human newborns (Di Giorgio et al. 2017, 2021) orient to things that start from rest without a visible cause or that accelerate abruptly. For a moving first-person eye this only makes sense *after* discounting the image motion predicted by one's own felt motion (efference copy / optic-flow prediction) — the residual motion that no external impact explains is the "self-propelled" cue (Premack 1990; Saxe et al. 2005 "secret agents" uses the converse: inert things need a cause). Ullman et al. 2012's "mover" bias (a moving region that sets another region in motion) is a closely related one-line prior that bootstrapped hands and gaze from video.

2. **A contingency detector between own action and seen change, sorting the world into self / responsive other / inert.** Watson's contingency analysis (Watson 1972, 1985; Bahrick & Watson 1985; Gergely & Watson 1999): perfectly predicted visual change = own body; *imperfect but reliable* response to one's own actions = a social agent (Johnson, Slaughter & Carey 1998: infants treat a faceless blob as having a "gaze" only if it reacted contingently); no contingency = background. Computationally this is the residual of a forward model plus a running statistic of how much a seen thing's motion depends on one's own recent actions. It doubles as a self-recognition scaffold (Rochat; Filippetti et al. 2013).

3. **A drive to predict other movers' trajectories, with attention allocated by learning progress, under an efficiency (goal-directed) assumption.** Infants expect agents to move efficiently toward targets (Gergely et al. 1995; Csibra 2008; Liu et al. 2017), and this is improved by own goal-directed action experience (Sommerville et al. 2005) — suggesting "predict others with the same machinery you use to reach your own goals" (inverse planning: Baker, Saxe & Tenenbaum 2009; learned form: Rabinowitz et al. 2018). Learning-progress curiosity alone already pulls attention toward other agents because their dynamics are complex but learnable (Kim et al. 2020; Haber et al. 2018; Kidd et al. 2012 Goldilocks in infants). For a creature with inside levels, the natural "goal" to test first is: does that mover end up where the things that restore *my* inside levels are? (This is a hypothesis linking the literature to the creature, not a finding.)

Optional fourth, weaker evidence: a coarse **biological-motion / face-like orienting bias** (Simion et al. 2008; Johnson & Morton CONSPEC). These are well documented but are tied to specific body plans; for rendered peers that share the creature's own body, contingency and self-propulsion (1-2) are more general and better supported as *cause* of agent categorization, with faces learned as predictors.

Cautions from the literature: (a) helper/hinderer social evaluation did not replicate at scale (ManyBabies 4, 2025), so do not use it as a target test; (b) neonatal imitation is not established, so imitation should be expected to emerge from contingency/association learning (Heyes ASL), not be present at birth; (c) belief-tracking in toddlers did not show up in ManyBabies 2 — goal prediction, not belief, is the right early benchmark.

---

## Key computational/developmental-robotics references (by role)

- Start-up software: Lake, Ullman, Tenenbaum & Gershman 2017 BBS.
- Inverse planning: Baker, Saxe & Tenenbaum 2009 Cognition; Baker et al. 2017 Nat Hum Behav; Jara-Ettinger et al. 2016 TICS.
- Learned ToM: Rabinowitz et al. 2018 ICML.
- Intrinsic motivation / learning progress: Oudeyer, Kaplan & Hafner 2007; Gottlieb et al. 2013 TICS; Kim et al. 2020 ICML; Haber et al. 2018 NeurIPS; Pathak et al. 2017 ICML (prediction-error curiosity; white-noise problem).
- Developmental robotics: Asada et al. 2009 IEEE TAMD; Cangelosi & Schlesinger 2015 *Developmental Robotics: From Babies to Robots* (MIT Press); Weng et al. 2001 Science ("Autonomous mental development by robots and animals").
- Counter-view / constructivism: Elman et al. 1996; Karmiloff-Smith 1998; Thelen & Smith 1994; Schöner & Thelen 2006.
- Priors in animals vs AI: Versace et al. 2018 TICS; Vallortigara 2021; Zador 2019 Nat Comms; Wood lab (Wood 2013 PNAS; Pandey et al. 2024 PLOS Comp Bio; Wood et al. 2024 arXiv).
- Infant-like ML benchmarks: Gandhi et al. 2021 BIB; Stojnić et al. 2023 Cognition; IntPhys (Riochet et al. 2018/2021); Garrido et al. 2025.
- Child-perspective SSL: Orhan, Gupta & Lake 2020; Orhan & Lake 2024 NMI; Vong et al. 2024 Science; Sheybani et al. 2023 (infant egocentric video curricula) [UNSURE venue].

## Web sources checked in this pass

- ManyBabies 4 helper/hinderer: https://dare.uva.nl/id/6bd26b80-9423-4dfb-9c14-acaa7e3f0d3a ; https://osf.io/qntyd/wiki
- ManyBabies 2: https://eprints.lancs.ac.uk/id/eprint/235468 ; https://www.lmu.de/psy/en/chairs/clinical-psychology-of-childhood-and-adolescence/research/manybabies2/
- Reid et al. 2017 and correction: https://eprints.lancs.ac.uk/id/eprint/86654 ; https://insight.cumbria.ac.uk/id/eprint/3670 ; critique: https://pubmed.ncbi.nlm.nih.gov/29787718/
- Ronga et al. 2025 fetal eye movements: https://iris.unito.it/ (Developmental Science 2025 PDF)
- Wood et al. 2024 object permanence: https://arxiv.org/abs/2402.14641
- Pandey et al. 2024 chicks vs DNNs: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11651591/
- Garrido et al. 2025 V-JEPA intuitive physics: https://arxiv.org/abs/2502.11831
- Davis et al. 2021 neonatal imitation meta-analysis: https://www.psychologicalscience.org/journals/perspectives/1745691620959834/ ; https://langcog.github.io/metalab/dataset/neonatal_imitation
- Stahl & Feigenson 2015: https://pubmed.ncbi.nlm.nih.gov/25838378/
- Kim et al. 2020 progress curiosity: https://proceedings.mlr.press/v119/kim20e.html
- Di Giorgio et al. 2021 newborn speed change: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7803759/
