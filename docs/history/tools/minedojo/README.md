# MineDojo in WSL2 (user space only, no sudo)

Order (each is safe to re-run):
1. `bash mc_setup1.sh` - micromamba, env `mc` with python 3.9 + openjdk 8 (conda-forge), in `~/atanor-mc`
2. `bash mc_xrandr.sh` - Ubuntu's own `x11-xserver-utils` .deb downloaded and unpacked (not installed): old LWJGL calls `xrandr`
3. `bash mc_setup2.sh` - pins (pip<24.1, setuptools 65.5, wheel 0.38 for gym 0.21; numpy<2 for minedojo 0.1), installs minedojo, first run
   - if gradle stops at `MixinGradle:dcfaf61`: `python mc_fix_gradle.py <minedojo>/sim/Malmo/Minecraft/build.gradle` (keeps `.orig`)
   - if Minecraft assets fail with HTTP 400 (old http URL): `python mc_assets.py` (fetches the 1.11 assets over https, sha1-checked)
4. viewer: `bash ../mcwatch.sh` -> http://localhost:8770

First run (2026-11-22): world ready in 24 s, first-person RGB 160x256, life/food 20/20, 42 steps/s (one instance).

## Sound (ear)
Malmo ships `run/options.txt` with every sound category at 0. For the ear, set them to 1.0 (music stays 0):
`sed -i -E "s/^soundCategory_(master|record|weather|block|hostile|neutral|player|ambient|voice):0\.0/soundCategory_\1:1.0/" <minedojo>/sim/Malmo/Minecraft/run/options.txt` (original kept as options.txt.orig).
The world server then runs each body with `ALSOFT_CONF` pointing the game's own OpenAL Soft (1.15.1) at a wave file (`ATANOR_MC_EAR=1`).
