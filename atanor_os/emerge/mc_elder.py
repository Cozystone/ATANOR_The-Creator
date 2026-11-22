"""목소리 어른 v0 (창립자 2026-11-22; 장부: 손으로 준 어른, 두뇌는 로컬 SLM, 차츰 뗀다). 존재의 마음 밖에 있다.
아기가 이미 바라보는 것에 이름을 붙여 준다 (따라가며 이름 붙이기, Tomasello & Farrar 1986): 바로 앞 빛살이 닿은 짐승 (12칸 안) 또는 가까운 블록 (3칸 안).
어른만 기호를 안다 (게임의 이름). 말은 로컬 SLM (Ollama, exaone3.5)이 짓고, 한국어 목소리 (Windows Heami)로 소리가 되어 존재의 귀에 들린다.
존재에게는 글자가 들어가지 않는다: 같은 귀로 듣는 소리 파형뿐.
  from .mc_elder import Elder; el = Elder(w); el.step()   # 걸음마다 (막지 않음, 말 짓기와 소리 만들기는 뒤에서)
"""
import json, os, subprocess, tempfile, threading, time, urllib.request, wave
import numpy as np

MODEL = os.environ.get("ATANOR_ELDER_MODEL", "exaone3.5:7.8b")
VOICE = os.environ.get("ATANOR_ELDER_VOICE", "Microsoft Heami Desktop")
DULL = {"air", "grass", "dirt", "stone", "", "None", "none", "null", "grass_block", "bedrock", "unknown"}
KO = {"sheep": "양", "cow": "소", "pig": "돼지", "chicken": "닭", "horse": "말", "rabbit": "토끼", "wolf": "늑대", "squid": "오징어", "villager": "사람", "bat": "박쥐",   # 어른의 말 (장부)
      "llama": "라마", "ocelot": "고양이", "polar_bear": "곰", "zombie": "좀비", "skeleton": "해골", "creeper": "크리퍼", "spider": "거미", "cave_spider": "거미", "witch": "마녀",
      "enderman": "엔더맨", "slime": "슬라임", "husk": "좀비", "stray": "해골", "drowned": "좀비", "zombie_villager": "좀비", "silverfish": "벌레", "item": "물건",
      "log": "나무", "log2": "나무", "wood": "나무", "oak_wood": "나무", "birch_wood": "나무", "spruce_wood": "나무", "jungle_wood": "나무", "flower": "꽃", "mushroom": "버섯", "sugar_cane": "갈대", "leaves": "나뭇잎", "leaves2": "나뭇잎", "water": "물", "flowing_water": "물", "lava": "용암", "flowing_lava": "용암", "sand": "모래",
      "gravel": "자갈", "red_flower": "꽃", "yellow_flower": "꽃", "tallgrass": "풀", "double_plant": "풀", "brown_mushroom": "버섯", "red_mushroom": "버섯",
      "brown_mushroom_block": "버섯", "red_mushroom_block": "버섯", "cactus": "선인장", "snow_layer": "눈", "snow": "눈", "ice": "얼음", "pumpkin": "호박", "melon_block": "수박",
      "reeds": "갈대", "vine": "덩굴", "vines": "덩굴", "lily_pad": "연잎", "waterlily": "연잎", "seagrass": "풀", "kelp": "풀", "dandelion": "꽃", "poppy": "꽃", "oak_leaves": "나뭇잎", "birch_leaves": "나뭇잎", "spruce_leaves": "나뭇잎", "jungle_leaves": "나뭇잎", "acacia_leaves": "나뭇잎", "dark_oak_leaves": "나뭇잎", "clay": "흙", "sandstone": "모래", "coal_ore": "돌", "iron_ore": "돌", "planks": "나무판", "cobblestone": "돌", "torch": "횃불", "wheat": "밀", "farmland": "밭"}
HOSTILE = {"zombie", "skeleton", "creeper", "spider", "cave_spider", "witch", "enderman", "slime", "husk", "stray", "drowned", "zombie_villager", "silverfish"}


def slm(prompt, timeout=60):
    req = urllib.request.Request("http://127.0.0.1:11434/api/generate", data=json.dumps(dict(model=MODEL, prompt=prompt, stream=False, options=dict(temperature=0.8, num_predict=24))).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())["response"].strip().split("\n")[0].strip(" \"'")


def tts(text):
    """한국어 목소리 -> 모노 44.1kHz float 파형 (Windows 기본 음성, 설치 없음)."""
    d = tempfile.mkdtemp(prefix="atanor_voice_"); p = os.path.join(d, "v.wav")
    ps = ("Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
          f"$s.SelectVoice('{VOICE}'); $f = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(44100, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono); "
          f"$s.SetOutputToWaveFile('{p}', $f); $s.Speak([Console]::In.ReadToEnd()); $s.Dispose()")
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], input=text.encode("utf-8"), capture_output=True, timeout=60)
    with wave.open(p) as f:
        x = np.frombuffer(f.readframes(f.getnframes()), np.int16).astype(np.float32) / 32768.0
    return x * 0.5


class Elder:
    def __init__(self, world, gap=6.0, again=30.0, fade=0.0, steps=1, log=print):
        self.w, self.gap, self.again, self.fade, self.steps, self.log = world, gap, again, fade, steps, log
        self.cache, self.busy = {}, set()                     # 이름마다 지어 둔 말과 그 소리 (여러 가지)
        self.last_t = np.zeros(world.n); self.last_name = [dict() for _ in range(world.n)]
        self.k, self.said, self.spoke = 0, 0, []
        self.rng = np.random.default_rng(0)

    def _make(self, name, kind):
        try:
            key = name.lower().replace("minecraft:", "").replace(" ", "_")
            danger = key in HOSTILE                           # 어른이 아는 것 (장부): 해치는 것
            ko = KO.get(key)
            if ko is None:                                    # 어른의 말에 없는 것은 말하지 않는다 (SLM이 지어낸 이름이 엉뚱했음: 'traffic light' 같은 것, 2026-11-22)
                self.cache[name] = [("", None)]
                return
            # 전보식 (창립자 2026-11-22): 어근만 짧게 끊어 ("양", "나무", "좀비, 위험"). 늘 같은 소리여야 소리와 것이 빨리 이어진다 (홀로 들린 낱말이 배움을 앞당김, Brent & Siskind 2001)
            text = f"{ko}, 위험!" if danger else ko
            text = text[:30]
            wav = tts(text)
            self.cache.setdefault(name, []).append((text, wav))
        except Exception as e:
            self.log(f"elder: could not make words for {name}: {e}")
        finally:
            self.busy.discard(name)

    def step(self):
        self.k += 1
        if self.fade and self.rng.random() < self.fade * self.k / max(self.steps, 1):
            return                                            # 차츰 뗀다: 날이 갈수록 덜 말한다
        now = time.time()
        for i, o in enumerate(self.w.obs):
            lk = o.get("look")
            if not lk or now - self.last_t[i] < self.gap:
                continue
            name, kind = None, None
            norm = lambda s: str(s).lower().replace("minecraft:", "").strip().replace(" ", "_")
            for e, dd in zip(lk["entity"], lk["entity_d"]):
                e = norm(e)
                if e and e not in DULL and dd < 12:
                    name, kind = e, "entity"; break
            if name is None:
                for b, dd in zip(lk["block"], lk["block_d"]):
                    b = norm(b)
                    if b and b not in DULL and dd < 3:
                        name, kind = b, "block"; break
            if name is None or now - self.last_name[i].get(name, -1e9) < self.again:
                continue
            got = self.cache.get(name, [])
            if len(got) < 1 and name not in self.busy:       # 이름마다 한 번 소리를 지어 둔다 (전보식: 늘 같은 낱말)
                self.busy.add(name); threading.Thread(target=self._make, args=(name, kind), daemon=True).start()
            if got and got[0][1] is None:                      # 모르는 이름
                continue
            if got:
                text, wav = got[self.rng.integers(len(got))]
                self.w.voice([i], wav, text=f"몸 {i + 1}: {text}")
                self.last_t[i] = now; self.last_name[i][name] = now; self.said += 1
                self.spoke.append((self.k, i, text))                # 재는 것: 몇 걸음째, 어느 몸에, 무슨 말 (존재는 소리만 들음)
