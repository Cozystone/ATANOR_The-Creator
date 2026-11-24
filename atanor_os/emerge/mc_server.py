"""MineDojo 세계 서버 (WSL2 안, MineDojo 환경의 python 3.9에서 돈다). 몸 여러 개를 따로 된 프로세스로 돌리고, 존재(윈도 쪽, GPU)와 로컬 TCP로 잇는다.
존재에게 주는 것: 1인칭 그림 (작게 줄인 것), 몸 (체력, 배부름, 숨), 이번 걸음에 맞은 양, 삶이 끝났는가. 기호 정보(블록, 위치, 가진 것)는 주지 않는다.
예외 하나 (장부): 제 걸음 감각으로 쓰려고 위치·바라보는 쪽의 변화만 몸 감각으로 준다 (절대 위치는 주지 않는다).
행동 (존재 쪽 번호): 0 가만히, 1 앞으로, 2 왼쪽으로 돌기, 3 오른쪽으로 돌기, 4 때리기, 5 쓰기, 6 뛰며 앞으로.
뷰어: 첫 몸의 큰 그림과 몸 상태를 포트 8770에 (존재 쪽이 보낸 '지금 한 것'과 이름표도 함께).
  (WSL) python mc_server.py --bodies 4 --port 8790 --view 8770
"""
import argparse, base64, io, json, pickle, socket, struct, threading, time
import multiprocessing as mp

import os
EYE = os.environ.get("ATANOR_EYE_PATH", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eye", "fovea.py"))
CROP = int(os.environ.get("ATANOR_MC_CROP", "44"))       # 그린 그림 아래에서 잘라 낼 줄 수 (HUD)
POINTER = os.environ.get("ATANOR_MC_POINTER", "/mnt/d/ATANOR-experiments/probe-work/mcviewer.json")      # 판 번호와 지금 판 설명 (bump_mc.py가 고친다)
USE_TICKS = int(os.environ.get("ATANOR_MC_USE_TICKS", "32"))   # 쓰기를 누르고 있는 틱 수 (먹기 = 32)
TURN = 2          # 고개 칸 수 (한 칸 15도): 한 번에 30도
BRANCH_ON = os.environ.get("ATANOR_MC_BRANCH") == "1"
NOCROSS = os.environ.get("ATANOR_MC_NOCROSS") == "1"             # 십자 조준점 지우기 (256x204 그림에서 가로·세로 9칸, 두께 1칸; 넉넉히 3칸 둘레)
H0_ = 204
import numpy as _np
XMASK = _np.zeros((13, 13), bool); XMASK[5:8, 1:12] = True; XMASK[1:12, 5:8] = True
_LEGS = ["서 있음", "앞", "뒤", "왼옆", "오른옆", "뛰며 앞"]; _NECK = ["그대로", "왼쪽 30도", "오른쪽 30도", "위 15도", "아래 15도"]; _HANDS = ["빈손", "때림", "씀"]


def actname(k):
    return f"{_LEGS[k // 15]} / {_NECK[(k // 3) % 5]} / {_HANDS[k % 3]}" if BRANCH_ON else ACTS[k]
ACTS = ["가만히", "앞으로", "왼쪽으로 돌기", "오른쪽으로 돌기", "때리기", "쓰기", "뛰며 앞으로"]


class Ear:
    """달팽이관처럼: 이 몸의 Minecraft가 내는 소리 (게임 안 OpenAL Soft의 '파일로 쓰기' 방식)를, 걸음마다 방금 0.25초만
    왼쪽·오른쪽 귀마다 주파수 띠 8개의 세기 (로그)로. 이름표 없음. 띠의 경계는 사람 귀처럼 로그 간격 (손으로 준 것, 장부).
    파일은 쌓이지 않게 가끔 비운다 (쓰는 쪽은 제 자리에서 계속 쓰므로 빈 앞부분은 디스크를 차지하지 않는 구멍이 된다)."""
    RATE, WIN, NB, HDR = 44100, 11025, 8, 68

    def __init__(self):
        import numpy as np, tempfile
        self.np = np
        d = tempfile.mkdtemp(prefix="atanor_ear_"); self.path = os.path.join(d, "ear.wav")
        conf = os.path.join(d, "alsoft.conf")
        open(conf, "w").write("\n".join(["[general]", "drivers = wave", "sample-type = float32", f"frequency = {self.RATE}", "channels = stereo", "[wave]", f"file = {self.path}", ""]))
        os.environ["ALSOFT_CONF"] = conf; os.environ["ALSOFT_DRIVERS"] = "wave"
        edges = np.geomspace(60, 16000, self.NB + 1); fq = np.fft.rfftfreq(self.WIN, 1 / self.RATE)
        self.band = [(fq >= edges[i]) & (fq < edges[i + 1]) for i in range(self.NB)]; self.win = np.hanning(self.WIN).astype(np.float32)
        self.calls = 0
        self.voice, self.voice_t0 = None, 0.0

    def say(self, wav):
        """어른의 목소리 (모노, 44.1kHz)가 지금부터 이 몸의 귀에 들린다: 게임의 소리와 한 귀에서 섞인다 (글자가 아니라 소리로)."""
        self.voice = self.np.asarray(wav, self.np.float32); self.voice_t0 = time.time()

    def bands(self):
        np = self.np
        self.calls += 1
        x = np.zeros((self.WIN, 2), np.float32)
        if self.voice is not None:                                # 어른의 목소리 가운데 방금 0.25초에 흘러나온 부분 (실제 시간으로)
            e = max(int((time.time() - self.voice_t0) * self.RATE), 0); s0 = max(e - self.WIN, 0); s1 = min(e, len(self.voice))
            if s0 >= len(self.voice):
                self.voice = None
            elif s1 > s0:                                         # 창의 끝 = 지금(e). 목소리의 [s0, s1)는 창의 [WIN-(e-s0), WIN-(e-s1))에 놓인다
                a0 = self.WIN - (e - s0); x[a0:a0 + (s1 - s0)] += self.voice[s0:s1, None]
        try:
            size = os.path.getsize(self.path)
            n = (size - self.HDR) // 8
            if n > self.WIN:
                with open(self.path, "rb") as f:
                    f.seek(self.HDR + (n - self.WIN) * 8); x = x + np.frombuffer(f.read(self.WIN * 8), np.float32).reshape(-1, 2)
            if self.calls % 3000 == 0:
                os.truncate(self.path, 0)
        except OSError:
            pass
        x = np.nan_to_num(x)
        sp = np.abs(np.fft.rfft(x * self.win[:, None], axis=0)) ** 2
        e = np.stack([sp[b].sum(0) for b in self.band], 1)               # [2 귀, 8 띠]
        return np.log10(e + 1e-9).astype(np.float32).ravel()            # [16]: 왼 귀 띠 8, 오른 귀 띠 8


def body(conn, seed, h, w, small, big):
    import numpy as np, minedojo
    from PIL import Image
    import importlib.util
    from PIL import ImageDraw
    ear = Ear() if os.environ.get("ATANOR_MC_EAR") == "1" else None      # 귀: 게임 안의 소리 엔진이 스피커 대신 이 몸의 관(FIFO)으로 쓰게 한다 (minedojo.make 전에)
    food0 = os.environ.get("ATANOR_MC_FOOD")                     # 태어날 때의 배부름 (세계 설정, 견주는 두 판에 똑같이). 없으면 MineDojo 기본 (가득)
    SKIN = os.environ.get("ATANOR_MC_SKIN") == "1"              # 살갗: 몸 둘레 3x3x3 칸이 막는가 / 물인가 (블록 이름은 쓰지 않는다)
    JUDGE_DEPTH = os.environ.get("ATANOR_MC_TRUEDEPTH") == "1"   # 참 깊이: 깊이 눈을 '판정'할 때만 (배우는 데는 쓰지 않는다)
    extra = {}
    if SKIN:
        extra.update(use_voxel=True, voxel_size=dict(xmin=-1, ymin=-1, zmin=-1, xmax=1, ymax=1, zmax=1))
    if JUDGE_DEPTH:                                              # MineDojo 0.1의 깊이 그림은 막혀 있다 (assert): 눈높이 빛살 5개로 잰다 (깊이 눈의 5갈래와 같은 각)
        import math as _m
        HF = _m.degrees(_m.atan(_m.tan(_m.radians(35.0)) * w / h))         # 가로 반시야 (세로 70도에서)
        extra.update(use_lidar=True, lidar_rays=[(0.0, float(_m.radians(HF * (2 * k + 1 - 5) / 5)), 30.0) for k in range(5)])      # Malmo 빛살의 각은 라디안 (Vec3d.rotateYaw). 예전엔 도로 넣어 옆 빛살이 엉뚱한 쪽을 봄 (2026-11-22 검토)
    JUDGE = os.environ.get("ATANOR_MC_JUDGE") == "1"            # 판정에만 (존재에게 주지 않음): 캐서 얻은 것, 시야를 부채꼴로 가르는 빛살 7개의 짐승 (따라가기, 다가가기)
    FAN = [-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0]
    if JUDGE:
        extra.update(use_lidar=True, lidar_rays=list(extra.get("lidar_rays", [])) + [(0.0, float(np.radians(y)), 32.0) for y in FAN])
    NFAN0 = len(extra.get("lidar_rays", [])) - (len(FAN) if JUDGE else 0)
    ELDER = os.environ.get("ATANOR_MC_ELDER") == "1"            # 목소리 어른이 '아기가 바라보는 것'을 알려고 (어른만 기호를 안다, 존재의 마음에는 들어가지 않는다)
    if ELDER:
        extra.update(use_lidar=True, lidar_rays=list(extra.get("lidar_rays", [])) + [(0.0, 0.0, 24.0), (float(np.radians(-12.0)), 0.0, 24.0), (float(np.radians(12.0)), 0.0, 24.0)])
    WTYPE = os.environ.get("ATANOR_MC_WORLD", "default")         # 판정 쪽 세계 꼴 (2026-11-23 계곡: 'flat' 위에 판정 명령으로 짓는다)
    env = minedojo.make(task_id="open-ended", image_size=(h, w), generate_world_type=WTYPE, world_seed=seed, allow_mob_spawn=True,
                        **({"start_food": int(food0)} if food0 else {}), **extra)

    def skin(obs, yaw):
        """[15]: 닿음 8 (발높이·머리높이 x 앞·오른쪽·뒤·왼쪽: 막는 것이 있나) + 발밑이 막혀 있나 + 발·머리가 물에 잠겼나 2
        + 아픔이 온 쪽 4 (앞·오른쪽·뒤·왼쪽, 맞은 양만큼). 방향은 바라보는 쪽에 맞춰 가장 가까운 네 방위로 돌림 (장부)."""
        out = np.zeros(15, np.float32)
        vx = obs.get("voxels")
        if vx is not None:
            mv = np.asarray(vx["blocks_movement"]).astype(np.float32); lq = np.asarray(vx["is_liquid"]).astype(np.float32)      # [x, y, z] (-1..1)
            q = int(np.round(((yaw % 360) / 90.0))) % 4                         # 0: +z를 봄, 1: -x, 2: -z, 3: +x (Minecraft yaw)
            dirs = [(1, 2), (0, 1), (1, 0), (2, 1)]                            # +z, -x, -z, +x 칸 (x, z 자리)
            order = [dirs[(q + k) % 4] for k in (0, 1, 2, 3)]                   # 앞, 오른쪽, 뒤, 왼쪽 (Minecraft는 yaw가 늘면 오른쪽으로 돈다: +z를 볼 때 오른쪽은 -x)
            for hi, yy in enumerate((1, 2)):
                for k, (ix, iz) in enumerate(order):
                    out[hi * 4 + k] = mv[ix, yy, iz]
            out[8] = mv[1, 0, 1]; out[9] = lq[1, 1, 1]; out[10] = lq[1, 2, 1]
        dm = obs.get("damage_source", {})
        amt = float(np.asarray(dm.get("damage_amount", 0)).sum()) if isinstance(dm, dict) else 0.0
        if amt > 0:
            rel = np.radians(float(np.asarray(dm.get("damage_yaw", 0)).ravel()[0]) - yaw)
            for k, a in enumerate((0, np.pi / 2, np.pi, -np.pi / 2)):
                out[11 + k] = min(amt / 5, 1.0) * max(np.cos(rel - a), 0.0)
        return out

    def light(obs):
        """[5]: 둘레의 밝음, 하늘빛, 해의 밝기 (밤과 낮을 몸으로), 따뜻함, 비에 젖음."""
        loc = obs.get("location_stats", {}); g = lambda k: float(np.asarray(loc.get(k, [0])).ravel()[0])
        return np.array([g("light_level") / 15, g("sky_light_level") / 15, g("sun_brightness"), min(max(g("temperature"), -0.5), 2.0) / 2,
                         float(g("is_raining") > 0 and g("can_see_sky") > 0)], np.float32)
    prev = {"rgb": None}
    # 눈: 창립자의 정본 눈 기관 (읽기 전용으로 불러 씀). 가운데 16px는 곱게, 고리 4개는 갈수록 넓고 거칠게 (16x16 같은 칸에), 고리마다 가장 센 빛깔도.
    # 초점은 눈 기관 스스로 옮긴다 (정본의 규칙: 내다본 것(앞 그림)과 달라진 곳으로). 마음이 고르지 않는다.
    import sys
    sp = importlib.util.spec_from_file_location("canon_fovea", EYE); fv = importlib.util.module_from_spec(sp); sys.modules["canon_fovea"] = fv; sp.loader.exec_module(fv)
    eye = fv.Retina(spec=fv.RetinaSpec(fovea_px=16, rings=4, ring_px=16, falloff=2.0, ring_peak=True))

    def retina(rgb):
        s = eye.look(rgb)
        fov = np.asarray(Image.fromarray(np.ascontiguousarray(s["fovea"]).astype(np.uint8)).resize((16, 16), Image.BILINEAR), np.float32)
        lv = [fov] + [np.asarray(l["px"], np.float32) for l in s["levels"]]
        while len(lv) < 5:
            lv.append(lv[-1])
        pk = [np.asarray(l["peak"], np.float32) for l in s["levels"]]
        while len(pk) < 4:
            pk.append(pk[-1])
        stack = np.concatenate([np.stack(lv[:5]), np.stack(pk[:4])], 0)                 # [9, 16, 16, 3]: 가운데 + 고리 4 + 고리의 센 빛깔 4
        return (stack.transpose(0, 3, 1, 2).reshape(27, 16, 16) / 255.0).astype(np.float16), s

    def kept(s, H, W):
        """눈이 실제로 받는 것을 사람에게 보이려고 빛깔로 되그림 (거친 바깥 고리부터, 가운데가 마지막)."""
        canvas = np.zeros((H, W, 3), np.float32)
        for l in reversed(s["levels"]):
            x0, y0, x1, y1 = l["box"]
            canvas[y0:y1, x0:x1] = np.asarray(Image.fromarray(np.asarray(l["px"]).clip(0, 255).astype(np.uint8)).resize((x1 - x0, y1 - y0), Image.NEAREST), np.float32)
        x0, y0, x1, y1 = s["fovea_box"]; canvas[y0:y1, x0:x1] = s["fovea"]
        return Image.fromarray(canvas.clip(0, 255).astype(np.uint8))

    def bug(rgb, s, k):
        """초점 안의 작은 움직이는 것 (개구리 망막의 '벌레 탐지'처럼 가운데-둘레 움직임): 같은 자리에서 앞 그림과 견주어,
        가운데(초점)가 둘레(첫 고리)보다 얼마나 더 바뀌었나. 제 몸이 돌거나 앞으로 갈 때 온 그림이 바뀌는 것은 둘레도 함께 바뀌어 지워진다."""
        p = prev["rgb"]; prev["rgb"] = rgb
        if p is None or p.shape != rgb.shape or k in (2, 3):
            return 0.0
        g, q = rgb.mean(2).astype(np.float32), p.mean(2).astype(np.float32)
        H_, W_ = g.shape; cy, cx = H_ // 2, W_ // 2                # 바로 앞 (시야 한가운데 = 손이 닿는 곳): 개구리도 바로 앞의 벌레만 문다. 초점은 화면을 떠돌므로 쓰지 않는다
        x0, y0, x1, y1 = cx - 12, cy - 12, cx + 12, cy + 12; c = np.abs(g[y0:y1, x0:x1] - q[y0:y1, x0:x1]).mean()
        X0, Y0, X1, Y1 = cx - 36, cy - 36, cx + 36, cy + 36; ring = np.abs(g[Y0:Y1, X0:X1] - q[Y0:Y1, X0:X1]); sur = (ring.sum() - np.abs(g[y0:y1, x0:x1] - q[y0:y1, x0:x1]).sum()) / max(ring.size - (y1 - y0) * (x1 - x0), 1)
        return float(max(c - sur, 0.0) / 64.0)

    def pack(obs, done, hurt, k=None):
        rgb = np.asarray(obs["rgb"]).transpose(1, 2, 0)
        if CROP:                                                  # 화면 아래의 HUD (하트, 배부름, 손에 든 칸, 손)를 잘라 낸다: 몸 신호로 이미 받으므로 눈은 바깥세상만
            rgb = np.ascontiguousarray(rgb[:-CROP])
        if NOCROSS:                                               # 십자 조준점 (사람 눈에는 없는, 게임이 그려 넣은 겨눔 표시)을 둘레 빛깔로 메운다
            H_, W_ = rgb.shape[:2]; y0, x0 = H0_ // 2 - 6, W_ // 2 - 6      # 조준점은 (잘라 내기 전) 그린 그림의 한가운데
            f = rgb[y0:y0 + 13, x0:x0 + 13].astype(np.float32); known = ~XMASK.copy()
            for _ in range(8):                                    # 메울 칸을, 이미 아는 이웃의 평균으로 바깥부터 차례로 메운다
                kp = np.pad(known, 1).astype(np.float32); fp = np.pad(f * known[..., None], ((1, 1), (1, 1), (0, 0)))
                nb = [(0, 1), (2, 1), (1, 0), (1, 2)]
                s = sum(fp[a:a + 13, b:b + 13] for a, b in nb); c = sum(kp[a:a + 13, b:b + 13] for a, b in nb)
                new = ~known & (c > 0)
                f[new] = (s[new] / c[new][:, None]); known |= new
            rgb = rgb.copy(); rgb[y0:y0 + 13, x0:x0 + 13] = f.clip(0, 255).astype(np.uint8)
        im = Image.fromarray(rgb)
        ret, s = retina(rgb)
        mv = bug(rgb, s, k)
        ls = obs["life_stats"]; loc = obs.get("location_stats", {})
        pos = np.asarray(loc.get("pos", [0, 0, 0]), dtype=np.float32).ravel(); yaw = float(np.asarray(loc.get("yaw", [0])).ravel()[0])
        out = dict(ret=ret, bug=mv, gaze=(float(eye.cx), float(eye.cy)), health=float(np.asarray(ls["life"]).ravel()[0]) / 20, food=float(np.asarray(ls["food"]).ravel()[0]) / 20,
                   air=float(np.asarray(ls["oxygen"]).ravel()[0]) / 300, hurt=float(hurt), done=bool(done), pos=pos, yaw=yaw,
                   pitch=float(np.asarray(loc.get("pitch", [0])).ravel()[0]),
                   small=np.asarray(im.resize((80, 48), Image.BILINEAR), np.uint8))      # 깊이 눈이 받는 작은 온 그림 (HUD 잘라 낸 것, 48x80): 제 걸음으로 깊이를 배운다
        if ear is not None:
            out["ear"] = ear.bands()
        if SKIN:
            out["skin"] = np.concatenate([skin(obs, yaw), light(obs)])         # [20]
        if os.environ.get("ATANOR_MC_FRAME") == "1":                    # 보편 몸 계약: 날 그림 (HUD 잘라 냄, 조준점 지움을 켰으면 지운 것) - 눈은 존재 안에서 돈다
            out["frame"] = rgb
        if JUDGE_DEPTH and "rays" in obs:                            # 판정에만: 빛살마다 처음 닿은 블록까지의 거리
            out["rays"] = {k: np.asarray(v).ravel() for k, v in obs["rays"].items() if np.asarray(v).dtype.kind in "fiu"}
        if JUDGE:
            r = obs.get("rays", {}); sl = slice(NFAN0, NFAN0 + len(FAN))
            en = [str(v).lower() for v in np.asarray(r.get("entity_name", [""] * 99)).ravel()[sl]]
            ed = np.asarray(r.get("entity_distance", [99.0] * 99), np.float32).ravel()[sl]
            di = obs.get("delta_inv", {}); q = np.asarray(di.get("inc_quantity_by_other", [0])).ravel() if isinstance(di, dict) else np.zeros(1)
            bd_ = np.asarray(r.get("block_distance", [99.0] * 99), np.float32).ravel()[sl]
            out["judge"] = dict(ent=[(e not in ("", "null", "none", "item")) and d < 32 for e, d in zip(en, ed)], ent_d=ed.tolist(), got=float(np.clip(q.astype(np.float32), 0, None).sum()),
                                reach=float(bd_[3]))                # 바로 앞 블록까지 (손이 닿는 거리는 약 4.5칸: 그보다 멀면 헛손질)
        if ELDER and "rays" in obs:                                 # 어른에게만: 바로 앞 (가운데, 조금 위, 조금 아래) 빛살이 처음 닿은 짐승·블록의 이름과 거리
            r = obs["rays"]; L = 3
            g = lambda k: [str(v) for v in np.asarray(r.get(k, [""] * 99)).ravel()[-L:]]
            f_ = lambda k: [float(v) for v in np.asarray(r.get(k, [99] * 99)).ravel()[-L:]]
            out["look"] = dict(entity=g("entity_name"), entity_d=f_("entity_distance"), block=g("block_name"), block_d=f_("block_distance"))
        if big:
            show = im.copy(); ImageDraw.Draw(show).rectangle(s["fovea_box"], outline=(255, 210, 0), width=2)
            b = io.BytesIO(); show.save(b, format="JPEG", quality=85); out["big"] = b.getvalue()
            b2 = io.BytesIO(); kept(s, rgb.shape[0], rgb.shape[1]).save(b2, format="PNG"); out["kept"] = b2.getvalue()
        return out

    def use_ticks(o):
        """쓰기 (먹기) = 32틱 누르기. 배가 가득 차면 마크는 먹지 않으므로 (세계의 성질) 1틱으로 끝낸다 - 겪는 결과는 같고 헛된 시간만 줄어듦 (효율, 장부).
        재 보니 '쓰기'의 32틱이 판 시간의 대부분이었다 (먹여 줘서 배가 거의 늘 가득)."""
        try:
            full = float(np.asarray(o["life_stats"]["food"]).ravel()[0]) >= 20
        except Exception:
            full = False
        return 1 if full else USE_TICKS

    obs = env.reset(); conn.send(pack(obs, False, 0.0))
    while True:
        cmd = conn.recv()
        if cmd == "reset":
            obs = env.reset(); conn.send(pack(obs, False, 0.0)); continue
        if cmd == "close":
            env.close(); return
        if isinstance(cmd, tuple) and cmd[0] == "move":      # 실험자가 몸을 들어 옮김 (그림을 고르게 모으려고. 존재의 고름이 아니다): 땅 위 아무 데로
            r = cmd[1]; x, z = np.random.randint(-r, r + 1, 2)
            ex = getattr(env, "execute_cmd", None) or env.unwrapped.execute_cmd
            obs, _, done, _ = ex(f"/spreadplayers {x} {z} 0 8 false @p")
            for _ in range(10):                                   # 떨어져 자리 잡을 때까지 몇 걸음 그대로
                obs, _, done, _ = env.step(env.action_space.no_op())
            conn.send(pack(obs, done, 0.0)); continue
        if isinstance(cmd, tuple) and cmd[0] == "exec":       # 판정 쪽 세계 명령 (계곡 짓기, 시간, 몸 옮기기). 존재의 고름이 아니다
            ex = getattr(env, "execute_cmd", None) or env.unwrapped.execute_cmd
            for c in cmd[1]:
                obs, _, done, _ = ex(c)
            for _ in range(int(cmd[2]) if len(cmd) > 2 else 2):
                obs, _, done, _ = env.step(env.action_space.no_op())
            conn.send(pack(obs, done, 0.0)); continue
        if isinstance(cmd, tuple) and cmd[0] == "voice":     # 어른의 목소리를 이 몸의 귀에 (세계는 한 걸음도 가지 않는다)
            if ear is not None:
                ear.say(cmd[1])
            conn.send(None); continue
        if isinstance(cmd, tuple) and cmd[0] == "feed":      # 먹여 주기 (돌봄 단계, 장부): 부모가 손(첫 칸)에 먹을 것을 쥐여 준다
            ex = getattr(env, "execute_cmd", None) or env.unwrapped.execute_cmd
            obs, _, done, _ = ex(f"/replaceitem entity @p {cmd[3] if len(cmd) > 3 else 'slot.hotbar.0'} minecraft:{cmd[1]} {int(cmd[2])}")      # 자리: 주 손 (slot.hotbar.0) 또는 다른 손 (slot.weapon.offhand)
            conn.send(pack(obs, done, 0.0)); continue
        a = env.action_space.no_op()
        k = int(cmd)
        if BRANCH_ON:                                         # 갈래 몸: 몸짓 번호 = 다리*15 + 목*3 + 손. 다리 (서 있음, 앞, 뒤, 왼옆, 오른옆, 뛰며 앞) x 목 (그대로, 왼 30, 오른 30, 위 15, 아래 15) x 손 (빈손, 때림, 씀)
            l, nk, hd = k // 15, (k // 3) % 5, k % 3
            if l == 1: a[0] = 1
            elif l == 2: a[0] = 2
            elif l == 3: a[1] = 1
            elif l == 4: a[1] = 2
            elif l == 5: a[0] = 1; a[2] = 1
            if nk == 1: a[4] = 12 - TURN
            elif nk == 2: a[4] = 12 + TURN
            elif nk == 3: a[3] = 11
            elif nk == 4: a[3] = 13
            kk = 2 if nk != 0 else 0                     # 벌레 탐지: 고개를 돌린 걸음은 온 그림이 바뀌므로 건너뜀
            if hd == 2:                                       # 쓰기 = 32틱 누르기 (먹는 몸짓): 다리·목은 첫 틱에만
                hurt = 0.0
                for t_ in range(use_ticks(obs)):
                    if t_ > 0:
                        a = env.action_space.no_op()
                    a[5] = 1
                    obs, r, done, info = env.step(a)
                    dm = obs.get("damage_source", {}); hurt += float(np.asarray(dm.get("damage_amount", 0)).sum()) if isinstance(dm, dict) else 0.0
                    if done:
                        break
                conn.send(pack(obs, done, hurt, kk)); continue
            if hd == 1: a[5] = 3
            obs, r, done, info = env.step(a)
            dm = obs.get("damage_source", {})
            hurt = float(np.asarray(dm.get("damage_amount", 0)).sum()) if isinstance(dm, dict) else 0.0
            conn.send(pack(obs, done, hurt, kk)); continue
        if k == 5:                                           # '쓰기' = 1.6초 (32틱) 누르고 있기 (먹는 몸짓. 마크에서 먹으려면 이만큼 눌러야 한다): 몸의 움직임 단위
            hurt = 0.0
            for _ in range(use_ticks(obs)):
                a = env.action_space.no_op(); a[5] = 1
                obs, r, done, info = env.step(a)
                dm = obs.get("damage_source", {}); hurt += float(np.asarray(dm.get("damage_amount", 0)).sum()) if isinstance(dm, dict) else 0.0
                if done:
                    break
            conn.send(pack(obs, done, hurt, k)); continue
        if k == 1: a[0] = 1
        elif k == 2: a[4] = 12 - TURN
        elif k == 3: a[4] = 12 + TURN
        elif k == 4: a[5] = 3
        elif k == 5: a[5] = 1
        elif k == 6: a[0] = 1; a[2] = 1
        obs, r, done, info = env.step(a)
        dm = obs.get("damage_source", {})
        hurt = float(np.asarray(dm.get("damage_amount", 0)).sum()) if isinstance(dm, dict) else 0.0
        conn.send(pack(obs, done, hurt, k))


VIEW, VLOCK = {}, threading.Lock()


def send(s, obj):
    b = pickle.dumps(obj, protocol=4); s.sendall(struct.pack("!Q", len(b)) + b)


def recv(s):
    n = struct.unpack("!Q", recvall(s, 8))[0]; return pickle.loads(recvall(s, n))


def recvall(s, n):
    buf = b""
    while len(buf) < n:
        c = s.recv(min(1 << 20, n - len(buf)))
        if not c:
            raise ConnectionError("closed")
        buf += c
    return buf


PAGE = """<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ATANOR MineDojo</title>
<style>:root{--bg:#f6f5f1;--card:#fff;--ink:#222;--mute:#666;--line:#e2dfd6}@media (prefers-color-scheme:dark){:root{--bg:#1b1b1b;--card:#262626;--ink:#eee;--mute:#aaa;--line:#3a3a3a}}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,sans-serif}main{max-width:1000px;margin:0 auto;padding:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin-bottom:14px}img{width:100%;image-rendering:pixelated;border-radius:6px;display:block}
.bar{height:12px;background:var(--line);border-radius:6px;overflow:hidden;margin:4px 0 10px}.bar>div{height:100%}h1{font-size:20px;margin:0 0 4px}h2{font-size:14px;margin:0 0 8px;color:var(--mute)}
.row{display:flex;justify-content:space-between;gap:12px}.two{display:grid;grid-template-columns:1fr 1fr;gap:14px}@media(max-width:700px){.two{grid-template-columns:1fr}}</style></head><body><main>
<h1>ATANOR · MineDojo</h1><div id="ver" style="font-weight:600;font-size:13px"></div><div id="label" style="color:var(--mute);font-size:12px"></div>
<div class="two"><div class="card"><h2>1인칭 그림 (노란 상자 = 초점, 곱게 보는 곳)</h2><img id="eye" alt="1인칭 그림"></div>
<div class="card"><h2>눈이 실제로 받는 것 (같은 크기로 되그림: 초점은 곱게, 바깥 고리로 갈수록 거칠게)</h2><img id="small" alt="눈이 받는 것"></div></div>
<div class="card" id="dcard" style="display:none"><h2>깊이 눈이 보는 것 (제 걸음으로 배운 깊이: 밝을수록 가까움, 아래 막대 = 존재가 받는 눈높이 5갈래의 가까움)</h2><img id="depth" alt="깊이"></div>
<div class="card"><h2>몸 (첫 몸)</h2>
<div class="row"><span>체력</span><span id="hv"></span></div><div class="bar"><div id="hb" style="background:#d9534f"></div></div>
<div class="row"><span>배부름</span><span id="fv"></span></div><div class="bar"><div id="fb" style="background:#c98a2e"></div></div>
<div class="row"><span>숨</span><span id="ov"></span></div><div class="bar"><div id="ob" style="background:#4f8fe0"></div></div>
<div class="row"><span>지금 한 것</span><b id="act"></b></div><div class="row"><span>누가 고르나</span><span id="who"></span></div><div class="row"><span>어른의 목소리 (아기가 보는 것에 이름 붙임, 소리로만 들림)</span><b id="said"></b></div>
<div class="row"><span>삶 / 걸음</span><span id="ls"></span></div><div class="row"><span>맞은 양 (이 걸음 / 이 삶)</span><span id="hurt"></span></div>
<div class="row"><span>몸의 수 / 초당 걸음 (모두 합쳐)</span><span id="sps"></span></div></div>
</main><script>
const $=i=>document.getElementById(i);
async function tick(){try{const s=await (await fetch('/state')).json();if(s.big){$('eye').src='data:image/jpeg;base64,'+s.big;$('small').src='data:image/png;base64,'+s.kept;if(s.depth){$('dcard').style.display='block';$('depth').src='data:image/png;base64,'+s.depth;}
$('ver').textContent=(s.version?'버전 v'+s.version+' · ':'')+(s.note||'');$('label').textContent=s.label||'';const p=v=>Math.max(0,Math.min(100,100*v))+'%';
$('hv').textContent=(s.health*20).toFixed(0)+' / 20';$('hb').style.width=p(s.health);$('fv').textContent=(s.food*20).toFixed(0)+' / 20';$('fb').style.width=p(s.food);
$('ov').textContent=(s.air*300).toFixed(0)+' / 300';$('ob').style.width=p(s.air);$('act').textContent=s.act||'';$('who').textContent=s.who||'';$('said').textContent=s.said||'';
$('ls').textContent=s.life+'번째 / '+s.step;$('hurt').textContent=s.hurt.toFixed(1)+' / '+s.hurt_total.toFixed(1);$('sps').textContent=s.n+' / '+s.sps;}}catch(e){}
setTimeout(tick,250)}tick();</script></body></html>"""


class H:
    pass


def viewer(port):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class V(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path.startswith("/state"):
                with VLOCK:
                    try:
                        ptr = json.load(open(POINTER, encoding="utf-8"))
                    except Exception:
                        ptr = {}
                    body = json.dumps(dict(VIEW, version=ptr.get("version", ""), note=ptr.get("note", "")), ensure_ascii=False).encode(); ct = "application/json; charset=utf-8"
            else:
                body, ct = PAGE.encode(), "text/html; charset=utf-8"
            self.send_response(200); self.send_header("Content-Type", ct); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    ThreadingHTTPServer(("0.0.0.0", port), V).serve_forever()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bodies", type=int, default=4); ap.add_argument("--port", type=int, default=8790); ap.add_argument("--view", type=int, default=8770)
    ap.add_argument("--seed", type=int, default=7); ap.add_argument("--h", type=int, default=204); ap.add_argument("--w", type=int, default=256)
    ap.add_argument("--sh", type=int, default=40); ap.add_argument("--sw", type=int, default=64)
    ap.add_argument("--food", type=int, default=0)               # 태어날 때의 배부름 (0 = MineDojo 기본)
    a = ap.parse_args()
    if a.food:
        os.environ["ATANOR_MC_FOOD"] = str(a.food)               # 몸 프로세스(spawn)가 물려받는다
    threading.Thread(target=viewer, args=(a.view,), daemon=True).start()
    with VLOCK:
        VIEW.update(label="세계를 여는 중 (몸마다 Minecraft를 띄우는 데 수십 초)")
    pipes, procs, firsts = [], [], []
    ctx = mp.get_context("spawn")                                # 갈라 낸(fork) 프로세스 안에서는 MineDojo가 멈춘다: 새로 띄운다
    for i in range(a.bodies):                                    # 같은 세계 씨앗: 익숙한 세계 하나 (몸마다 따로 된 같은 세계)
        p, c = ctx.Pipe(); pr = ctx.Process(target=body, args=(c, a.seed, a.h, a.w, None, i == 0), daemon=True); pr.start(); pipes.append(p); procs.append(pr)
        firsts.append(p.recv())                                  # 하나씩: 앞 몸이 다 뜬 뒤에 다음 몸 (동시에 빌드하면 공유 캐시가 깨진다)
        print(f"body {i} up", flush=True)
        with VLOCK:
            VIEW.update(label=f"세계를 여는 중: 몸 {i + 1} / {a.bodies} 떴음")
    first = firsts
    print("bodies ready", len(first), flush=True)
    life, step, hurt_total, t0, nstep = 1, 0, 0.0, time.time(), 0

    def show(o, act="", who="", label=""):
        nonlocal step, hurt_total
        with VLOCK:
            VIEW.update(big=base64.b64encode(o["big"]).decode(), kept=base64.b64encode(o["kept"]).decode(), gaze=o["gaze"], health=o["health"], food=o["food"], air=o["air"], hurt=o["hurt"], hurt_total=hurt_total,
                        life=life, step=step, n=a.bodies, sps=round(nstep / max(time.time() - t0, 1e-6), 1), act=act, who=who, label=label or VIEW.get("label", ""))
    show(first[0], label="세계가 열림. 존재를 기다리는 중")
    srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); srv.bind(("0.0.0.0", a.port)); srv.listen(1)
    print("listening", a.port, flush=True)
    while True:
        s, _ = srv.accept(); s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            send(s, dict(n=a.bodies, obs=[{k: v for k, v in o.items() if k not in ("big", "kept")} for o in first]))
            while True:
                m = recv(s)
                if m.get("cmd") == "voice":
                    for i in m["which"]:
                        pipes[i].send(("voice", m["wav"]))
                    for i in m["which"]:
                        pipes[i].recv()
                    with VLOCK:
                        VIEW["said"] = m.get("text", "")
                    send(s, "ok"); continue
                if m.get("cmd") == "feed":
                    for i in m["which"]:
                        pipes[i].send(("feed", m.get("item", "bread"), int(m.get("qty", 8)), m.get("slot", "slot.hotbar.0")))
                    for i in m["which"]:
                        first[i] = pipes[i].recv()
                    send(s, [{k: v for k, v in first[i].items() if k not in ("big", "kept")} for i in m["which"]]); continue
                if m.get("cmd") == "exec":
                    for i in m["which"]:
                        pipes[i].send(("exec", list(m["cmds"]), int(m.get("settle", 2))))
                    for i in m["which"]:
                        first[i] = pipes[i].recv()
                    if 0 in m["which"]:
                        show(first[0], "", "실험자 (판정 쪽 세계 명령)", m.get("label", ""))
                    send(s, [{k: v for k, v in first[i].items() if k not in ("big", "kept")} for i in m["which"]]); continue
                if m.get("cmd") == "move":
                    for i in m["which"]:
                        pipes[i].send(("move", int(m.get("r", 400))))
                    for i in m["which"]:
                        first[i] = pipes[i].recv()
                    if 0 in m["which"]:
                        show(first[0], "", "실험자가 몸을 옮김 (그림을 고르게 모으려고)", m.get("label", ""))
                    send(s, [{k: v for k, v in first[i].items() if k not in ("big", "kept")} for i in m["which"]]); continue
                if m.get("cmd") == "reset":
                    for i in m["which"]:
                        pipes[i].send("reset")
                    for i in m["which"]:
                        first[i] = pipes[i].recv()
                    if 0 in m["which"]:
                        life += 1; step = 0; hurt_total = 0.0
                    send(s, [{k: v for k, v in first[i].items() if k not in ("big", "kept")} for i in m["which"]]); continue
                acts = m["acts"]
                for p, k in zip(pipes, acts):
                    p.send(int(k))
                obs = [p.recv() for p in pipes]
                step += 1; nstep += a.bodies; hurt_total += obs[0]["hurt"]
                show(obs[0], actname(int(acts[0])), m.get("who", ""), m.get("label", ""))
                if m.get("depth"):                                   # 존재의 깊이 눈이 본 것 (뷰어에만)
                    with VLOCK:
                        VIEW["depth"] = base64.b64encode(m["depth"]).decode()
                send(s, [{k: v for k, v in o.items() if k not in ("big", "kept")} for o in obs])
        except (ConnectionError, OSError, EOFError) as e:
            print("client gone:", e, flush=True)
            with VLOCK:
                VIEW["who"] = "존재가 떠남 (다시 붙기를 기다림)"


if __name__ == "__main__":
    main()
