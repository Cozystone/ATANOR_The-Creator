"""1인칭 3D 세계의 뷰어: 눈만으로 아는 존재 하나가 사는 것을 제 시계대로 보여 준다 (고를 때마다 TICK초가 실제로 흐른다. 빨리 감지 않는다).

보여 주는 것: 그 존재의 눈앞 그림, 위에서 본 세계(사람이 보라고 그린 것. 존재는 이것을 받지 않는다), 몸 안 저장고, 지금 하는 것,
그리고 마음 안의 신호 (이름은 우리가 붙인 것이고 존재 안에는 없다): 가장 급한 모자람, 끌림이 있는가, 행동마다의 끌림.

  python -m atanor_os.emerge.room3dwatch --model <core.pt> --port 8769 --walls 3
"""
import argparse, base64, io, json, math, os, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np, torch
from PIL import Image
from . import room3d_live as R
from .room3d import Room, COLOR
from .core import Core
from .plastic import Plastic
from .inner import Inner, chunk_pull


def now_only(e):
    """지금 눈에 보이는 것만 (적어 둔 자리와 지도에서 온 것은 뺀다)."""
    kept, way = e.kept, e.way
    e.kept, e.way = [None] * R.CHROMA, [None] * R.CHROMA
    x = e.sense(); e.kept, e.way = kept, way
    return x

ACTS = ["가만히", "앞으로", "왼쪽으로 돌기", "오른쪽으로 돌기", "하기", "눈동자 왼쪽으로", "눈동자 오른쪽으로"]
STATE, LOCK = {}, threading.Lock()


def live_gpu(a):
    """GPU 판의 한 몸을 실제 시간으로. 가리킴 파일이 바뀌면 돌아간다."""
    import torch as T_
    from . import room3d_gpu as G
    from .life_gpu import Life
    dev = "cuda" if T_.cuda.is_available() else "cpu"
    st0 = os.path.getmtime(a.pointer) if a.pointer else 0
    m0 = os.path.getmtime(a.model)
    R.set_slots(R.CHROMA, 1 + R.CHROMA)
    sd = T_.load(a.model, map_location="cpu")
    body = getattr(a, "body", "")                                 # 몸에 더한 감각: "touch" 또는 "touch+depth"
    E = {"": 0, "touch": 3, "touch+depth": 8}[body]
    net = Core(R.DX + E, R.M, R.INSIDE, R.PLACE); net.load_state_dict(sd, strict=False); net = net.to(dev).eval()
    net.far = float(getattr(a, "far", 0.0))                   # 헛됨을 넓힘: 떠올린 순간에 가까워지리라 내다봤는데 안 가까워져도 헛되다 (알맹이의 배움, 손 규칙 아님)
    L = Life(net, 1, R.HOME, dev, seed=a.seed, pool=getattr(a, "pool", "mean"), drain=(1 / 60.0) / float(getattr(a, "slow", 4.0)))
    L.dishab = bool(getattr(a, "dishab", False))
    if body:
        from .body_gpu import touch as _touch, DepthSense
        _ds = DepthSense(getattr(a, "depth_net", "models/depth0.pt"), dev) if "depth" in body else None
        L.extra = (lambda w_: T_.cat([_touch(w_), _ds(w_)], 1)) if _ds is not None else _touch
    dep = None
    if getattr(a, "depth", ""):                                  # 깊이 눈: 정본 depth_learner를 읽기 전용으로, 제 걸음으로 배운 무게
        import importlib.util, math as _m
        _cp = os.environ.get("ATANOR_DEPTH_PATH", "depth_learner")
        def _ld(nm):
            sp = importlib.util.spec_from_file_location("canon_depth_" + nm, os.path.join(_cp, nm + ".py")); mm = importlib.util.module_from_spec(sp); sp.loader.exec_module(mm); return mm
        _SS, _MD = _ld("selfsup"), _ld("model")
        dep = _MD.DepthNet(width=16).to(dev); dep.load_state_dict(T_.load(a.depth, map_location=dev)); dep.eval()
        _f = 0.5 * 80 / _m.tan(0.5 * G.FOV); _offs = T_.atan((T_.arange(80, device=dev) + 0.5 - 40) / _f); _rows = T_.arange(48, device=dev).float()[:, None] + 0.5 - 24
        def pinhole(w):
            kind, dist = w.cast(w.head[:1, None] + _offs[None], step=0.05)
            z = dist[:1] * T_.cos(_offs)[None]; half = 0.5 * _f / z.clamp(min=0.05); wall = _rows.abs()[None] <= half[:, None, :]
            col = G.COLOR.to(dev)[kind[:1].long()] * (1.0 - z / 12.0).clamp(0.25, 1.0)[..., None]
            base = T_.where((_rows < 0)[None, :, :, None], G.SKY.to(dev), G.GROUND.to(dev)).expand(1, 48, 80, 3)
            return T_.where(wall[..., None], col[:, None].expand(-1, 48, -1, -1), base).permute(0, 3, 1, 2).clamp(0, 1)
    social = False
    im = None
    if getattr(a, "wm", ""):                                     # 상상으로 고르기: 모자랄 때 행동 줄을 속으로 굴려 보고, 풀리기 직전 장면에 가장 가까이 가는 줄의 첫 행동
        from .eyewm import WM, Imagine
        wm = WM().to(dev).eval(); wm.load_state_dict(T_.load(a.wm, map_location=dev)); im = Imagine(wm, 1, device=dev, seed=a.seed)
    saved = time.strftime("%m-%d %H:%M", time.localtime(os.path.getmtime(a.model)))
    label = (f"{('v' + str(a.version) + ' · ') if getattr(a, 'version', '') else ''}{getattr(a, 'note', '')} · {os.path.basename(a.model)} · 저장 {saved} · GPU 판 · "
             f"지금 보이는 것만 · 눈의 반사 · 헛되면 놓음{(' (넓힘 ' + str(net.far) + ')') if net.far else ''} · 손으로 넣은 행동 규칙 없음{' · 몸에 더한 감각: ' + body + ' (닿음 = 앞/왼/오른 가까움, 깊이 = 제 걸음으로 배운 깊이 5갈래)' if body else ''}{' · 탈습관화 (내다본 것과 크게 다른 것이 오면 쌓인 헛됨이 풀림)' if L.dishab else ''}{' · 상상으로 고름 (눈을 거꾸로 뒤집은 상상 기관으로 8걸음 앞을 그려 봄)' if im is not None else ''} · 익숙한 세계 하나 · 1틱 {G.TICK}초 (실제 시간)")
    span = float(getattr(a, "life", 900)); life = 0
    while True:
        life += 1; L.begin(); w = L.world; t0 = time.time()
        while not bool(w.dead[0]) and float(w.clock[0]) < span:
            if a.pointer and os.path.exists(a.pointer) and os.path.getmtime(a.pointer) != st0:
                return
            if os.path.exists(a.model) and os.path.getmtime(a.model) != m0:      # 기르는 중인 존재가 다시 저장되면 그 존재로
                return
            if im is not None:
                with T_.no_grad():
                    zc = im.wm.encode(w.picture().permute(0, 3, 1, 2).clamp(0, 1))[0]
                pick, _ = im.choose(zc, w.store.argmin(1), (w.store.min(1).values < 0.75) & ~w.dead)
                L.override = lambda a_, pick=pick: T_.where(pick >= 0, pick, a_)
            act, sc, strong, filled = L.tick()
            depth_img = ""
            if dep is not None:
                with T_.no_grad():
                    dd = _SS.bounded_log_depth(dep(pinhole(w)), 0.1, 30.0)[0]
                import matplotlib.cm as _cm
                v = ((dd - _m.log(0.3)) / (_m.log(15) - _m.log(0.3))).clamp(0, 1).cpu().numpy()
                dimg = (_cm.turbo_r(v)[..., :3] * 255).astype(np.uint8)
                b3 = io.BytesIO(); Image.fromarray(dimg).resize((192, 115), Image.NEAREST).save(b3, format="PNG"); depth_img = base64.b64encode(b3.getvalue()).decode()
            if im is not None:
                im.remember(zc, filled)
            img = (L.world.picture(192, 120, step=0.04)[0].clamp(0, 1) * 255).byte().cpu().numpy()
            buf = io.BytesIO(); Image.fromarray(img).save(buf, format="PNG")
            eimg = (w.picture()[0].clamp(0, 1) * 255).byte().cpu().numpy()
            b2 = io.BytesIO(); Image.fromarray(eimg).resize((192, 120), Image.NEAREST).save(b2, format="PNG")
            start = G.parse(R.HOME)[2]; sx, sy = start[0] + 0.5, start[1] + 0.5
            seen = []
            where, size, vis = L.eye.last
            gx = float(L.eye.gx[0]); fw = 8 / 64
            store = w.store[0].cpu().numpy(); need = 1 - store; a_ = int(act[0])
            with LOCK:
                STATE.update(label=label, life=life, clock=round(float(w.clock[0]), 1), span=span, store=[round(float(v), 3) for v in store], act=ACTS[a_],
                             fills=int(L.fills[0]), watched=None, dead=bool(w.dead[0]), pos=[float(w.pos[0, 0]), float(w.pos[0, 1])], head=float(w.head[0]), map=w.map[0].cpu().numpy().tolist(),
                             colors=(COLOR * 255).astype(int).tolist(), seen=seen, frame=base64.b64encode(buf.getvalue()).decode(),
                             lack=["마실 것", "먹을 것"][int(need.argmax())], lack_size=round(float(need.max()), 2), pulled=bool(strong[0]), seeking=False,
                             pull=[round(float(v), 2) for v in sc[0]], acts=ACTS[:5], fov=G.FOV, lines=[], reach=G.REACH, retina=True,
                             gaze=[max(gx - fw / 2, 0), 0.4, min(gx + fw / 2, 1), 0.6], kept=[[j, float(where[0, j]), float(size[0, j]), 1] for j in range(3) if bool(vis[0, j])],
                             kept_img=base64.b64encode(b2.getvalue()).decode(), depth_img=depth_img, mind=None, futile=[round(float(v), 2) for v in L.fut[0]], mended=0, inner_norm=None,
                             mend_loss=0, infront=[])
            time.sleep(max(0.0, G.TICK - (time.time() - t0))); t0 = time.time()
        with LOCK:
            STATE["ended"] = "모자라서 끝남" if bool(w.dead[0]) else "끝까지 온전"
        time.sleep(2.0)


def live(a):
    gen = torch.Generator().manual_seed(a.seed); life = 0; stamp = None; mem = None
    while True:
        if a.pointer:                                              # 가리킴 파일이 바뀌면 다음 삶부터 그 존재와 그 세계를 보여 준다 (새 갈래가 나올 때마다 이 파일만 고친다)
            try:
                st = (os.path.getmtime(a.pointer), os.path.getmtime(a.model) if a.model and os.path.exists(a.model) else 0)      # 가리킴이 바뀌었거나, 가리킨 존재가 더 자라 다시 저장되었거나
                if st != stamp:
                    for k, v in json.load(open(a.pointer, encoding="utf-8")).items():
                        setattr(a, k, v)
                    stamp, net = (os.path.getmtime(a.pointer), os.path.getmtime(a.model)), None
                    mem = None
            except Exception:
                pass
        if getattr(a, "gpu", False):                               # GPU 판: 몸 수천 개를 키우는 그 고리(life_gpu)로 이 한 몸을 산다
            live_gpu(a)
            stamp = None; net = None; continue
        if life == 0 or net is None:
            sd = torch.load(a.model, map_location="cpu")
            dx = sd["heavy"].shape[0]
            ret = getattr(a, "eye", "rays") == "retina"
            if ret:
                back = bool(getattr(a, "back", False)); R.set_world(bool(getattr(a, "pattern", False)))
                R.Retina.reflex = bool(getattr(a, "reflex", False)); R.Retina.OPPONENT = getattr(a, "opponent", "old"); R.Retina.POOL = getattr(a, "pool", "mean")
                R.Retina.EAR = bool(getattr(a, "ear", False))
                R.Retina.SPOTS = int(getattr(a, "spots", 1)); R.Retina.ROUTES = bool(getattr(a, "routes", False)); R.Retina.MAP = bool(getattr(a, "map", False)); R.Retina.IMAGINE = bool(getattr(a, "imagine", False))
                both = bool(getattr(a, "both", False))
                R.set_slots(R.CHROMA, dx - 3 - 6 * R.CHROMA); R.set_motors(sd["f.0.weight"].shape[1] - dx > 5 + back + 2 * both, back, both)
            else:
                R.set_slots(R.TINTS, dx - 3 - 6 * R.TINTS); R.set_motors(False); R.set_world(False)
            net = Core(R.DX, R.M, R.INSIDE, R.PLACE); net.load_state_dict(sd, strict=False); net.eval()
            net.also = slice(R.DX - 4, R.DX) if (ret and R.Retina.EAR) else None
            aw = Plastic(net) if getattr(a, "plastic", False) else None; hist = []; lived = 0      # 살면서 스스로를 고친다 (뷰어의 이 한 존재가)
            inner = None
            if getattr(a, "inner", "") and os.path.exists(a.inner):                 # 속마음: 지은 기억 없이, 지금 보이는 것과 이어지는 속 상태로 고른다
                isd = torch.load(a.inner); mo = isd["g.weight_ih"].shape[1] - R.DX - R.M - 1
                inner = Inner(R.DX, R.M, mo=mo); inner.load_state_dict(isd); inner.eval()
                R.Retina.SPOTS, R.Retina.MAP, R.Retina.IMAGINE = 1, False, False
            saved = time.strftime("%m-%d %H:%M", time.localtime(os.path.getmtime(a.model)))
            label = (f"{('v' + str(a.version) + ' · ') if getattr(a, 'version', '') else ''}{getattr(a, 'note', '') or os.path.basename(a.model)} · {os.path.basename(a.model)} · 저장 {saved} · {('그림으로만 앎 (가운데 곱게, 주변 거칠게' + (', 눈의 반사로 초점이 감)' if R.Retina.reflex else ', 눈동자가 움직임)' if R.M > len(R.MOTOR) else ', 초점 고정)')) if ret else f'눈만으로 앎 (빛깔과 거리, 시선 {R.RAYS}줄)'} · 벽 {a.walls}개 · "
                     f"{'샘은 채워 주면 옮겨 감' if a.roam else '샘은 제자리'}{' · 찾아냄의 기억' if a.seek else ''}{' · 헛되면 놓음' if getattr(a, 'tire', 0) else ''}{' · 길의 기억 (걸어 본 길을 적고 풀렸던 길을 다시 걷는다)' if getattr(a, 'routes', False) else ''}{' · 살면서 스스로를 고침 (되새김 + 느린 나)' if getattr(a, 'plastic', False) else ''}{' · 귀 (샘마다 소리, 벽을 지나면 준다, 두 귀의 크기만)' if getattr(a, 'ear', False) else ''}{' · 지금 보이는 것만 (지은 기억 없음)' if getattr(a, 'nowonly', False) else ''}{' · 속마음 (지은 기억 없이 이어지는 속 상태, 행동 묶음을 눈 감고 내다봄)' if getattr(a, 'inner', '') else ''}{' · 머릿속 지도와 속으로 걸어 보기' if getattr(a, 'imagine', False) else ''}{' · 익숙한 세계 하나' + (' (지도와 자리를 삶마다 이어 듦)' if getattr(a, 'map', False) else '') if getattr(a, 'home', False) else ''}{' · 저장고 ' + str(getattr(a, 'slow', 1.0)) + '배 느리게' if getattr(a, 'slow', 1.0) != 1.0 else ''} · 1틱 {R.TICK}초 (실제 시간)")
        life += 1
        home = bool(getattr(a, "home", False)); slow = float(getattr(a, "slow", 1.0))
        w = Room(seed=a.seed * 1000 + life, walls=a.walls, roam=a.roam, layout=R.HOME if home else None)
        w.drain = (1 / 60.0) / slow
        span = float(getattr(a, "life", a.span))
        eye = R.Retina(w) if ret else R.Eye(w)
        if ret and home and mem is not None:                      # 익숙한 세계: 앞선 삶의 지도와 자리들을 이어 든다
            eye.spots, eye.t, eye.back = mem[0], mem[1], mem[3]
            if len(mem) > 4 and mem[4] is not None and eye.road is not None:
                eye.road, eye.link = mem[4], mem[5]; eye.road.newlife()
            if mem[2] is not None and eye.map is not None:
                eye.map = mem[2]
        R.look_retina([w], [eye]) if ret else R.look_all([w], [eye])
        fills = 0; t0 = time.time(); futile = torch.zeros(1, R.M)
        h = inner.zero(1) if inner is not None else None
        pi_ = R.PathIntegrator()                                   # 제 걸음을 더해 가는 타고난 셈 (삶마다 0에서)
        while not w.dead and w.clock < span:
            if a.pointer and stamp is not None and os.path.exists(a.pointer) and os.path.getmtime(a.pointer) != stamp[0]:
                break                                              # 가리킴이 바뀌었다: 이 삶을 접고 곧바로 새 존재로 (다음 삶까지 기다리지 않는다)
            x = torch.tensor(now_only(eye) if (inner is not None or getattr(a, 'nowonly', False)) else eye.sense())[None]
            with torch.no_grad():
                sc, strong = net.pull(x)
                if inner is not None:
                    mu, wt, scale = net._last
                    sc = sc + float(getattr(a, "chunk", 0.3)) * chunk_pull(inner, h, x, mu, wt, scale, keep=(4,))
                    strong = strong | (scale > 0)
                sc = sc - float(getattr(a, 'tire', 0.0)) * futile
                sc2 = sc + a.curious * net.wonder(x) if a.curious else sc
                lost = a.seek and bool(strong[0]) and not bool(net.sees(x)[0])
                sk = net.seek(x) if lost else None
            if sk is not None:
                act = int(torch.multinomial(torch.softmax(sk[0] / a.loose, 0), 1, generator=gen))
            elif (bool(strong[0]) and not lost) or a.curious:
                act = int(torch.multinomial(torch.softmax(sc2[0] / a.loose, 0), 1, generator=gen))
            else:
                act = int(torch.randint(0, R.M, (1,), generator=gen))
            e0 = w.store.copy()
            w.set_motor(*(R.MOTOR[act] if act < len(R.MOTOR) else R.MOTOR[0])); w.advance(R.TICK)
            R.look_retina([w], [eye], [-1.0 if act == len(R.MOTOR) else 1.0 if act == len(R.MOTOR) + 1 else 0.0]) if ret else R.look_all([w], [eye])
            if inner is not None:
                with torch.no_grad():
                    h, _ = inner.step(h, x, torch.tensor([act]), motion=(torch.tensor(pi_.feel(w))[None] if inner.mo == 7 else torch.tensor([[w._felt[0] / 0.4, w._felt[5] / 0.4, w._felt[1] / 0.42]], dtype=torch.float32)) if inner.mo else None)
            fills += int(((w.store - e0) > 0.05).any())
            if aw is not None:
                x2_ = torch.tensor(eye.sense())[None]
                hist = (hist + [(x[0], act, x2_[0])])[-8:]; lived += 1
                aw.step(torch.stack([h[0] for h in hist])[None], torch.tensor([h[1] for h in hist])[None], torch.stack([h[2] for h in hist])[None], torch.ones(1, len(hist)))
                if lived % 1500 == 0 and getattr(a, "save", ""):
                    torch.save(net.state_dict(), a.save)
            if getattr(a, "tire", 0.0):
                more, eased = net.unmet(x, torch.tensor([act]), torch.tensor(now_only(eye) if (inner is not None or getattr(a, 'nowonly', False)) else eye.sense())[None])
                futile = futile * R.LEAK; futile[0, act] += float(more[0])
                if bool(eased[0]):
                    futile.zero_()
            img = (R.pictures([w], step=0.03, PW=192, PH=120)[0] * 255).astype(np.uint8) if ret else (w.eye(160, 100) * 255).astype(np.uint8)
            buf = io.BytesIO(); Image.fromarray(img).save(buf, format="PNG")
            seen, kept, kept_img = [], [], ""
            if ret:                                                 # 눈이 실제로 받는 것: 초점 둘레는 곱게, 바깥은 거칠게 (보여 주려고 다시 펴 그린다)
                o = R.canon_eye().sample(R.pictures([w])[0], eye.gx, 0.5, eye.spec)
                canvas = np.zeros((R.PH, R.PW, 3), np.float32)
                for lv in reversed(o["levels"]):
                    x0, y0, x1, y1 = lv["box"]
                    canvas[y0:y1, x0:x1] = np.asarray(Image.fromarray((np.clip(lv["px"], 0, 1) * 255).astype(np.uint8)).resize((x1 - x0, y1 - y0), Image.NEAREST), np.float32) / 255
                x0, y0, x1, y1 = o["fovea_box"]; canvas[y0:y1, x0:x1] = o["fovea"]
                b2 = io.BytesIO(); Image.fromarray((canvas * 255).astype(np.uint8)).save(b2, format="PNG"); kept_img = base64.b64encode(b2.getvalue()).decode()
                kept = [[j, float(eye.now[j][0]), float(eye.now[j][1]), 1] if eye.now[j] is not None else
                        [j, float(eye.recall(j)[0]), float(eye.recall(j)[1]), 0] for j in range(R.CHROMA) if eye.now[j] is not None or eye.kept[j] is not None]
                c, s_ = math.cos(-eye.head + w.head), math.sin(-eye.head + w.head)
                for j in range(R.CHROMA):
                    for sp in eye.spots[j]:
                        d = sp[0] - eye.pos
                        seen.append([float(w.pos[0] + d[0] * c - d[1] * s_), float(w.pos[1] + d[0] * s_ + d[1] * c), j + 1, int(sp[1] > sp[2])])
            else:
                c, s_ = math.cos(-eye.head + w.head), math.sin(-eye.head + w.head)      # 제 걸음으로 적어 둔 자리를 세계 그림 위로 옮긴다 (보여 주려고만)
                for j, pts in enumerate(eye.at):
                    for q in pts:
                        d = q - eye.pos
                        seen.append([float(w.pos[0] + d[0] * c - d[1] * s_), float(w.pos[1] + d[0] * s_ + d[1] * c), j])
            mind = None
            if ret and eye.map is not None:                         # 머릿속 지도 (제 걸음의 자리로. 보여 주려고 그대로 내보낸다)
                m = eye.map; r_ = 13; cx, cy = m.ix(eye.pos)
                x0, y0 = int(np.clip(cx - r_, 0, m.size - 2 * r_ - 1)), int(np.clip(cy - r_, 0, m.size - 2 * r_ - 1))
                mind = dict(cells=["".join(str(int(v)) for v in m.cell[x0:x0 + 2 * r_ + 1, y]) for y in range(y0, y0 + 2 * r_ + 1)],
                            me=[float(eye.pos[0]) + m.c - x0, float(eye.pos[1]) + m.c - y0], head=float(eye.head),
                            ways=[[float(v[0]) + m.c - x0, float(v[1]) + m.c - y0, j] for j, v in enumerate(eye.way) if v is not None],
                            spots=[[float(sp[0][0]) + m.c - x0, float(sp[0][1]) + m.c - y0, j, int(sp[1] > sp[2])] for j in range(R.CHROMA) for sp in eye.spots[j]],
                            known=round(float((m.cell != 0).mean()), 3), thought=eye.thought)
            need = 1 - w.store
            with LOCK:
                STATE.update(label=label, life=life, clock=round(w.clock, 1), span=span, store=[round(float(v), 3) for v in w.store], act=(R.NAMES if ret else ACTS)[act],
                             fills=fills, dead=bool(w.dead), pos=[float(w.pos[0]), float(w.pos[1])], head=float(w.head), map=w.map.tolist(),
                             colors=(COLOR * 255).astype(int).tolist(), seen=seen, frame=base64.b64encode(buf.getvalue()).decode(),
                             lack=["마실 것", "먹을 것"][int(need.argmax())], lack_size=round(float(need.max()), 2), pulled=bool(strong[0]) and not lost, mended=(aw.steps if aw is not None else 0), inner_norm=(round(float(h.norm()), 2) if h is not None else None), mend_loss=(round(aw.last, 4) if aw is not None else 0), futile=[round(float(v), 2) for v in futile[0]], seeking=sk is not None,
                             pull=[round(float(v), 2) for v in sc[0]], acts=(R.NAMES if ret else ACTS)[:R.M], fov=R.FOV, lines=[] if ret else eye.lines, reach=R.REACH,
                             mind=mind, retina=ret, gaze=[v / d for v, d in zip(eye.box, (R.PW, R.PH, R.PW, R.PH))] if ret else None, kept=kept, kept_img=kept_img)
            time.sleep(max(0.0, R.TICK - (time.time() - t0))); t0 = time.time()
        mem = (eye.spots, eye.t, eye.map, eye.back, eye.road, eye.link) if ret else None
        with LOCK:
            STATE["dead"] = bool(w.dead); STATE["ended"] = "모자라서 끝남" if w.dead else "끝까지 온전"
        time.sleep(2.0)


PAGE = """<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ATANOR 3D 세계</title>
<style>
:root{--bg:#f6f5f1;--fg:#1d1d1b;--mut:#6b6a66;--card:#fff;--line:#dddbd3;--a:#2f6fe4;--b:#d9583b}
@media (prefers-color-scheme:dark){:root{--bg:#161614;--fg:#ecebe6;--mut:#a09f99;--card:#20201d;--line:#34332f}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,"Malgun Gothic",sans-serif}
main{max-width:980px;margin:0 auto;padding:16px}
h1{font-size:18px;margin:0 0 2px} #label{color:var(--mut);font-size:12.5px;margin-bottom:12px}
.row{display:flex;gap:14px;flex-wrap:wrap} .card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;flex:1 1 300px;min-width:0}
.card h2{font-size:13px;color:var(--mut);margin:0 0 8px;font-weight:600}
#eye{width:100%;image-rendering:pixelated;border-radius:6px;display:block} .view{position:relative} #ov{position:absolute;inset:0;width:100%;height:100%;max-width:none;margin:0} canvas{width:100%;max-width:420px;display:block;margin:0 auto;border-radius:6px}
.bar{height:14px;background:var(--line);border-radius:7px;overflow:hidden;margin:3px 0 9px} .bar i{display:block;height:100%}
.kv{display:flex;justify-content:space-between;gap:8px;font-variant-numeric:tabular-nums} .kv span:first-child{color:var(--mut)}
.pl{display:grid;grid-template-columns:110px 1fr 48px;gap:6px;align-items:center;font-size:13px;font-variant-numeric:tabular-nums}
.pl .t{height:9px;background:var(--line);border-radius:5px;overflow:hidden} .pl .t i{display:block;height:100%;background:var(--a)}
.now{font-weight:700} small{color:var(--mut)}
</style></head><body><main>
<h1>ATANOR · 1인칭 3D 세계</h1><div id="label">연결 중…</div>
<div class="row">
 <div class="card"><h2>존재의 눈앞 (이것과 몸의 느낌만 받는다)</h2><div class="view"><img id="eye" alt="존재의 눈앞 그림"><canvas id="ov" width="640" height="400"></canvas></div>
  <div id="keptwrap" style="display:none;margin-top:8px"><h2>눈이 실제로 받는 것 (초점 둘레만 곱게, 바깥은 거칠게)</h2><img id="kept" alt="눈이 받는 거친 그림" style="width:100%;image-rendering:pixelated;border-radius:6px;display:block"></div>
  <div id="depthwrap" style="display:none;margin-top:8px"><h2>제 걸음으로 배운 깊이 (파랑 = 멀다, 빨강 = 가깝다. 아직 존재의 고름에는 쓰지 않음)</h2><img id="depth" alt="배운 깊이" style="width:100%;image-rendering:pixelated;border-radius:6px;display:block"></div>
  <small id="legend" data-rays="노란 상자 = 가운데 시선 ('하기'가 닿는 곳. 닿는 거리 안이면 굵게). 작은 표시 = 나머지 시선 8줄 (주변)."></small></div>
 <div class="card"><h2>위에서 본 세계 (사람이 보라고 그린 것. 존재는 받지 않는다)</h2><canvas id="map" width="420" height="420"></canvas>
  <div id="mindwrap" style="display:none;margin-top:12px"><h2>머릿속 지도 (존재가 제 몸과 눈으로 그린 것. 제 걸음 기준)</h2><canvas id="mind" width="420" height="420"></canvas>
  <small>밝은 칸 = 트였다고 아는 곳, 짙은 칸 = 막혔다고 아는 곳, 빈 곳 = 모름. 고리 = 속으로 걸어 보고 고른 "저리로". 점 = 적어 둔 자리 (빈 점 = 가 보니 없던 곳).</small></div>
  <small>파란 칸 = 마실 것, 붉은 칸 = 먹을 것, 잿빛 = 벽. 작은 점 = 존재가 제 걸음으로 적어 둔 자리.</small></div>
</div>
<div class="row" style="margin-top:14px">
 <div class="card"><h2>몸 안</h2>
  <div class="kv"><span>마실 것</span><span id="s0"></span></div><div class="bar"><i id="b0" style="background:var(--a)"></i></div>
  <div class="kv"><span>먹을 것</span><span id="s1"></span></div><div class="bar"><i id="b1" style="background:var(--b)"></i></div>
  <div class="kv"><span>삶</span><span id="life"></span></div><div class="kv"><span>이 삶의 시계</span><span id="clock"></span></div>
  <div class="kv"><span>채운 횟수</span><span id="fills"></span></div><div class="kv"><span>살면서 고친 횟수 (내다보기의 틀림)</span><span id="mended"></span></div><div class="kv" id="watchrow" style="display:none"><span>어른이 먹는 것을 본 횟수 (그 길을 적음)</span><span id="watchv"></span></div><div class="kv"><span>지금 하는 것</span><span id="act" class="now"></span></div></div>
 <div class="card"><h2>마음 안의 신호 (이름은 우리가 붙인 것. 존재 안에는 이름이 없다)</h2>
  <div class="kv"><span>가장 급한 모자람</span><span id="lack"></span></div><div class="kv" id="innerrow" style="display:none"><span>속 상태의 세기 (순간에서 순간으로 이어지는 것)</span><span id="innerv"></span></div><div class="kv"><span>끌림</span><span id="pulled"></span></div>
  <div id="pull" style="margin-top:8px"></div></div>
</div></main><script>
const $=id=>document.getElementById(id);
function draw(s){const c=$('map'),g=c.getContext('2d'),n=s.map.length,u=c.width/n;g.clearRect(0,0,c.width,c.height);
 for(let x=0;x<n;x++)for(let y=0;y<n;y++){const k=s.map[x][y];g.fillStyle=k?`rgb(${s.colors[k].join(',')})`:'rgba(128,128,128,.10)';g.fillRect(x*u,y*u,u-1,u-1);}
 const tc=['#888','#d9583b','#3a3','#2f6fe4'];for(const p of s.seen){g.fillStyle=tc[p[2]];g.beginPath();g.arc(p[0]*u,p[1]*u,3,0,7);g.fill();g.strokeStyle='#fff';g.lineWidth=1;g.stroke();}
 const px=s.pos[0]*u,py=s.pos[1]*u;g.fillStyle='rgba(240,200,60,.22)';g.beginPath();g.moveTo(px,py);g.arc(px,py,u*5,s.head-s.fov/2,s.head+s.fov/2);g.closePath();g.fill();
 g.fillStyle='#f0c83c';g.beginPath();g.arc(px,py,u*.32,0,7);g.fill();g.strokeStyle='#222';g.lineWidth=2;g.beginPath();g.moveTo(px,py);g.lineTo(px+Math.cos(s.head)*u*.7,py+Math.sin(s.head)*u*.7);g.stroke();}
function over(s){const c=$('ov'),g=c.getContext('2d'),W=c.width,H=c.height,n=s.lines.length,tc=['#bbb','#ff8a6a','#6c6','#7fb0ff'];g.clearRect(0,0,W,H);
 if(s.retina){const b=s.gaze;g.strokeStyle='#ffd400';g.lineWidth=4;g.strokeRect(b[0]*W,b[1]*H,(b[2]-b[0])*W,(b[3]-b[1])*H);const cc=['#ff8a6a','#6c6','#7fb0ff'];
  for(const k of s.kept){const x=(k[1]/s.fov+0.5)*W;if(x<0||x>W)continue;g.beginPath();g.arc(x,H*0.9-k[0]*14,7,0,7);g.lineWidth=2;g.strokeStyle='#ffd400';g.stroke();if(k[3]){g.fillStyle=cc[k[0]];g.fill();}}return;}
 s.lines.forEach((l,i)=>{const x=(i/(n-1))*(W-1),y=H/2;if(i==(n-1)/2){const near=l[0]>0&&l[1]<=s.reach;g.strokeStyle='#ffd400';g.lineWidth=near?6:3;g.strokeRect(x-22,y-22,44,44);}
  else{g.beginPath();g.arc(Math.min(Math.max(x,7),W-7),y,6,0,7);g.lineWidth=2;g.strokeStyle='#ffd400';g.stroke();if(l[0]>0){g.fillStyle=tc[l[0]];g.fill();}}});}
function mindmap(s){const w=$('mindwrap');if(!s.mind){w.style.display='none';return;}w.style.display='block';
 const c=$('mind'),g=c.getContext('2d'),m=s.mind,n=m.cells.length,u=c.width/n;g.clearRect(0,0,c.width,c.height);
 for(let y=0;y<n;y++)for(let x=0;x<n;x++){const v=m.cells[y][x];if(v=='0')continue;g.fillStyle=v=='1'?'rgba(150,190,150,.55)':'rgba(90,85,80,.95)';g.fillRect(x*u,y*u,u-1,u-1);}
 const cc=['#d9583b','#3a3','#2f6fe4'];
 for(const p of m.spots){g.beginPath();g.arc((p[0]+.5)*u,(p[1]+.5)*u,4,0,7);g.lineWidth=2;g.strokeStyle=cc[p[2]];g.stroke();if(p[3]){g.fillStyle=cc[p[2]];g.fill();}}
 for(const p of m.ways){g.beginPath();g.arc((p[0]+.5)*u,(p[1]+.5)*u,8,0,7);g.lineWidth=2.5;g.strokeStyle=cc[p[2]];g.stroke();}
 const px=(m.me[0]+.5)*u,py=(m.me[1]+.5)*u;g.fillStyle='#f0c83c';g.beginPath();g.arc(px,py,u*.4,0,7);g.fill();g.strokeStyle='#222';g.lineWidth=2;g.beginPath();g.moveTo(px,py);g.lineTo(px+Math.cos(m.head)*u,py+Math.sin(m.head)*u);g.stroke();}
async function tick(){try{const s=await (await fetch('state')).json();if(s.label){
 $('label').textContent='버전 '+s.label;$('keptwrap').style.display=s.retina?'block':'none';if(s.retina)$('kept').src='data:image/png;base64,'+s.kept_img;$('depthwrap').style.display=s.depth_img?'block':'none';if(s.depth_img)$('depth').src='data:image/png;base64,'+s.depth_img;$('legend').textContent=s.retina?'노란 상자 = 초점 (곱게 보는 곳). 아래 동그라미 = 빛깔의 맞섬마다 지금 느끼는 쪽 (채워짐 = 지금 보임, 빈 것 = 마지막으로 본 쪽을 기억).':$('legend').dataset.rays;$('eye').src='data:image/png;base64,'+s.frame;draw(s);over(s);mindmap(s);
 for(const i of [0,1]){$('s'+i).textContent=Math.round(s.store[i]*100)+'%';$('b'+i).style.width=(s.store[i]*100)+'%';}
 $('life').textContent=s.life+'번째'+(s.ended&&(s.dead||s.clock>=s.span)?' · '+s.ended:'');$('clock').textContent=s.clock+'초 / '+s.span+'초';$('fills').textContent=s.fills+'번';$('watchrow').style.display=(s.watched!=null)?'flex':'none';if(s.watched!=null)$('watchv').textContent=s.watched+'번';$('innerrow').style.display=(s.inner_norm!==null&&s.inner_norm!==undefined)?'flex':'none';if(s.inner_norm!=null)$('innerv').textContent=s.inner_norm;$('mended').textContent=s.mended?(s.mended+'번 ('+s.mend_loss+')'):'고치지 않음';$('act').textContent=s.act;
 $('lack').textContent=s.lack+' ('+Math.round(s.lack_size*100)+'% 비었음)';$('pulled').textContent=s.pulled?'있음 (풀렸던 순간을 떠올림)':(s.seeking?'갈 데를 모름 (찾아냈던 순간에 기댐)':'없음 (헤맴)');
 const lo=Math.min(...s.pull),hi=Math.max(...s.pull),sp=(hi-lo)||1;
 $('pull').innerHTML=s.acts.map((a,i)=>`<div class="pl"><span class="${a==s.act?'now':''}">${a}</span><span class="t"><i style="width:${s.pulled?Math.round((s.pull[i]-lo)/sp*100):0}%"></i></span><span>${s.pull[i].toFixed(2)}</span></div>`).join('')+(s.futile&&Math.max(...s.futile)>0.01?'<div style="margin-top:8px;color:var(--mut);font-size:12.5px">헛됨이 쌓인 것: '+s.acts.map((a,i)=>s.futile[i]>0.01?a+' '+s.futile[i].toFixed(2):'').filter(Boolean).join(', ')+'</div>':'');
 }}catch(e){$('label').textContent='연결이 끊겼습니다. 다시 시도 중…';}setTimeout(tick,200);}
tick();</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.split("?")[0].endswith("state"):
            with LOCK:
                body = json.dumps(STATE).encode(); ct = "application/json"
        else:
            body = PAGE.encode("utf-8"); ct = "text/html; charset=utf-8"
        self.send_response(200); self.send_header("Content-Type", ct); self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=""); ap.add_argument("--port", type=int, default=8769)
    ap.add_argument("--walls", type=int, default=3); ap.add_argument("--seed", type=int, default=int(time.time()) % 100000)
    ap.add_argument("--span", type=float, default=120.0); ap.add_argument("--loose", type=float, default=0.05); ap.add_argument("--curious", type=float, default=0.0)
    ap.add_argument("--roam", action="store_true"); ap.add_argument("--seek", action="store_true"); ap.add_argument("--pointer", default="")
    a = ap.parse_args()
    threading.Thread(target=live, args=(a,), daemon=True).start()
    print(f"room3d viewer on http://127.0.0.1:{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
