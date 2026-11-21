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
TURN = 2          # 고개 칸 수 (한 칸 15도): 한 번에 30도
ACTS = ["가만히", "앞으로", "왼쪽으로 돌기", "오른쪽으로 돌기", "때리기", "쓰기", "뛰며 앞으로"]


def body(conn, seed, h, w, small, big):
    import numpy as np, minedojo
    from PIL import Image
    import importlib.util
    from PIL import ImageDraw
    food0 = os.environ.get("ATANOR_MC_FOOD")                     # 태어날 때의 배부름 (세계 설정, 견주는 두 판에 똑같이). 없으면 MineDojo 기본 (가득)
    env = minedojo.make(task_id="open-ended", image_size=(h, w), generate_world_type="default", world_seed=seed, allow_mob_spawn=True,
                        **({"start_food": int(food0)} if food0 else {}))
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
        x0, y0, x1, y1 = s["fovea_box"]; c = np.abs(g[y0:y1, x0:x1] - q[y0:y1, x0:x1]).mean()
        X0, Y0, X1, Y1 = s["levels"][0]["box"]; ring = np.abs(g[Y0:Y1, X0:X1] - q[Y0:Y1, X0:X1]); sur = (ring.sum() - np.abs(g[y0:y1, x0:x1] - q[y0:y1, x0:x1]).sum()) / max(ring.size - (y1 - y0) * (x1 - x0), 1)
        return float(max(c - sur, 0.0) / 64.0)

    def pack(obs, done, hurt, k=None):
        rgb = np.asarray(obs["rgb"]).transpose(1, 2, 0)
        if CROP:                                                  # 화면 아래의 HUD (하트, 배부름, 손에 든 칸, 손)를 잘라 낸다: 몸 신호로 이미 받으므로 눈은 바깥세상만
            rgb = np.ascontiguousarray(rgb[:-CROP])
        im = Image.fromarray(rgb)
        ret, s = retina(rgb)
        mv = bug(rgb, s, k)
        ls = obs["life_stats"]; loc = obs.get("location_stats", {})
        pos = np.asarray(loc.get("pos", [0, 0, 0]), dtype=np.float32).ravel(); yaw = float(np.asarray(loc.get("yaw", [0])).ravel()[0])
        out = dict(ret=ret, bug=mv, gaze=(float(eye.cx), float(eye.cy)), health=float(np.asarray(ls["life"]).ravel()[0]) / 20, food=float(np.asarray(ls["food"]).ravel()[0]) / 20,
                   air=float(np.asarray(ls["oxygen"]).ravel()[0]) / 300, hurt=float(hurt), done=bool(done), pos=pos, yaw=yaw)
        if big:
            show = im.copy(); ImageDraw.Draw(show).rectangle(s["fovea_box"], outline=(255, 210, 0), width=2)
            b = io.BytesIO(); show.save(b, format="JPEG", quality=85); out["big"] = b.getvalue()
            b2 = io.BytesIO(); kept(s, rgb.shape[0], rgb.shape[1]).save(b2, format="PNG"); out["kept"] = b2.getvalue()
        return out

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
        a = env.action_space.no_op()
        k = int(cmd)
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
<div class="card"><h2>몸 (첫 몸)</h2>
<div class="row"><span>체력</span><span id="hv"></span></div><div class="bar"><div id="hb" style="background:#d9534f"></div></div>
<div class="row"><span>배부름</span><span id="fv"></span></div><div class="bar"><div id="fb" style="background:#c98a2e"></div></div>
<div class="row"><span>숨</span><span id="ov"></span></div><div class="bar"><div id="ob" style="background:#4f8fe0"></div></div>
<div class="row"><span>지금 한 것</span><b id="act"></b></div><div class="row"><span>누가 고르나</span><span id="who"></span></div>
<div class="row"><span>삶 / 걸음</span><span id="ls"></span></div><div class="row"><span>맞은 양 (이 걸음 / 이 삶)</span><span id="hurt"></span></div>
<div class="row"><span>몸의 수 / 초당 걸음 (모두 합쳐)</span><span id="sps"></span></div></div>
</main><script>
const $=i=>document.getElementById(i);
async function tick(){try{const s=await (await fetch('/state')).json();if(s.big){$('eye').src='data:image/jpeg;base64,'+s.big;$('small').src='data:image/png;base64,'+s.kept;
$('ver').textContent=(s.version?'버전 v'+s.version+' · ':'')+(s.note||'');$('label').textContent=s.label||'';const p=v=>Math.max(0,Math.min(100,100*v))+'%';
$('hv').textContent=(s.health*20).toFixed(0)+' / 20';$('hb').style.width=p(s.health);$('fv').textContent=(s.food*20).toFixed(0)+' / 20';$('fb').style.width=p(s.food);
$('ov').textContent=(s.air*300).toFixed(0)+' / 300';$('ob').style.width=p(s.air);$('act').textContent=s.act||'';$('who').textContent=s.who||'';
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
                show(obs[0], ACTS[int(acts[0])], m.get("who", ""), m.get("label", ""))
                send(s, [{k: v for k, v in o.items() if k not in ("big", "kept")} for o in obs])
        except (ConnectionError, OSError, EOFError) as e:
            print("client gone:", e, flush=True)
            with VLOCK:
                VIEW["who"] = "존재가 떠남 (다시 붙기를 기다림)"


if __name__ == "__main__":
    main()
