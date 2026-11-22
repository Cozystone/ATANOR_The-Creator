"""깊이 눈 뷰어 (재는 것, 존재에게 아무것도 주지 않는다): 마크 뷰어(8770)의 첫 몸 1인칭 그림을 받아, 존재의 깊이 눈 (제 걸음으로 배운 정본 depth_learner)으로
어떻게 보이는지를 나란히 보인다. 깊이 눈을 붙이지 않은 판에서도 '그 눈이 있다면 이렇게 보인다'를 보려고.
  python -m atanor_os.emerge.mc_depthwatch --model mc/mc_depth.pt --port 8772
"""
import argparse, base64, io, json, threading, time, urllib.request
import numpy as np, torch
from PIL import Image
from .mc_depth import DepthEye, picture

PAGE = """<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ATANOR 깊이 눈</title>
<style>:root{--bg:#f6f5f1;--card:#fff;--ink:#222;--mute:#666;--line:#e2dfd6}@media (prefers-color-scheme:dark){:root{--bg:#1b1b1b;--card:#262626;--ink:#eee;--mute:#aaa;--line:#3a3a3a}}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,sans-serif}main{max-width:1000px;margin:0 auto;padding:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin-bottom:14px}img{width:100%;image-rendering:pixelated;border-radius:6px;display:block}
h1{font-size:20px;margin:0 0 4px}h2{font-size:14px;margin:0 0 8px;color:var(--mute)}.two{display:grid;grid-template-columns:1fr 1fr;gap:14px}@media(max-width:700px){.two{grid-template-columns:1fr}}</style></head><body><main>
<h1>ATANOR · 깊이 눈</h1><div id="ver" style="font-weight:600;font-size:13px"></div><div id="label" style="color:var(--mute);font-size:12px"></div>
<div class="two"><div class="card"><h2>1인칭 그림 (첫 몸)</h2><img id="eye" alt="1인칭"></div>
<div class="card"><h2>깊이 눈이 보는 것: 밝을수록 가까움 (제 걸음으로 배움, 이름표 없음). 아래 막대 = 눈높이 5갈래의 가까움. 판정: 바로 앞 갈래는 참 거리와 r 0.95, 양옆 갈래는 아직 맞지 않음</h2><img id="depth" alt="깊이"></div></div>
<div class="card" style="color:var(--mute);font-size:12px">이 판의 존재가 깊이 눈을 받는지: <b id="uses"></b>. 받지 않는 판에서는 '그 눈이 있다면 이렇게 보인다'를 보여 줄 뿐이다.</div>
</main><script>const $=i=>document.getElementById(i);
async function tick(){try{const s=await (await fetch('/state')).json();if(s.big){$('eye').src='data:image/jpeg;base64,'+s.big;}if(s.depth){$('depth').src='data:image/png;base64,'+s.depth;}
$('ver').textContent=(s.version?'버전 v'+s.version+' · ':'')+(s.note||'');$('label').textContent=s.label||'';$('uses').textContent=s.uses?'받음':'받지 않음 (보기만)';}catch(e){}setTimeout(tick,400)}tick();</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--port", type=int, default=8772); ap.add_argument("--src", default="http://127.0.0.1:8770/state")
    a = ap.parse_args()
    E = DepthEye(a.model, "cuda"); VIEW, LOCK = {}, threading.Lock()

    def loop():
        while True:
            try:
                s = json.loads(urllib.request.urlopen(a.src, timeout=5).read())
                if s.get("big"):
                    im = Image.open(io.BytesIO(base64.b64decode(s["big"]))).convert("RGB")
                    pic = torch.tensor(np.asarray(im.resize((80, 48), Image.BILINEAR), np.float32) / 255.0).permute(2, 0, 1)[None].cuda()
                    d = E.depth(pic); png = picture(d[0], E.near(d)[0])
                    with LOCK:
                        VIEW.update(big=s["big"], depth=base64.b64encode(png).decode(), version=s.get("version", ""), note=s.get("note", ""), label=s.get("label", ""), uses=bool(s.get("depth")))
            except Exception:
                pass
            time.sleep(0.3)
    threading.Thread(target=loop, daemon=True).start()
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class V(BaseHTTPRequestHandler):
        def log_message(self, *x):
            pass

        def do_GET(self):
            if self.path.startswith("/state"):
                with LOCK:
                    body = json.dumps(VIEW, ensure_ascii=False).encode(); ct = "application/json; charset=utf-8"
            else:
                body, ct = PAGE.encode(), "text/html; charset=utf-8"
            self.send_response(200); self.send_header("Content-Type", ct); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    ThreadingHTTPServer(("0.0.0.0", a.port), V).serve_forever()


if __name__ == "__main__":
    main()
