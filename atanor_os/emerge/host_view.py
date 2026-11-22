"""숙주 삶의 뷰어 (재는 것, 존재에게 아무것도 주지 않는다). host_live가 몇 걸음마다 넣어 준 것을 보여 줄 뿐:
몸마다 1인칭 그림 (존재가 받는 것과 같은 그림), 방을 위에서 본 지도 (몸의 자리·향함), 몸 안 (먹이·물), 삶의 상태 (표준시, 걸음, 삶, 얻은 믿음).
127.0.0.1에만 열린다 (밖에서는 ssh -L로 넘겨 봄). numpy와 표준 라이브러리만 씀 (숙주에 PIL 없음).
  ViewServer(port).update(frames [n,H,W,3] uint8, state dict)"""
import json, struct, threading, zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np


def png(img):
    """[H, W, 3] uint8 -> PNG 바이트."""
    h, w, _ = img.shape
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))

    def chunk(t, d):
        c = struct.pack(">I", len(d)) + t + d
        return c + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")


PAGE = """<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ATANOR host life</title><style>
:root{--bg:#0f1115;--fg:#e6e6e6;--mut:#8a93a3;--card:#171a21;--line:#262b36;--food:#e0a050;--water:#4aa3ff;--body:#ff5a7a}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
main{max-width:1100px;margin:0 auto;padding:16px}h1{font-size:18px;margin:0 0 4px}.mut{color:var(--mut)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px;margin-top:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px}
img,canvas{width:100%;image-rendering:pixelated;border-radius:6px;display:block}
.bar{height:8px;background:var(--line);border-radius:4px;overflow:hidden;margin:3px 0 6px}.bar i{display:block;height:100%}
table{border-collapse:collapse;width:100%}td{padding:2px 4px;border-bottom:1px solid var(--line)}td:first-child{color:var(--mut)}
</style></head><body><main>
<h1>ATANOR - 숙주에서의 연속 시간 삶</h1><div class="mut" id="sub">불러오는 중...</div>
<div class="grid" id="bodies"></div>
<div class="mut" style="margin-top:6px">지도: 회색 = 벽, 주황 = 먹이 샘, 파랑 = 물 샘, 분홍 = 몸 (선 = 향함). 몸마다 제 방 (같은 꼴, 샘 자리는 다름).</div>
<div class="grid"><div class="card"><b>상태</b><table id="st"></table><div class="mut" style="margin-top:6px">뷰어는 재기만 함. 존재는 이 화면을 보지 않음.</div></div></div>
</main><script>
let n=0;
function bars(b){return `<div>먹이</div><div class="bar"><i style="width:${(b.food*100).toFixed(0)}%;background:var(--food)"></i></div>
<div>물</div><div class="bar"><i style="width:${(b.water*100).toFixed(0)}%;background:var(--water)"></i></div>`}
async function tick(){try{const s=await (await fetch('state.json?'+Date.now())).json();
document.getElementById('sub').textContent=`${s.time} | 산 시간 ${s.lived} | 걸음 ${s.step} | 삶 ${s.life} (방 ${s.room}) | 판단 ${s.spd}s`;
const B=document.getElementById('bodies');if(n!==s.bodies.length){n=s.bodies.length;B.innerHTML=s.bodies.map((b,i)=>`<div class="card"><b>몸 ${i+1}</b> <span class="mut" id="d${i}"></span><img id="v${i}"><canvas id="m${i}" width="240" height="240" style="width:60%;margin:8px auto 4px"></canvas><div id="b${i}"></div></div>`).join('')}
s.bodies.forEach((b,i)=>{document.getElementById('v'+i).src=`frame${i}.png?${s.k}`;document.getElementById('b'+i).innerHTML=bars(b);
document.getElementById('d'+i).textContent=b.dead?'죽음':(b.dark?'어둠':'')});
const t=document.getElementById('st');t.innerHTML=Object.entries(s.table).map(([k,v])=>`<tr><td>${k}</td><td>${v}</td></tr>`).join('');
s.bodies.forEach((b,i)=>{const c=document.getElementById('m'+i),g=c.getContext('2d'),R=s.rows,h=R.length,w=R[0].length,u=Math.min(c.width/w,c.height/h);
g.fillStyle='#0b0d11';g.fillRect(0,0,c.width,c.height);
for(let y=0;y<h;y++)for(let x=0;x<w;x++){if(R[y][x]==='#'){g.fillStyle='#3a4050';g.fillRect(x*u,y*u,u,u)}}
g.fillStyle='#4aa3ff';g.fillRect(b.water_at[0]*u,b.water_at[1]*u,u,u);g.fillStyle='#e0a050';g.fillRect(b.food_at[0]*u,b.food_at[1]*u,u,u);
if(b.dead)return;const X=b.x*u,Y=b.y*u;g.fillStyle='#ff5a7a';g.beginPath();g.arc(X,Y,u*0.32,0,7);g.fill();
g.strokeStyle='#ff5a7a';g.lineWidth=2;g.beginPath();g.moveTo(X,Y);g.lineTo(X+Math.cos(b.h)*u*1.2,Y+Math.sin(b.h)*u*1.2);g.stroke()});
}catch(e){document.getElementById('sub').textContent='숙주 응답 없음: '+e}}
setInterval(tick,700);tick();
</script></body></html>"""


class ViewServer:
    def __init__(self, port=8773, host="127.0.0.1", page=None):
        self.lock = threading.Lock(); self.frames = []; self.state = b"{}"; self.page = page or PAGE
        vs = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                p = self.path.split("?")[0]
                with vs.lock:
                    if p in ("/", "/index.html"):
                        body, ct = vs.page.encode("utf-8"), "text/html; charset=utf-8"
                    elif p == "/state.json":
                        body, ct = vs.state, "application/json"
                    elif p.startswith("/frame") and p.endswith(".png") and p[6:-4].isdigit() and int(p[6:-4]) < len(vs.frames):
                        body, ct = vs.frames[int(p[6:-4])], "image/png"
                    else:
                        self.send_response(404); self.end_headers(); return
                self.send_response(200); self.send_header("Content-Type", ct); self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        self.srv = ThreadingHTTPServer((host, port), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def update(self, frames, state):
        fr = [png(np.ascontiguousarray(f)) for f in frames]
        js = json.dumps(state, ensure_ascii=False).encode("utf-8")
        with self.lock:
            self.frames, self.state = fr, js
