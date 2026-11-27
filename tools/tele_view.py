"""Inner-state dashboard (founder 2026-11-27; Copilot: plain axes of inner state, not "brain waves"), port 8777.
Judge-side display only. Left: time series of a few axes over the last 600 beats. Right: the being's own map - its place units where its
own path sense says they are (its birth frame), size = how often it has been there, colour = how much relief it remembers there for each
need, lines = crossings it has lived, the body's own believed position and its current goal.
  from tools import tele_view as T; T.start(); T.push(row_dict, map_dict)"""
import json, threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATE, LOCK = {"rows": deque(maxlen=600), "map": None, "label": ""}, threading.Lock()
AXES = [("trust", "귀가 믿음", "#e0b040"), ("lack", "모자람", "#e05050"), ("stuck", "고착 (맴돎)", "#b070e0"), ("again", "다시 생각", "#50b0e0"),
        ("explore", "탐색 (최근 칸 수 / 30)", "#50d080"), ("mismatch", "어긋남 (최근 300박자, /10)", "#ff8040"), ("attend", "시선이 배운 중요도로 끌린 몫", "#e0e0e0")]
PAGE = """<!doctype html><meta charset=utf-8><title>ATANOR inner state</title><style>body{background:#111;color:#ddd;font:13px sans-serif;margin:16px}
canvas{background:#181818;border:1px solid #333;margin:4px;max-width:100%}.k{display:inline-block;margin-right:14px}</style>
<div id=lab style="font-size:16px;margin-bottom:6px"></div><div id=keys></div>
<div><canvas id=ts width=760 height=360></canvas><canvas id=mp width=420 height=420></canvas></div>
<div id=note style="color:#999;max-width:1200px">왼쪽: 속 상태 축 (최근 600박자, 0~1). 오른쪽: 존재 자신의 지도 - 제 걸음 셈으로 기억하는 장소들 (태어난 자리 기준).
점 크기 = 들른 횟수, 파랑 = 물을 기억하는 정도, 주황 = 먹이를 기억하는 정도, 선 = 실제로 건너 본 길, 흰 점 = 제가 믿는 지금 자리, 빨간 고리 = 지금 목표.
갈색 칸 = 제 벽 지도 (가려다 막힌 곳). 판정 쪽 정보는 하나도 없음 - 존재 머릿속에 있는 것만.</div>
<script>
const AX=__AXES__;keys.innerHTML=AX.map(a=>`<span class=k style="color:${a[2]}">■ ${a[1]}</span>`).join('');
async function f(){try{const r=await fetch('state');const d=await r.json();lab.textContent=d.label||'';draw(d.rows||[]);mapd(d.map)}catch(x){}setTimeout(f,300)}
function draw(rows){const c=ts.getContext('2d');c.clearRect(0,0,ts.width,ts.height);const W=ts.width,H=ts.height,n=rows.length;if(!n)return;
c.strokeStyle='#333';for(let y=0;y<=4;y++){c.beginPath();c.moveTo(0,H*y/4);c.lineTo(W,H*y/4);c.stroke()}
for(const a of AX){c.strokeStyle=a[2];c.lineWidth=1.6;c.beginPath();rows.forEach((r,i)=>{const v=Math.max(0,Math.min(1,r[a[0]]??0));const x=W*(i/(Math.max(n-1,1))),y=H-v*H;i?c.lineTo(x,y):c.moveTo(x,y)});c.stroke()}
c.fillStyle='#888';c.fillText('박자 '+rows[n-1].t,6,12)}
function mapd(m){const c=mp.getContext('2d');c.clearRect(0,0,mp.width,mp.height);if(!m||!m.units.length)return;
const xs=m.units.map(u=>u[0]).concat([m.me[0]]),ys=m.units.map(u=>u[1]).concat([m.me[1]]);
const x0=Math.min(...xs)-1,x1=Math.max(...xs)+1,y0=Math.min(...ys)-1,y1=Math.max(...ys)+1,s=Math.min(400/(x1-x0),400/(y1-y0));
const P=(x,y)=>[10+(x-x0)*s,10+(y-y0)*s];c.fillStyle='#8a6d52';for(const q of (m.walls||[])){const [wx,wy]=P(q[0]-0.5,q[1]-0.5);c.fillRect(wx,wy,s,s)}c.strokeStyle='#444';c.lineWidth=1;
for(const [i,j] of m.links){const a=P(m.units[i][0],m.units[i][1]),b=P(m.units[j][0],m.units[j][1]);c.beginPath();c.moveTo(...a);c.lineTo(...b);c.stroke()}
m.units.forEach((u,i)=>{const [x,y]=P(u[0],u[1]);const r=2+Math.min(8,Math.sqrt(u[2])*0.6);c.fillStyle=`rgb(${Math.round(60+190*u[4])},${Math.round(60+120*u[4])},${Math.round(60+190*u[3])})`;
c.beginPath();c.arc(x,y,r,0,7);c.fill();if(i===m.goal){c.strokeStyle='#f33';c.lineWidth=2;c.beginPath();c.arc(x,y,r+4,0,7);c.stroke()}});
const [mx,my]=P(m.me[0],m.me[1]);c.fillStyle='#fff';c.beginPath();c.arc(mx,my,4,0,7);c.fill()}
f()</script>"""


def start(port=8777):
    page = PAGE.replace("__AXES__", json.dumps(AXES, ensure_ascii=False))

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path.startswith("/state"):
                with LOCK:
                    body = json.dumps(dict(rows=list(STATE["rows"]), map=STATE["map"], label=STATE["label"]), ensure_ascii=False).encode(); ct = "application/json"
            else:
                body = page.encode(); ct = "text/html; charset=utf-8"
            self.send_response(200); self.send_header("Content-Type", ct); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(body)
    threading.Thread(target=lambda: ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever(), daemon=True).start()


def push(row, mind_map=None, label=None):
    with LOCK:
        if row is not None:
            STATE["rows"].append(row)
        if mind_map is not None:
            STATE["map"] = mind_map
        if label is not None:
            STATE["label"] = label
