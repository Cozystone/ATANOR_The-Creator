"""Small live viewer for judge-side room runs (tools/a1_data.py --view): the body's own eye, a top-down map (judge only: springs, body, trail),
the two stores, and what the run is doing. Port 8768 (the room viewer address).
  from tools import room_live_view as V; V.start(); V.push(eye, rows, pos, head, stores, stock, label, text)"""
import base64, io, json, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
from PIL import Image, ImageDraw

STATE, LOCK = {"label": "시작 중"}, threading.Lock()
PAGE = """<!doctype html><meta charset=utf-8><title>ATANOR room</title><style>body{background:#111;color:#ddd;font:14px sans-serif;margin:16px}
img{image-rendering:pixelated;border:1px solid #333;margin:4px;max-width:100%}#l{font-size:16px;margin:8px 0}</style>
<div id=l></div><div><img id=e width=512><img id=m width=360></div><pre id=t></pre>
<script>async function f(){try{const r=await fetch('state');const d=await r.json();l.textContent=d.label||'';
if(d.e)e.src='data:image/jpeg;base64,'+d.e;if(d.m)m.src='data:image/png;base64,'+d.m;t.textContent=d.text||''}catch(x){}setTimeout(f,500)}f()</script>"""


def start(port=8768):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path.startswith("/state"):
                with LOCK:
                    body = json.dumps(STATE).encode(); ct = "application/json"
            else:
                body = PAGE.encode(); ct = "text/html; charset=utf-8"
            self.send_response(200); self.send_header("Content-Type", ct); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(body)
    threading.Thread(target=lambda: ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever(), daemon=True).start()


TRAIL = []


def push(eye, rows, pos, head, stores, stock, label, text):
    e = Image.fromarray(np.asarray(eye, np.uint8)); b = io.BytesIO(); e.save(b, format="JPEG", quality=80)
    S = len(rows); c = 16; im = Image.new("RGB", (S * c, S * c), (25, 25, 25)); d = ImageDraw.Draw(im)
    col = {"#": (90, 90, 90), "w": (60, 120, 255), "f": (230, 150, 40)}
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch in col:
                k = col[ch]
                if ch in "wf" and stock is not None:
                    a = float(stock[0 if ch == "w" else 1]); k = tuple(int(25 + (v - 25) * (0.25 + 0.75 * a)) for v in k)
                d.rectangle([x * c, y * c, x * c + c - 1, y * c + c - 1], fill=k)
    TRAIL.append((float(pos[0]), float(pos[1]))); del TRAIL[:-400]
    for i in range(1, len(TRAIL)):
        d.line([TRAIL[i - 1][0] * c, TRAIL[i - 1][1] * c, TRAIL[i][0] * c, TRAIL[i][1] * c], fill=(120, 200, 120), width=1)
    px, py = float(pos[0]) * c, float(pos[1]) * c
    d.ellipse([px - 5, py - 5, px + 5, py + 5], fill=(240, 240, 240))
    d.line([px, py, px + 12 * np.cos(head), py + 12 * np.sin(head)], fill=(240, 60, 60), width=2)
    m = io.BytesIO(); im.save(m, format="PNG")
    with LOCK:
        STATE.update(e=base64.b64encode(b.getvalue()).decode(), m=base64.b64encode(m.getvalue()).decode(), label=label, text=text)
