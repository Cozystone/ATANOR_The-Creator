"""노트북 몸으로 산다 (연속 시간). 몸짓의 모드:
  sense  : 몸짓 없음 (배관만 - 칸이 들어오나, 셈 비용)
  random : 무작위 몸짓 (1-5걸음 붙잡음) - 갈래 -> 칸 영향이 신호에 실제로 있나 (재는 쪽에서)
  mind   : 존재가 고름 (다음 단계)
재는 쪽 (judge): 진짜 배선을 쥐고, 레버의 움직임 (t)과 각 장치 값의 바뀜 (t -> t+k)의 상관을 메모리에서 셈. 파일에는 상관 같은 모음 수치만 (그림·소리 날것 없음).
뷰어 (127.0.0.1): 존재가 보는 낮은 해상도 (80x48), 소리 세기 막대, 장치 값 (재는 쪽 이름), 레버 상태, 영향 표.
  python -m atanor_os.emerge.laptop_live --mode random --minutes 30 --view_port 8775"""
import argparse, json, os, time
import numpy as np, torch, torch.nn.functional as F
from .laptop_body import LaptopBody
from .body_contract import Organs
from .mind import Mind
from .host_view import ViewServer

PAGE = """<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ATANOR laptop body</title><style>
:root{--bg:#0f1115;--fg:#e6e6e6;--mut:#8a93a3;--card:#171a21;--line:#262b36;--acc:#4aa3ff}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}main{max-width:1100px;margin:0 auto;padding:16px}
h1{font-size:18px;margin:0 0 4px}.mut{color:var(--mut)}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px;margin-top:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px;overflow-x:auto;min-width:0}td{word-break:break-word}img{width:100%;image-rendering:pixelated;border-radius:6px}
table{border-collapse:collapse;width:100%;font-size:13px}td{padding:2px 4px;border-bottom:1px solid var(--line);vertical-align:top}td:first-child{color:var(--mut)}
.bars{display:flex;gap:2px;align-items:flex-end;height:60px}.bars i{flex:1;background:var(--acc);border-radius:2px 2px 0 0}
</style></head><body><main><h1>ATANOR - 노트북 몸</h1><div class="mut" id="sub">불러오는 중...</div>
<div class="grid"><div class="card"><b>존재가 보는 것</b> <span class="mut">(흰 테 = 중심와 16px 그대로, 둘레 고리 4개는 갈수록 흐림, 그 밖은 큰세포 길 80x48)</span><img id="eye"></div>
<div class="card"><b>존재가 듣는 것</b> <span class="mut">(16칸 세기)</span><div class="bars" id="ear"></div><b style="display:block;margin-top:8px">깊이 (왼쪽 -> 오른쪽 5갈래, 가까움)</b><div class="bars" id="dep" style="height:40px"></div>
<b style="display:block;margin-top:10px">레버 (재는 쪽 이름)</b><table id="lev"></table></div></div>
<div class="grid"><div class="card"><b>장치 값</b> <span class="mut">(이름은 재는 쪽만 앎 - 존재는 섞인 이름 없는 칸으로 받음)</span><table id="ch"></table></div>
<div class="card"><b>영향 (재는 쪽)</b> <span class="mut">레버 움직임 -> 2걸음 안 값의 바뀜, 상관이 큰 것</span><table id="inf"></table></div></div>
</main><script>
async function tick(){try{const s=await (await fetch('state.json?'+Date.now())).json();
document.getElementById('sub').textContent=`${s.time} | 모드 ${s.mode} | 걸음 ${s.step} | 존재의 나이 ${s.age} | 온전함 ${s.whole} | 값의 믿음 ${s.trust} | 몸짓 간격 ${s.gap}s`;
document.getElementById('eye').src='frame0.png?'+s.step;
document.getElementById('ear').innerHTML=s.ear.map(v=>`<i style="height:${Math.max(2,Math.min(100,(v+9)*9))}%"></i>`).join('');
document.getElementById('dep').innerHTML=(s.depth||[]).map(v=>`<i style="height:${Math.max(2,Math.min(100,v*100))}%;background:#e0a050"></i>`).join('');
document.getElementById('lev').innerHTML=s.levers.map(r=>`<tr><td>${r[0]}</td><td>${r[1]}</td></tr>`).join('');
document.getElementById('ch').innerHTML=s.channels.map(r=>`<tr><td>${r[0]}</td><td>${r[1]}</td></tr>`).join('');
document.getElementById('inf').innerHTML=s.influence.map(r=>`<tr><td>${r[0]}</td><td>${r[1]}</td><td>${r[2]}</td></tr>`).join('')||'<tr><td>아직 모으는 중</td></tr>';
}catch(e){document.getElementById('sub').textContent='응답 없음: '+e}}setInterval(tick,800);tick();</script></body></html>"""


def elements(body):
    """재는 쪽: 장치 값을 이름 붙은 낱값으로 (그림은 밝기 하나, 소리는 세기 하나)."""
    names, vals = [], []
    for p, v in zip(body.phys, body.last_phys):
        if p[1] == "image":
            names.append("camera mean brightness"); vals.append(float(np.asarray(v).mean()))
        elif p[1] == "audio":
            names.append("microphone mean level"); vals.append(float(np.mean(v)))
        else:
            for i, x in enumerate(np.asarray(v).ravel()):
                names.append(f"{p[0].split(':')[0]}[{i}]"); vals.append(float(x))
    return names, np.array(vals)


def tempo_heat(B, names, L, E):
    """재는 쪽: 뜨거울수록 제 박자를 늦추나 - 가장 뜨거운 온도 칸과 박자 레버 (느림 -1 .. 빠름 +1)의 상관, 그리고 뜨거운 때 / 식은 때의 박자 평균."""
    j = [k for k, lv in enumerate(B.levers) if lv[0].startswith("own")][0]
    ti = [i for i, n in enumerate(names) if n.startswith("sysfs[") and int(n[6:-1]) < len(B.thermal)]
    if not ti or len(L) < 50:
        return None
    hot = E[:, ti].max(1); s = L[:, j]; hi = hot > np.median(hot)
    c = float(np.corrcoef(hot, s)[0, 1]) if hot.std() > 0 and s.std() > 0 else 0.0
    return dict(corr=round(c, 3), tempo_when_hot=round(float(s[hi].mean()), 3), tempo_when_cool=round(float(s[~hi].mean()), 3), hot_median_mC=float(np.median(hot)))


def foveated(O, img):
    """존재의 작은세포 길이 받는 것을 그림으로 (재는 쪽): 고리 4 -> 1은 갈수록 덜 흐린 16x16을 제 상자에 펴고, 가운데는 중심와 16px 그대로.
    고리 밖은 큰세포 길 (온 화면 80x48)을 어둡게. 중심와 상자에 흰 테."""
    i = O.img[0]; H_, W_ = img.shape[:2]; fr = torch.as_tensor(np.ascontiguousarray(img), dtype=torch.float32).permute(2, 0, 1)[None] / 255.0
    can = F.interpolate(F.adaptive_avg_pool2d(fr, (48, 80)), size=(H_, W_), mode="nearest")[0] * 0.55
    r = O.r[0]; boxes = O.ret[i].boxes
    for k in range(min(4, len(boxes) - 1), 0, -1):
        x0, y0, h, w = [int(v[0]) if torch.is_tensor(v) else int(v) for v in boxes[k]]
        can[:, y0:y0 + h, x0:x0 + w] = F.interpolate(r[3 * k:3 * k + 3][None], size=(h, w), mode="bilinear", align_corners=False)[0]
    x0, y0, h, w = [int(v[0]) if torch.is_tensor(v) else int(v) for v in boxes[0]]
    can[:, y0:y0 + h, x0:x0 + w] = r[0:3]
    out = (can.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    out[max(y0 - 1, 0), x0:x0 + w] = 255; out[min(y0 + h, H_ - 1), x0:x0 + w] = 255
    out[y0:y0 + h, max(x0 - 1, 0)] = 255; out[y0:y0 + h, min(x0 + w, W_ - 1)] = 255
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", default="sense"); ap.add_argument("--minutes", type=float, default=0); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--view_port", type=int, default=0); ap.add_argument("--out", default="being/laptop/judge.json"); ap.add_argument("--hold", type=int, default=5)
    ap.add_argument("--load", default=""); ap.add_argument("--being", default="being/atanor_laptop.pt"); ap.add_argument("--depth", default="model/mc_depth.pt")
    ap.add_argument("--threads", type=int, default=4); ap.add_argument("--save_every", type=int, default=600)
    ap.add_argument("--loki", default="", help="LOKI v4 checkout (pinned 24e2313); empty = off (the baseline life)")
    ap.add_argument("--loki_state", default="being/laptop/loki_v4.npz"); ap.add_argument("--loki_k", type=int, default=256)
    ap.add_argument("--body", default="laptop", help="laptop | mock (tools/loki/mock_laptop.py, no hardware; judge-side practice)")
    ap.add_argument("--no_sleep", action="store_true", help="mock only: do not wait the gap in real time (accelerated practice, kept apart)")
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    if a.body == "mock":
        from tools.loki.mock_laptop import MockLaptop
        B = MockLaptop(seed=a.seed)
    else:
        B = LaptopBody(seed=a.seed)
    view = ViewServer(a.view_port, page=PAGE) if a.view_port else None
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(B.wiring(), open(os.path.join(os.path.dirname(a.out) or ".", "judge_wiring.json"), "w"), ensure_ascii=False, indent=1)
    O = Organs(B.c, 1, dev="cpu", depth=a.depth); Mi = None
    if a.mode == "mind":
        Mi = Mind(O, B.c, 1, dev="cpu")
        if os.path.exists(a.being):
            print("continue laptop being at step", Mi.load(a.being), flush=True)
        elif a.load and os.path.exists(a.load):
            info = Mi.wear(a.load); json.dump(info, open(os.path.join(os.path.dirname(a.out) or ".", "wear.json"), "w"), ensure_ascii=False, indent=1)
            print("wore", a.load, "steps", info["steps"], "eye dims", info.get("eye_dims"), "moved", len(info.get("moved", {})), flush=True)
        if Mi.wo is not None:
            Mi.wo.on = True
        Mi.begin(torch.ones(1, dtype=torch.bool))
    M = len(B.combos); rng = np.random.default_rng(a.seed); act, left = 0, 0
    lever_sig = {"backlight": lambda o: o - 1, "volume": lambda o: o - 1, "sound": lambda o: float(o > 0), "own": lambda o: o - 1}
    LS, EV, XS = [], [], []; t0 = time.time(); step = 0; last_w = time.time(); last_v = 0.0
    no = torch.zeros(1, dtype=torch.bool)
    raw = B.raw(); xr = O(raw); x = Mi.feel(xr) if Mi else xr
    LB = LE = None
    if a.loki:                                                              # LOKI organ, judged in parallel - it does not take part in choosing
        from .loki_bridge import LokiBridge, LokiEval
        LB = LokiBridge(B.c, O, path=a.loki, wiring_id=B.wiring_id())
        loaded = os.path.exists(a.loki_state) and LB.restore(a.loki_state)
        print("loki v4 organ", "restored" if loaded else "fresh", "matrix bytes", LB.mem.state_bytes, "width", LB.width, "grid", LB.grid, flush=True)
        LB.begin(raw, xr, B.available())
        wi = B.wiring(); dn = []                                            # judge side: LOKI dim -> device (names never reach the organ)
        for k in O.bed + O.vec:
            dev = wi["sensors"][k]["device"].split(":")[0]; n = int(np.prod(B.c.sensors[k].shape)); perm = wi["sensors"][k]["within_order"] or list(range(n))
            dn += [f"{'microphone' if dev.startswith('microphone') else dev}[{perm[q]}]" for q in range(n)]
        nth = len(B.thermal); G = {}
        for i, nm in enumerate(dn):
            g = {"lever readback[0]": "backlight readback", "lever readback[1]": "volume readback", "lever readback[2]": "tempo readback", "insides[0]": "heat margin (bedrock)"}.get(nm)
            if g is None:
                g = ("thermal zones" if nm.startswith("sysfs[") and int(nm[6:-1]) < nth else "microphone" if nm.startswith("microphone") else
                     "accelerometer" if nm.startswith("accelerometer") else "other vector")
            G.setdefault(g, []).append(i)
        G["camera grid"] = list(range(LB.width - LB.gsize, LB.width))
        LE = LokiEval(LB, G, k=a.loki_k); last_l = time.time(); last_e = time.time(); emap = {}
        TRUEDEV = {"backlight": ["lever readback[0]"], "volume": ["lever readback[1]"], "own": ["lever readback[2]"], "sound": [f"microphone[{q}]" for q in range(16)]}
        true_dims = {j: [dn.index(d) for d in TRUEDEV[lv[0].split(" ")[0].rstrip(":")] if d in dn] for j, lv in enumerate(B.levers)}
    while not a.minutes or time.time() - t0 < a.minutes * 60:
        ts = time.time(); names, ev = elements(B)
        if a.mode == "random":
            if left <= 0:
                act, left = int(rng.integers(M)), int(rng.integers(1, a.hold + 1))
            left -= 1
        elif a.mode == "mind":
            act = int(Mi.choose(x)[0])
        else:
            act = None
        if act is not None:
            B.step(torch.tensor([act]))
            sig = [lever_sig[lv[0].split(" ")[0].rstrip(":")](o) for lv, o in zip(B.levers, B.last_opts)]
        else:
            sig = [0.0] * len(B.levers)
        if LB is not None:                                                  # forecast after the act, before the next raw
            tl = time.time(); LB.forecast(act)
            fpred = Mi.net.foresee(x)[0, act].detach().numpy() if (Mi is not None and act is not None) else None
            lat = time.time() - tl
        spent = time.time() - ts
        if spent < B.gap and not a.no_sleep:
            time.sleep(B.gap - spent)                                       # 세계가 흐른 뒤에 본다 (연속 시간)
        raw = B.raw(); xr2 = O(raw)
        if LB is not None:                                                  # the next raw arrived: score, then learn (same raw, same O output as the Mind's)
            tl = time.time(); scored = LB.arrive(raw, xr2, act, B.available())
            LE.beat(scored, x, fpred, act, xr2, time.time(), gap=B.gap)
            if scored is not None:
                def roll(acts, z0=(Mi.feel(xr2) if Mi else xr2).clone()):
                    with torch.no_grad():
                        z = z0
                        for q in acts:
                            if q is not None:
                                z = Mi.net.foresee(z)[:, q]
                                z = torch.cat([z[:, :LB.nbed].clamp(0, 1), z[:, LB.nbed:].clamp(-1, 1)], 1)
                        return z[0].numpy()
                LE.snapshot(time.time(), scored[1], roll if Mi is not None else None)
            LE.lat.append(1000 * (lat + time.time() - tl)); LE.lat = LE.lat[-2000:]
            if time.time() - last_e > 300:                                  # judge: LOKI's own branch -> channel map (read-only)
                from .loki_bridge import effect_ranks
                last_e = time.time(); er = effect_ranks(LB, true_dims, seed=LE.t)
                emap = {B.levers[j][0]: dict(true_rank=v["true_rank"], of=v["of"], top=[dn[i] for i in v["top"]]) for j, v in er.items()}
            if time.time() - last_l > 60:
                last_l = time.time(); json.dump(dict(LE.report(), effect_map=emap, effect_path=LB.effect([0], [0])[1]), open(os.path.join(os.path.dirname(a.out) or ".", "loki_eval.json"), "w"), ensure_ascii=False, indent=1)
            if LE.t % a.save_every == 0:
                LB.save(a.loki_state)                                       # last-sense anchors blanked (no camera / mic moment on disk)
        if Mi is not None:
            Mi.carry(xr, torch.tensor([act]), no); x2 = Mi.feel(xr2)
            Mi.after(x, torch.tensor([act]), x2, no, no); x, xr = x2, xr2
            XS.append(x[0].detach().cpu().numpy().astype(np.float32)); XS = XS[-512:]   # 재는 쪽: 요즘 느낌 (판정이 진짜 느낌 위에서 잼)
            if Mi.steps % a.save_every == 0:
                Mi.save(a.being)
        else:
            xr = xr2
        LS.append(sig); EV.append(ev); LS, EV = LS[-4000:], EV[-4000:]; step += 1
        if view is not None and time.time() - last_v > 0.5:
            last_v = time.time(); img = np.asarray(B.last_phys[[p[1] for p in B.phys].index("image")])
            bed = B.last_phys[[p[4] for p in B.phys].index(True)]; dep = xr2[0, -5:].tolist() if O.DE is not None else []
            view.update([foveated(O, img)], dict(time=time.strftime("%Y-%m-%d %H:%M:%S %z"), mode=a.mode, step=step, age=(Mi.steps if Mi else "-"), gap=B.gap,
                        whole=round(float(1 - np.mean((1 - bed) ** 2)), 3), ear=[round(float(v), 2) for v in B.last_phys[[p[1] for p in B.phys].index("audio")]],
                        depth=[round(float(v), 2) for v in dep], trust=(round(float(Mi.wo.earned()), 3) if Mi and Mi.wo is not None else "-"),
                        levers=[[lv[0], (B.last_opts[j] if hasattr(B, "last_opts") else "-")] for j, lv in enumerate(B.levers)],
                        channels=[[n, round(float(v), 3)] for n, v in zip(names, ev)], influence=getattr(B, "_inf", [])))
        if time.time() - last_w > 60 and len(LS) > 200:
            last_w = time.time(); L = np.array(LS); E = np.array(EV); k = 2
            dE = E[k:] - E[:-k]; Ls = L[:-k]; inf = []
            for j, lv in enumerate(B.levers):
                xx = Ls[:, j] - Ls[:, j].mean()
                for i, nm in enumerate(names):
                    y = dE[:, i] - dE[:, i].mean(); c = float((xx * y).sum() / (np.linalg.norm(xx) * np.linalg.norm(y) + 1e-9))
                    inf.append((lv[0], nm, round(c, 3)))
            inf.sort(key=lambda r: -abs(r[2])); B._inf = [list(r) for r in inf[:12]]
            if XS:
                np.save(os.path.join(os.path.dirname(a.out) or ".", "recent_feel.npy"), np.stack(XS))
            if Mi is not None and len(Mi.buf) >= 2000 and step % 2400 < 300:             # 재는 쪽: 겪음 되감기 고리를 가끔 남김 (알맹이 배움 진단용, 노트북 안에만)
                Xb, Ab, X2b = [torch.cat([q[k] for q in Mi.buf]).cpu().numpy() for k in range(3)]
                np.savez(os.path.join(os.path.dirname(a.out) or ".", "recent_buf.npz"), x=Xb, a=Ab, x2=X2b)
            json.dump(dict(time=time.strftime("%Y-%m-%d %H:%M:%S %z"), mode=a.mode, steps=step, age=(Mi.steps if Mi else None), window=len(LS), top_influence=B._inf,
                           lever_use=[np.bincount(L[:, j].astype(int) + 1, minlength=3).tolist() for j in range(len(B.levers))],
                           trust=(float(Mi.wo.earned()) if Mi and Mi.wo is not None else None), camera_ok=B.cam.ok, mic_ok=B.mic.ok,
                           decided_by=(dict(zip(["strong pull", "value", "uniform random", "hold", "try"], [int(v) for v in Mi.src_count[0].tolist()]))
                                       if Mi is not None and getattr(Mi, "src_count", None) is not None else None),
                           tempo_vs_heat=tempo_heat(B, names, L, E)), open(a.out, "w"), ensure_ascii=False, indent=1)
    if Mi is not None:
        Mi.save(a.being)
    if LB is not None:
        LB.save(a.loki_state); json.dump(LE.report(), open(os.path.join(os.path.dirname(a.out) or ".", "loki_eval.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
