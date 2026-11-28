"""LV1 two-lever device (Astra's whole review 2026-11-28; first gate of the intellectual axis: its own causal experiment). Outside space.
A body with two insides that drain. Two unnamed levers. Pressing one of them brings, 8 beats later, a rise of whichever inside is lower;
pressing the other brings, 14 beats later, a fall of both. At first the levers are linked (they can only be pressed together): the rise and
the fall always come as a pair, and pressing both does not keep a body alive for long. Then they come apart: each can be pressed alone.
Nothing rewards trying a lever alone. Later the wiring is swapped without any change in how the device looks.
The being here is the common principle alone - not the room being: "on this cue, keeping this action, this is what later changed in my
body" (Astra's SomaticReadout, LOKI de50f9f, as wired for the gut in the room) with wholeness as the only judge. One continuing being:
a death refills the body, the memory goes on.
  PYTHONPATH=. python -m tools.lever2 OUT.json [--arm gut|both|random] [--seeds 24]      measured, no display
  PYTHONPATH=. python -m tools.lever2 --view 1                                             live on port 8779"""
import argparse, json, sys, time, threading
import torch

CHUNK, RISE_AT, FALL_AT, RISE, FALL, DRAIN = 16, 8, 14, 0.30, 0.10, 0.004
WAIT, BOTH, ONE, TWO = 0, 1, 2, 3
NAMES = ["기다림", "둘 다 누름", "첫째만 누름", "둘째만 누름"]


def gut():
    sys.path.insert(0, "ext/LOKI-v13")
    from loki_memory.somatic_readout import SomaticReadout
    return SomaticReadout(cue_width=2, channels=2, slots=32, events=128)


def life(arm, seed, T1=1500, T2=5000, T3=9000, view=None):
    g = torch.Generator().manual_seed(seed); G = gut() if arm == "gut" else None
    cue = torch.tensor([1.0, 0.0]); store = torch.tensor([0.75, 0.75]); good = ONE          # which single lever brings the rise (hidden)
    log = dict(deaths=[0, 0, 0], picks=[[0] * 4 for _ in range(3)], first_single=None, first_good_run=None, after_swap_wrong=None, after_swap_recovered=None)
    t = 0; run_good = 0; wrong_after = 0; chunks_after = 0
    while t < T3:
        ph = 0 if t < T1 else (1 if t < T2 else 2)
        if ph == 2 and good == ONE:
            good = TWO                                                                # the wiring is swapped; the device looks the same
        acts = torch.tensor([WAIT, BOTH] if ph == 0 else [WAIT, BOTH, ONE, TWO])
        df = 1.0 - store
        if arm == "gut":
            a, conf, called = G.choose(cue, acts, df, generator=g)
        elif arm == "both":
            a, conf, called = BOTH, 1.0, False
        else:
            a, conf, called = int(acts[torch.randint(len(acts), (1,), generator=g)]), 0.0, False
        log["picks"][ph][a] += 1
        if a in (ONE, TWO) and log["first_single"] is None:
            log["first_single"] = t - T1
        is_good = a == good
        if ph == 1:
            run_good = run_good + 1 if is_good else 0
            if run_good >= 5 and log["first_good_run"] is None:
                log["first_good_run"] = t - T1
        if ph == 2:
            chunks_after += 1
            if log["after_swap_recovered"] is None:
                wrong_after += int(a in (ONE, TWO) and not is_good)
                run_good = run_good + 1 if is_good else 0
                if run_good >= 5:
                    log["after_swap_recovered"] = chunks_after; log["after_swap_wrong"] = wrong_after
        before = df.clone(); died = False
        rise = a == BOTH or a == good; fall = a == BOTH or (a in (ONE, TWO) and a != good)
        for b in range(CHUNK):
            store = store - DRAIN
            if rise and b == RISE_AT:
                i = int(store.argmin()); store[i] = min(1.0, float(store[i]) + RISE)
            if fall and b == FALL_AT:
                store = store - FALL
            t += 1
            if view is not None:
                view(dict(t=t, phase=ph, store=store.clamp(0, 1).tolist(), act=a, beat=b, rise=bool(rise and b >= RISE_AT), fall=bool(fall and b >= FALL_AT), conf=round(conf, 2), called=bool(called),
                          deaths=log["deaths"], picks=log["picks"], good=int(good), first_single=log["first_single"], recovered=log["after_swap_recovered"],
                          gut=([dict(a=int(G.fast_action[j]), d=[round(float(v), 3) for v in G.fast_delta[j].tolist()], n=int(G.visits[j])) for j in range(len(G.visits)) if int(G.fast_action[j]) >= 0] if G is not None else [])))
                time.sleep(0.05)
            if float(store.min()) <= 0:
                died = True; break
        after = torch.ones(2) if died else 1.0 - store.clamp(0, 1)
        if arm == "gut":
            G.remember(cue, a, before, after)
        if died:
            log["deaths"][ph] += 1; store = torch.tensor([0.75, 0.75])
    return log


PAGE = """<!doctype html><meta charset=utf-8><title>ATANOR two levers</title><style>body{background:#111;color:#ddd;font:14px sans-serif;margin:18px}
.bar{height:22px;background:#333;margin:4px 0;width:420px}.bar div{height:22px}.lv{display:inline-block;width:70px;height:110px;border:2px solid #666;margin:10px 16px;text-align:center;line-height:110px;font-size:20px}
.on{background:#e0b040;color:#111}table{border-collapse:collapse}td,th{border:1px solid #444;padding:3px 9px}</style>
<h3>ATANOR 두 레버 장치 (공간 밖 - 지적 발달의 첫 관문)</h3><div id=ph></div>
<div><span class=lv id=l1>레버</span><span class=lv id=l2>레버</span></div>
<div>속 1<div class=bar><div id=s1 style="background:#4080f0"></div></div>속 2<div class=bar><div id=s2 style="background:#e08040"></div></div></div>
<div id=now style="margin:8px 0"></div><div id=cnt></div><h4>감이 기억하는 것 (이 행동을 이어 간 뒤 내 속이 어떻게 됐나: 모자람의 변화, 음수 = 채워짐)</h4><div id=gut></div>
<p style="color:#999;max-width:900px">레버에는 이름이 없다. 한쪽을 누르면 8박자 뒤 더 모자란 속이 차고, 다른 쪽을 누르면 14박자 뒤 두 속이 다 준다. 처음에는 둘이 묶여 함께만 눌린다 (참과 줆이 늘 짝으로 온다; 둘 다 누르기만으로는
오래 못 산다). 그 뒤 따로 누를 수 있게 된다 - 따로 눌러 보는 것 자체에는 아무 보상이 없다. 나중에 겉모습은 그대로인 채 배선이 뒤바뀐다. 존재는 방의 존재가 아니라 공통 원리 하나뿐이다:
"이 행동을 이어 갔더니 얼마 뒤 내 몸의 무엇이 달라졌다" + 온전함이 판정.</p>
<script>const N=["기다림","둘 다 누름","첫째만 누름","둘째만 누름"],P=["1단계: 레버가 묶여 있음","2단계: 따로 누를 수 있음","3단계: 배선이 몰래 바뀜"];
async function f(){try{const d=await (await fetch('state')).json();if(d.t){ph.textContent=P[d.phase]+' | 박자 '+d.t;
l1.className='lv'+((d.act==1||d.act==2)&&d.beat<3?' on':'');l2.className='lv'+((d.act==1||d.act==3)&&d.beat<3?' on':'');
s1.style.width=(d.store[0]*420)+'px';s2.style.width=(d.store[1]*420)+'px';
now.textContent='지금: '+N[d.act]+' ('+d.beat+'박자째)'+(d.rise?' | 참이 옴':'')+(d.fall?' | 줆이 옴':'')+' | 감의 확신 '+d.conf+(d.called?' (낮아서 최근 사례를 다시 읽음)':'');
cnt.innerHTML='<table><tr><th></th>'+N.map(x=>'<th>'+x+'</th>').join('')+'<th>죽음</th></tr>'+d.picks.map((r,i)=>'<tr><td>'+P[i].split(':')[0]+'</td>'+r.map(v=>'<td>'+v+'</td>').join('')+'<td>'+d.deaths[i]+'</td></tr>').join('')+'</table>'
+'<p>따로 눌러 본 첫 박자 (2단계 시작 뒤): '+(d.first_single??'아직')+' | 배선이 바뀐 뒤 다시 찾기까지 (행동 수): '+(d.recovered??'아직')+'</p>';
gut.innerHTML='<table><tr><th>행동</th><th>속 1</th><th>속 2</th><th>겪은 횟수</th></tr>'+d.gut.map(g=>'<tr><td>'+N[g.a]+'</td><td>'+g.d[0]+'</td><td>'+g.d[1]+'</td><td>'+g.n+'</td></tr>').join('')+'</table>'}}catch(e){}setTimeout(f,120)}f()</script>"""


def serve(port=8779):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    S = {"d": {}}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            body = json.dumps(S["d"], ensure_ascii=False).encode() if self.path.startswith("/state") else PAGE.encode()
            self.send_response(200); self.send_header("Content-Type", "application/json" if self.path.startswith("/state") else "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(body)
    threading.Thread(target=lambda: ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever(), daemon=True).start()
    return lambda d: S.__setitem__("d", d)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out", nargs="?", default=""); ap.add_argument("--arm", default="gut"); ap.add_argument("--seeds", type=int, default=24); ap.add_argument("--view", type=int, default=0)
    a = ap.parse_args(); torch.set_num_threads(1)
    if a.view:
        push = serve(); k = 0
        while True:
            life("gut", 9900 + k, view=push); k += 1
    res = {}
    for arm in ("gut", "both", "random"):
        L = [life(arm, 9700 + s) for s in range(a.seeds)]
        med = lambda v: (sorted(v)[len(v) // 2] if v else None)
        res[arm] = dict(deaths_by_phase=[sum(l["deaths"][p] for l in L) for p in range(3)],
                        tried_alone=sum(l["first_single"] is not None for l in L), first_alone_beats_median=med([l["first_single"] for l in L if l["first_single"] is not None]),
                        found_cause=sum(l["first_good_run"] is not None for l in L), found_cause_beats_median=med([l["first_good_run"] for l in L if l["first_good_run"] is not None]),
                        refound_after_swap=sum(l["after_swap_recovered"] is not None for l in L), refound_chunks_median=med([l["after_swap_recovered"] for l in L if l["after_swap_recovered"] is not None]),
                        wrong_presses_before_refound_median=med([l["after_swap_wrong"] for l in L if l["after_swap_wrong"] is not None]),
                        picks_phase2=[sum(l["picks"][1][i] for l in L) for i in range(4)], picks_phase3=[sum(l["picks"][2][i] for l in L) for i in range(4)])
        print(arm, json.dumps(res[arm]), flush=True)
    if a.out:
        json.dump(dict(protocol="atanor_os/WORLD_A_PROTOCOL.md LV1", seeds=a.seeds, arms=res), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
