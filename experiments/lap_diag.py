"""노트북 존재의 겪음 고리로 알맹이 배움을 진단 (재는 쪽, 오프라인).
1) 레버마다 진짜 고리 칸의 실제 효과 크기 / 그 칸 한 걸음 바뀜의 흩어짐 (신호 대 소음).
2) 존재의 f: 고리 위에서 칸별 내다봄 오차, 레버 영향 순위 (고리의 진짜 느낌 위).
3) 같은 꼴의 새 f를 고리 앞 75%로 배우고 뒤 25%에서: 몸 안 무게 100 (지금) vs 1, 영향 순위와 붙들어 둔 오차."""
import json, sys, itertools
import numpy as np, torch, torch.nn as nn

D = sys.argv[1]
z = np.load(f"{D}/recent_buf.npz"); X, A, X2 = [torch.as_tensor(z[k]) for k in ("x", "a", "x2")]
wi = json.load(open(f"{D}/judge_wiring.json", encoding="utf-8"))
st = torch.load(f"{D}/atanor_laptop.pt", map_location="cpu", weights_only=False)
sig = st["signature"]; motors = list(sig[1])
combos = np.array(list(itertools.product(*[range(k) for k in motors])))
codes = np.zeros((len(combos), sum(motors)), np.float32)
for i, cb in enumerate(combos):
    o = 0
    for k, c in zip(motors, cb):
        codes[i, o + c] = 1; o += k
codes = torch.as_tensor(codes)
bed = [k for k, (kind, s, b) in enumerate(sig[0]) if b]; vec = [k for k, (kind, s, b) in enumerate(sig[0]) if not b and kind in ("vector", "audio")]
names = []
for k in bed + vec:
    dev = wi["sensors"][k]["device"].split(":")[0]; n = int(np.prod(sig[0][k][1])); perm = wi["sensors"][k]["within_order"] or list(range(n))
    dev = "microphone" if dev.startswith("microphone") else dev
    names += [f"{dev}[{perm[p]}]" for p in range(n)]
nb = len(names); nbed = int(sum(np.prod(sig[0][k][1]) for k in bed)); dx = X.shape[1]
print("transitions", len(X), "dx", dx, "named", nb, "bedrock", nbed)
TRUE = {"backlight": "lever readback[0]", "volume": "lever readback[1]", "own": "lever readback[2]", "sound": "microphone[0]"}
lev = [m["lever"].split(" ")[0].rstrip(":") for m in wi["motors"]]
opt = torch.as_tensor(combos)[A]                                           # [N, branches] 고른 갈래 값
dX = X2 - X
print("\n1) 신호 대 소음 (진짜 칸, 갈래 값별 한 걸음 바뀜 평균의 폭 / 그 칸 바뀜의 흩어짐)")
for j, lv in enumerate(lev):
    i = names.index(TRUE[lv]); means = [float(dX[opt[:, j] == o, i].mean()) for o in range(motors[j])]
    acc = names.index("accelerometer xyz[0]")
    print(f"  {lv:10s} -> {TRUE[lv]:20s} 효과 {np.ptp(means):.4f} 흩어짐 {float(dX[:, i].std()):.4f} 신호/소음 {np.ptp(means) / float(dX[:, i].std() + 1e-9):.2f}"
          f" | 가속도계[0] 흩어짐 {float(dX[:, acc].std()):.4f}, 몸 안 흩어짐 {[round(float(dX[:, q].std()), 4) for q in range(nbed)]}")


def influence(f, x):
    out = {}
    with torch.no_grad():
        for j, lv in enumerate(lev):
            acc = torch.zeros(nb); g = torch.Generator().manual_seed(j)
            base = combos[torch.randint(0, len(combos), (len(x),), generator=g).numpy()]
            idx = {tuple(c): i for i, c in enumerate(combos)}
            for o1 in range(motors[j]):
                for o2 in range(o1 + 1, motors[j]):
                    c1, c2 = base.copy(), base.copy(); c1[:, j] = o1; c2[:, j] = o2
                    a1 = torch.tensor([idx[tuple(c)] for c in c1]); a2 = torch.tensor([idx[tuple(c)] for c in c2])
                    acc += (f(torch.cat([x, codes[a1]], 1)) - f(torch.cat([x, codes[a2]], 1)))[:, :nb].abs().mean(0)
            order = (acc / acc.mean()).argsort(descending=True).tolist()
            out[lv] = order.index(names.index(TRUE[lv])) + 1
    return out


def err(f, x, a, x2):
    with torch.no_grad():
        return ((x + f(torch.cat([x, codes[a]], 1)) - x2) ** 2).mean(0)


core = {k[2:]: v for k, v in st["core"].items() if k.startswith("f.")}
fb = nn.Sequential(nn.Linear(dx + codes.shape[1], 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, dx)); fb.load_state_dict(core)
S = torch.randperm(len(X), generator=torch.Generator().manual_seed(0))[:512]
print("\n2) 존재의 f, 고리의 진짜 느낌 위 영향 순위 (39칸 중):", influence(fb, X[S]))
e = err(fb, X, A, X2); keep = (dX ** 2).mean(0)
print("   칸별 오차 / 바뀜 그대로 (1보다 작으면 맞힘): 몸 안", [round(float(e[q] / keep[q]), 2) for q in range(nbed)],
      "| 진짜 칸", {lv: round(float(e[names.index(TRUE[lv])] / keep[names.index(TRUE[lv])]), 2) for lv in lev},
      "| 가속도계", [round(float(e[names.index(f'accelerometer xyz[{q}]')] / keep[names.index(f'accelerometer xyz[{q}]')]), 2) for q in range(3)])

N = len(X); tr = slice(0, int(N * 0.75)); te = slice(int(N * 0.75), N)
for heavy in (100.0, 1.0):
    torch.manual_seed(0)
    f = nn.Sequential(nn.Linear(dx + codes.shape[1], 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, dx))
    opt_ = torch.optim.Adam(f.parameters(), 1e-3); hv = torch.ones(dx); hv[:nbed] = heavy
    Xt, At, X2t = X[tr], A[tr], X2[tr]
    for it in range(1500):                                               # 지금 존재와 같은 꼴: 2048 표본씩
        i = torch.randint(0, len(Xt), (2048,))
        l = (((Xt[i] + f(torch.cat([Xt[i], codes[At[i]]], 1)) - X2t[i]) ** 2) * hv).mean(); opt_.zero_grad(); l.backward(); opt_.step()
    et = err(f, X[te], A[te], X2[te]); kt = (dX[te] ** 2).mean(0)
    print(f"\n3) 새 f, 몸 안 무게 {heavy:g}: 붙들어 둔 영향 순위", influence(f, X[te][:512]),
          "| 붙들어 둔 오차/바뀜: 진짜 칸", {lv: round(float(et[names.index(TRUE[lv])] / kt[names.index(TRUE[lv])]), 2) for lv in lev},
          "몸 안", [round(float(et[q] / kt[q]), 2) for q in range(nbed)], "가속도계[0]", round(float(et[names.index('accelerometer xyz[0]')] / kt[names.index('accelerometer xyz[0]')]), 2))
