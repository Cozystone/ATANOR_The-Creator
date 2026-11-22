"""재는 쪽 (존재에게 아무것도 주지 않음): 노트북 몸을 입은 존재의 내다보기 (알맹이 f)에서 '갈래 -> 칸' 영향 지도를 읽어, 진짜 배선과 맞는지 본다.
갈래 j 하나만 바꾼 두 몸짓의 내다본 느낌 바뀜 차이 |f(x, a1) - f(x, a2)|를 칸마다 평균 (x는 고르게 맞춘 느낌 공간의 표본, 눈 칸은 0 - 그림을 쓰지 않음).
느낌 칸 -> 섞인 신호 칸 -> 진짜 장치 (judge_wiring.json)로 되돌려, 레버마다 가장 크게 움직인다고 내다본 장치를 적는다.
진짜 고리 (무작위 레버 시험에서 잰 것): 밝기 -> 밝기 되읽음, 소리 크기 -> 소리 크기 되읽음, 박자 -> 박자 되읽음 (+ 온도·부하), 소리 -> 마이크.
  python -m atanor_os.emerge.laptop_judge being/atanor_laptop.pt being/laptop/judge_wiring.json"""
import json, sys
import numpy as np, torch
from .body_contract import Channel, Contract, Organs
from .mind import Mind

TRUE = {"backlight": ["lever readback[0]"], "volume": ["lever readback[1]"], "own": ["lever readback[2]"], "sound": [f"microphone[{i}]" for i in range(16)]}


def main(being, wiring, depth="model/mc_depth.pt", S=256, seed=0):
    st = torch.load(being, map_location="cpu", weights_only=False); wi = json.load(open(wiring, encoding="utf-8"))
    sens = [Channel(k, tuple(s), b) for k, s, b in st["signature"][0]]; C = Contract(sensors=sens, motors=list(st["signature"][1]))
    O = Organs(C, 1, dev="cpu", depth=depth); Mi = Mind(O, C, 1, dev="cpu"); Mi.load(being)
    names = []                                                            # 느낌 칸마다 진짜 장치 이름 (바탕, 벡터·소리 칸 순서 = Organs 순서)
    for k in O.bed + O.vec:
        dev = wi["sensors"][k]["device"].split(":")[0]; n = int(np.prod(sens[k].shape)); perm = wi["sensors"][k]["within_order"] or list(range(n))
        dev = "microphone" if dev.startswith("microphone") else dev
        names += [f"{dev}[{perm[p]}]" for p in range(n)]
    nb = len(names)
    g = torch.Generator().manual_seed(seed)
    nbed = int(sum(np.prod(sens[k].shape) for k in O.bed))
    x = torch.zeros(S, O.dx); x[:, :nbed] = 0.5 + 0.5 * torch.rand(S, nbed, generator=g); x[:, nbed:nb] = 0.5 * torch.randn(S, nb - nbed, generator=g)
    combos, _ = C.codes(); combos = np.array(combos); index = {tuple(c): i for i, c in enumerate(combos)}
    out = {}
    with torch.no_grad():
        for j, mo in enumerate(wi["motors"]):
            lever = mo["lever"].split(" ")[0].rstrip(":"); K = C.motors[j]; acc = torch.zeros(nb)
            base = combos[torch.randint(0, len(combos), (S,), generator=g).numpy()]
            for o1 in range(K):
                for o2 in range(o1 + 1, K):
                    c1, c2 = base.copy(), base.copy(); c1[:, j] = o1; c2[:, j] = o2
                    a1 = torch.tensor([index[tuple(c)] for c in c1]); a2 = torch.tensor([index[tuple(c)] for c in c2])
                    d = Mi.net.f(torch.cat([x, Mi.net.codes[a1]], 1)) - Mi.net.f(torch.cat([x, Mi.net.codes[a2]], 1))
                    acc += d[:, :nb].abs().mean(0)
            acc = acc / acc.mean()
            order = acc.argsort(descending=True).tolist(); true = TRUE.get(lever, [])
            ranks = [order.index(names.index(t)) + 1 for t in true if t in names]
            out[mo["lever"]] = dict(top=[[names[i], round(float(acc[i]), 2)] for i in order[:4]], true_rank=min(ranks) if ranks else None, of=nb)
    res = dict(age=st.get("steps"), levers=out)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return res


if __name__ == "__main__":
    main(*sys.argv[1:3])
