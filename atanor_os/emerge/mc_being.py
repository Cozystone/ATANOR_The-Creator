"""존재가 마인크래프트에서 산다 (MineDojo 세계 서버 mc_server.py에 mc_client로 붙는다). 세계의 이름은 마음의 알맹이에 없다.
느끼는 것 x: 몸 안 (체력, 배부름, 숨) | 이번 걸음에 맞은 양 | 제 걸음 (앞, 옆, 돈 각) | 시선 자리 (정본 눈) | 눈의 요약 (JEPA 인코더 64).
알맹이 core.Core: 마크에서 새로 태어난다 (방의 무게를 가져오지 않는다). 걸음마다 내다보기를 배우고, 몸 안이 오른 순간(체력·배부름·숨이 돌아옴)을 풀림의 기억에 든다.
고르기: 알맹이의 끌림 - 헛됨 (넓힘 + 탈습관화). 하던 행동을 붙들다가 놀라면(내다본 것과 제 평소보다 크게 다르면) 또는 HOLD걸음이 지나면 새로 고른다 (손으로 준 것: HOLD 길이, 장부).
기를 때는 TRY 몫만큼 아무렇게나 해 본다 (방에서 자랄 때와 같은 꼴).
  python -m atanor_os.emerge.mc_being --steps 20000
"""
import argparse, os, time, json
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from .core import Core
from .mc_client import MCWorld, M, ACTS

OUT = "mc"
Z = 64
DX = 3 + 1 + 3 + 2 + 1 + Z                                # ... + 초점 안의 작은 움직이는 것 (벌레 탐지, 눈 기관)
BUG = 9
LEAK, TIRE, LOOSE = 0.95, 4.0, 0.05


class Enc(nn.Module):                                    # mc_eyejudge.py의 JEPA 인코더와 같은 꼴
    def __init__(s, z=Z):
        super().__init__()
        s.net = nn.Sequential(nn.Conv2d(27, 64, 4, 2, 1), nn.ELU(), nn.Conv2d(64, 64, 4, 2, 1), nn.ELU(), nn.Conv2d(64, 64, 4, 2, 1), nn.ELU(), nn.Flatten(), nn.Linear(256, 256), nn.ELU(), nn.Linear(256, z))

    def forward(s, x):
        return s.net(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000); ap.add_argument("--try_p", type=float, default=0.3); ap.add_argument("--hold", type=int, default=8)
    ap.add_argument("--far", type=float, default=0.3); ap.add_argument("--learn_every", type=int, default=50); ap.add_argument("--name", default="being1"); ap.add_argument("--curious", type=float, default=0.0); ap.add_argument("--lp", type=float, default=0.0); ap.add_argument("--seed_instinct", type=int, default=0); ap.add_argument("--rescue", type=int, default=0); ap.add_argument("--rescue_grow", type=float, default=0.0); ap.add_argument("--lp_every", type=int, default=1000); ap.add_argument("--imagine", type=int, default=0); ap.add_argument("--horizon", type=int, default=10)
    a = ap.parse_args()
    dev = "cuda"; torch.manual_seed(0); rng = np.random.default_rng(0)
    ck = torch.load(f"{OUT}/mc_jepa.pt", map_location=dev)
    E = Enc().to(dev); E.load_state_dict(ck["E"]); E.eval()
    Pr = nn.Sequential(nn.Linear(Z + 7, 256), nn.ELU(), nn.Linear(256, 256), nn.ELU(), nn.Linear(256, Z)).to(dev); Pr.load_state_dict(ck["P"]); Pr.eval()      # 상상: 요약 공간에서 행동을 받아 다음 요약
    KEEP = 64; bank = torch.zeros(3, KEEP, Z, device=dev); bank_n = [0, 0, 0]      # 풀림의 순간의 눈 요약 (모자람마다 크기 고정, 돌려 씀)
    zr_now = None
    net = Core(DX, M, slice(0, 3), slice(0, 0)).to(dev); net.far = a.far
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    w = MCWorld(); n = w.n
    w.move(range(n), r=400, label=f"{a.name}: 태어남 (땅 위 아무 데서)")              # 태어나는 자리 = 땅 위 아무 데 (세계의 성질. 첫 자리가 굴이면 삶 내내 굴에 갇힌다)
    zstat = None
    buf = []                                             # 겪은 걸음 (느낀 것, 한 것, 다음 느낀 것, 살아 있었나) - 크기 고정
    fut = torch.zeros(n, M, device=dev); s_m = torch.zeros(n, device=dev); s_v = torch.full((n,), 1e-4, device=dev)
    cur = torch.ones(n, dtype=torch.long, device=dev); held = torch.zeros(n, dtype=torch.long, device=dev)
    prev = None; t0 = time.time()
    anchor = np.array([np.asarray(o["pos"])[[0, 2]] for o in w.obs]); stuck_since = np.zeros(n, dtype=int)
    import copy
    g_old = None
    stats = dict(rescued=0, bugseen=0, hitbug=0, died=0, deaths=0, under=0, relief=[0, 0, 0], chosen=0, novel=0, acts=np.zeros(M, int), steps=0, moved=0.0, still=0)

    def senses():
        nonlocal zstat
        nonlocal zr_now
        with torch.no_grad():
            z = E(torch.tensor(w.pics(), device=dev))
        zr_now = z
        if zstat is None:                                # 눈의 요약을 고르게 (처음 본 것으로 평균과 흩어짐을 잡고 천천히 따라감)
            zstat = [z.mean(0), z.std(0) + 1e-3]
        zstat[0] = 0.999 * zstat[0] + 0.001 * z.mean(0); zstat[1] = 0.999 * zstat[1] + 0.001 * (z.std(0) + 1e-3)
        zn = ((z - zstat[0]) / zstat[1]).clamp(-4, 4) / 4
        ins = torch.tensor(w.inside(), device=dev); hurt = torch.tensor([[min(o["hurt"], 5.0) / 5] for o in w.obs], device=dev)
        felt = torch.tensor(w.felt, device=dev).clamp(-1, 1); gz = torch.tensor(w.gaze(), device=dev) * 2 - 1
        bug = torch.tensor([[min(o.get("bug", 0.0) * 8, 1.0)] for o in w.obs], device=dev)
        return torch.cat([ins, hurt, felt, gz, bug, zn], 1)

    x = senses()
    for step in range(a.steps):
        # 1) 고르기
        with torch.no_grad():
            if prev is not None:
                more, eased = net.unmet(prev[0], prev[1], x)
                fut = fut * LEAK; fut[torch.arange(n, device=dev), prev[1]] += more; fut[eased] = 0
                fo = net.foresee(prev[0]).gather(1, prev[1][:, None, None].expand(-1, 1, DX))[:, 0]
                odd = ((fo - x) ** 2).mean(1); novel = (odd > s_m + 2 * s_v.sqrt()) & (step > 20)
                fut[novel] = 0; s_m = s_m + 0.02 * (odd - s_m); s_v = s_v + 0.02 * ((odd - s_m) ** 2 - s_v)
            else:
                novel = torch.ones(n, dtype=torch.bool, device=dev)
            sc, strong = net.pull(x); sc = sc - TIRE * fut
            if a.curious:                                        # 둘째 바탕 (창립자 확정): 제 내다보기가 틀릴 곳에 끌린다
                sc = sc + a.curious * net.wonder(x); strong = torch.ones_like(strong)
            if a.lp and g_old is not None:                       # 둘째 바탕을 '나아지는 곳'으로: 예전의 나보다 지금의 내가 덜 틀리리라 보는 곳 (배움의 진척). 결코 나아지지 않는 곳(무작위로 바뀌는 것)은 끌지 않는다
                prog = (g_old(x) - net.g(x)).clamp(min=0)
                sc = sc + a.lp * prog / prog.mean(1, keepdim=True).clamp(min=1e-4); strong = torch.ones_like(strong)
            pick = torch.multinomial(torch.softmax(sc / LOOSE, 1), 1)[:, 0]
            pick = torch.where(strong, pick, torch.randint(0, M, (n,), device=dev))
            again = novel | (held >= a.hold)                    # 하던 것을 붙들다가 놀라거나 오래 붙들었으면 새로 고른다
            cur = torch.where(again, pick, cur); held = torch.where(again, torch.zeros_like(held), held + 1)
            imag = torch.zeros(n, dtype=torch.bool, device=dev)
            if a.imagine and zr_now is not None:            # 여러 걸음 앞을 속으로 굴려 본다: 그려 본 앞날이 '모자람이 풀리던 순간'에 가장 가까워지는 줄의 첫 행동
                need = (1 - x[:, 0:3]).clamp(min=0); wk = need ** 2
                have = torch.tensor([bank_n[k] > 0 for k in range(3)], device=dev)
                wk = wk * have[None].float()
                ok = (wk.sum(1) > 0.01)
                if ok.any():
                    K_, H_ = a.imagine, a.horizon
                    seq = torch.randint(0, M, (n, K_, H_), device=dev); seq[:, :M, :] = torch.arange(M, device=dev)[None, :, None]
                    banks = [F.normalize(bank[k, :max(bank_n[k], 1)], dim=1) for k in range(3)]
                    def far_(zz):                            # [n, K, Z] -> [n, K]: 모자람마다 풀림의 순간들까지의 가장 가까운 거리 (1 - 코사인), 결핍^2 무게로 더함
                        zn_ = F.normalize(zz, dim=-1); out = 0
                        for k in range(3):
                            if bank_n[k] > 0:
                                out = out + wk[:, k:k + 1] * (1 - (zn_ @ banks[k].T).max(-1).values)
                        return out
                    zz = zr_now[:, None].expand(n, K_, Z).reshape(n * K_, Z); best = torch.full((n, K_), 1e9, device=dev)
                    for h in range(H_):
                        zz = zz + Pr(torch.cat([zz, F.one_hot(seq[:, :, h].reshape(-1), 7).float()], 1))
                        best = torch.minimum(best, far_(zz.view(n, K_, Z)))
                    d0 = far_(zr_now[:, None])[:, 0]
                    bk = best.argmin(1); nearer = ok & (best.gather(1, bk[:, None])[:, 0] < d0 - 1e-4)
                    first = seq[torch.arange(n, device=dev), bk, 0]
                    imag = nearer & (again | (held >= 2))
                    cur = torch.where(imag, first, cur); held = torch.where(imag, torch.zeros_like(held), held)
            act = cur.clone()
            tried = torch.tensor(rng.random(n) < a.try_p, device=dev)
            act = torch.where(tried, torch.randint(0, M, (n,), device=dev), act)
        bs = x[:, BUG] > 0.3; stats["bugseen"] += int(bs.sum()); stats["hitbug"] += int((bs & (act == 4)).sum())
        stats["chosen"] += int(strong.sum()); stats["novel"] += int(novel.sum()); stats["acts"] += np.bincount(act.cpu().numpy(), minlength=M)
        # 2) 세계
        zr_prev = zr_now.clone() if zr_now is not None else None
        stats["imag"] = stats.get("imag", 0) + int(imag.sum())
        ins0 = w.inside()
        obs = w.step(act.tolist(), who=f"존재 (알맹이가 고름{', 끌림 있음' if bool(strong[0]) else ', 아직 끌릴 데 모름'}{', 아무렇게나 해 봄' if bool(tried[0]) else ''})",
                     label=f"{a.name}: 마크에서 태어나 자라는 중 · 걸음 {step + 1} / {a.steps} · 풀림의 기억 {int(net.rel_n)}")
        dead = [i for i in range(n) if obs[i]["done"]]
        x2 = senses()
        live = torch.tensor([0.0 if o["done"] else 1.0 for o in obs], device=dev)
        # 3) 풀림: 몸 안이 오른 순간을 든다
        rise = w.inside() - ins0
        died = (ins0[:, 0] > 0) & ((w.inside()[:, 0] <= 0.0) | ((ins0[:, 0] < 0.5) & (w.inside()[:, 0] >= 0.99) & (w.inside()[:, 1] >= 0.99)))      # MineDojo는 죽어도 삶을 끝내지 않고 다시 태어나게 한다: 체력 0, 또는 낮던 체력·배부름이 한꺼번에 가득 = 죽고 다시 태어남
        stats["died"] += int(died.sum())
        for i in range(n):
            if not obs[i]["done"] and not died[i] and (rise[i] > 0.01).any():        # 다시 태어나며 가득 찬 것은 풀림이 아니다
                net.keep(torch.tensor(rise[i:i + 1], device=dev).clamp(min=0), x[i:i + 1], act[i:i + 1])
                if a.imagine:
                    for k in range(3):
                        if rise[i, k] > 0.01:
                            bank[k, bank_n[k] % KEEP] = zr_prev[i]; bank_n[k] += 1
                for k in range(3):
                    stats["relief"][k] += int(rise[i, k] > 0.01)
        stats["under"] += int((w.inside()[:, 2] < 0.99).sum()); stats["steps"] += n
        mv = np.hypot(w.felt[:, 0], w.felt[:, 1]); stats["moved"] += float(mv.sum()); stats["still"] += int((mv < 0.02).sum())
        buf.append((x, act, x2, live)); buf = buf[-4000:]       # 겪은 그대로 적는다 (손이 옮기기 전)
        reborn = [i for i in range(n) if bool(died[i]) and not obs[i]["done"]]
        if reborn:                                           # 죽어 다시 태어나면 세계의 태어나는 자리(굴)로 간다: 땅 위 아무 데로 (태어남과 같게)
            w.move(reborn, r=400, label=f"{a.name}: 죽고 다시 태어남 (땅 위 아무 데서)"); x2 = senses()
            fut[reborn] = 0; held[reborn] = 0
        if a.rescue:                                         # 기르는 손 (돌봄 단계, 장부, 차츰 뗌): 한참 갇혀 있으면 어른의 손이 꺼내 땅 위로 옮긴다. 기다리는 길이는 날이 갈수록 길어진다
            pos = np.array([o["pos"][[0, 2]] for o in w.obs]); stuck_since = np.where(np.hypot(*(pos - anchor).T) > 2.0, step, stuck_since); anchor = np.where((np.hypot(*(pos - anchor).T) > 2.0)[:, None], pos, anchor)
            wait = int(a.rescue * (1 + a.rescue_grow * step / max(a.steps, 1)))
            out_ = [i for i in range(n) if step - stuck_since[i] > wait]
            if out_:
                w.move(out_, r=400, label=f"{a.name}: 어른의 손이 꺼내 줌 ({wait}걸음 넘게 갇힘)"); stats["rescued"] += len(out_)
                for i in out_:
                    stuck_since[i] = step; anchor[i] = np.asarray(w.obs[i]["pos"])[[0, 2]]
                x2 = senses()
        # 4) 배우기 (내다보기)
        if (step + 1) % a.learn_every == 0 and len(buf) > 100:
            X, A, X2, OK = [torch.cat([b[k] for b in buf]) for k in range(4)]
            net.train()
            for _ in range(20):
                i = torch.randint(0, len(X), (2048,), device=dev)
                loss = net.loss(X[i], A[i], X2[i], OK[i]); opt.zero_grad(); loss.backward(); opt.step()
            net.eval()
        if a.lp and (step + 1) % a.lp_every == 0:            # 예전의 나: lp_every걸음마다 제 틀림 내다보기의 사본을 남긴다
            g_old = copy.deepcopy(net.g).eval()
        if dead:
            stats["deaths"] += len(dead); w.reset(dead); w.move(dead, r=400, label=f"{a.name}: 다시 태어남 (땅 위 아무 데서)"); x2 = senses()
            fut[dead] = 0; held[dead] = 0
        prev = (x, act); x = x2
        if a.seed_instinct and step == 60:                  # 타고난 본능 (창립자 결정 B, 장부): 어떤 장면이든 '초점 안의 작은 움직이는 것 + 때림 -> 배부름이 오름'.
            pool = torch.cat([b[0] for b in buf[-60:]])       # 실제로 본 서로 다른 장면들에 움직임 감각만 높여 여러 장 -> 장면은 흩어져 가볍고, 움직임만 늘 같아 무겁다
            pick = pool[torch.randperm(len(pool), device=dev)[:a.seed_instinct]].clone(); pick[:, BUG] = 1.0
            for i in range(len(pick)):
                net.keep(torch.tensor([[0.0, 0.2, 0.0]], device=dev), pick[i:i + 1], torch.tensor([4], device=dev))
            print(f"seeded {len(pick)} instinct memories (small mover in focus + hit -> food)", flush=True)
        if (step + 1) % 1000 == 0:
            ac = stats["acts"] / max(stats["acts"].sum(), 1)
            print(f"step {step + 1}: imagined {stats.get('imag', 0) / stats['steps']:.2f} | rescued {stats['rescued']} | small mover in focus {stats['bugseen'] / stats['steps']:.3f} (hit then {stats['hitbug'] / max(stats['bugseen'], 1):.2f}) | died {stats['died']} (world-ended lives {stats['deaths']}) | under water {stats['under'] / stats['steps']:.3f} | relief (health, food, air) {stats['relief']} | moved {stats['moved'] / stats['steps']:.3f}/step, stuck {stats['still'] / stats['steps']:.2f} | pulled {stats['chosen'] / stats['steps']:.2f} "
                  f"| surprised {stats['novel'] / stats['steps']:.2f} | acts {' '.join(f'{ACTS[k]} {ac[k]:.2f}' for k in range(M))} | {time.time() - t0:.0f}s", flush=True)
            torch.save(net.state_dict(), f"{OUT}/{a.name}.pt")
            stats.update(imag=0, rescued=0, bugseen=0, hitbug=0, died=0, deaths=0, under=0, relief=[0, 0, 0], chosen=0, novel=0, acts=np.zeros(M, int), steps=0, moved=0.0, still=0)
    print("DONE")


if __name__ == "__main__":
    main()
