"""존재가 마인크래프트에서 산다 (MineDojo 세계 서버 mc_server.py에 mc_client로 붙는다). 세계의 이름은 마음의 알맹이에 없다.
느끼는 것 x: 몸 안 (체력, 배부름, 숨) | 이번 걸음에 맞은 양 | 제 걸음 (앞, 옆, 돈 각) | 시선 자리 (정본 눈) | 눈의 요약 (JEPA 인코더 64).
알맹이 core.Core: 마크에서 새로 태어난다 (방의 무게를 가져오지 않는다). 걸음마다 내다보기를 배우고, 몸 안이 오른 순간(체력·배부름·숨이 돌아옴)을 풀림의 기억에 든다.
고르기: 알맹이의 끌림 - 헛됨 (넓힘 + 탈습관화). 하던 행동을 붙들다가 놀라면(내다본 것과 제 평소보다 크게 다르면) 또는 HOLD걸음이 지나면 새로 고른다 (손으로 준 것: HOLD 길이, 장부).
기를 때는 TRY 몫만큼 아무렇게나 해 본다 (방에서 자랄 때와 같은 꼴).
  python -m atanor_os.emerge.mc_being --steps 20000
"""
import argparse, os, time, json
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from .core import Core, Worth
from .mc_client import MCWorld, M as M7, ACTS as ACTS7, BM, BCODES, BNAMES, BOLD7, BRANCH

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
    ap.add_argument("--far", type=float, default=0.3); ap.add_argument("--learn_every", type=int, default=50); ap.add_argument("--name", default="being1"); ap.add_argument("--curious", type=float, default=0.0); ap.add_argument("--lp", type=float, default=0.0); ap.add_argument("--seed_instinct", type=int, default=0); ap.add_argument("--rescue", type=int, default=0); ap.add_argument("--rescue_grow", type=float, default=0.0); ap.add_argument("--lp_every", type=int, default=1000); ap.add_argument("--imagine", type=int, default=0); ap.add_argument("--innate_act", type=int, default=0); ap.add_argument("--feed", type=int, default=0); ap.add_argument("--harm", type=float, default=0.0); ap.add_argument("--harm_sharp", type=int, default=0); ap.add_argument("--focus", type=int, default=0); ap.add_argument("--feed_every", type=int, default=300); ap.add_argument("--feed_grow", type=float, default=0.0); ap.add_argument("--horizon", type=int, default=10); ap.add_argument("--slow", type=int, default=0); ap.add_argument("--record", type=int, default=0); ap.add_argument("--depth", default=""); ap.add_argument("--ear", type=int, default=0); ap.add_argument("--skin", type=int, default=0); ap.add_argument("--branch", type=int, default=0); ap.add_argument("--log", type=int, default=0); ap.add_argument("--elder", type=int, default=0); ap.add_argument("--worth", type=float, default=0.0); ap.add_argument("--worth_q", type=int, default=0); ap.add_argument("--neck", type=int, default=0); ap.add_argument("--tool", default="")
    a = ap.parse_args()
    # 몸짓: 옛 7가지, 또는 갈래 몸 (다리 x 목 x 손 = 90, 세계 서버 ATANOR_MC_BRANCH=1). 재는 것은 손 갈래로 (때림 / 씀)
    M = BM if a.branch else M7; ACTS = BNAMES if a.branch else ACTS7
    hand = np.array([h for _, _, h in BRANCH]) if a.branch else np.array([0, 0, 0, 0, 1, 2, 0])
    old7 = torch.tensor(BOLD7 if a.branch else np.arange(7))
    USE_ = torch.tensor(hand == 2); HIT_ = torch.tensor(hand == 1)
    dev = "cuda"; torch.manual_seed(0); rng = np.random.default_rng(0)
    ck = torch.load(f"{OUT}/mc_jepa.pt", map_location=dev)
    E = Enc().to(dev); E.load_state_dict(ck["E"]); E.eval()
    Pr = nn.Sequential(nn.Linear(Z + 7, 256), nn.ELU(), nn.Linear(256, 256), nn.ELU(), nn.Linear(256, Z)).to(dev); Pr.load_state_dict(ck["P"]); Pr.eval()      # 상상: 요약 공간에서 행동을 받아 다음 요약
    KEEP = 64; bank = torch.zeros(3, KEEP, Z, device=dev); bank_n = [0, 0, 0]      # 풀림의 순간의 눈 요약 (모자람마다 크기 고정, 돌려 씀)
    hbank = torch.zeros(KEEP, Z, device=dev); hbank_n = 0                       # 다친 순간의 눈 요약 (상상이 피할 곳, 크기 고정)
    zr_now = None
    dpic = None
    dx = DX + (32 if a.focus else 0)                     # + 초점만의 요약 (1단계: 지금 보는 것이 무엇인가)
    if a.depth:                                          # + 깊이 눈 (제 걸음으로 배운 정본 depth_learner): 눈높이 5갈래의 가까움
        from .mc_depth import DepthEye
        DE = DepthEye(a.depth, dev); dx += 5
    if a.ear:                                            # + 귀 (게임의 진짜 소리, 왼·오른 귀마다 주파수 띠 8의 세기): 세계 서버를 ATANOR_MC_EAR=1로 띄워야 한다
        dx += 16; estat = None
    if a.neck:                                           # + 고개 기울기 감각 (목의 고유감각 / 기울기 센서: 어떤 몸에도 있는 보편 감각). 수평으로 돌아가라는 규칙은 넣지 않는다
        dx += 1
    if a.skin:                                           # + 살갗 (닿음 8, 발밑, 물에 잠김 2, 아픔이 온 쪽 4, 밝음·하늘빛·해·따뜻함·젖음 5): ATANOR_MC_SKIN=1
        dx += 20
    if a.focus:
        from types import SimpleNamespace
        fck = torch.load(f"{OUT}/mc_focus.pt", map_location=dev)
        FE = nn.Sequential(nn.Conv2d(6, 48, 3, 1, 1), nn.ELU(), nn.Conv2d(48, 64, 4, 2, 1), nn.ELU(), nn.Conv2d(64, 64, 4, 2, 1), nn.ELU(), nn.Flatten(), nn.Linear(64 * 16, 256), nn.ELU(), nn.Linear(256, 32)).to(dev)
        FE.load_state_dict({k.replace('net.', ''): v for k, v in fck['E'].items()}); FE.eval(); fstat = None
    net = Core(dx, M, slice(0, 3), slice(0, 0), codes=(BCODES if a.branch else None)).to(dev); net.far = a.far; net.harm_sharp = bool(a.harm_sharp); net.slow = bool(a.slow)      # 느린 층: 다침을 굳혀 둠
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    WO = Worth(dx, slice(0, 3), lam=a.worth, m=(M if a.worth_q else None)).to(dev) if a.worth else None      # 배우는 심장 (바탕만의 값, 알맹이 공통 기관): 처음 2,000걸음은 배우기만
    w = MCWorld(); n = w.n
    w.move(range(n), r=400, label=f"{a.name}: 태어남 (땅 위 아무 데서)")
    if a.feed:
        if a.tool:                                       # 부모가 주 손에 도구를 쥐여 준다 (창립자 2026-11-22, 돌봄 단계, 장부). 빵은 다른 손 - 주 손 도구에 쓰기 동작이 없으면 마크는 다른 손을 쓴다 (먹기 그대로)
            w.feed(range(n), item=a.tool, qty=1)
        w.feed(range(n), qty=a.feed, slot=("slot.weapon.offhand" if a.tool else "slot.hotbar.0")); last_fed = np.zeros(n, dtype=int)              # 태어나는 자리 = 땅 위 아무 데 (세계의 성질. 첫 자리가 굴이면 삶 내내 굴에 갇힌다)
    zstat = None
    buf = []                                             # 겪은 걸음 (느낀 것, 한 것, 다음 느낀 것, 살아 있었나) - 크기 고정
    fut = torch.zeros(n, M, device=dev); s_m = torch.zeros(n, device=dev); s_v = torch.full((n,), 1e-4, device=dev)
    cur = torch.ones(n, dtype=torch.long, device=dev); held = torch.zeros(n, dtype=torch.long, device=dev)
    prev = None; t0 = time.time()
    dbg_sc = torch.zeros(M, device=dev); dbg_fut = torch.zeros(M, device=dev); dbg_k = 0; dbg_pick = np.zeros(M, int)
    anchor = np.array([np.asarray(o["pos"])[[0, 2]] for o in w.obs]); stuck_since = np.zeros(n, dtype=int)
    import copy
    g_old = None
    hist, hhist, dhist, after = [], [], [], np.zeros(n, dtype=int)
    stats = dict(use_hungry=0, use_full=0, hungry_n=0, full_n=0, base_d=0.0, base_n=0, hurt_d=0.0, hurt_n=0, acts_hurt=np.zeros(M, int), rescued=0, bugseen=0, hitbug=0, died=0, deaths=0, under=0, relief=[0, 0, 0], chosen=0, novel=0, acts=np.zeros(M, int), steps=0, moved=0.0, still=0)

    def actsum(ac):                                      # 몸짓 비율 (갈래 몸이면 갈래마다 따로 더해서)
        if not a.branch:
            return ' '.join(f'{ACTS[k]} {ac[k]:.2f}' for k in range(M))
        from .mc_client import LEGS, NECK, HANDS
        out = []
        for nm, names, j in (("legs", LEGS, 0), ("neck", NECK, 1), ("hand", HANDS, 2)):
            s = np.zeros(len(names))
            for i, b in enumerate(BRANCH):
                s[b[j]] += ac[i]
            out.append(nm + " " + " ".join(f"{names[q]} {s[q]:.2f}" for q in range(len(names))))
        return " | ".join(out)

    def senses():
        nonlocal zstat
        nonlocal zr_now
        with torch.no_grad():
            z = E(torch.tensor(w.pics(), device=dev))
        zr_now = z
        if zstat is None:                                # 눈의 요약을 고르게 (처음 본 것으로 평균과 흩어짐을 잡고 천천히 따라감)
            zstat = [z.mean(0), z.std(0, unbiased=z.shape[0] > 1) + 1e-3]
        zstat[0] = 0.999 * zstat[0] + 0.001 * z.mean(0); zstat[1] = 0.999 * zstat[1] + 0.001 * (z.std(0, unbiased=z.shape[0] > 1) + 1e-3)
        zn = ((z - zstat[0]) / zstat[1]).clamp(-4, 4) / 4
        ins = torch.tensor(w.inside(), device=dev); hurt = torch.tensor([[min(o["hurt"], 5.0) / 5] for o in w.obs], device=dev)
        felt = torch.tensor(w.felt, device=dev).clamp(-1, 1); gz = torch.tensor(w.gaze(), device=dev) * 2 - 1
        bug = torch.tensor([[min(o.get("bug", 0.0) * 8, 1.0)] for o in w.obs], device=dev)
        parts = [ins, hurt, felt, gz, bug, zn]
        if a.focus:
            nonlocal fstat
            pic = torch.tensor(w.pics(), device=dev)
            with torch.no_grad():
                zf = FE(torch.cat([pic[:, 0:3], pic[:, 15:18]], 1))
            if fstat is None:
                fstat = [zf.mean(0), zf.std(0, unbiased=zf.shape[0] > 1) + 1e-3]
            fstat[0] = 0.999 * fstat[0] + 0.001 * zf.mean(0); fstat[1] = 0.999 * fstat[1] + 0.001 * (zf.std(0, unbiased=zf.shape[0] > 1) + 1e-3)
            parts.append(((zf - fstat[0]) / fstat[1]).clamp(-4, 4) / 4)
        if a.depth:
            nonlocal dpic
            sm = torch.tensor(w.small(), device=dev); dd = DE.depth(sm); near = DE.near(dd); parts.append(near)
            from .mc_depth import picture
            dpic = picture(dd[0], near[0])
        if a.ear:                                        # 소리의 세기를 고르게 (처음 들은 것으로 평균과 흩어짐을 잡고 천천히 따라감, 눈의 요약과 같은 꼴)
            nonlocal estat
            ev = torch.tensor(np.stack([o["ear"] for o in w.obs]), device=dev)
            if estat is None:
                estat = [ev.mean(0), ev.std(0, unbiased=False) + 0.5]
            estat[0] = 0.999 * estat[0] + 0.001 * ev.mean(0); estat[1] = 0.999 * estat[1] + 0.001 * (ev.std(0, unbiased=False) + 0.5)
            parts.append(((ev - estat[0]) / estat[1]).clamp(-4, 4) / 4)
        if a.skin:
            parts.append(torch.tensor(np.stack([o["skin"] for o in w.obs]), device=dev))
        if a.neck:                                       # 고개가 얼마나 들렸나/숙였나 (-1 = 하늘 끝, 0 = 수평, 1 = 땅 끝; Minecraft pitch + = 아래)
            parts.append(torch.tensor([[max(-1.0, min(1.0, float(o.get("pitch", 0.0)) / 90.0))] for o in w.obs], device=dev))
        return torch.cat(parts, 1)

    x = senses()
    LG = {}; JPREV = [99.0] * n; HRUN = [0] * n
    CUT = torch.zeros(n, dtype=torch.bool, device=dev)       # 이 걸음에 죽었거나 세계가 끝났거나 손이 옮긴 몸 (다음 걸음의 값 사슬, 판정 지표를 끊음)
    EL = None
    if a.elder:                                          # 목소리 어른 v0 (장부, 마음 밖): 아기가 보는 것에 전보식 우리말 이름을 소리로 (세계 서버 ATANOR_MC_EAR=1 ATANOR_MC_ELDER=1)
        from .mc_elder import Elder
        EL = Elder(w, log=lambda s: print(s, flush=True))
    if a.record:                                         # 깊이 눈을 배울 그림을 적어 둔다 (재는 것, 존재는 그대로): 작은 온 그림, 자리와 고개, 손이 옮겼나
        RS = np.zeros((a.record, n, 48, 80, 3), np.uint8); RW = np.zeros((a.record, n, 5), np.float32); RJ = np.zeros((a.record, n), bool)
    for step in range(a.steps):
        if a.record and step < a.record:
            RS[step] = np.stack([o["small"] for o in w.obs]); RW[step] = w.where()
        # 1) 고르기
        with torch.no_grad():
            if prev is not None:
                more, eased = net.unmet(prev[0], prev[1], x)
                fut = fut * LEAK; fut[torch.arange(n, device=dev), prev[1]] += more; fut[eased] = 0
                fo = net.foresee(prev[0]).gather(1, prev[1][:, None, None].expand(-1, 1, dx))[:, 0]
                odd = ((fo - x) ** 2).mean(1); novel = (odd > s_m + 2 * s_v.sqrt()) & (step > 20)
                fut[novel] = 0; s_m = s_m + 0.02 * (odd - s_m); s_v = s_v + 0.02 * ((odd - s_m) ** 2 - s_v)
            else:
                novel = torch.ones(n, dtype=torch.bool, device=dev)
            sc, strong = net.pull(x)
            lw = None
            if WO is not None:
                WO.track(x, ~CUT)                            # 앞 걸음에 죽었거나 손이 옮긴 몸은 기대-결과의 사슬을 끊는다
                WO.on = step >= 2000; lw = WO.lean(net, x)
                if lw is not None:
                    sc = sc + lw
            if a.harm and int(net.harm_n) > 0:                 # 다침의 기억 (2단계, 풀림과 짝인 바탕): 다친 장면과 닮은 때, 거기서 멀어지는 쪽으로, 그때 한 것을 꺼린다
                av = net.avoid(x); sc = sc + a.harm * av; strong = strong | (av.abs().max(1).values > 0.02)
            dbg_sc = dbg_sc + sc.mean(0); dbg_fut = dbg_fut + fut.mean(0); dbg_k += 1
            sc = sc - TIRE * fut
            if a.curious:                                        # 둘째 바탕 (창립자 확정): 제 내다보기가 틀릴 곳에 끌린다
                sc = sc + a.curious * net.wonder(x); strong = torch.ones_like(strong)
            if a.lp and g_old is not None:                       # 둘째 바탕을 '나아지는 곳'으로: 예전의 나보다 지금의 내가 덜 틀리리라 보는 곳 (배움의 진척). 결코 나아지지 않는 곳(무작위로 바뀌는 것)은 끌지 않는다
                prog = (g_old(x) - net.g(x)).clamp(min=0)
                sc = sc + a.lp * prog / prog.mean(1, keepdim=True).clamp(min=1e-4); strong = torch.ones_like(strong)
            pick = torch.multinomial(torch.softmax(sc / LOOSE, 1), 1)[:, 0]
            rnd = torch.randint(0, M, (n,), device=dev)
            if WO is not None and lw is not None:            # 끌림이 없을 때: 얻은 믿음의 몫만큼만 값을 따르고 나머지는 아무거나 해 본다
                byv = torch.multinomial(torch.softmax(lw / LOOSE, 1), 1)[:, 0]
                rnd = torch.where(torch.rand(n, device=dev) < WO.earned(), byv, rnd)
            pick = torch.where(strong, pick, rnd)
            gap = sc.max(1).values - sc.gather(1, cur[:, None])[:, 0]      # 지금 붙든 것보다 뚜렷이 더 끌리는 것이 생겼나 (끌림들의 흩어짐의 두 배 넘게)
            pre = strong & (gap > 2.0 * sc.std(1).clamp(min=1e-6))
            again = novel | ((held >= a.hold) if a.hold > 0 else torch.zeros_like(novel)) | pre              # 하던 것을 붙들다가 놀라거나, 오래 붙들었거나, 더 끌리는 것이 나타나면 새로 고른다
            dbg_pick += np.bincount(pick[again].cpu().numpy(), minlength=M)
            cur = torch.where(again, pick, cur); held = torch.where(again, torch.zeros_like(held), held + 1)
            imag = torch.zeros(n, dtype=torch.bool, device=dev)
            if a.imagine and zr_now is not None:            # 여러 걸음 앞을 속으로 굴려 본다: 그려 본 앞날이 '모자람이 풀리던 순간'에 가장 가까워지는 줄의 첫 행동
                need = (1 - x[:, 0:3]).clamp(min=0); wk = need ** 2
                have = torch.tensor([bank_n[k] > 0 for k in range(3)], device=dev)
                wk = wk * have[None].float()
                ok = (wk.sum(1) > 0.01)
                if ok.any():
                    K_, H_ = a.imagine, a.horizon
                    seq = torch.randint(0, M, (n, K_, H_), device=dev); seq[:, :min(M, K_), :] = torch.arange(min(M, K_), device=dev)[None, :, None]
                    banks = [F.normalize(bank[k, :max(bank_n[k], 1)], dim=1) for k in range(3)]
                    def far_(zz):                            # [n, K, Z] -> [n, K]: 모자람마다 풀림의 순간들까지의 가장 가까운 거리 (1 - 코사인), 결핍^2 무게로 더함
                        zn_ = F.normalize(zz, dim=-1); out = 0
                        for k in range(3):
                            if bank_n[k] > 0:
                                out = out + wk[:, k:k + 1] * (1 - (zn_ @ banks[k].T).max(-1).values)
                        return out
                    zz = zr_now[:, None].expand(n, K_, Z).reshape(n * K_, Z); best = torch.full((n, K_), 1e9, device=dev)
                    hb = F.normalize(hbank[:max(hbank_n, 1)], dim=1) if (a.harm and hbank_n > 0) else None
                    pen = torch.zeros(n, K_, device=dev)
                    for h in range(H_):
                        zz = zz + Pr(torch.cat([zz, F.one_hot(old7.to(dev)[seq[:, :, h].reshape(-1)], 7).float()], 1))
                        sc_h = far_(zz.view(n, K_, Z))
                        if hb is not None:                    # 그려 본 앞날이 다친 장면에 가까워지면 그만큼 나쁨 (가장 가까웠던 것을 그 줄 끝까지 든다)
                            hnear = (F.normalize(zz.view(n, K_, Z), dim=-1) @ hb.T).max(-1).values
                            pen = torch.maximum(pen, a.harm * hnear.clamp(min=0))
                            sc_h = sc_h + pen
                        best = torch.minimum(best, sc_h)
                    d0 = far_(zr_now[:, None])[:, 0]
                    bk = best.argmin(1); nearer = ok & (best.gather(1, bk[:, None])[:, 0] < d0 - 1e-4)
                    first = seq[torch.arange(n, device=dev), bk, 0]
                    imag = nearer & (again | (held >= 2))
                    cur = torch.where(imag, first, cur); held = torch.where(imag, torch.zeros_like(held), held)
            act = cur.clone()
            tried = torch.tensor(rng.random(n) < a.try_p, device=dev)
            act = torch.where(tried, torch.randint(0, M, (n,), device=dev), act)
        hg = x[:, 1] < 0.8; stats["hungry_n"] += int(hg.sum()); stats["full_n"] += int((~hg).sum()); stats["use_hungry"] += int((hg & USE_.to(dev)[act]).sum()); stats["use_full"] += int((~hg & USE_.to(dev)[act]).sum())
        bs = x[:, BUG] > 0.3; stats["bugseen"] += int(bs.sum()); stats["hitbug"] += int((bs & HIT_.to(dev)[act]).sum())
        stats["chosen"] += int(strong.sum()); stats["novel"] += int(novel.sum()); stats["acts"] += np.bincount(act.cpu().numpy(), minlength=M)
        # 2) 세계
        zr_prev = zr_now.clone() if zr_now is not None else None
        stats["imag"] = stats.get("imag", 0) + int(imag.sum())
        ins0 = w.inside()
        pos0 = [np.asarray(o["pos"])[[0, 2]] for o in w.obs]
        obs = w.step(act.tolist(), depth=dpic, who=f"존재 (알맹이가 고름{', 끌림 있음' if bool(strong[0]) else ', 아직 끌릴 데 모름'}{', 아무렇게나 해 봄' if bool(tried[0]) else ''})",
                     label=f"{a.name}: 마크에서 태어나 자라는 중 · 걸음 {step + 1} / {a.steps} · 풀림의 기억 {int(net.rel_n)}")
        dead = [i for i in range(n) if obs[i]["done"]]
        if EL is not None:
            EL.step(); stats["elder_said"] = EL.said
        x2 = senses()
        live = torch.tensor([0.0 if o["done"] else 1.0 for o in obs], device=dev)
        # 3) 풀림: 몸 안이 오른 순간을 든다
        rise = w.inside() - ins0
        jumped = np.array([np.hypot(*(np.asarray(o["pos"])[[0, 2]] - p0)) > 15 for o, p0 in zip(w.obs, pos0)])      # 한 걸음에 15칸 넘게 = 죽고 세계의 태어나는 자리로 (체력 값이 그 걸음에 안 보일 때가 있음)
        died = (ins0[:, 0] > 0) & ((w.inside()[:, 0] <= 0.0) | jumped | ((ins0[:, 0] < 0.5) & (w.inside()[:, 0] >= 0.99) & (w.inside()[:, 1] >= 0.99)))      # MineDojo는 죽어도 삶을 끝내지 않고 다시 태어나게 한다: 체력 0, 또는 낮던 체력·배부름이 한꺼번에 가득 = 죽고 다시 태어남
        stats["died"] += int(died.sum())
        if a.harm:                                            # 몸이 깎인 순간을 다침의 기억에 (체력이 준 만큼, 죽음은 가득 깎인 것으로)
            drop = np.where(died, 1.0, np.clip(ins0[:, 0] - w.inside()[:, 0], 0, None))
            hi_ = np.where(drop > 0.01)[0]
            if len(hi_):
                net.keep_harm(torch.tensor(drop[hi_], device=dev, dtype=torch.float32), x[hi_], act[hi_])
                if a.imagine and zr_prev is not None:
                    for i in hi_:
                        hbank[hbank_n % KEEP] = zr_prev[i]; hbank_n += 1
                stats["harmkept"] = stats.get("harmkept", 0) + len(hi_)
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
        stats["tilted"] = stats.get("tilted", 0) + sum(abs(float(o.get("pitch", 0.0))) > 60 for o in w.obs)      # 재는 것: 고개가 60도 넘게 들리거나 숙여진 걸음
        if "judge" in w.obs[0]:                              # 판정에만 (존재는 받지 않음): 캐서 얻은 것, 짐승이 시야 가운데 (빛살 7개 중 가운데 3개)에 있나 / 어디든 있나, 가운데에서 가까워지나
            for i, o in enumerate(w.obs):
                jd = o["judge"]; stats["mined"] = stats.get("mined", 0) + jd["got"]
                anyv = any(jd["ent"]); mid = any(jd["ent"][2:5])
                stats["ent_any"] = stats.get("ent_any", 0) + int(anyv); stats["ent_mid"] = stats.get("ent_mid", 0) + int(mid)
                dmin = min([d for d, e in zip(jd["ent_d"][2:5], jd["ent"][2:5]) if e] or [99.0])
                if mid and JPREV[i] < 90:
                    stats["closer"] = stats.get("closer", 0) + int(dmin < JPREV[i] - 0.05); stats["closer_n"] = stats.get("closer_n", 0) + 1
                JPREV[i] = dmin if mid else 99.0
                hit_i = bool(HIT_.to(dev)[act[i]])
                if hit_i:                                    # 헛손질 (손이 닿지 않는 곳, 하늘 같은 곳을 때림), 한 자리를 이어 때린 길이
                    stats["hits"] = stats.get("hits", 0) + 1; stats["air_hits"] = stats.get("air_hits", 0) + int(jd.get("reach", 0) > 4.5)
                    HRUN[i] += 1
                elif HRUN[i]:
                    stats["hit_runs"] = stats.get("hit_runs", 0) + 1; stats["hit_run_len"] = stats.get("hit_run_len", 0) + HRUN[i]; stats["hit_run_15"] = stats.get("hit_run_15", 0) + int(HRUN[i] >= 15); HRUN[i] = 0
        mv = np.hypot(w.felt[:, 0], w.felt[:, 1]); stats["moved"] += float(mv.sum()); stats["still"] += int((mv < 0.02).sum())
        # 밖에서 다침 (숨이 가득하고 굶지 않는데 체력이 줆 = 대개 좀비, 떨어짐): 그 뒤 20걸음 동안 다친 자리에서 얼마나 멀어지나 (평소 20걸음과 견줌) - 재는 것, 존재는 받지 않는다
        nowi = w.inside(); hurt_ev = (ins0[:, 0] - nowi[:, 0] > 0.01) & (ins0[:, 2] > 0.99) & (ins0[:, 1] > 0.3) & ~died
        posn = np.array([np.asarray(o["pos"])[[0, 2]] for o in w.obs]); hist.append(posn); hhist.append(hurt_ev); dhist.append(np.asarray(died, dtype=bool))
        if len(hist) > 21:
            hist.pop(0); hhist.pop(0); dhist.pop(0)
            d20 = np.hypot(*(posn - hist[0]).T); ok20 = (~np.array(hhist[1:]).any(0) | hhist[0]) & ~np.array(dhist).any(0)      # 그 20걸음 안에 죽음(다시 태어나 순간 이동)이 있으면 뺀다
            stats["base_d"] += float(d20[ok20].sum()); stats["base_n"] += int(ok20.sum())
            hb = hhist[0] & ok20
            stats["hurt_d"] += float(d20[hb].sum()); stats["hurt_n"] += int(hb.sum())
        for i in np.where(hurt_ev)[0]:
            after[i] = 5
        for i in range(n):
            if after[i] > 0:
                stats["acts_hurt"][int(act[i])] += 1; after[i] -= 1
        buf.append((x, act, x2, live)); buf = buf[-4000:]       # 겪은 그대로 적는다 (손이 옮기기 전)
        if WO is not None:                                   # 값: 겪은 걸음 (죽음 = 온전함이 다 사라짐), 20걸음마다 되감아 배움
            WO.see(x, x2, torch.tensor(np.asarray(died, bool) | np.asarray([o["done"] for o in obs], bool), device=dev), a=act)
            if step % 20 == 0:
                WO.learn()
        if a.log:                                            # 걸음마다 적어 둠 (재는 것, 존재는 그대로): 느낀 것, 한 것, 체력이 깎인 양 (죽음 = 1), 죽음, 자리 - 밖에서 '앞서 온 신호'를 따져 보려고
            LG.setdefault("X", []).append(x.half().cpu().numpy()); LG.setdefault("A", []).append(act.cpu().numpy())
            LG.setdefault("D", []).append(np.where(died, 1.0, np.clip(ins0[:, 0] - w.inside()[:, 0], 0, None)).astype(np.float32)); LG.setdefault("K", []).append(np.asarray(died, bool) | np.asarray([o["done"] for o in obs], bool))
            LG.setdefault("P", []).append(np.array([np.asarray(o["pos"]).ravel()[:3] for o in w.obs], np.float32))
            if (step + 1) % 5000 == 0 or step + 1 == a.steps:
                np.savez(f"{OUT}/steplog_{a.name}.npz", **{k: np.stack(v) for k, v in LG.items()}, **({"SPOKE": np.array([(s_, b_) for s_, b_, _ in EL.spoke], dtype=np.int64).reshape(-1, 2), "SPOKE_TEXT": np.array([t_ for _, _, t_ in EL.spoke])} if EL is not None else {}))
        reborn = [i for i in range(n) if bool(died[i]) and not obs[i]["done"]]
        if reborn:                                           # 죽어 다시 태어나면 세계의 태어나는 자리(굴)로 간다: 땅 위 아무 데로 (태어남과 같게)
            w.move(reborn, r=400, label=f"{a.name}: 죽고 다시 태어남 (땅 위 아무 데서)"); x2 = senses()
            fut[reborn] = 0; held[reborn] = 0; hist.clear(); hhist.clear(); dhist.clear()      # 손이 옮긴 걸음은 거리 재기에서 뺀다
        if a.feed:                                           # 먹여 주기 (돌봄 단계, 장부): feed_every걸음마다 손에 빵을 쥐여 줌. 사이가 점점 길어짐 (끝에는 feed_grow배) - 차츰 뗀다
            every = int(a.feed_every * (1 + a.feed_grow * step / max(a.steps, 1)))
            due = [i for i in range(n) if step - last_fed[i] >= every or i in reborn]
            if due:
                w.feed(due, qty=a.feed, slot=("slot.weapon.offhand" if a.tool else "slot.hotbar.0")); stats["fed"] = stats.get("fed", 0) + len(due)
                if a.tool:                                   # 죽으면 가진 것을 잃는다: 다시 태어난 몸에 도구를 다시 쥐여 줌
                    regive = [i for i in due if i in reborn or i in dead]
                    if regive:
                        w.feed(regive, item=a.tool, qty=1)
                for i in due:
                    last_fed[i] = step
        if a.rescue:                                         # 기르는 손 (돌봄 단계, 장부, 차츰 뗌): 한참 갇혀 있으면 어른의 손이 꺼내 땅 위로 옮긴다. 기다리는 길이는 날이 갈수록 길어진다
            pos = np.array([o["pos"][[0, 2]] for o in w.obs]); stuck_since = np.where(np.hypot(*(pos - anchor).T) > 2.0, step, stuck_since); anchor = np.where((np.hypot(*(pos - anchor).T) > 2.0)[:, None], pos, anchor)
            wait = int(a.rescue * (1 + a.rescue_grow * step / max(a.steps, 1)))
            out_ = [i for i in range(n) if step - stuck_since[i] > wait]
            if out_:
                w.move(out_, r=400, label=f"{a.name}: 어른의 손이 꺼내 줌 ({wait}걸음 넘게 갇힘)"); stats["rescued"] += len(out_); hist.clear(); hhist.clear(); dhist.clear()
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
        if a.record and step < a.record:
            RJ[step] = np.asarray(died, bool) | np.asarray([i in dead for i in range(n)])      # 다음 그림과 잇지 않는다 (죽음, 다시 태어남)
            if a.rescue and out_:
                RJ[step, out_] = True
            if reborn:
                RJ[step, reborn] = True
            if step + 1 == a.record:
                np.savez(f"{OUT}/depthframes_{a.name}.npz", S=RS, W=RW, J=RJ); print(f"recorded {a.record} steps of small pictures for the depth eye", flush=True)
        moved_now = set(reborn) | set(dead) | (set(out_) if a.rescue else set())
        CUT = torch.tensor([bool(died[i]) or (i in moved_now) for i in range(n)], device=dev)
        for i in moved_now:
            JPREV[i] = 99.0; HRUN[i] = 0
        prev = (x, act); x = x2
        if a.seed_instinct and step == 300:                  # 타고난 본능 (창립자 결정 B, 장부): 어떤 장면이든 '초점 안의 작은 움직이는 것 + 때림 -> 배부름이 오름'.
            pool = torch.cat([b[0] for b in buf[-300:]])      # '어떤 장면이든': 갈래(느낌의 칸)마다 서로 다른 순간에서 따로 뽑아 장면을 고루 흩뜨린다 -> 장면은 가볍고, 움직임만 늘 같아 무겁다
            pick = torch.stack([pool[torch.randint(len(pool), (a.seed_instinct,), device=dev), d] for d in range(dx)], 1)
            hi = float(torch.quantile(pool[:, BUG], 0.9).clamp(min=0.3)); pick[:, BUG] = hi      # 겪어 본 높은 움직임 값 (위 10%)
            for i in range(len(pick)):
                net.keep(torch.tensor([[0.0, 0.2, 0.0]], device=dev), pick[i:i + 1], torch.tensor([4], device=dev), innate=a.innate_act)
            print(f"seeded {len(pick)} instinct memories (small mover straight ahead {hi:.2f} + hit -> food, innate act {a.innate_act})", flush=True)
            with torch.no_grad():                            # 바로 확인: 지금 장면에 바로 앞 움직임이 없을 때와 있을 때 끌림 (때리기 = 4)
                for bv in (0.0, hi):
                    xx = x[:1].clone(); xx[:, BUG] = bv; xx[:, 1] = 0.5; sc_, _ = net.pull(xx)
                    print(f"  check: mover {bv:.2f} -> pull {[round(v, 3) for v in sc_[0].tolist()]}", flush=True)
        if (step + 1) % 1000 == 0:
            ac = stats["acts"] / max(stats["acts"].sum(), 1)
            ah = stats['acts_hurt'] / max(stats['acts_hurt'].sum(), 1)
            print(f"  why: mean pull {[round(v, 3) for v in (dbg_sc / max(dbg_k, 1)).tolist()]} | mean futility x{TIRE} {[round(v * TIRE, 3) for v in (dbg_fut / max(dbg_k, 1)).tolist()]} | new picks {(dbg_pick / max(dbg_pick.sum(), 1)).round(2).tolist()} ({int(dbg_pick.sum())})", flush=True)
            dbg_sc.zero_(); dbg_fut.zero_(); dbg_k = 0; dbg_pick[:] = 0
            print(f"step {step + 1}: fed {stats.get('fed', 0)} | use when hungry {stats['use_hungry'] / max(stats['hungry_n'], 1):.3f} vs when full {stats['use_full'] / max(stats['full_n'], 1):.3f} | hurt-from-outside {stats['hurt_n']} -> moved away {stats['hurt_d'] / max(stats['hurt_n'], 1):.2f} in 20 steps (usual {stats['base_d'] / max(stats['base_n'], 1):.2f}); acts right after hurt {' '.join(f'{k}:{ah[k]:.2f}' for k in range(M))} | imagined {stats.get('imag', 0) / stats['steps']:.2f} | rescued {stats['rescued']} | small mover in focus {stats['bugseen'] / stats['steps']:.3f} (hit then {stats['hitbug'] / max(stats['bugseen'], 1):.2f}) | died {stats['died']} (world-ended lives {stats['deaths']}) | under water {stats['under'] / stats['steps']:.3f} | relief (health, food, air) {stats['relief']} | moved {stats['moved'] / stats['steps']:.3f}/step, stuck {stats['still'] / stats['steps']:.2f} | pulled {stats['chosen'] / stats['steps']:.2f} "
                  f"| surprised {stats['novel'] / stats['steps']:.2f} | elder spoke {EL.said if EL is not None else 0} | head tilted past 60 deg {stats.get('tilted', 0) / stats['steps']:.2f} | value trust earned {WO.earned() if WO is not None else 0:.2f} | judge: mined {stats.get('mined', 0):.0f}, animal in view {stats.get('ent_any', 0) / stats['steps']:.3f} of which in the middle {stats.get('ent_mid', 0) / max(stats.get('ent_any', 0), 1):.2f} (chance about 0.43), getting closer {stats.get('closer', 0) / max(stats.get('closer_n', 0), 1):.2f} | hits into empty air {stats.get('air_hits', 0) / max(stats.get('hits', 0), 1):.2f}, hitting run {stats.get('hit_run_len', 0) / max(stats.get('hit_runs', 0), 1):.1f} steps (runs of 15+ {stats.get('hit_run_15', 0)}) | acts {actsum(ac)} | {time.time() - t0:.0f}s", flush=True)
            if a.slow and int(net.slow_n):                       # 느린 층 안: 밖에서 맞은 꼴(숨 가득, 굶지 않음 - 맞기 직전의 느낌이라 맞은 감각은 대개 0)이 몇이나 남았나 - 재는 것
                sx = net.slow_x[:int(net.slow_n)]; print(f"  slow layer: {int(net.slow_n)} kinds from {int(net.slow_cnt.sum())} moments | outside kinds (air full, not starving) {int(((sx[:, 2] > 0.99) & (sx[:, 1] > 0.3)).sum())} | under-water kinds {int((sx[:, 2] < 0.99).sum())}", flush=True)
            torch.save(net.state_dict(), f"{OUT}/{a.name}.pt")
            stats.update(tilted=0, hits=0, air_hits=0, hit_runs=0, hit_run_len=0, hit_run_15=0, mined=0, ent_any=0, ent_mid=0, closer=0, closer_n=0, fed=0, use_hungry=0, use_full=0, hungry_n=0, full_n=0, base_d=0.0, base_n=0, hurt_d=0.0, hurt_n=0, acts_hurt=np.zeros(M, int), imag=0, rescued=0, bugseen=0, hitbug=0, died=0, deaths=0, under=0, relief=[0, 0, 0], chosen=0, novel=0, acts=np.zeros(M, int), steps=0, moved=0.0, still=0)
    print("DONE")


if __name__ == "__main__":
    main()
