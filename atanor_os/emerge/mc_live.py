"""마크에서 같은 존재 파일로 산다 (공통 접면: 보편 몸 계약 + 하나의 마음). 방에서 기른 파일을 --load로 올리면 그 존재가 이어서 산다.
세계 서버를 ATANOR_MC_FRAME=1 ATANOR_MC_EAR=1 ATANOR_MC_SKIN=1 ATANOR_MC_BRANCH=1 (+ NOCROSS, JUDGE)로 띄워야 계약이 방과 같다.
  python -m atanor_os.emerge.mc_live --steps 12000 --load being/atanor.pt --out ...
돌봄 (장부, 차츰 뗌): 먹여 주기 (빵은 다른 손), 주 손에 곡괭이, 오래 갇히면 꺼내 줌. 죽음 알아채기는 mc_being과 같음.
"""
import argparse, os, time, numpy as np, torch
from .mc_client import MCWorld
from .room_body import contract
from .body_contract import Organs
from .mind import Mind


def raw(w, dev):
    o = w.obs
    bed = torch.tensor([[q["health"], q["food"], q["air"]] for q in o], device=dev, dtype=torch.float32)
    hurt = torch.tensor([[q["hurt"]] for q in o], device=dev, dtype=torch.float32).clamp(max=5)
    felt = torch.tensor(w.felt, device=dev, dtype=torch.float32)
    skin = torch.tensor(np.stack([q["skin"] for q in o]), device=dev, dtype=torch.float32)
    tilt = torch.tensor([[q.get("pitch", 0.0) / 90.0] for q in o], device=dev, dtype=torch.float32).clamp(-1, 1)
    ear = torch.tensor(np.stack([q["ear"] for q in o]), device=dev, dtype=torch.float32)
    frame = torch.tensor(np.stack([q["frame"] for q in o]), device=dev, dtype=torch.float32)
    return [bed, hurt, felt, skin, tilt, ear, frame]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=12000); ap.add_argument("--load", default=""); ap.add_argument("--out", default="")
    ap.add_argument("--name", default="mc_live"); ap.add_argument("--feed", type=int, default=1); ap.add_argument("--feed_every", type=int, default=300); ap.add_argument("--tool", default="stone_pickaxe")
    ap.add_argument("--rescue", type=int, default=300); ap.add_argument("--depth", default="mc/mc_depth.pt"); ap.add_argument("--learn", type=int, default=1)
    a = ap.parse_args(); dev = "cuda"; torch.manual_seed(0)
    w = MCWorld(); n = w.n; C = contract()
    O = Organs(C, n, depth=a.depth); Mi = Mind(O, C, n)
    if a.load:
        print("loaded being at step", Mi.load(a.load), flush=True)
        if Mi.wo is not None:
            Mi.wo.on = True
    w.move(range(n), r=400, label=f"{a.name}: 태어남")
    if a.tool:
        w.feed(range(n), item=a.tool, qty=1)
    w.feed(range(n), qty=a.feed, slot="slot.weapon.offhand"); last_fed = np.zeros(n, int)
    anchor = np.array([np.asarray(q["pos"])[[0, 2]] for q in w.obs]); stuck = np.zeros(n, int)
    st = dict(died=0, hurt=0, under=0, steps=0, relief=0, rescued=0)
    x = O(raw(w, dev)); t0 = time.time()
    for step in range(a.steps):
        act = Mi.choose(x)
        ins0 = w.inside(); pos0 = [np.asarray(q["pos"])[[0, 2]] for q in w.obs]
        obs = w.step(act.tolist(), who="하나의 마음 (방과 같은 존재 파일)" if a.load else "하나의 마음", label=f"{a.name}: 걸음 {step + 1} / {a.steps}")
        now = w.inside()
        jumped = np.array([np.hypot(*(np.asarray(q["pos"])[[0, 2]] - p)) > 15 for q, p in zip(w.obs, pos0)])
        died = (ins0[:, 0] > 0) & ((now[:, 0] <= 0) | jumped | ((ins0[:, 0] < 0.5) & (now[:, 0] >= 0.99) & (now[:, 1] >= 0.99)))
        x2 = O(raw(w, dev))
        st["died"] += int(died.sum()); st["hurt"] += int(((ins0[:, 0] - now[:, 0]) > 0.01).sum()); st["under"] += int((now[:, 2] < 0.99).sum()); st["steps"] += n
        st["relief"] += int(((now - ins0) > 0.01).any(1).sum())
        moved = set(i for i in range(n) if died[i] or obs[i]["done"])
        reborn = [i for i in moved if not obs[i]["done"]]
        if reborn:
            w.move(reborn, r=400, label=f"{a.name}: 다시 태어남")
            if a.tool:
                w.feed(reborn, item=a.tool, qty=1)
        dead = [i for i in range(n) if obs[i]["done"]]
        if dead:
            w.reset(dead); w.move(dead, r=400)
        pos = np.array([np.asarray(q["pos"])[[0, 2]] for q in w.obs]); far = np.hypot(*(pos - anchor).T) > 2.0
        stuck = np.where(far, step, stuck); anchor = np.where(far[:, None], pos, anchor)
        out_ = [i for i in range(n) if step - stuck[i] > a.rescue and i not in moved]
        if out_:
            w.move(out_, r=400, label=f"{a.name}: 어른의 손이 꺼내 줌"); st["rescued"] += len(out_); moved |= set(out_)
            for i in out_:
                stuck[i] = step; anchor[i] = np.asarray(w.obs[i]["pos"])[[0, 2]]
        due = [i for i in range(n) if step - last_fed[i] >= a.feed_every or i in reborn]
        if due:
            w.feed(due, qty=a.feed, slot="slot.weapon.offhand")
            for i in due:
                last_fed[i] = step
        cut = torch.tensor([i in moved for i in range(n)], device=dev)
        if moved:
            x2 = O(raw(w, dev)); O.reset(cut)
        if a.learn:
            Mi.after(x, act, x2, torch.tensor(died, device=dev), cut)
        else:
            Mi.prev = (x, act); Mi.cut = cut
        x = x2
        if (step + 1) % 1000 == 0:
            print(f"step {step + 1}: died {st['died']} | hurt {st['hurt']} | under water {st['under'] / st['steps']:.3f} | relief {st['relief']} | rescued {st['rescued']} | "
                  f"value trust earned {Mi.wo.earned() if Mi.wo is not None else 0:.2f} | relief memories {int(Mi.net.rel_n)} harm {int(Mi.net.harm_n)} | {time.time() - t0:.0f}s", flush=True)
            st.update(died=0, hurt=0, under=0, steps=0, relief=0, rescued=0)
            if a.out:
                Mi.save(a.out)
    print("DONE")


if __name__ == "__main__":
    main()
