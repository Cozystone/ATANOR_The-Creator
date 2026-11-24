"""채널별 앞날 (A1 시험, atanor_os/A1_CHANNEL_PROTOCOL.md, 2026-11-23). 기존 Worth는 그대로 두고 대조군으로 쓴다.

ChannelP: '여기서 이 몸짓을 하면, 몸 안의 칸마다 앞으로 얼마나 바뀌나' (할인된 합, gamma 0.97). 제 삶의 걸음에서만 배운다 - 칸의 뜻 (물, 먹이)도 쪽 (왼, 오른)도 모른다.
  목표: db = b(x2) - b(x) (죽음이면 -b(x)) + gamma psi_old(x2, a*), a* = 다음 장면에서 점수가 가장 나은 몸짓 (successor features, Barreto 2017).
ChannelWorth(mode):
  'bedrock' (갈래 3) - 몸짓의 점수 S = whole(clip(b + psi, 0, 1)) - whole(b). 이 바꿈 (내다본 칸 바뀜 -> 몸짓 점수)은 설계자가 준 부분이다 (장부).
             칸마다의 무게는 바탕 (온전함 = 1 - 결핍^2의 평균)의 정의에서 나온다. exact=False면 작은 바뀜의 근사 sum 2(1-b)/n psi (주 판정에 쓰지 않음).
  'aux'     (갈래 2) - 같은 P의 줄기 특징 (얼려 둠)과 느낌 위에 몸짓 값 머리 Q2를 Worth와 똑같이 (온전함 바뀜 보상, TD) 배운다.
Worth의 고리·믿음·기댐 (see, track, earned, lean)을 그대로 물려받아 Mind에 그대로 꽂힌다 (Mind(value_arm=...))."""
import copy
import torch, torch.nn as nn
from .core import Worth


class ChannelP(nn.Module):
    def __init__(self, dx, m, ni, d=128):
        super().__init__()
        self.m, self.ni = m, ni
        self.trunk = nn.Sequential(nn.Linear(dx, d), nn.ELU(), nn.Linear(d, d), nn.ELU())
        self.head = nn.Linear(d, m * ni)

    def forward(self, x):
        h = self.trunk(x)
        return h, self.head(h).view(*x.shape[:-1], self.m, self.ni)


class ChannelWorth(Worth):
    def __init__(self, dx, inside, lam=2.0, gamma=0.97, keep=300000, d=128, m=None, mode="bedrock", P=None, exact=True):
        super().__init__(dx, inside, lam=lam, gamma=gamma, keep=keep, d=d, m=m)
        del self._modules["v"], self._modules["vt"]                  # 값 그물 대신 아래의 점수를 쓴다 (고리, 믿음, 기댐은 그대로)
        self.mode, self.exact, self.ni = mode, exact, inside.stop - inside.start
        self.P = P if P is not None else ChannelP(dx, m, self.ni, d)
        self.Pt = copy.deepcopy(self.P)
        for p in self.Pt.parameters():
            p.requires_grad_(False)
        self.train_P = P is None                                     # 남의 P를 빌리면 (갈래 2가 갈래 3의 P를) 그 P는 배우지 않는다
        if mode == "aux":
            mk = lambda: nn.Sequential(nn.Linear(dx + d, d), nn.ELU(), nn.Linear(d, m))
            self.q, self.qt = mk(), mk(); self.qt.load_state_dict(self.q.state_dict())
            for p in self.qt.parameters():
                p.requires_grad_(False)
        object.__setattr__(self, "v", self.score); object.__setattr__(self, "vt", self.score_target)
        self.popt = None

    def _whole_b(self, b):
        return 1 - ((1 - b).clamp(min=0) ** 2).mean(-1)

    def _S(self, x, psi):
        b = x[..., self.inside]
        if self.exact:
            return self._whole_b((b[..., None, :] + psi).clamp(0, 1)) - self._whole_b(b)[..., None]
        return (2 * (1 - b).clamp(min=0) / self.ni)[..., None, :].mul(psi).sum(-1)

    def psi(self, x):
        return self.P(x)[1]

    def score(self, x):
        h, psi = self.P(x)
        if self.mode == "aux":
            return self.q(torch.cat([x, h.detach()], -1))
        return self._S(x, psi)

    def score_target(self, x):
        h, psi = self.Pt(x)
        if self.mode == "aux":
            return self.qt(torch.cat([x, h], -1))
        return self._S(x, psi)

    def learn(self, n=4, batch=2048, lr=1e-3):
        top = int(self.bn)
        if top < 4096:
            return None
        if self.popt is None:
            ps = (list(self.P.parameters()) if self.train_P else []) + (list(self.q.parameters()) if self.mode == "aux" else [])
            self.popt = torch.optim.Adam(ps, lr)
        ins = self.inside; loss = None
        for _ in range(n):
            j = torch.randint(0, top, (batch,), device=self.bx.device)
            x, x2, a, d, r = self.bx[j], self.bx2[j], self.ba[j], self.bd[j], self.br[j]
            loss = 0.0
            if self.train_P:
                with torch.no_grad():
                    _, p2 = self.Pt(x2)
                    if getattr(self, "target", "max") == "sarsa":           # D12: the combo really done next (mean over combos where unknown)
                        an = self.ba2[j]; nxt = torch.where((an >= 0)[:, None], p2.gather(1, an.clamp(min=0)[:, None, None].expand(-1, 1, self.ni))[:, 0], p2.mean(1))
                    else:
                        a2 = self._S(x2, p2).argmax(1)
                        nxt = p2.gather(1, a2[:, None, None].expand(-1, 1, self.ni))[:, 0]
                    db = torch.where(d[:, None] > 0, -x[:, ins], x2[:, ins] - x[:, ins])
                    tg = db + self.gamma * nxt * (1 - d)[:, None]
                _, p = self.P(x)
                out = p.gather(1, a[:, None, None].expand(-1, 1, self.ni))[:, 0]
                loss = loss + ((out - tg) ** 2).mean()
            if self.mode == "aux":
                with torch.no_grad():
                    h2, _ = self.P(x2); h, _ = self.P(x)
                    tq = r + self.gamma * self.qt(torch.cat([x2, h2], -1)).max(1).values * (1 - d)
                q = self.q(torch.cat([x, h], -1)).gather(1, a[:, None])[:, 0]
                loss = loss + ((q - tq) ** 2).mean()
            self.popt.zero_grad(); loss.backward(); self.popt.step()
            self.steps += 1
            if self.steps % 40 == 0:
                if self.train_P:
                    self.Pt.load_state_dict(self.P.state_dict())
                if self.mode == "aux":
                    self.qt.load_state_dict(self.q.state_dict())
        return float(loss)


class ReliefWorth(ChannelWorth):
    """Latent learning (D17-D23, 2026-11-24): per-channel futures as [share of the lack relieved (0..1), loss (<= 0)], learned by SARSA from the being's own
    steps at horizon gamma (.99, about 100 beats). Score of a combo: S = whole(b + (1 - b) clip(psi+, 0, 1) + psi-) - whole(b) - 'what is where' is learned
    whatever the need was; the current need enters only through b. The conversion is designer-given (ledger), reward is bedrock only, no behaviour rule."""
    def __init__(self, dx, inside, m, d=128, keep=300000, gamma=0.99, sync=25, lam=2.0, per_call=4):
        super().__init__(dx, inside, lam=lam, gamma=gamma, m=m, keep=keep, d=d, mode="bedrock")
        self.sync, self.per_call, self.target = sync, per_call, "sarsa"
        self.P = ChannelP(dx, m, 2 * self.ni, d); self.Pt = copy.deepcopy(self.P)
        for p in self.Pt.parameters():
            p.requires_grad_(False)

    def _S(self, x, psi):
        b = x[..., self.inside]; pp = psi[..., :self.ni].clamp(0, 1); pm = psi[..., self.ni:]
        return self._whole_b((b[..., None, :] + (1 - b)[..., None, :] * pp + pm).clamp(0, 1)) - self._whole_b(b)[..., None]

    def learn(self, n=4, batch=2048, lr=1e-3):
        top = int(self.bn)
        if top < 4096:
            return None
        if self.popt is None:
            self.popt = torch.optim.Adam(self.P.parameters(), lr)
        ins = self.inside; ni2 = 2 * self.ni; loss = None
        for _ in range(n * self.per_call):
            j = torch.randint(0, top, (batch,), device=self.bx.device); x, x2, a, d = self.bx[j], self.bx2[j], self.ba[j], self.bd[j]
            with torch.no_grad():
                _, p2 = self.Pt(x2); an = self.ba2[j]
                nxt = torch.where((an >= 0)[:, None], p2.gather(1, an.clamp(min=0)[:, None, None].expand(-1, 1, ni2))[:, 0], p2.mean(1))
                db = x2[:, ins] - x[:, ins]; dead = d[:, None] > 0
                fp = torch.where(dead, torch.zeros_like(db), (db.clamp(min=0) / (1 - x[:, ins] + 0.02)).clamp(max=1))
                fm = torch.where(dead, -x[:, ins], db.clamp(max=0))
                tg = torch.cat([fp, fm], 1) + self.gamma * nxt * (1 - d)[:, None]
            _, p = self.P(x); out = p.gather(1, a[:, None, None].expand(-1, 1, ni2))[:, 0]
            loss = ((out - tg) ** 2).mean(); self.popt.zero_grad(); loss.backward(); self.popt.step(); self.steps += 1
            if self.steps % self.sync == 0:
                self.Pt.load_state_dict(self.P.state_dict())
        return float(loss.detach())


def params(mod):
    return sum(p.numel() for p in mod.parameters())
