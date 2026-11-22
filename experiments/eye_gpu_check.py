"""Does the GPU retina give the same 27x16x16 as the canon fovea eye (as the world server uses it) on real Minecraft-like frames?
Uses recorded small frames upscaled? No - uses synthetic + real full frames: we render random frames of 160x256 from the recorded 48x80 pictures (bilinear up)
plus pure random noise, run both eyes over a sequence (so the fixation moves the same way), and compare."""
import importlib.util, sys, numpy as np, torch
from PIL import Image
sys.modules.setdefault("x", None)
EYE = "atanor_os/eye/fovea.py"
sp = importlib.util.spec_from_file_location("canon_fovea", EYE); fv = importlib.util.module_from_spec(sp); sys.modules["canon_fovea"] = fv; sp.loader.exec_module(fv)
from atanor_os.emerge.eye_gpu import RetinaGPU
d = np.load("mc/depthframes_being28_slow_rep.npz")
S = d["S"][:200, :2]                                      # [T, 2 bodies, 48, 80, 3]
T, n = S.shape[:2]
frames = np.stack([[np.asarray(Image.fromarray(S[t, b]).resize((256, 160), Image.BILINEAR)) for b in range(n)] for t in range(T)]).astype(np.float32)
canon = [fv.Retina(spec=fv.RetinaSpec(fovea_px=16, rings=4, ring_px=16, falloff=2.0, ring_peak=True)) for _ in range(n)]
gpu = RetinaGPU(n, 160, 256)


def canon_ret(eye, rgb):                                    # exactly as mc_server.retina()
    s = eye.look(rgb.astype(np.uint8))
    fov = np.asarray(Image.fromarray(np.ascontiguousarray(s["fovea"]).astype(np.uint8)).resize((16, 16), Image.BILINEAR), np.float32)
    lv = [fov] + [np.asarray(l["px"], np.float32) for l in s["levels"]]
    while len(lv) < 5: lv.append(lv[-1])
    pk = [np.asarray(l["peak"], np.float32) for l in s["levels"]]
    while len(pk) < 4: pk.append(pk[-1])
    st = np.concatenate([np.stack(lv[:5]), np.stack(pk[:4])], 0)
    return st.transpose(0, 3, 1, 2).reshape(27, 16, 16) / 255.0, (eye.cx, eye.cy)


errs, gaze = [], []
for t in range(T):
    a = np.stack([canon_ret(canon[b], frames[t, b])[0] for b in range(n)])
    g = gpu.look(torch.tensor(frames[t], device="cuda")).cpu().numpy()
    errs.append(np.abs(a - g).reshape(n, 9, 3, 16, 16).mean((0, 2, 3, 4)))
    gaze.append([abs(canon[b].cx - float(gpu.cx[b])) + abs(canon[b].cy - float(gpu.cy[b])) for b in range(n)])
e = np.array(errs).mean(0)
print("mean abs difference per channel group (0 = identical; values are 0..1):", np.round(e, 4).tolist())
print("fixation difference (sum |dx|+|dy|, fractions of the frame): mean", round(float(np.mean(gaze)), 4), "max", round(float(np.max(gaze)), 4))
