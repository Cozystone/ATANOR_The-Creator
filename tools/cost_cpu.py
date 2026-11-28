"""EFF1 efficiency ledger, first line (Astra's four columns; founder 2026-11-28): one body of the newest being, no graphics card, a few
threads - time of a beat split into the world (physics and drawing; not the being's cost on a real device), the senses, the choice and the
learning; peak memory of the process; the being's file size. A desktop CPU is a stand-in for the laptop, not the laptop. Power is NOT measured.
  CUDA_VISIBLE_DEVICES= PYTHONPATH=. python -m tools.cost_cpu OUT.json --being s3_e --eye 1.1 [--threads 4 --beats 300]"""
import argparse, json, os, time
import numpy as np, torch


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--being", required=True); ap.add_argument("--eye", type=float, default=0.0)
    ap.add_argument("--threads", type=int, default=4); ap.add_argument("--beats", type=int, default=300); ap.add_argument("--warm", type=int, default=60)
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    from tools.need_fork import t_room, place, DEV
    from tools.a1_channel import V2
    from tools.l1_life import make
    from tools.s4_kin import being, rebirth
    from atanor_os.emerge.room_body import RoomBody
    assert DEV == "cpu", "run with CUDA_VISIBLE_DEVICES= (no graphics card)"
    V2["on"] = True; RoomBody.default_cues = True
    if a.eye:
        RoomBody.default_eye = a.eye
    rows, info = t_room(9104, False); torch.manual_seed(1); n = 1
    B = make(rows, n, 9104, V2["drain"]); O, Mi = being(B, n, a.being)
    one = torch.ones(1, dtype=torch.bool); B.w.store[:] = 0.75; place(B, info); O.reset(one); Mi.begin(one); Mi.prev = None
    x = O(B.raw()); T = dict(world=[], senses=[], choice=[], learning=[]); tot = []
    for t in range(a.warm + a.beats):
        t0 = time.perf_counter(); act = Mi.choose(x); t1 = time.perf_counter()
        al = ~B.dead.clone(); B.step(act); raw = B.raw(); t2 = time.perf_counter()
        x2 = O(raw); t3 = time.perf_counter(); died = al & B.dead
        Mi.after(x, act, x2, died, B.dead & ~died); t4 = time.perf_counter(); x = x2
        if B.dead.any():
            rebirth(B, O, Mi, info); B.felt[:] = 0; x = O(B.raw())
        if t >= a.warm:
            T["choice"].append(t1 - t0); T["world"].append(t2 - t1); T["senses"].append(t3 - t2); T["learning"].append(t4 - t3); tot.append((t1 - t0) + (t3 - t2) + (t4 - t3))
    tr = time.perf_counter(); rest = Mi.rest() if hasattr(Mi, "rest") else (0, 0); rest_s = time.perf_counter() - tr
    ms = lambda v: dict(mean_ms=round(1000 * float(np.mean(v)), 2), p95_ms=round(1000 * float(np.percentile(v, 95)), 2), p99_ms=round(1000 * float(np.percentile(v, 99)), 2))
    try:
        import psutil
        pr = psutil.Process(); mem = dict(rss_mb=round(pr.memory_info().rss / 2 ** 20, 1), peak_mb=round(getattr(pr.memory_info(), "peak_wset", 0) / 2 ** 20, 1))
    except Exception as e:
        mem = dict(error=str(e))
    files = {f: round(os.path.getsize(os.path.join(a.being, f)) / 2 ** 20, 2) for f in os.listdir(a.being)}
    r = dict(protocol="atanor_os/WORLD_A_PROTOCOL.md EFF1", device="desktop CPU stand-in, no graphics card", threads=a.threads, beats=a.beats, bodies=1, being=a.being,
             flags={k: v for k, v in os.environ.items() if k.startswith("ATANOR_")}, world=ms(T["world"]), senses=ms(T["senses"]), choice=ms(T["choice"]), learning=ms(T["learning"]),
             being_beat=ms(tot), over_100ms_share=round(float(np.mean(np.array(tot) > 0.1)), 3), memory=mem, being_files_mb=files, rest=dict(seconds=round(rest_s, 2), steps=rest[0], bursts_lost=rest[1], waking_seconds=round(sum(tot), 2)), power="not measured")
    print(json.dumps(r), flush=True); json.dump(r, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
