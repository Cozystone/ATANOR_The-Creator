"""D23 (registered in atanor_os/A1_LATENT_PROTOCOL.md before running): the relief-share futures at several horizons (gamma .97 / .99 / .997), trained longer
(10,000 updates, target sync 25) on the saved 64-body lives; junction fork and far fork under the being's own choosing, and the greedy + 10% probe.
  PYTHONPATH=. python -m tools.a1_horizon OUT.json 9101_0 ..."""
import json, os, sys, time
import numpy as np, torch
from tools.need_fork import t_room, DEV, DEPTH
from tools.a1_channel import V2, side_sets, fork
from tools.a1_relief import relief_arm, shuffled, far_fork
from tools.a1_sf import paths_of
from tools.a1_greedy import greedy
from atanor_os.emerge.room_body import RoomBody
from atanor_os.emerge.body_contract import Organs
from atanor_os.emerge.mind import Mind

if __name__ == "__main__":
    V2["on"] = True; RoomBody.default_cues = True; res = []; t0 = time.time(); out = sys.argv[1]
    for lay in sys.argv[2:]:
        seed, mirror = map(int, lay.split("_")); d = os.environ.get("CACHE_ROOT", "a1cache_many") + f"/{lay}"; rows, info = t_room(seed, bool(mirror)); sd = seed * 10 + mirror
        ring = torch.load(os.path.join(d, "life_ring.pt"), weights_only=False); ring["bx2_true"], ring["bd_true"] = ring["bx2"], ring["bd"]; ck = os.path.join(d, "mind.pt")
        st = torch.load(ck, map_location=DEV, weights_only=False)
        c = RoomBody(1, rows, drain=0.0, device=DEV).c; tmp = Mind(Organs(c, 1, dev=DEV, depth=DEPTH), c, 1, dev=DEV, worth=0); dx, ins, M = tmp.net.dx, tmp.o.inside, tmp.M; del tmp
        L, R = side_sets(rows, info, seed); P = paths_of(rows, info); rec = dict(layout=lay, arms={})
        for g in (0.97, 0.99, 0.997):
            for shuf in (False, True):
                arm = relief_arm(shuffled(ring, ins, sd) if shuf else ring, dx, ins, M, sd, updates=10000, gamma=g, sync=25); nm = f"g{g}" + ("-shuf" if shuf else "")
                fo = fork(ck, arm, rows, info, sd, L, R); ff = far_fork(ck, arm, rows, info, sd, P)
                gj = greedy(arm, st["organs"], rows, info, sd, P, "junction", eps=0.1); gf = greedy(arm, st["organs"], rows, info, sd, P, "far", eps=0.1)
                rec["arms"][nm] = dict(value_pref_right=fo["value_pref_right"], first_match=fo["first_match"], recover_share=fo["recover_share"], far=ff["recover"],
                                       greedy_junction=gj, greedy_far=gf["recover"])
                print(lay, nm, "being: first", [round(v, 2) for v in fo["first_match"]], "rec", fo["recover_share"], "far", ff["recover"], "| greedy: first", gj["first_match"], "rec", gj["recover"], "far", gf["recover"], round(time.time() - t0), flush=True)
                del arm; torch.cuda.empty_cache()
        res.append(rec); json.dump(dict(protocol="atanor_os/A1_LATENT_PROTOCOL.md (D23)", rows=res), open(out, "w"), indent=1)
