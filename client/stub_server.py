"""11-18: a stand-in for runtime_server.py in protocol tests without a GPU (fake_robot.py): same line protocol, but the "student"
replays the recorded labels of one teacher demo (64 token | 14 finger targets, every 4th row) from the step the episode is at.
usage: stub_server.py <demo w###.npz>"""
import json, sys
import numpy as np
z = np.load(sys.argv[1]); tn = [str(x) for x in z["target_names"]]; fin = [str(n) for n in z["fin_names"]]
L = np.concatenate([z["latent"], z["target"][:, [tn.index(j) for j in fin]]], 1).astype(np.float32)[::4]; step = 0
print(json.dumps(dict(ready=True, type="stub", state_dim=43, action_dim=78, image_hw=[168, 224], n_obs=2, horizon=13, official_format=True)), flush=True)
for line in sys.stdin:
    req = json.loads(line)
    if req.get("reset"): step = 0
    k, x = int(req.get("first", 1)), int(req.get("extra", 0)); idx = np.clip(step + np.arange(k + x), 0, len(L) - 1); step += k
    print(json.dumps(dict(actions=L[idx][None].round(5).tolist())), flush=True)
