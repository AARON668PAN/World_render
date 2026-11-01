"""Step 7 of the data pipeline (was inline Python in chain_g1i.sh, 10-29): jobs_robot.json = the leg-free segments
(leg_check.json "leg" false) with the SAM3 prompt "robotic arm" for track_prompt.py, and val_segments.json = every segment of
one random episode per dataset (random.seed(0)), the held-out split of every run v1..v6b (shipped as meta/val_segments.json).
usage: python3 make_jobs.py"""
import json
import os
R = os.environ.get("G1I_ROOT", "/data/datasets/unitree_g1_inspire")  # dataset work dir
segs = json.load(open(f"{R}/segments.json")); chk = json.load(open(f"{R}/leg_check.json"))
jobs = [dict(clip=s["id"], rgb=f"{R}/segments/{s['id']}.mp4", prompts=["robotic arm"]) for s in segs if s["id"] in chk and not chk[s["id"]]["leg"]]
json.dump(jobs, open(f"{R}/jobs_robot.json", "w")); print(len(jobs), "robot jobs")
import random; random.seed(0)
eps = sorted({(s["dataset"], s["episode"]) for s in segs}); ds = sorted({d for d, _ in eps})
val_eps = [random.choice([e for e in eps if e[0] == d]) for d in ds]
json.dump([s["id"] for s in segs if (s["dataset"], s["episode"]) in val_eps], open(f"{R}/val_segments.json", "w"))
