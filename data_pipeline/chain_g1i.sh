#!/bin/bash
# HISTORICAL (repo note): the 10-29 pilot chain, kept as a record. It calls the working-copy paths ($R/*.py, sam3's
# local/track_prompt.py, roborender's local/g1src/), skips several steps, and trains the 1-GPU pilot config, not v5/v6.
# The current order with repo paths is data_pipeline/README.md; training is cluster/train_v*.sbatch.
# G1 + Inspire pilot: wait for the video download (files 0-3), cut segments, SAM3 leg spot check, SAM3 robot masks,
# cache (2 workers), train (LoRA 128, 4000 steps). (The old header also promised a quick eval; the script has none.)
R=/data/datasets/unitree_g1_inspire
M=/data/models/roborender_g1src/g1i
C=$R/cache
mkdir -p $M $C/qa
while [ $(ls $R/official/*/videos/observation.images.head_stereo_left/chunk-000/*.mp4 2>/dev/null | wc -l) -lt 104 ] && systemctl --user is-active --quiet g1i-fetch; do sleep 60; done
cd /data/repos/sam3
.venv/bin/python $R/extract_windows.py ${N:-1300} > $R/extract.log 2>&1 || exit 1
.venv/bin/python $R/leg_check.py > $R/leg_check.log 2>&1 || exit 1
python3 - <<'PY'
import json
R = "/data/datasets/unitree_g1_inspire"
segs = json.load(open(f"{R}/segments.json")); chk = json.load(open(f"{R}/leg_check.json"))
jobs = [dict(clip=s["id"], rgb=f"{R}/segments/{s['id']}.mp4", prompts=["robotic arm"]) for s in segs if s["id"] in chk and not chk[s["id"]]["leg"]]
json.dump(jobs, open(f"{R}/jobs_robot.json", "w")); print(len(jobs), "robot jobs")
import random; random.seed(0)
eps = sorted({(s["dataset"], s["episode"]) for s in segs}); ds = sorted({d for d, _ in eps})
val_eps = [random.choice([e for e in eps if e[0] == d]) for d in ds]
json.dump([s["id"] for s in segs if (s["dataset"], s["episode"]) in val_eps], open(f"{R}/val_segments.json", "w"))
PY
for K in 0 1; do JOB_RANK=$K JOB_WORLD=2 EDGE=1 .venv/bin/python local/track_prompt.py $R/jobs_robot.json $R robot > $R/track_robot_r$K.log 2>&1 & done; wait
cd /data/repos/roborender
for K in 0 1; do .venv/bin/python local/g1src/cache_g1i.py $C --rank $K --world 2 > $C/cache_r$K.log 2>&1 & done; wait
grep -q CACHE_DONE $C/cache_r0.log && grep -q CACHE_DONE $C/cache_r1.log || exit 1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/torchrun --nproc_per_node 1 --master_port 29534 local/g1src/train_src.py $M ${STEPS:-4000} --cache $C --val $R/val_segments.json --balance --batch 1 --accum 4 --save_every 1000 --ckpt_grad --ckpt_blocks 24 > $M/train.log 2>&1
echo CHAIN_DONE
