#!/bin/bash
# Push the G1 + Inspire pilot to Isambard (u6ym): leg-free segments re-encoded (crf 16, ~41 dB, 1/4.7 the size), the
# segment lists, the human hand masks for the hole shapes, the code (upstream RoboRender core/ -> $D/roborender, this repo's
# renderer/ and cluster/*.sbatch -> $D/World_render), then robot/, obj1/, obj_depth/ and full_depth/ (v6b).
# Run it once before any cluster/train_*.sbatch. The Wan weights, prompt cache and venv are set up separately (README).
set -e
REPO=$(cd "$(dirname "$0")/.." && pwd)  # this repo
R=${G1I_ROOT:-/data/datasets/unitree_g1_inspire}; H=u6ym.aip2.isambard; D=/scratch/u6ym/aaron668.u6ym/g1i
# hand/forearm masks of the tissue clip (sim_bridge/sam3_union_masks.py, run_tissue_chain.sh step 3): cache_g1i.py --hands
HANDS=${HANDS:-/data/repos/chord/local/ego/tissue_box/align/human_table_masks.npz}
cd $R
mkdir -p seg_up
python3 -c "import json;[print(j['clip']) for j in json.load(open('jobs_robot.json'))]" > seg_up/list.txt
cat seg_up/list.txt | xargs -P 16 -I{} sh -c '[ -s seg_up/{}.mp4 ] || ffmpeg -v error -y -i segments/{}.mp4 -c:v libx264 -crf 16 -preset slow -pix_fmt yuv420p seg_up/{}.mp4'
echo ENCODED $(ls seg_up/*.mp4 | wc -l) $(du -sh seg_up | cut -f1)
SSH="ssh -o ConnectTimeout=60"
$SSH $H "mkdir -p $D/roborender $D/World_render/renderer $D/World_render/cluster $D/data/segments $D/data/robot $D/logs $D/runs"
rsync -az -e "$SSH" ${ROBORENDER_ROOT:-/data/repos/roborender}/core $H:$D/roborender/
rsync -az -e "$SSH" --exclude __pycache__ $REPO/renderer/ $H:$D/World_render/renderer/
rsync -az -e "$SSH" $REPO/cluster/*.sbatch $H:$D/World_render/cluster/
rsync -az -e "$SSH" segments.json leg_check.json val_segments.json jobs_robot.json duplicates.json $HANDS $H:$D/data/
rsync -a -e "$SSH" seg_up/*.mp4 $H:$D/data/segments/
# masks and depth (were separate glue rsyncs: handoff2.sh robot/, handoff_obj1.sh obj1/, upload_v4.sh obj_depth/, unit upload-fulldepth full_depth/)
rsync -a -e "$SSH" $R/robot/ $H:$D/data/robot/
rsync -a -e "$SSH" $R/obj1/ $H:$D/data/obj1/
rsync -a --partial -e "$SSH" $R/obj_depth $H:$D/data/
rsync -a --partial -e "$SSH" $R/full_depth/ $H:$D/data/full_depth/
echo UPLOAD_DONE $(date)
