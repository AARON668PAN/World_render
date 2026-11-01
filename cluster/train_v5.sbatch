#!/bin/bash
#SBATCH --job-name=g1i_obj4_2g
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=128
#SBATCH --time=14:00:00
#SBATCH --output=/scratch/u6ym/aaron668.u6ym/g1i/logs/obj4_%j.out
# v5 (10-31): v3 (aligned source p .5, object holes p .15, flat object colour) + 3 scene (info) frames per window from other
# moments of the segment, robot holed (ref_conv + learned mem_embed, dropped p .2), + holes glued to the handled object for
# the whole window (p .4; the human thumbs on the tissue box left grey patches in v3). Code: this repo's renderer/ in $D/World_render (was code5/), upstream core/ in $D/roborender.
# 2 GPUs, 5000 steps, LoRA 128, global batch 16 (8 per GPU), save every 250. ~+23 % tokens -> ~6.5 s/step, ~9 h.
# NOTE (repo copy): v5 ran from a code5/ snapshot taken before the v6 flags existed. This copy runs the current renderer/,
# whose v6 additions are all behind flags whose defaults keep the v5 path (--hole_px 5, --hand_stamps 0.7, --ghost 0,
# --mask_pool max, no --depth, --mask_drop 0, --band 0.4). Checked by reading the code against the v4 snapshot only; code5/
# itself is not available locally, so bit-identity with the run that made g1i_obj4_2g is unverified.
module load cuda/12.6 2>/dev/null
D=/scratch/u6ym/aaron668.u6ym/g1i; V=/scratch/u6ym/aaron668.u6ym/vis; PY=$V/venv/bin/python; cd $D/World_render/renderer
export PYTHONUNBUFFERED=1 TMPDIR=$V/../tmp ROBORENDER_ROOT=$D/roborender WAN_MODEL_DIR=$V/models/Wan2.1-Fun-V1.1-1.3B-Control PROMPT_CACHE=$V/data/prompt_cache.pt
N=8; echo CACHE_START obj1 maps $(ls $D/data/obj1/*.npz | wc -l) $(date)
for r in $(seq 0 $((N - 1))); do CUDA_VISIBLE_DEVICES=$((r % 2)) OMP_NUM_THREADS=8 $PY cache_g1i.py $D/data/cache_obj4 --root $D/data --obj_dir obj1 --info 3 --obj_stick 0.4 --hands $D/data/human_table_masks.npz --rank $r --world $N > $D/logs/cacheobj4_${SLURM_JOB_ID}_$r.log 2>&1 & done; wait
n=$(ls $D/data/cache_obj4/*.pt | wc -l); echo CACHE_DONE files=$n $(date)
if [ "$n" -lt 700 ]; then echo CACHE_TOO_FEW; exit 1; fi
$PY -m torch.distributed.run --nproc_per_node 2 train_src.py $D/runs/${RUN:-g1i_obj4_2g} ${STEPS:-5000} --cache $D/data/cache_obj4 --val $D/data/val_segments.json --balance --batch 8 --accum 1 --ckpt_grad --lora_rank 128 --lr 1e-4 --lr_patch 2e-5 --save_every 250 --warmup 200 --info_drop 0.2 --resume $EXTRA
echo TRAIN_JOB_END $(date)
