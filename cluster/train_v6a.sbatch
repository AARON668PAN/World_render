#!/bin/bash
#SBATCH --job-name=g1i_v6a
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=128
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/u6ym/aaron668.u6ym/g1i/logs/v6a_%j.out
# v6a: v5 + new holes (thin 1 px, no hand stamps, ghost p .5, glued holes p .2, hole channel = share per cell). Starts from v5 (runs/g1i_obj4_2g lora_05000.pt) via --init_from, 3000 steps, 2 GPUs, LoRA 128, global batch 16. Code: this repo's renderer/ in $D/World_render (was code6/), upstream RoboRender core/ in $D/roborender (cluster/upload_isambard.sh).
module load cuda/12.6 2>/dev/null
D=/scratch/u6ym/aaron668.u6ym/g1i; V=/scratch/u6ym/aaron668.u6ym/vis; PY=$V/venv/bin/python; cd $D/World_render/renderer
export PYTHONUNBUFFERED=1 TMPDIR=$V/../tmp ROBORENDER_ROOT=$D/roborender WAN_MODEL_DIR=$V/models/Wan2.1-Fun-V1.1-1.3B-Control PROMPT_CACHE=$V/data/prompt_cache.pt
# v5's trainable weights: lora_05000.pt is what the v6 runs used (made on Isambard by cluster/fetch_steps.sh g1i_obj4_2g 5000,
# LoRA A/B in bf16); without it, resume.pt of the finished v5 run holds the same step-5000 weights (fp32).
INIT=$D/runs/g1i_obj4_2g/lora_05000.pt; [ -f $INIT ] || INIT=$D/runs/g1i_obj4_2g/resume.pt; echo INIT_FROM $INIT
N=8; echo CACHE_START $(date)
for r in $(seq 0 $((N - 1))); do CUDA_VISIBLE_DEVICES=$((r % 2)) OMP_NUM_THREADS=8 $PY cache_g1i.py $D/data/cache_v6a --root $D/data --obj_dir obj1 --info 3 --obj_stick 0.2 --hole_px 1 --hand_stamps 0 --ghost 0.5 --mask_pool mean  --hands $D/data/human_table_masks.npz --rank $r --world $N > $D/logs/cachev6a_${SLURM_JOB_ID}_$r.log 2>&1 & done; wait
n=$(ls $D/data/cache_v6a/*.pt | wc -l); echo CACHE_DONE files=$n $(date)
if [ "$n" -lt 700 ]; then echo CACHE_TOO_FEW; exit 1; fi
$PY -m torch.distributed.run --nproc_per_node 2 train_src.py $D/runs/g1i_v6a 3000 --cache $D/data/cache_v6a --val $D/data/val_segments.json --balance --batch 8 --accum 1 --ckpt_grad --lora_rank 128 --lr 1e-4 --lr_patch 2e-5 --save_every 250 --warmup 100 --info_drop 0.2  --init_from $INIT --resume
echo TRAIN_JOB_END $(date)
