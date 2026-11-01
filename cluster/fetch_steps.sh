#!/bin/bash
# Pull only chosen checkpoints of an Isambard run, as the small trainable part (LoRA A/B + input layer + ref_conv, bf16,
# ~350 MB) instead of the 3.1 GB merged ckpt, and merge them locally with the base Wan weights (merge_lora_g1i.py).
# resume.pt on Isambard is overwritten at every save (every 250 steps, ~22 min), so the export has to happen while it
# holds the wanted step: poll every 4 min; when ckpt_<step>.pt exists and resume.pt is at that step, export
# lora_<step>.pt there, download it, merge it here. Stops when all steps are done.
# usage: fetch_steps.sh <run> <step> [<step> ...]     e.g. fetch_steps.sh g1i_obj1_2g 3000 4000 5000
RUN=$1; shift; STEPS="$*"
REPO=$(cd "$(dirname "$0")/.." && pwd); RR=${ROBORENDER_ROOT:-/data/repos/roborender}  # this repo; RoboRender clone with .venv
H=u6ym.aip2.isambard; D=/scratch/u6ym/aaron668.u6ym/g1i/runs/$RUN; L=/data/models/roborender_g1src/$RUN
PY=/scratch/u6ym/aaron668.u6ym/vis/venv/bin/python; SSH="ssh -o ConnectTimeout=60"; mkdir -p $L
left="$STEPS"
while [ -n "$left" ]; do
  for s in $left; do
    S=$(printf %05d $s)
    if [ -f $L/ckpt_$S.pt ]; then left=$(echo $left | tr ' ' '\n' | grep -vx $s | tr '\n' ' '); continue; fi
    got=$(timeout 300 $SSH $H "cd $D 2>/dev/null || exit 0; [ -f lora_$S.pt ] && echo have && exit 0; [ -f ckpt_$S.pt ] || exit 0
      $PY -c \"
import torch
rs = torch.load('resume.pt', map_location='cpu')
if rs['step'] != $s: print('resume_at', rs['step']); raise SystemExit
keep = {k: (v.to(torch.bfloat16) if k.endswith(('.A.weight', '.B.weight')) else v) for k, v in rs['model'].items()}
torch.save(dict(step=rs['step'], keep=keep), 'lora_$S.pt'); print('have')\"" 2>&1 | tail -1)
    echo "$(date +%H:%M) step $s: ${got:-not yet}"
    if [ "$got" = "have" ]; then
      rsync -a --partial -e "$SSH" $H:$D/lora_$S.pt $L/ && \
        ROBORENDER_ROOT=$RR $RR/.venv/bin/python $REPO/renderer/merge_lora_g1i.py $L/lora_$S.pt $L/ckpt_$S.pt 2>&1 | grep -v -i warn | tail -1 && echo "READY $L/ckpt_$S.pt"
    elif echo "$got" | grep -q resume_at; then
      echo "MISSED step $s (resume.pt already moved on); recover it from ckpt_$S.pt if needed"
      left=$(echo $left | tr ' ' '\n' | grep -vx $s | tr '\n' ' ')
    fi
  done
  [ -n "$left" ] && sleep 240
done
echo FETCH_DONE
