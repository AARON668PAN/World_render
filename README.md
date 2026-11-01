# World_render: human ego video -> Unitree G1 + Inspire video

A video model that takes a first-person video of a person doing a manipulation task and renders the same scene with
a Unitree G1 humanoid with Inspire five-finger hands doing it. The robot's motion comes from simulation: the human
hands and the object are reconstructed, retargeted to the G1, and rendered in Newton. The video model then paints a
real-looking robot into the human video, where the sim says it is.

- Model: Wan2.1-Fun-V1.1-1.3B-Control + LoRA rank 128. The Wan model code is RoboRender's `core/`
  ([robo-render/RoboRender-Video-Model](https://github.com/robo-render/RoboRender-Video-Model) @16850eb); everything in
  this repo is ours.
- Training data: real G1 + Inspire video only (Unitree's official `G1_WBT_Inspire_*` datasets, 1043 segments where
  the robot's legs are not visible). Human video is used only at inference.
- Input layer, 65 channels: noise 16 | control A (robot cut-out + handled object in one colour) 16 | first-frame slot
  16 (zeros; v6b: depth) | source video with grey holes 16 | hole mask 1. Three scene frames from other moments and
  the last frame of the previous chunk enter as extra tokens.
- Current best: **v6a** (step 3000). v6b (adds whole-frame depth) is best on the grey patches under the human thumbs.
  Numbers in [`docs/RESULTS.md`](docs/RESULTS.md).

A picture-by-picture explainer of every input (Chinese) is
[`docs/pipeline/g1_renderer_v6.html`](docs/pipeline/g1_renderer_v6.html). It was written on 10-31 before v6 was
trained, so its plan and status parts are out of date.

## Layout

| Folder | What |
|---|---|
| `renderer/` | the model: cache building, training, LoRA merge, human-clip input prep, inference |
| `cluster/` | Isambard-AI (GH200, SLURM) upload, training jobs, checkpoint fetch |
| `data_pipeline/` | download, leg filtering, SAM3 robot masks, Claude object boxes + SAM3 tracking, VDA depth ([README](data_pipeline/README.md)) |
| `sim_bridge/` | human video -> sim render: camera, object fit, Inspire finger retargeting, whole-body IK, Newton render |
| `tools/` | evaluation (`eval/`), page figures, previews, data checks, the native CHORD reconstruction setup (`chord_recon/`) |
| `env/` | package lists of the envs |
| `docs/` | results, changes from the working copy, CHORD setup notes, the explainer page |

## External code and weights

| What | Where | Pin |
|---|---|---|
| RoboRender (Wan model code `core/`) | https://github.com/robo-render/RoboRender-Video-Model | 16850eb |
| Wan2.1-Fun-V1.1-1.3B-Control | HF `alibaba-pai/Wan2.1-Fun-V1.1-1.3B-Control` | DiT, VAE, CLIP, T5 (T5 only for the prompt cache) |
| SAM3 | https://github.com/facebookresearch/sam3 | f1647e8, `sam3.pt` |
| Video-Depth-Anything | https://github.com/DepthAnything/Video-Depth-Anything | 4f5ae23, `video_depth_anything_vitl.pth` (relative, large) |
| CHORD (video_to_data) | https://github.com/nvidia-isaac/video_to_data | origin/main 7eb9721 when cloned; gives `tissue_box.mp4`, its processed parquet and the retarget code |
| unitree_ros | https://github.com/unitreerobotics/unitree_ros | deeb607 (G1 + Inspire FTP URDF and meshes) |
| xr_teleoperate | https://github.com/unitreerobotics/xr_teleoperate | 817fb00 (DexPilot `inspire_hand.yml`) |
| Training data | HF `unitreerobotics/G1_WBT_Inspire_*` | 26 datasets, video files 000-003 each |

`sim_bridge/assets/head_cam_calib.json` is the G1 head-camera pose relative to the torso, calibrated earlier on the
Unitree G1 Dex3 `pick_tissue` data (`tools/head_cam_calib/calibrate_head_cam.py`; its inputs are not shipped). It is
used as is.

## Environments

| Env | Spec | Runs |
|---|---|---|
| renderer | `env/renderer_reqs.txt` (RoboRender's requirements + opencv, scipy, lpips) | `renderer/*` except `prep_human_depth.py`, `tools/eval/*` |
| SAM3 + VDA | `env/sam3_vda_reqs.txt` | `data_pipeline/` SAM3 and VDA steps, `renderer/prep_human_depth.py`, `sim_bridge/sam3_*.py` |
| retarget | `sim_bridge/env/retarget_reqs.txt` (pinocchio, pink, mediapipe, pyarrow) | `sim_bridge/` camera / fit / IK steps, `data_pipeline/prep_index.py`, `leg_clf.py` |
| Newton | `env/newton_reqs.txt` (newton 1.1.0.dev0, warp, dex_retargeting) | `sim_bridge/mano_to_inspire.py`, `inspire_contact_ik.py`, `render_sim_depth.py`, `recolor_robot.py` |
| CHORD reconstruction | `tools/chord_recon/` | only for new human videos (not run end to end yet) |

### Environment variables

The defaults are the paths on the original machine.

| Variable | Default | Used by |
|---|---|---|
| `ROBORENDER_ROOT` | `/data/repos/roborender` | renderer, `tools/eval/diag_gap.py`, `cluster/fetch_steps.sh` (also expects `$ROBORENDER_ROOT/.venv`) |
| `WAN_MODEL_DIR` | `/data/models/roborender/Wan2.1-Fun-V1.1-1.3B-Control` | renderer (`inst_model.py`), `cache_empty_prompt.py` |
| `PROMPT_CACHE` | `/data/models/roborender/prompt_cache.pt` | renderer (empty-prompt T5 embedding) |
| `G1I_ROOT` | `/data/datasets/unitree_g1_inspire` | `data_pipeline/*`, `cluster/upload_isambard.sh` (renderer scripts take `--root`) |
| `SAM3_CKPT`, `SAM3_TMP` | `/data/models/sam3/sam3/sam3.pt` | SAM3 scripts |
| `VDA_ROOT`, `VDA_CKPT` | `/data/repos/video_depth_anything`, `/data/models/video_depth_anything/large/video_depth_anything_vitl.pth` | depth scripts |
| `CHORD`, `UNITREE_ROS`, `XR_TELEOP` | `/data/repos/chord`, `/data/repos/unitree_assets/{unitree_ros,xr_teleoperate}` | `sim_bridge/` |
| `E`, `V`, `SIM` | the tissue-clip dir, video and sim render of the original run | `sim_bridge/run_tissue_chain.sh`, `tools/eval/compare_runs.py` |
| `HANDS` | `/data/repos/chord/local/ego/tissue_box/align/human_table_masks.npz` | `cluster/upload_isambard.sh` |
| `G1_FISH_F` | 256.7 | `renderer/cache_g1i.py` (G1 head-lens focal, only for the unused ring hole) |
| `RUN`, `STEPS`, `EXTRA` | `g1i_obj4_2g`, 5000, empty | `cluster/train_v5.sbatch` |
| `FETCH_MIRROR`, `FETCH_PROXY` | hf-mirror, `http://127.0.0.1:10900` | `data_pipeline/fetch_all.py` |

The Isambard host, user and scratch paths are written into `cluster/*` (`u6ym.aip2.isambard`,
`/scratch/u6ym/aaron668.u6ym/...`). Change them there for another account.

## How to run it

### 0. Setup (local machine)

```bash
git clone https://github.com/robo-render/RoboRender-Video-Model /data/repos/roborender && git -C /data/repos/roborender checkout 16850eb
export WAN_MODEL_DIR=/data/models/roborender/Wan2.1-Fun-V1.1-1.3B-Control PROMPT_CACHE=/data/models/roborender/prompt_cache.pt
hf download alibaba-pai/Wan2.1-Fun-V1.1-1.3B-Control --local-dir $WAN_MODEL_DIR \
    --include "diffusion_pytorch_model.safetensors" "Wan2.1_VAE.pth" "models_t5_umt5-xxl-enc-bf16.pth" \
              "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" "google/umt5-xxl/*"
python renderer/cache_empty_prompt.py            # writes $PROMPT_CACHE (renderer env, one GPU)
```

(This is the download line of RoboRender's README; RoboRender's own LoRA is not used.) Then set up SAM3, VDA, the
retarget venv and the Newton env from the lists above.

### 1. Data

Follow [`data_pipeline/README.md`](data_pipeline/README.md). In short: download -> index -> leg filtering -> cut
segments -> SAM3 robot masks -> Claude boxes the handled object -> SAM3 tracks it -> object depth -> whole-frame depth.
The lists in `data_pipeline/meta/` pin the exact segment set, and `data_pipeline/labeling/results/` holds the boxes.

The cache also needs the tissue clip's human hand/forearm masks (`cache_g1i.py --hands`; v5 draws random hand-shaped
holes from them, v6 does not use them). Make them first:

```bash
# V = the human video, E = the tissue-clip work dir (as in sim_bridge/run_tissue_chain.sh, which does this as step 3)
$SAM3_PY sim_bridge/sam3_union_masks.py $V $E/align/human_table_masks.npz hand forearm table
```

### 2. Training on Isambard

One-time setup on Isambard (not scripted here). The sbatch files expect, under `/scratch/u6ym/aaron668.u6ym/vis`
(`$V` in the sbatch files): a venv `venv/` with the renderer packages for aarch64 + CUDA 12.6, the Wan weights in
`models/Wan2.1-Fun-V1.1-1.3B-Control/` (same `hf download` as above) and the prompt cache `data/prompt_cache.pt` (copy
the local one, or run `cache_empty_prompt.py` there). The package list of that venv was not recorded.

```bash
cluster/upload_isambard.sh                       # once; code + data -> $D/roborender, $D/World_render, $D/data
ssh u6ym.aip2.isambard "cd /scratch/u6ym/aaron668.u6ym/g1i/World_render/cluster && sbatch train_v5.sbatch"
cluster/fetch_steps.sh g1i_obj4_2g 5000           # exports lora_05000.pt on Isambard, downloads and merges it
ssh u6ym.aip2.isambard "cd /scratch/u6ym/aaron668.u6ym/g1i/World_render/cluster && sbatch train_v6a.sbatch && sbatch train_v6b.sbatch"
cluster/fetch_steps.sh g1i_v6a 3000; cluster/fetch_steps.sh g1i_v6b 3000
```

Each job builds its cache first (8 workers on 2 GPUs), then trains. v5: 5000 steps, about 9 h. v6a / v6b: 3000
steps from v5's step-5000 weights. `fetch_steps.sh` downloads only the trainable part (about 350 MB) and rebuilds the
3.1 GB checkpoint locally with `renderer/merge_lora_g1i.py`. The v6 jobs start from `runs/g1i_obj4_2g/lora_05000.pt`
(made by `fetch_steps.sh`) or, if that is missing, from the finished v5 run's `resume.pt`.

The sbatch files are path-adapted copies of the jobs that ran; see
[`docs/CHANGES_FROM_WORKING_COPY.md`](docs/CHANGES_FROM_WORKING_COPY.md).

### 3. A human clip

Only CHORD's `tissue_box` sample has gone through this so far. A new video first needs CHORD's reconstruction
(`docs/chord_recon_native.md`).

```bash
E=<work dir> bash sim_bridge/run_tissue_chain.sh   # 2D hands, masks, camera, object fit, Inspire retarget, IK, Newton render
```

Then `prep_human_g1i.py` (and for v6b `prep_human_depth.py` on the same npz), `infer_g1i.py`, and
`tools/eval/compare_runs.py`. The exact commands of the reported runs are in [`docs/RESULTS.md`](docs/RESULTS.md).
`run_tissue_chain.sh` was written afterwards from the outputs and has not been run as one script.

## Trained weights

Not in this repo and not released yet. On the training machine they are in `/data/models/roborender_g1src/`:
`g1i_obj4_2g/lora_05000.pt` (v5), `g1i_v6a/lora_03000.pt`, `g1i_v6b/lora_03000.pt` (352 MB each, LoRA + input layer
+ ref_conv) and the merged `ckpt_*.pt`. To rebuild a checkpoint from a lora file:

```bash
python renderer/merge_lora_g1i.py lora_03000.pt ckpt_03000.pt
```

## Things to know

- The data, figure and preview scripts write into the dataset dir: `tools/figures/*` and `tools/previews/*` overwrite
  files in `$G1I_ROOT/check/` and `pipeline_page/img/`, and `tools/figures/page_outputs_v5.py` overwrites
  `/data/models/roborender_g1src/compare_v3v4v5/compare_5000.mp4`. Point them elsewhere before re-running.
- `data_pipeline/chain_g1i.sh` is the 10-29 pilot chain, kept as a record. Use `data_pipeline/README.md` instead.
- `tools/chord_recon/` records the native CHORD reconstruction env, which lives in `/data/repos/chord/local/`.
- Held-out results cover 3 segments; see the caveats in `docs/RESULTS.md`.
