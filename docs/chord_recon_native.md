# CHORD ego reconstruction: native setup (no Docker)

> Repo note (11-01). This is the setup record from the CHORD working copy (`/data/repos/chord/local/RECON_SETUP.md`).
> The env it describes lives there, so the paths below point there. The copies in this repo: `$L/recon_env.sh`,
> `link_mano.sh`, `recon_reqs.txt`, `recon_smoke_partial.py` -> `tools/chord_recon/`; `$L/recon_patches/...` ->
> `tools/chord_recon/patches/...`; the build/download scripts of `$L/logs/` -> `tools/chord_recon/setup/`
> (`tools/chord_recon/README.md`). MANO and the FoundationPose weights arrived on 10-31 evening (see Blockers), but
> the full pipeline has not been run end to end yet. The renderer's tissue clip uses CHORD's shipped processed parquet.

Goal: run `reconstruction/modules/v2d_pipelines/run_ego_reconstruction.py --hand_tracking hamer`
(WiLoR + SAM2 + HaMeR hands, Grounding DINO + SAM2 + FoundationPose object, MoGe depth,
AnyCalib undistortion) on the RTX 5090 without Docker. Upstream images are
`pytorch/pytorch:2.5.1-cuda12.4` with `TORCH_CUDA_ARCH_LIST<=9.0` and cannot run on sm_120.

Status as of 2026-10-31: everything for the `--hand_tracking hamer --object_mesh ...` path is
installed and compiled for sm_120. The two things only the user can provide, `MANO_RIGHT.pkl` and the
FoundationPose weights, were added by the user late on 10-31 (see "Blockers"); nothing has been run with them yet.

## Environment

| What | Where |
|---|---|
| venv (uv, Python 3.11.14) | `/data/repos/chord/local/.venv-recon` (14 GB) |
| activate | `source /data/repos/chord/local/recon_env.sh` (PATH, CUDA_HOME, CPATH, TORCH_CUDA_ARCH_LIST=12.0, runtime env, network env) |
| CUDA toolkit for building extensions | `/data/repos/chord/local/.cuda128` (conda-forge nvcc 12.8.93, 1.5 GB; `lib64` symlink added by hand) |
| upstream source clones | `/data/repos/chord/local/third_party/` (hamer @3a01849, WiLoR-mini, manotorch, chumpy, nvdiffrast, pytorch3d, GroundingDINO, sam2, MoGe @925b8ed, AnyCalib) |
| out-of-tree builds | `/data/repos/chord/local/build/` (mycpp, mycuda) |
| patches | `/data/repos/chord/local/recon_patches/` |
| pip list of plain deps | `/data/repos/chord/local/recon_reqs.txt` |
| build / download logs | `/data/repos/chord/local/logs/` (`build_*.log`, `dl_*.log`, `recon_*.log`) |

Key versions: torch 2.8.0+cu128 (arch list includes sm_120), torchvision 0.23.0, numpy 1.26.4,
opencv 4.11.0.86, pytorch3d 0.7.9 (source, sm_120), nvdiffrast 0.4.0 (source, sm_120),
GroundingDINO 0.1.0 (source + CUDA op, sm_120), SAM-2 1.0 (source + `_C`, sm_120),
tensorrt-cu12 10.13.3.9, transformers 4.27.4 (pinned like the GroundingDINO image),
ultralytics 8.1.34, smplx 0.1.28, manotorch 0.0.2, chumpy 0.71 (git master, numpy-1.26 safe),
pytorch-lightning 2.6.6, open3d 0.20.0, warp-lang 1.18.0, kornia 0.9.0, moge 2.0.0, anycalib 1.0.

CHORD packages are installed editable from the repo (`v2d_common`, `v2d_depth`, `v2d_mv`,
`v2d_mesh/lib`, the `lib/` of hamer, wilor, foundation_pose, sam2, grounding_dino, moge,
anycalib, all host-side `docker/` wrappers that `install_ego_reconstruction_packages.sh`
lists, and `v2d_pipelines`). The CHORD repo itself is unmodified (`git status` shows only `local/`).

### How "Docker" runs natively

Every stage calls `v2d.docker.container.run_in_container(image=..., module=...)`. Instead of
`modules/v2d_docker`, the venv has `local/recon_patches/v2d_docker_native` installed under the
same name. It runs `python -m <module>` in this venv with host paths, maps env values that point
into container mounts back to host paths (FoundationPose `FOUNDATIONPOSE_WEIGHTS_DIR`), rewrites
container paths inside JSON inputs (SAM2 mask prompts mounted via `extra_volumes`), and applies
the env the Dockerfiles set. Native images: `v2d_hamer, v2d_wilor, v2d_foundation_pose,
v2d_sam2, v2d_grounding_dino, v2d_moge, v2d_anycalib`. Any other image (DROID-SLAM, GeoCalib,
SAM3D, gsplat, HaWoR, ...) raises a clear error unless `V2D_NATIVE_PYTHON_<IMAGE>` points at an
interpreter that has it.

### Patches (all outside the CHORD repo)

| Patch | Why |
|---|---|
| `recon_patches/v2d_docker_native/` | native replacement for `modules/v2d_docker` (above) |
| `recon_patches/groundingdino_torch28_scalar_type.patch` | `AT_DISPATCH_*(x.type(), ...)` no longer compiles with torch 2.8 → `x.scalar_type()`; applied to the clone in `third_party/GroundingDINO` |
| `recon_patches/foundationpose_mycuda/torch28_scalar_type.patch` | same fix for FoundationPose `bundlesdf/mycuda/common.cu`; applied to the copy in `build/mycuda_src` |
| `recon_patches/foundationpose_mycpp/CMakeLists.txt` | upstream asks for compiled Boost `system program_options`, which are not installed (no sudo); the code only uses header-only Boost (`assign`, `format`), so link `Boost::headers` |
| `recon_patches/trtexec_py.sh` / `trtexec_py.py` | pip TensorRT has no `trtexec`; FoundationPose builds engines by calling it. This stand-in supports exactly the flags `trt_engine.build_engine` passes and builds with the TensorRT Python API. `TRTEXEC_PATH` points at it (set in `recon_env.sh`) |
| `Pillow<12` (in `recon_reqs.txt`) | Pillow 13 rejects `Image.fromarray(uint16, mode="I;16")`, which `v2d_common.DepthImage` uses to write every depth PNG (MoGe stage crashed until pinned to 11.3.0) |
| `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` (in `recon_env.sh`) | torch>=2.6 defaults `torch.load(weights_only=True)`; HaMeR's Lightning ckpt (yacs config), WiLoR and ultralytics checkpoints need the old behaviour |

Compiled FoundationPose extensions were built out of tree and only the `.so` files were copied
into the repo tree, where `*.so` is git-ignored:
`modules/v2d_foundation_pose/lib/FoundationPose/mycpp/build/mycpp.cpython-311-x86_64-linux-gnu.so`,
`.../bundlesdf/mycuda/{common,gridencoder}.cpython-311-x86_64-linux-gnu.so`.

## Component status

Stages that `run_ego_reconstruction.py --hand_tracking hamer --object_mesh X --skip_object_scale_estimation [--undistort]` runs, in order:

| Stage (image) | Needed? | Installed / built | Weights | Gate | Verified |
|---|---|---|---|---|---|
| AnyCalib undistortion (`v2d_anycalib`) | only with `--undistort` (airplane.mp4 is fisheye, so yes) | yes | yes, 1.2 GB | none (Apache-2.0) | ran on 60 frames, offline (proxy blocked) |
| frames, stabilised intrinsics (host) | yes | yes | - | - | ran |
| MoGe-2 depth + intrinsics (`v2d_moge`) | yes | yes (MoGe @925b8ed) | yes, 1.3 GB | none (MIT) | 60 frames in 12.7 s |
| WiLoR detection + MANO (`v2d_wilor`) | yes (finds hands, left/right, SAM2 hand prompts) | yes (wilor-mini, ultralytics 8.1.34) | yes except MANO | **MANO** | YOLO hand detector ran on 6 airplane frames (L/R boxes 0.34-0.82); full model needs MANO; ckpt loads (641 M params) |
| Grounding DINO (`v2d_grounding_dino`) | yes (object box on the reference frame) | yes, CUDA op sm_120 (patched) | yes, 0.66 GB + BERT 0.42 GB | none (Apache-2.0) | "a toy airplane" 0.85, offline |
| SAM2.1 large (`v2d_sam2`) | yes (object + hand masks) | yes, `_C` sm_120 | yes, 0.9 GB `.pt` | none (Apache-2.0) | 60 frames at 34 fps, box and mask prompts |
| SAM3D mesh, FP scale estimation | no (skipped by `--object_mesh` + `--skip_object_scale_estimation`) | no | no | SAM3D is HF-gated | - |
| FoundationPose tracking + EKF (`v2d_foundation_pose`) | yes | yes: pytorch3d, nvdiffrast, mycpp, mycuda, warp, TensorRT 10.13 + `trtexec_py` | **no** | **NVIDIA Open Model License** | imports; nvdiffrast raster, pytorch3d ops, mycpp `cluster_poses`, warp depth filters ran on GPU; TensorRT engine build + `TensorRTModule` inference checked on a dummy ONNX with the same I/O names (max err 1.8e-6) |
| HaMeR regression, align, render (`v2d_hamer`) | yes | yes (hamer @3a01849, no ViTDet/ViTPose) | yes, 2.6 GB ckpt (+3.6 GB unused ViTPose from the same tarball) | **MANO** | real `hamer.ckpt` loaded (all backbone and head keys match); ViT-H backbone + MANO transformer head ran on a WiLoR right-hand crop from airplane frame 0 (153 ms, outputs hand_pose 15x3x3, betas, cam). Only the MANO layer (mesh vertices) is missing |
| result bundle, three.js export (host) | yes / optional | yes | uses MANO via hamer root | - | imports |
| kaolin | no: only `OctreeManager` (BundleSDF/NeRF path) imports it, lazily | not installed, no stub needed | - | - | - |
| DROID-SLAM, GeoCalib, gsplat refinement, HaWoR, DynHaMR | not for this run | no | no | - | - |

pyrender offscreen rendering works with EGL (`PYOPENGL_PLATFORM=egl`, PyOpenGL 3.1.0).

## Weights

`reconstruction/data/` is git-ignored; `reconstruction/data/weights/<name>` are symlinks so the
pipeline defaults (`--*_weights data/weights/<name>`) work unchanged.

| Name (`data/weights/...` link) | Location | Size | Source | License |
|---|---|---|---|---|
| `hamer` | `/data/models/hamer/_DATA/` (`hamer_ckpts/checkpoints/hamer.ckpt`, `data/mano_mean_params.npz`, `vitpose_ckpts/` unused) | 6.1 GB (ViTPose 3.6 GB can be deleted) | `v2d.hamer.lib.download_weights` (cs.utexas.edu tarball, through the proxy ~6 MB/s; direct is ~10 KB/s) | HaMeR MIT |
| `wilor` | `/data/models/wilor/pretrained_models/{detector.pt, wilor_final.ckpt, mano_mean_params.npz}` | 2.5 GB | HF `warmshao/WiLoR-mini` via hf-mirror, file by file (MANO_RIGHT.pkl deliberately skipped) | WiLoR upstream CC BY-NC-ND 4.0 |
| `sam2` | `/data/models/sam2/sam2.1-hiera-large/sam2.1_hiera_large.pt` (+ yaml; same dir already had the transformers copy) | 0.86 GB | HF `facebook/sam2.1-hiera-large` via hf-mirror | Apache-2.0 |
| `moge` | `/data/models/moge/moge-2-vitl-normal/model.pt` | 1.3 GB | HF `Ruicheng/moge-2-vitl-normal` via hf-mirror | MIT |
| `grounding_dino` | `/data/models/grounding_dino/swint_ogc/groundingdino_swint_ogc.pth` | 0.66 GB | `v2d.grounding_dino.lib.download_weights` (GitHub release, proxy) | Apache-2.0 |
| (BERT for GDINO) | `/data/models/grounding_dino/hf_home/hub/models--bert-base-uncased` | 0.42 GB | hf-mirror; the shim sets `HF_HOME` here + `HF_HUB_OFFLINE=1` for `v2d_grounding_dino` | Apache-2.0 |
| `anycalib` | `/data/models/anycalib/hub/anycalib/anycalib_gen.pt` (TORCH_HOME layout) | 1.2 GB | `v2d.anycalib.lib.download_weights` (GitHub release, proxy) | Apache-2.0 |
| `foundation_pose` | `/data/models/foundation_pose/nvidia_tensorrt/deployable_v1.0/{score_net,refiner_net}.onnx` | 132 MB (added 10-31 23:37; TensorRT engines not built yet) | see Blockers | NVIDIA Open Model License |
| MANO | `/data/models/mano/MANO_RIGHT.pkl` | 3.8 MB (added 10-31 23:23, linked by `link_mano.sh`) | see Blockers | MANO non-commercial |

Catalog entries were added/updated in `/data/catalog/datasets.md` (hamer, wilor, mano, foundation_pose, moge, anycalib; sam2, grounding_dino, chord updated).

Disk: weights 13 GB under `/data/models`; env 17 GB (`.venv-recon` 14 GB, `.cuda128` 1.5 GB, `third_party` 1.3 GB, `build`, `recon_test`).

## Blockers (user side) - both cleared on 10-31

Both files are now on disk: `/data/models/mano/MANO_RIGHT.pkl` (linked into the HaMeR and WiLoR folders by
`link_mano.sh`) and `/data/models/foundation_pose/nvidia_tensorrt/deployable_v1.0/`. The steps below are kept as the
record of how they were obtained. Next: the model-load checks and the 60-frame run under "After both blockers are cleared".


1. **MANO_RIGHT.pkl** (MANO license, non-commercial; only from the official site).
   Register at https://mano.is.tue.mpg.de/, download "Models & Code" (`mano_v1_2.zip`), then
   ```bash
   mkdir -p /data/models/mano && cp mano_v1_2/models/MANO_RIGHT.pkl /data/models/mano/
   /data/repos/chord/local/link_mano.sh
   ```
   `link_mano.sh` symlinks it to every place the code looks:
   `/data/models/hamer/_DATA/data/mano/MANO_RIGHT.pkl` (HaMeR, smplx),
   `/data/models/hamer/_DATA/data/models/MANO_RIGHT.pkl` (manotorch in v2d_hamer align/render, three.js export),
   `/data/models/wilor/pretrained_models/MANO_RIGHT.pkl` (wilor-mini),
   `/data/models/wilor/pretrained_models/models/MANO_RIGHT.pkl` (manotorch in v2d_wilor),
   `reconstruction/data/weights/hand/models/MANO_RIGHT.pkl` (DynHaMR / hand_alignment layout, not used by the hamer path).
   Without it these stages fail: WiLoR per-frame detection + MANO, WiLoR prompt-mask rendering,
   HaMeR regression, hand alignment and all hand renders. Only the right-hand model is used.

   Warning: `wilor-mini` fetches `MANO_RIGHT.pkl` from its Hugging Face repo when the file is
   missing, and CHORD's `v2d.wilor.lib.download_weights` triggers exactly that. Do not run that
   downloader. The WiLoR weights here were fetched file by file without it, and the native shim
   sets `HF_HUB_OFFLINE=1` for the `v2d_wilor` image so a pipeline run fails instead of
   downloading. (`wilor_final.ckpt` itself already carries MANO-derived buffers in its state dict;
   that is how upstream ships it.)

2. **FoundationPose weights** (NVIDIA Open Model License, the NGC "TAO FoundationPose"
   `nvidia/tao/foundationpose:deployable_v1.0`; 2 files, `score_net.onnx` 63.9 MB +
   `refiner_net.onnx` 68.2 MB = 132 MB). Not downloaded: the download needs the user's own
   acceptance of the license (an agent-relayed OK was refused by the permission system). After
   reading the terms, run:
   ```bash
   source /data/repos/chord/local/recon_env.sh && cd /data/repos/chord/reconstruction
   env -u HTTPS_PROXY -u HTTP_PROXY NO_PROXY='*' no_proxy='*' \
     python -m v2d.foundation_pose.lib.download_weights \
       --output_dir /data/models/foundation_pose --accept_nvidia_model_eula
   ```
   NGC must be reached directly: through the proxy the signed `xfiles.ngc.nvidia.com` redirect
   returns 403; directly it is 200. Files land in
   `/data/models/foundation_pose/nvidia_tensorrt/deployable_v1.0/` (sha256 checked by the
   downloader). The first pipeline run builds `*_trt_10_13_3_9_sm_120_fp32.engine` next to them
   via `recon_patches/trtexec_py.sh`. To build them ahead of time (GPU, a few minutes):
   `python -c "from v2d.foundation_pose.lib.trt_engine import ensure_engines; print(ensure_engines('/data/models/foundation_pose'))"`.
   Alternative backend `--backend nvlabs_pytorch` uses the original NVlabs checkpoints from Google
   Drive (gdown; NVIDIA Source Code License, non-commercial); also not downloaded.

## Smoke tests

### Already run (2026-10-31, no MANO / no FoundationPose weights needed)

```bash
source /data/repos/chord/local/recon_env.sh && cd /data/repos/chord/reconstruction
python /data/repos/chord/local/recon_smoke_partial.py \
  --video /data/repos/chord/local/recon_test/airplane_60f.mp4 \
  --out /data/repos/chord/local/recon_test/partial --prompt "A toy airplane"
```
`airplane_60f.mp4` = first 60 frames of `assets/airplane.mp4` (1280x800). Runs AnyCalib
undistortion → frame extraction → MoGe → stabilised intrinsics → Grounding DINO on frame 0 →
SAM2 (box prompt, then a mask prompt through the `extra_volumes` path) with the pipeline's own
wrappers. Result: 29 s wall; AnyCalib fisheye `opencv_fisheye`, undistorted fx 449; GDINO
"a toy airplane" 0.81; SAM2 60/60 frames, mask-prompt run IoU 0.998 with the box run; MoGe
depth median 0.85 m. Log `logs/recon_smoke_partial.log`, overlay `recon_test/partial/overlay.png`.

### After both blockers are cleared

1. Model-load checks (seconds, GPU):
   ```bash
   source /data/repos/chord/local/recon_env.sh && cd /data/repos/chord/reconstruction
   python -c "from v2d.hamer.lib._hamer import get_model; m,c=get_model('/data/models/hamer'); print('hamer OK')"
   HF_HUB_OFFLINE=1 python -c "from v2d.wilor.lib._wilor import get_pipeline; get_pipeline('/data/models/wilor'); print('wilor OK')"
   python -c "from v2d.foundation_pose.lib.trt_engine import ensure_engines; print(ensure_engines('/data/models/foundation_pose'))"
   ```
2. Full pipeline on the 60-frame clip (fixed camera, provided mesh, HaMeR hands):
   ```bash
   systemd-run --user --unit=recon-smoke60 --collect -p WorkingDirectory=/data/repos/chord/reconstruction \
     bash -c 'source /data/repos/chord/local/recon_env.sh; python modules/v2d_pipelines/run_ego_reconstruction.py \
       --video /data/repos/chord/local/recon_test/airplane_60f.mp4 --object_prompt "A toy airplane" \
       --object_mesh assets/textured_mesh.obj --skip_object_scale_estimation \
       --output_dir /data/repos/chord/local/recon_test/full60 --reference_frame 0 --undistort \
       --hand_tracking hamer --export_threejs_result > /data/repos/chord/local/logs/recon_smoke60.log 2>&1'
   ```
   Then the whole clip: same command with `--video assets/airplane.mp4 --output_dir data/outputs/airplane_hamer_mesh_native`.
   Output bundle: `<output_dir>/result/` (`result.npz`, `mesh.obj`), viewer
   `<output_dir>/result/threejs_scene/index.html`, plus overlay videos in `<output_dir>`.
   The pipeline caches stages, so a rerun resumes. Drop `--undistort` for pinhole footage.
   Do not add `--run_droid_slam`, `--run_gravity_alignment` or `--run_gsplat_refinement`:
   DROID-SLAM, GeoCalib and gsplat are not installed natively (the shim stops with a clear error).

## Rebuild from scratch (reference)

```bash
L=/data/repos/chord/local; M=/data/repos/chord/reconstruction/modules
# 1. toolkit + venv + torch (PyPI default torch 2.8 wheels are cu128)
/home/yan/miniconda3/bin/conda create -y -p $L/.cuda128 --override-channels \
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge \
  cuda-nvcc=12.8 cuda-cudart-dev=12.8 cuda-cccl=12.8 cuda-version=12.8 cuda-profiler-api=12.8 cuda-nvrtc-dev=12.8
ln -sfn $L/.cuda128/targets/x86_64-linux/lib $L/.cuda128/lib64
uv venv $L/.venv-recon --python 3.11
source $L/recon_env.sh
uv pip install --index-url https://mirrors.bfsu.edu.cn/pypi/web/simple -r $L/recon_reqs.txt
uv pip install --index-url https://mirrors.bfsu.edu.cn/pypi/web/simple onnx   # only for the TensorRT dummy-model test
# 2. CHORD packages (editable, no deps) + native docker shim
uv pip install --no-deps -e $M/v2d_common -e $M/v2d_depth -e $M/v2d_mv -e $M/v2d_mesh/lib \
  -e $M/v2d_{hamer,wilor,foundation_pose,sam2,grounding_dino,moge,anycalib}/lib \
  -e $M/v2d_{anycalib,geocalib,droid_slam,ego_hand_reconstruction,foundation_pose,grounding_dino,gsplat_refinement,hamer,hand_alignment,hawor,mediapipe,moge,sam2,sam3d,wilor}/docker \
  -e $M/v2d_pipelines -e $L/recon_patches/v2d_docker_native
# 3. git deps (clones in third_party; GitHub via proxy)
cd $L/third_party
uv pip install --no-build-isolation --no-deps ./chumpy ./manotorch ./WiLoR-mini -e ./hamer ./MoGe ./AnyCalib \
  "utils3d @ git+https://github.com/EasternJournalist/utils3d.git@3fab839f0be9931dac7c8488eb0e1600c236e183" \
  "pipeline @ git+https://github.com/EasternJournalist/pipeline.git@866f059d2a05cde05e4a52211ec5051fd5f276d6"
# 4. CUDA extensions for sm_120 (each takes 1-2 min with MAX_JOBS=12-16)
git -C GroundingDINO apply $L/recon_patches/groundingdino_torch28_scalar_type.patch
FORCE_CUDA=1 MAX_JOBS=16 pip install --no-deps --no-build-isolation ./pytorch3d ./nvdiffrast ./GroundingDINO
SAM2_BUILD_CUDA=1 SAM2_BUILD_ALLOW_ERRORS=0 pip install --no-deps --no-build-isolation ./sam2
# 5. FoundationPose mycpp / mycuda (out of tree, copy .so back)
FP=$M/v2d_foundation_pose/lib/FoundationPose
cp -r $FP/mycpp $L/build/mycpp_src && cp $L/recon_patches/foundationpose_mycpp/CMakeLists.txt $L/build/mycpp_src/
cmake -S $L/build/mycpp_src -B $L/build/mycpp_src/build -DPYTHON_EXECUTABLE=$(which python) -Dpybind11_DIR=$(python -m pybind11 --cmakedir)
cmake --build $L/build/mycpp_src/build -j 16 && mkdir -p $FP/mycpp/build && cp $L/build/mycpp_src/build/mycpp*.so $FP/mycpp/build/
cp -r $FP/bundlesdf/mycuda $L/build/mycuda_src
sed -i 's/AT_DISPATCH_\([A-Z_0-9]*\)(\([A-Za-z_]*\)\.type()/AT_DISPATCH_\1(\2.scalar_type()/g' $L/build/mycuda_src/common.cu
(cd $L/build/mycuda_src && python setup.py build_ext --inplace) && cp $L/build/mycuda_src/*.so $FP/bundlesdf/mycuda/
```
(`foundationpose_mycuda/torch28_scalar_type.patch` records the same change as a `diff -ru`.)
