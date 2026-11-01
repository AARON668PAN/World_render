# Source this before using or building the native CHORD reconstruction env.
#   source /data/repos/chord/local/recon_env.sh
# Env: /data/repos/chord/local/.venv-recon (uv, Python 3.11, torch 2.8.0+cu128)
# Toolkit for building extensions: /data/repos/chord/local/.cuda128 (conda-forge CUDA 12.8 nvcc)

RECON_LOCAL=/data/repos/chord/local
export VIRTUAL_ENV=$RECON_LOCAL/.venv-recon
export PATH=$VIRTUAL_ENV/bin:$RECON_LOCAL/.cuda128/bin:$PATH
export CUDA_HOME=$RECON_LOCAL/.cuda128
export TORCH_CUDA_ARCH_LIST="12.0"

# Headers for g++-compiled parts of torch extensions: the conda toolkit keeps
# them under targets/, and cuBLAS/cuSPARSE/cuSOLVER headers come from torch's
# pip nvidia-* wheels.
_NV=$VIRTUAL_ENV/lib/python3.11/site-packages/nvidia
export CPATH=$CUDA_HOME/targets/x86_64-linux/include:$_NV/cublas/include:$_NV/cusparse/include:$_NV/cusolver/include:$_NV/curand/include:$_NV/cufft/include:$_NV/cuda_runtime/include:$_NV/nvtx/include${CPATH:+:$CPATH}
unset _NV

# Runtime defaults that the Docker images set.
export PYOPENGL_PLATFORM=egl
export OPENCV_IO_ENABLE_OPENEXR=1
# torch>=2.6 defaults torch.load(weights_only=True); HaMeR (Lightning ckpt with
# yacs config), WiLoR and ultralytics checkpoints need the old behaviour.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
# FoundationPose TensorRT backend: trtexec replacement built on the TensorRT
# Python API (pip tensorrt has no trtexec binary).
export TRTEXEC_PATH=$RECON_LOCAL/recon_patches/trtexec_py.sh

# Network rules on this machine. systemd-run units do not inherit the login shell's proxy, so set it
# here (only the three.js exporter fetches from a CDN at run time; NGC must bypass it, see RECON_SETUP.md).
export HTTPS_PROXY=${HTTPS_PROXY:-http://127.0.0.1:10900} HTTP_PROXY=${HTTP_PROXY:-http://127.0.0.1:10900}
export NO_PROXY="localhost,127.0.0.1,::1,.local,mirrors.bfsu.edu.cn,hf-mirror.com"
export HF_ENDPOINT=https://hf-mirror.com
