#!/bin/bash
# Link the user-provided MANO right-hand model into every place the CHORD
# reconstruction modules look for it. Run once after placing the file:
#   /data/models/mano/MANO_RIGHT.pkl   (from https://mano.is.tue.mpg.de/, "Models & Code")
# Only the right-hand model is needed (left hands are mirrored).
set -e
SRC=/data/models/mano/MANO_RIGHT.pkl
if [ ! -s "$SRC" ]; then
  echo "missing $SRC (download mano_v1_2.zip after registering, unzip, copy mano_v1_2/models/MANO_RIGHT.pkl here)" >&2
  exit 1
fi
link() { mkdir -p "$(dirname "$1")"; ln -sfn "$SRC" "$1"; echo "  $1 -> $SRC"; }
# HaMeR model (smplx MANO, cfg MANO.MODEL_PATH=./_DATA/data/mano)
link /data/models/hamer/_DATA/data/mano/MANO_RIGHT.pkl
# manotorch ManoLayer(mano_assets_root=<hamer>/_DATA/data) used by v2d_hamer align/render + three.js export
link /data/models/hamer/_DATA/data/models/MANO_RIGHT.pkl
# wilor-mini pipeline (<wilor_weights>/pretrained_models/MANO_RIGHT.pkl)
link /data/models/wilor/pretrained_models/MANO_RIGHT.pkl
# manotorch ManoLayer(mano_assets_root=<wilor>/pretrained_models) used by v2d_wilor renders/matching
link /data/models/wilor/pretrained_models/models/MANO_RIGHT.pkl
# DynHaMR / v2d_hand_alignment layout (data/weights/hand/models/MANO_RIGHT.pkl), not needed for --hand_tracking hamer
link /data/repos/chord/reconstruction/data/weights/hand/models/MANO_RIGHT.pkl
python3 - <<'EOF'
import hashlib
h = hashlib.md5(open("/data/models/mano/MANO_RIGHT.pkl", "rb").read()).hexdigest()
print("MANO_RIGHT.pkl md5:", h)
EOF
