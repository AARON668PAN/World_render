#!/bin/bash
# The student's weights (627 MB) are in this repository as 7 pieces under 100 MB (GitHub's limit for one file):
# student/model.safetensors.part00 .. part06. This joins them into student/model.safetensors and checks the SHA-256.
# usage: ./assemble_weights.sh
set -euo pipefail
cd "$(dirname "$0")/student"
[ -f model.safetensors ] || cat model.safetensors.part?? > model.safetensors
sha256sum -c model.sha256
echo "WEIGHTS_OK"
