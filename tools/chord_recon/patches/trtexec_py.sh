#!/bin/bash
# TRTEXEC_PATH target for FoundationPose (see trtexec_py.py). Runs the trtexec_py.py next to this file with the recon venv
# (RECON_VENV, default the native env in /data/repos/chord/local).
exec ${RECON_VENV:-/data/repos/chord/local/.venv-recon}/bin/python "$(dirname "$(readlink -f "$0")")/trtexec_py.py" "$@"
