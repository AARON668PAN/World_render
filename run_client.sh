#!/bin/bash
# The robot client. usage: ROBOT_IP=192.168.123.164 ./run_client.sh [--hand-check | --dry-run | --no-student] [more g1_client.py options]
# env CAMERA (default official:$ROBOT_IP:5555; needs GR00T_WBC=<checkout of NVIDIA's GR00T-WholeBodyControl>; or cv:0)
set -euo pipefail
cd "$(dirname "$0")"
ROBOT_IP=${ROBOT_IP:?set ROBOT_IP=<the robot computer that runs deploy.sh>}
exec .venv/bin/python client/g1_client.py --robot-host "$ROBOT_IP" --camera "${CAMERA:-official:$ROBOT_IP:5555}" "$@"
