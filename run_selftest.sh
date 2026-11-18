#!/bin/bash
# No robot needed. 1) the wire format's own round trip, 2) the student loads, reproduces the training machine's numbers and is
# fast enough, 3) the whole client against a fake robot: nothing is sent without an instruction, 50 Hz while acting.
set -euo pipefail
cd "$(dirname "$0")"; PY=.venv/bin/python
./assemble_weights.sh
$PY client/selftest.py student
$PY client/protocol_test.py
echo "ALL_SELFTESTS_DONE"
