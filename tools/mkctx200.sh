#!/usr/bin/env bash
# mkctx200.sh -- build the 200-token fixture.
set -u
cd /Users/ruihe/disk-qwen35bA3B
unset PYTHONPATH
export PATH="/Volumes/prod/miniforge3/envs/ca_lpp/bin:$PATH"
PY="$PWD/.venv/bin/python3"
"$PY" tools/mkctx.py 200 /tmp/q35-200-ids.txt
