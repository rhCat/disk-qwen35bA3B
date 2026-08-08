#!/usr/bin/env bash
# mkctx-run.sh -- build the 500 and 1500 token fixtures.
set -u
cd /Users/ruihe/disk-qwen35bA3B
unset PYTHONPATH
export PATH="/Volumes/prod/miniforge3/envs/ca_lpp/bin:$PATH"
PY="$PWD/.venv/bin/python3"
"$PY" tools/mkctx.py 3000 /tmp/q35-3000-ids.txt
"$PY" tools/mkctx.py 
