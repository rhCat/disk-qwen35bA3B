#!/usr/bin/env bash
# run-bench-gpu3.sh -- run the extended GPU bench with mode flags.
set -u
B=/Users/ruihe/disk-qwen35bA3B/bench-gpu-experts
"$B" "$@" 2>&1 | grep -vE '^gpu: Metal ready' | tail -8
