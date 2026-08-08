#!/usr/bin/env bash
# dbg-batch1.sh -- single-job batched vs CPU comparison: first row of job 0.
set -u
B=/Users/ruihe/disk-qwen35bA3B/bench-gpu-experts
"$B" --verify 1 2>&1 | grep -E 'verify' | head -2
