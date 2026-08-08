#!/usr/bin/env bash
# run-bench-gpu.sh -- run the GPU expert microbenchmark (binary via var).
set -u
B=/Users/ruihe/disk-qwen35bA3B/bench-gpu-experts
"$B" "${1:-30}" 2>&1 | tail -7
