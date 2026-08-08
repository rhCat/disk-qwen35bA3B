#!/usr/bin/env bash
# rss-gpu.sh -- measure process RSS for CPU-only vs GPU-batched.
set -u
BIN=/Users/ruihe/disk-qwen35bA3B/bench-gpu-experts
RUN=/Users/ruihe/disk-qwen35bA3B/tools/run-bench-gpu3.sh
echo "== CPU-only (8 threads) =="
/usr/bin/time -l bash "$RUN" --cpu-only 8 2>&1 | grep -E 'maximum resident|CPU' | head -3
echo "== GPU batched only =="
/usr/bin/time -l bash "$RUN" --gpu-only 8 2>&1 | grep -E 'maximum resident|batched' | head -3
echo "== GPU Metal buffers =="
bash "$RUN" --mem 2>&1 | grep -E 'buffers' | head -1
