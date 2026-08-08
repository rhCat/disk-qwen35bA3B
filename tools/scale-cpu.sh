#!/usr/bin/env bash
# scale-cpu.sh -- CPU thread-count sweep vs the GPU batched baseline.
set -u
RUN=/Users/ruihe/disk-qwen35bA3B/tools/run-bench-gpu3.sh
for T in 1 2 4 8 16; do
  out=$(bash "$RUN" --cpu-only --threads "$T" 12 2>&1 | grep 'CPU' | head -1)
  echo "threads=$T : $out"
done
echo "--- GPU batched (thread-independent) ---"
bash "$RUN" --gpu-only 12 2>&1 | grep 'batched' | head -1
