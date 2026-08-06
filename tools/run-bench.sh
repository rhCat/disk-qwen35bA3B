#!/usr/bin/env bash
# run-bench.sh -- bench-kernels, guarded
cd /Users/ruihe/disk-qwen35bA3B
./bench-kernels 50 2>&1 | grep -E 'mlx4|bf16|matvec|decode' | head -6
