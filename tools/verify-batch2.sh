#!/usr/bin/env bash
# verify-batch2.sh -- compare batch vs scalar fallback (same acc order).
set -u
cd /Users/ruihe/disk-qwen35bA3B
mkdir -p build
cc -std=c99 -O2 -Wall -Wextra -pthread -Iinclude -Isrc \
    -o build/verify-batch2 tests/verify_batch2.c \
    src/kernels.c src/simd.c src/gpu_stub.c -lm
echo "build rc=$?"
./build/verify-batch2
echo "run rc=$?"
