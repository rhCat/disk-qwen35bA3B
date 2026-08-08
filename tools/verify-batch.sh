#!/usr/bin/env bash
# verify-batch.sh -- build + run a bit-fidelity check for the batched matvec.
set -u
cd /Users/ruihe/disk-qwen35bA3B
mkdir -p build
cc -std=c99 -O2 -Wall -Wextra -pthread -Iinclude -Isrc \
    -o build/verify-batch tests/verify_batch.c \
    src/kernels.c src/simd.c src/gpu_stub.c -lm
echo "build rc=$?"
./build/verify-batch
echo "run rc=$?"
