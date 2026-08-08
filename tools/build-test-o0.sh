#!/usr/bin/env bash
# build-test-o0.sh -- compile test_cache with the suite's exact -O0 flags.
set -u
cd /Users/ruihe/disk-qwen35bA3B
mkdir -p build
cc -std=c99 -O0 -Wall -Wextra -pthread -Iinclude -Isrc \
    -o build/test_cache tests/test_cache.c \
    src/cfg.c src/st.c src/trunk.c src/cache.c src/router.c src/mem.c \
    src/kernels.c src/moe.c src/simd.c src/attn.c src/head.c \
    src/tokenizer.c src/gpu_stub.c -lm
echo "build rc=$?"
./build/test_cache
echo "run rc=$?"
