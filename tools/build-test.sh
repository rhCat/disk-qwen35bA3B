#!/usr/bin/env bash
# build-test.sh -- compile ONE unit test (arg = test name, e.g. test_cache).
set -u
cd /Users/ruihe/disk-qwen35bA3B
CC="${CC:-cc}"
CFLAGS="-std=c99 -O2 -Wall -Wextra -pthread -Iinclude"
NAME="${1:-test_cache}"
mkdir -p build
$CC $CFLAGS -o "build/$NAME" "tests/$NAME.c" \
    src/cfg.c src/st.c src/trunk.c src/cache.c src/router.c src/mem.c \
    src/kernels.c src/moe.c src/simd.c src/attn.c src/head.c \
    src/gpu_stub.c -lm
echo "build rc=$?"
./build/$NAME
echo "run rc=$?"
