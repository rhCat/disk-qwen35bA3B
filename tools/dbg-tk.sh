#!/usr/bin/env bash
# dbg-tk.sh -- run the suite's test_kernels binary directly
cd /Users/ruihe/disk-qwen35bA3B
if [ ! -x build/test_kernels ]; then
  mkdir -p build
  cc $CFLAGS -o build/test_kernels tests/test_kernels.c \
     src/cfg.c src/st.c src/trunk.c src/cache.c src/router.c src/mem.c \
     src/kernels.c src/moe.c src/simd.c src/attn.c src/attn_qwen.c \
     src/head.c src/tokenizer.c -lm
fi
./build/test_kernels 2>&1 | tail -6
echo "exit: $?"
