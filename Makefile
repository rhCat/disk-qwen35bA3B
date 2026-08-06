CC      ?= cc
# macOS default -O2: the old -O0-on-Darwin rule existed for a clang
# codegen divergence in the DS-V4 fixture era, but the current gate
# (including e2e_text's byte-identical determinism check) passes at
# -O2, and the real Qwen3.5 model measured 9.14 s/token at -O0 vs
# 0.26 s/token at -O2 (35x). If the divergence ever reappears, the
# -O0 override is one flag away:
#   make CFLAGS="-std=c99 -O0 -Wall -Wextra -pthread"
ifeq ($(shell uname),Darwin)
CFLAGS  ?= -std=c99 -O2 -Wall -Wextra -pthread
else
CFLAGS  ?= -std=c99 -O2 -Wall -Wextra -pthread
endif
INC      = -Iinclude -Isrc
SRC      = src/cfg.c src/st.c src/trunk.c src/cache.c src/router.c src/mem.c src/kernels.c src/moe.c

HDR = include/ds4f/ds4f.h include/ds4f/kernels.h include/ds4f/moe.h \
      include/ds4f/simd.h include/ds4f/attn.h include/ds4f/head.h \
      include/ds4f/tokenizer.h src/json.h
SRC = src/cfg.c src/st.c src/trunk.c src/cache.c src/router.c src/mem.c \
      src/kernels.c src/moe.c src/simd.c src/attn.c src/attn_qwen.c \
      src/head.c src/tokenizer.c

all: ds4f pack-trunk make-fixture bench-kernels

ds4f: src/main.c $(SRC) $(HDR)
	$(CC) $(CFLAGS) $(INC) -DDS4F_GIT=\"$(shell git rev-parse --short HEAD 2>/dev/null)\" -o $@ src/main.c $(SRC) -lm

pack-trunk: tools/pack-trunk.c $(HDR)
	$(CC) $(CFLAGS) $(INC) -o $@ tools/pack-trunk.c -lm

make-fixture: tools/make-fixture.c $(HDR)
	$(CC) $(CFLAGS) $(INC) -o $@ tools/make-fixture.c -lm

bench-kernels: tools/bench-kernels.c $(SRC) $(HDR)
	$(CC) $(CFLAGS) $(INC) -o $@ tools/bench-kernels.c $(SRC) -lm

test: all
	./tests/run_tests.sh
	bash tools/test-memlimit.sh

# the disk/cpu loading experiment (A3B-shaped fixture, footprint report)
FIXDIR ?= /tmp/fix35
experiment: all
	bash tools/run-experiment.sh $(FIXDIR)

clean:
	rm -rf build ds4f pack-trunk make-fixture

.PHONY: all test clean
