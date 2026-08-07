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
GPU_SRC  = src/gpu_metal.mm
GPU_LIBS = -framework Metal -framework Foundation
OBJCXX  ?= clang++
OBJCXXFLAGS ?= -std=gnu++17 -O2 -Wall -Wextra
else
CFLAGS  ?= -std=c99 -O2 -Wall -Wextra -pthread
GPU_SRC  = src/gpu_stub.c
GPU_LIBS =
endif
INC      = -Iinclude -Isrc
SRC      = src/cfg.c src/st.c src/trunk.c src/cache.c src/router.c src/mem.c src/kernels.c src/moe.c

HDR = include/ds4f/ds4f.h include/ds4f/kernels.h include/ds4f/moe.h \
      include/ds4f/simd.h include/ds4f/attn.h include/ds4f/head.h \
      include/ds4f/tokenizer.h src/json.h
SRC = src/cfg.c src/st.c src/trunk.c src/cache.c src/router.c src/mem.c \
      src/kernels.c src/moe.c src/simd.c src/attn.c src/attn_qwen.c \
      src/head.c src/tokenizer.c $(GPU_SRC)

all: ds4f pack-trunk make-fixture bench-kernels

ds4f: src/main.c $(SRC) $(HDR) $(GPU_SRC:.mm=.o)
	$(CC) $(CFLAGS) $(INC) -DDS4F_GIT=\"$(shell git rev-parse --short HEAD 2>/dev/null)\" -o $@ src/main.c $(filter-out %.mm,$(SRC)) $(GPU_SRC:.mm=.o) -lm $(GPU_LIBS)

src/gpu_metal.o: src/gpu_metal.mm include/ds4f/gpu.h
	$(OBJCXX) $(OBJCXXFLAGS) $(INC) -c src/gpu_metal.mm -o src/gpu_metal.o

pack-trunk: tools/pack-trunk.c $(HDR)
	$(CC) $(CFLAGS) $(INC) -o $@ tools/pack-trunk.c -lm

make-fixture: tools/make-fixture.c $(HDR)
	$(CC) $(CFLAGS) $(INC) -o $@ tools/make-fixture.c -lm

bench-kernels: tools/bench-kernels.c $(SRC) $(HDR) $(GPU_SRC:.mm=.o)
	$(CC) $(CFLAGS) $(INC) -o $@ tools/bench-kernels.c $(filter-out %.mm,$(SRC)) $(GPU_SRC:.mm=.o) -lm $(GPU_LIBS)

# GPU expert offload microbenchmark (Darwin only): one expert layer,
# CPU 8-thread vs the existing per-matvec Metal API. Links only the
# kernels it needs (kernels.c + simd.c + the Metal shim) -- head.c and
# the rest of the engine are C99 and don't compile as C++.
BENCH_GPU_SRC = src/kernels.c src/simd.c $(GPU_SRC:.mm=.o)
bench-gpu-experts: tools/bench-gpu-experts.mm $(BENCH_GPU_SRC) $(HDR)
	$(CXX) $(CXXFLAGS) $(INC) -o $@ tools/bench-gpu-experts.mm $(BENCH_GPU_SRC) -lm $(GPU_LIBS)

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
