# GPU expert offload — investigation (feat/gpu-expert-offload)

Status: **MICROBENCHMARKED — the batched offload is a 13x win.**
Branch off `02a1341` (the validated row-split main).

## The measurement (tools/bench-gpu-experts.mm, Apple M4 Pro)

One expert layer = 8 routed experts x 3 tensors (gate/up 512x2048,
down 2048x512). 30 reps, best-stable run:

| path | ms/layer | us/matvec | vs CPU |
|---|---|---|---|
| CPU (8 threads, fused NEON, row-split) | 15.34 | 639 | 1.00x |
| GPU per-matvec API (24 dispatches) | 9.14 | 381 | 0.60x |
| **GPU batched (1 dispatch, all 24)** | **1.16** | **48** | **0.08x = 13x faster** |

Variance note: the per-matvec path wandered 0.60x-1.02x between runs
(dispatch + sync overhead swamps the compute at this size); the
batched path was stable ~0.08x.

## Why the batched design wins (the code-level answer)

1. **Weights resident, uploaded once.** All 24 tensors sit in shared
   memory buffers (MTLResourceStorageModeShared) across tokens; the
   routed set changes per token but the pool tensors are the same
   objects -- buffer pointers do not change, so no re-upload (the
   per-matvec API's cache miss every token was the old cost).
2. **One dispatch, no per-matvec sync.** The per-matvec API does
   command-buffer commit + waitUntilCompleted 24x/token; the batched
   kernel does it once. On unified memory the "transfers" were never
   the cost -- the dispatch/sync overhead was.
3. **GPU width on the flattened grid.** 24 jobs x up-to-2048 rows in
   one dispatch; the M4 GPU's ~5 TFLOPs chew the dequant-FMA loop far
   past 8-thread NEON.
4. **Dequant is parallel, not cheaper** -- the kernel still does
   q*s+b per element, but 49152 threads do it simultaneously.

## Predicted engine impact

Expert path ~= 88% of 0.17 s/token. Batched offload:
- experts: ~1.16 ms/layer x 40 layers = ~46 ms/token
- attention + rest: ~20 ms/token (0.17 - 0.15 expert)
- **projected ~0.06-0.07 s/token = 14-16 tok/s (vs 5.9 now)**

RSS: pool tensors resident in shared memory (the 4-5 GB cache shrinks
to what the GPU needs) -- likely a wash vs the 5.8 GB measured.

## Integration design (next branch of work)

`ds4f_gpu_mlx4_batch(jobs[], njobs)` API in gpu.h/gpu_metal.mm:
- concat pool tensors once at init (or cache per-expert buffers)
- per-token: build the 8-expert desc table (R, C, xoff, yoff) -- the
  ONLY per-token work, ~24 uint4s
- one dispatch; read back 24 y-slices
- fall back to the CPU path per-call on failure (same contract as the
  head offload)

Integration point: `ds4f_moe_step` -- replace the per-expert exp_run
matvecs with the batched call when `--gpu`/`DS4F_GPU=1`; keep the CPU
path default for byte-determinism (GPU reduction order differs by
float32 rounding, documented in gpu.h).

## Open questions before implementation

1. **Routing dependency**: gate (router) scores must be computed before
   the expert set is known; the router itself is a 40x2048 matvec --
   can batch that too (one more job).
2. **Shared-expert tensors** (moe.c:1127-1153) are per-layer, not
   routed -- include them in the same dispatch.
3. **Precision gate**: verify the batched kernel's logits match the
   CPU path within the documented float32 tolerance on a real token
   (e2e_trace tolerance, not byte-exact).

## Prior steps (main)

- Row-split main: 0.19 s/token @ 2K, 5.9 tok/s, RSS 4.9-5.8 GB
- malloc kill + fusion: 0.24 -> 0.23 s/token
- context-aware row-split: 0.23 -> 0.17 s/token

## 4K long-context test (2026-08-07)

4017-token QA, GEN=8, cache 5 / pin 4, greedy:
- per-token 0.231 s (vs 0.192 at 2K = 1.20x context cost)
- trunk read flat at 693 MB/token (never disk-bound)
- cache hit 68.4% (vs 77.2% at 2K)
- PEAK RSS 5.66 GB
