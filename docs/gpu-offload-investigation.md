# GPU expert offload — investigation (feat/gpu-expert-offload)

Status: **investigation only — no working offload yet.** Branch off
`02a1341` (the validated row-split main).

## Why the head-only Metal path measured as a non-win

`src/gpu_metal.mm` offloads ONE matvec: the 248320-row output head.
Measured parity (~0.26 vs 0.24 s/token) because the head is ~12% of the
per-token work. The expert matvecs — the ~88% — stayed on CPU.

## The expert offload problem, from the code

The Metal kernel is a single-shot `ds4f_gpu_mlx4_matvec` with **cached
weight buffers**. The expert path is fundamentally different:

1. **Shape volatility.** Experts are 512×2048 / 2048×512 per layer, but
   the *routed* set changes every token (top-8 of 256). The Metal code
   rebuilds weight buffers when `R/C/pointers` change — with experts,
   the pointer set changes EVERY token (8 experts × 3 tensors × 40
   layers = 960 buffer (re)creations/token). The cache-hit design that
   makes the head cheap is inapplicable.

2. **Buffer churn.** Each expert tensor is ~1-2 MB. 960 uploads/token at
   M4 bandwidth (~200 GB/s) = ~10 ms/token of pure transfer — HALF the
   current 17 ms/token budget. Uploading 4-bit weights that the GPU then
   dequantizes loses to the CPU's already-resident cache.

3. **Dequant is the same cost on GPU.** The kernel does `q*s+b` per
   element (line 40-47) — identical math to the NEON path. The GPU's
   win would be *width* (many rows in parallel), but expert rows are
   only 512-2048 wide; the CPU already does 8 rows in parallel per
   thread × 8 threads.

4. **Synchronization.** `waitUntilCompleted` per matvec serializes; a
   batched kernel over all 8 experts per layer would help, but the
   routing happens per-layer in C, so batching crosses the engine's
   control flow.

## What WOULD make GPU offload win (the real design)

| design | transfer/token | why it could win |
|---|---|---|
| **Whole-pool resident on GPU** | 0 (weights pre-uploaded once) | the 16.88 GB pool at ~200 GB/s = 84 ms once; then per-token only x/y (KB) |
| **Batched per-layer kernel** | 8 experts × 3 = 24 tensors, one dispatch | one launch, no per-matvec sync; GPU width on 24 parallel matvecs |
| **Persistent expert buffers** | rebuild only when a new expert first enters cache | after warmup, ~0 churn |

The whole-pool design is the only one that clears the transfer bound —
but it needs **16.88 GB of GPU memory**. M4 Pro (this Mac) has a unified
memory pool; the engine's 8 GB target means the pool would be
shared-memory resident anyway (MTLResourceStorageModeShared), so
"upload" is mostly a page-map, not a copy. That's the crux: on unified
memory, **the transfer argument partly evaporates** — the real question
is whether the GPU's SIMD width beats the CPU's 8-thread NEON on
512-wide rows.

## Verdict

**GPU expert offload is possible but not a clear win on this machine.**
The honest next step is a **microbenchmark**, not a full implementation:
time one expert layer's 8×3 matvecs on Metal (whole-pool resident,
batched single dispatch) vs the current CPU path (8 threads, fused
NEON). If the GPU doesn't beat ~0.3 ms/layer, the offload is
theoretically sound but practically pointless here — the answer would
then be "keep CPU, the pool stays on disk" (the original design intent).

## Next steps on this branch

1. `tools/bench-gpu-experts.mm` — microbenchmark (one layer, 24 tensors,
   batched Metal vs CPU)
2. If GPU wins: `ds4f_gpu_mlx4_batch()` API + integration in
   `ds4f_moe_step`
3. If not: document the measurement, close the branch with the verdict

## Measurements so far (from main)

- Row-split main: **0.19 s/token at 2K context** (2017-token QA bench)
- 5.9 tok/s short-context; RSS 4.9-5.8 GB
- Expert path = ~960 matvecs/token, the 0.17-0.19 s floor
