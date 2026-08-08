# GPU expert offload — FINAL VERDICT (feat/gpu-expert-offload)

Status: **IMPLEMENTED, TESTED, and REJECTED — the offload is 3x
slower than the CPU path on this machine.** The microbenchmark's 13x
did not survive integration.

## The measured truth (Apple M4 Pro, GEN=80, 5-token prompt)

| path | s/token | tok/s | PEAK RSS |
|---|---|---|---|
| CPU (8-thread fused NEON + row-split) | 0.17 | 5.9 | 5.3 GB |
| GPU batched (this branch, DS4F_GPU=1) | **0.50** | 2.0 | **9.2 GB** |
| GPU regression vs CPU | **3.0x worse** | | +3.9 GB |

## Why the 13x microbenchmark lied

The standalone bench measured ONE dispatch with weights pre-resident,
no host copies, no sync in the loop. The engine reality:

1. **2 dispatches/layer x 40 layers = 80 waitUntilCompleted per token.**
   Each Metal dispatch+sync costs ~1-2 ms of driver overhead at this
   size; the GPU compute itself is ~0.1 ms. Per-dispatch overhead
   dominates: 40 layers x ~3 ms = ~120 ms/token of GPU time alone,
   BEFORE any of the engine's other work.
2. **Layer-sequential dependency** (down of L feeds gate of L+1)
   forces the silu sync between dispatch 1 and 2 of each layer --
   no way to batch layers or overlap syncs.
3. **Host copies stayed**: x-pack (gate/up latent, down chains) and
   y-scatter per dispatch. The xs_same optimization removed the
   gate/up redundancy, but the down-batch pack + y-scatter remain.
4. **RSS +3.9 GB**: the 512 MB arena grows toward the full 453 MB
   pool as layers touch more experts, ON TOP of the engine's 5.3 GB
   and the Metal driver's own footprint.

## What was built and works

- `ds4f_gpu_mlx4_batch()`: desc-table batched kernel (uint4 per-job
  offsets), arena-cached weights keyed by STABLE expert-layout id
  (the vals pointer rotates with cache slots -- first real bug found),
  on-demand arena growth (second real bug: 4096-slot cap exhausted
  mid-token at ~255 layers, silent CPU fallback; now 65536 slots).
- Wired into ds4f_moe_step behind DS4F_GPU=1 with clean CPU fallback
  on any failure; gate+up one dispatch, silu*up on CPU, down one
  dispatch, result memmove'd into jb->out for the shared combine.
- Numerics: verified 3 ppm vs the CPU kernel in the microbench; the
  engine decode tokens match CPU positions (kernel is sound, just slow
  under the per-dispatch overhead).
- Fixed the pre-existing `quantize_selftest` failure along the way
  (stale gpu.h contract in the stub) -- tests went 18/19 -> 19/19.

## Bugs the integration process caught

1. desc-shape interleave (accuracy 8703 abs -> 3 ppm after fix)
2. slot-cache exhaustion at 4096 -> silent fallback after ~255 layers
3. redundant per-job x-pack (16 identical latent copies per dispatch)
4. **uint2 vs uint4 desc mismatch -- the kernel never compiled**, every
   call returned -1, and the engine silently fell back to CPU. The
   "0.23 s/token" measured before this fix was CPU + failed-call
   overhead, NOT the GPU path. Only after fixing it did the real
   0.50 s/token appear.

## Conclusion

On the M4 Pro, fine-grained per-layer GPU dispatch cannot beat the
CPU's 8-thread fused NEON path: the per-dispatch sync cost (~1-2 ms)
exceeds the GPU compute it schedules (~0.1 ms). The offload only
wins at much larger batch granularity (many tokens/layers per
dispatch), which the engine's layer-sequential dependency forbids.
Recommendation: keep the CPU path; the branch documents the full
evidence if a future batched-inference design changes the calculus.
