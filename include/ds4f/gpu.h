#ifndef DS4F_GPU_H
#define DS4F_GPU_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Optional GPU acceleration (macOS Metal; stub elsewhere).
 *
 * The engine's default path is CPU-only and byte-deterministic; GPU
 * mode is opt-in (--gpu / DS4F_GPU=1) and accelerates the output-head
 * MLX-4bit matvec (the single largest per-token matvec: 248320 x 2048).
 * GPU logits are float32-accumulated like the CPU SIMD path but the
 * reduction order differs, so logits may differ by float32 rounding --
 * acceptable for sampling, NOT for byte-identical determinism checks
 * (keep those on the CPU path).
 *
 * All functions return 0 on success, -1 on "unavailable/failed"
 * (caller falls back to the CPU path). */
int  ds4f_gpu_init(void);
void ds4f_gpu_free(void);

/* y = MLX-4bit_matvec(vals, scales, biases, R, C, x) -- identical
 * layout to ds4f_mlx4_matvec: U32 packed nibbles (8 per word),
 * per-64-element BF16 scale (+ optional BF16 bias). R rows, C cols;
 * y must hold R floats. Returns 0 if computed, -1 if not (caller
 * should run the CPU matvec). */
int ds4f_gpu_mlx4_matvec(const uint32_t *vals, const uint16_t *scales,
                         const uint16_t *biases, int R, int C,
                         const float *x, float *y);

/* Batched expert matvec: y[j] = MLX4_matvec(vals[j], scales[j],
 * biases[j], R, C, xs[j]) for j in [0, njobs). All jobs share R, C;
 * each job has its OWN x (the shared latent for gate/up, the per-
 * expert chain for down).
 *
 * ids[j] is a STABLE identity for job j's tensor (e.g. the expert
 * layout pointer). Weight buffers are cached per id in a growing
 * arena (first sight = one copy; thereafter zero per-token copies) --
 * the vals POINTER alone is NOT a stable key: the engine's cache
 * slots rotate, so the same expert sits at a different address every
 * token. The whole batch is ONE dispatch -- the design the
 * microbenchmark measured at 13x CPU. Returns 0 if computed, -1 to
 * fall back to the CPU path (Metal unavailable, arena full, OOM). */
int ds4f_gpu_mlx4_batch(const uint32_t *const *vals,
                        const uint16_t *const *scales,
                        const uint16_t *const *biases,
                        const float *const *xs, float *const *ys,
                        const void *const *ids,
                        int R, int C, int njobs);

#ifdef __cplusplus
}
#endif

#endif /* DS4F_GPU_H */
