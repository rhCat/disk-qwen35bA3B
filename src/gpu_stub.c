/* gpu_stub.c -- non-Darwin fallback: GPU acceleration unavailable.
 * Keeps the engine portable (Linux/EC2 quick-boot path stays CPU-only). */
#include "ds4f/gpu.h"

int ds4f_gpu_init(void) { return -1; }
void ds4f_gpu_free(void) {}
int ds4f_gpu_mlx4_matvec(const uint32_t *vals, const uint16_t *scales,
                         const uint16_t *biases, int R, int C,
                         const float *x, float *y) {
    (void)vals; (void)scales; (void)biases;
    (void)R; (void)C; (void)x; (void)y;
    return -1;
}
