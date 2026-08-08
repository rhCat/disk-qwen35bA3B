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

int ds4f_gpu_mlx4_batch(const uint32_t *const *vals,
                        const uint16_t *const *scales,
                        const uint16_t *const *biases,
                        const float *const *xs, float *const *ys,
                        const void *const *ids,
                        int R, int C, int njobs) {
    (void)vals; (void)scales; (void)biases;
    (void)xs; (void)ys; (void)ids; (void)R; (void)C; (void)njobs;
    return -1;
}
