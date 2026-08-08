/* verify_batch.c -- bit-fidelity gate: ds4f_mlx4_matvec_batch must
 * match ds4f_mlx4_matvec EXACTLY per (row, token). */
#include "ds4f/kernels.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned g_rng = 0x12345678u;
static unsigned rnd(void) {
    g_rng = g_rng * 1664525u + 1013904223u;
    return g_rng;
}

int main(void) {
    const int R = 512, C = 2048, B = 64;
    const size_t nvals = (size_t)R * C / 8;
    const size_t ngroups = (size_t)R * C / 64;
    uint32_t *vals = (uint32_t *)malloc(nvals * 4);
    uint16_t *scales = (uint16_t *)malloc(ngroups * 2);
    uint16_t *biases = (uint16_t *)malloc(ngroups * 2);
    for (size_t i = 0; i < nvals; i++) vals[i] = rnd();
    for (size_t i = 0; i < ngroups; i++) {
        scales[i] = (uint16_t)(0x3F80 + (rnd() % 64));
        biases[i] = (uint16_t)(0x3F80 + (rnd() % 64));
    }

    /* B distinct x vectors (the batch's per-token inputs) */
    float *xs = (float *)malloc((size_t)B * C * 4);
    for (int t = 0; t < B; t++)
        for (int c = 0; c < C; c++)
            xs[(size_t)t * C + c] = (float)(rnd() % 100) / 100.0f;

    /* reference: serial matvec per token (the current path) */
    float *ref = (float *)malloc((size_t)B * R * 4);
    float *got = (float *)malloc((size_t)B * R * 4);
    for (int t = 0; t < B; t++)
        ds4f_mlx4_matvec(vals, scales, biases, R, C, xs + (size_t)t * C,
                         ref + (size_t)t * R);

    if (ds4f_mlx4_matvec_batch(vals, scales, biases, R, C, B, xs, got) != 0) {
        printf("batch returned -1 (SIMD off?)\n");
        return 1;
    }

    int bad = 0;
    for (size_t i = 0; i < (size_t)B * R; i++) {
        if (got[i] != ref[i]) {
            if (bad < 3)
                printf("MISMATCH at %zu: got %.9g ref %.9g\n", i, got[i], ref[i]);
            bad++;
        }
    }
    if (bad == 0)
        printf("BIT-IDENTICAL: %d tokens x %d rows all match\n", B, R);
    else
        printf("FAIL: %d mismatches\n", bad);
    return bad == 0 ? 0 : 1;
}
