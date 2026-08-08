/* verify_batch2.c -- the batched kernel must match the SCALAR
 * reference exactly (same c-order accumulation). The SIMD matvec
 * uses an 8-accumulator tree so it differs by float32 rounding; the
 * scalar path is the true bit-fidelity anchor. */
#include "ds4f/kernels.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned g_rng = 0x12345678u;
static unsigned rnd(void) {
    g_rng = g_rng * 1664525u + 1013904223u;
    return g_rng;
}

/* scalar mlx4 matvec, one token: bit-fidelity anchor */
static void ref_scalar(const uint32_t *vals, const uint16_t *scales,
                       const uint16_t *biases, int R, int C,
                       const float *x, float *y) {
    for (int r = 0; r < R; r++) {
        float acc = 0.0f;
        for (int c = 0; c < C; c++) {
            int g = c / DS4F_MLX4_GROUP;
            uint32_t sb = (uint32_t)scales[g] << 16;
            uint32_t bb = biases ? (uint32_t)biases[g] << 16 : 0;
            float s, b;
            memcpy(&s, &sb, 4);
            memcpy(&b, &bb, 4);
            int q = (int)((vals[(size_t)r * (C / 8) + (c >> 3)] >>
                           (4 * (c & 7))) & 0xFu);
            acc += ((float)q * s + b) * x[c];
        }
        y[r] = acc;
    }
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
    float *xs = (float *)malloc((size_t)B * C * 4);
    for (int t = 0; t < B; t++)
        for (int c = 0; c < C; c++)
            xs[(size_t)t * C + c] = (float)(rnd() % 100) / 100.0f;
    float *ref = (float *)malloc((size_t)B * R * 4);
    float *got = (float *)malloc((size_t)B * R * 4);
    for (int t = 0; t < B; t++)
        ref_scalar(vals, scales, biases, R, C, xs + (size_t)t * C,
                   ref + (size_t)t * R);
    if (ds4f_mlx4_matvec_batch(vals, scales, biases, R, C, B, xs, got) != 0) {
        printf("batch returned -1\n");
        return 1;
    }
    int bad = 0;
    for (size_t i = 0; i < (size_t)B * R; i++)
        if (got[i] != ref[i]) {
            if (bad < 3)
                printf("MISMATCH at %zu: got %.9g ref %.9g\n", i, got[i], ref[i]);
            bad++;
        }
    if (bad == 0)
        printf("BIT-IDENTICAL to scalar: %d tokens x %d rows\n", B, R);
    else
        printf("FAIL: %d mismatches\n", bad);
    return bad == 0 ? 0 : 1;
}
