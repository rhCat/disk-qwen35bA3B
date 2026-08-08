/* verify_batch2.c -- the batch kernel must be BIT-IDENTICAL to the
 * engine's ds4f_simd_mlx4_matvec per (row, token): same 8-accumulator
 * topology, same (c>>2)&7 map, same reduction order. This is the
 * real reference (the earlier scalar-ref test certified the same
 * wrong group indexing as the kernel -- self-consistent but wrong). */
#include "ds4f/kernels.h"
#include "ds4f/simd.h"

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
    /* xs: ROW-MAJOR [B][C] (the batch kernel's layout) */
    float *xs = (float *)malloc((size_t)B * C * 4);
    for (int t = 0; t < B; t++)
        for (int c = 0; c < C; c++)
            xs[(size_t)t * C + c] = (float)(rnd() % 100) / 100.0f;
    /* reference: the ENGINE's SIMD matvec, one token at a time */
    float *ref = (float *)malloc((size_t)B * R * 4);
    float *got = (float *)malloc((size_t)B * R * 4);
    for (int t = 0; t < B; t++)
        ds4f_simd_mlx4_matvec(vals, scales, biases, R, C,
                              xs + (size_t)t * C, ref + (size_t)t * R,
                              0, R);
    if (ds4f_mlx4_matvec_batch(vals, scales, biases, R, C, B, xs, got) != 0) {
        printf("batch returned -1\n");
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
        printf("BIT-IDENTICAL to engine SIMD matvec: %d tokens x %d rows\n", B, R);
    else
        printf("FAIL: %d mismatches\n", bad);
    return bad == 0 ? 0 : 1;
}
