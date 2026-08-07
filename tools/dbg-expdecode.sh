#!/usr/bin/env bash
# dbg-expdecode.sh -- decode expert 0 layer 0 gate with the engine's kernel,
# check value magnitudes against the raw U32 + scale/BF16.
cd /Users/ruihe/disk-qwen35bA3B
cat > build/expdec.c <<'EOF'
#include "ds4f/kernels.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static float bf16f(uint16_t h){ uint32_t b=(uint32_t)h<<16; float f; memcpy(&f,&b,4); return f; }
int main(int argc, char **argv){
    /* pool file, slot for (layer 0, expert 0): off = 24 + 0 */
    FILE *f = fopen("/tmp/q35-pool/pool.bin", "rb");
    if (!f) { perror("pool"); return 1; }
    int slot_nbytes = 1769472;
    unsigned char *slot = malloc(slot_nbytes);
    fseek(f, 24, SEEK_SET);
    if (fread(slot, 1, slot_nbytes, f) != (size_t)slot_nbytes) return 1;
    fclose(f);
    /* gate: v at 0, s at 524312, b at 557080; shape [512, 2048] */
    uint32_t *v = (uint32_t *)(void *)(slot + 0);
    uint16_t *s = (uint16_t *)(void *)(slot + 524312);
    uint16_t *b = (uint16_t *)(void *)(slot + 557080);
    /* decode row 0 (2048 elems) with the engine kernel into y */
    float *x = malloc(2048 * 4);
    for (int i = 0; i < 2048; i++) x[i] = 0.0f;
    x[0] = 1.0f;
    float y[512];
    ds4f_kernels_set_simd(1);
    ds4f_mlx4_matvec(v, s, b, 512, 2048, x, y);
    /* reference: row0 elem0 = (q-8)*s0+b0, q from v[0] low nibble */
    int q = v[0] & 0xF;
    float ref = ((float)q - 8.0f) * bf16f(s[0]) + bf16f(b[0]);
    printf("SIMD y[0] = %.6f  ref = %.6f  %s\n", y[0], ref,
           fabsf(y[0]-ref) < 1e-4f ? "OK" : "MISMATCH");
    printf("scale[0] = %.6f bias[0] = %.6f  q=%d\n",
           bf16f(s[0]), bf16f(b[0]), q);
    /* range of row0 scales (groups 0..31) */
    float smin = 1e30f, smax = -1e30f;
    for (int g = 0; g < 32; g++) {
        float sv = bf16f(s[g]);
        if (sv < smin) smin = sv;
        if (sv > smax) smax = sv;
    }
    printf("row0 scale range [%.6f, %.6f]\n", smin, smax);
    return 0;
}
EOF
cc -std=c99 -O2 -pthread -Iinclude -Isrc -o build/expdec build/expdec.c \
   src/kernels.c src/simd.c src/moe.c 2>&1 | grep -c error
./build/expdec 2>&1 | tail -4
