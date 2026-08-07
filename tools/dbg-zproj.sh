#!/usr/bin/env bash
# dbg-zproj.sh -- decode in_proj_z row 0 with unit input, check magnitude
cd /Users/ruihe/disk-qwen35bA3B
cat > build/zproj.c <<'EOF'
#include "ds4f/kernels.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static float bf16f(uint16_t h){ uint32_t b=(uint32_t)h<<16; float f; memcpy(&f,&b,4); return f; }
int main(void){
    /* trunk.bin, layer 0: in_proj_z off 10105024 (weight), scales 9842880, biases 9580736 */
    FILE *f = fopen("/tmp/q35-trunk/trunk.bin", "rb");
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    unsigned char *tr = malloc(sz);
    fseek(f, 0, SEEK_SET);
    if (fread(tr, 1, sz, f) != (size_t)sz) return 1;
    fclose(f);
    uint32_t *w = (uint32_t *)(void *)(tr + 10105024);
    uint16_t *s = (uint16_t *)(void *)(tr + 9842880);
    uint16_t *b = (uint16_t *)(void *)(tr + 9580736);
    /* unit input (rms 1): all 1/sqrt(2048) */
    float x[2048];
    for (int i = 0; i < 2048; i++) x[i] = 1.0f / sqrtf(2048.0f);
    float y[4096];
    ds4f_kernels_set_simd(1);
    ds4f_mlx4_matvec(w, s, b, 4096, 2048, x, y);
    double y2 = 0.0;
    for (int i = 0; i < 4096; i++) y2 += (double)y[i] * y[i];
    printf("in_proj_z unit-input rms: %.6f (expect ~1-3)\n", sqrt(y2/4096));
    /* random unit vector */
    srand(42);
    for (int i = 0; i < 2048; i++) x[i] = (float)(rand() % 2000 - 1000) / 1000.0f;
    double xr = 0.0;
    for (int i = 0; i < 2048; i++) xr += (double)x[i] * x[i];
    float xs = sqrtf((float)(xr / 2048));
    for (int i = 0; i < 2048; i++) x[i] /= xs;
    ds4f_mlx4_matvec(w, s, b, 4096, 2048, x, y);
    y2 = 0.0;
    for (int i = 0; i < 4096; i++) y2 += (double)y[i] * y[i];
    printf("in_proj_z random-unit rms: %.6f\n", sqrt(y2/4096));
    /* also qkv row 0..2 */
    uint32_t *wq = (uint32_t *)(void *)(tr + 1192128);
    uint16_t *sq = (uint16_t *)(void *)(tr + 667840);
    float yq[8192];
    ds4f_mlx4_matvec(wq, sq, NULL, 8192, 2048, x, yq);
    double yq2 = 0.0;
    for (int i = 0; i < 8192; i++) yq2 += (double)yq[i] * yq[i];
    printf("in_proj_qkv unit-input rms: %.6f\n", sqrt(yq2/8192));
    /* expert down (pool slot 0): down_proj [2048, 512] U32,
     * v_off 1179672, s_off ?, b_off 1736728 -- need manifest offsets */
    return 0;
}
EOF
cat >> build/zproj.c <<'EOF'
/* appended: expert down decode from the pool */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static float bf16f2(uint16_t h){ uint32_t b=(uint32_t)h<<16; float f; memcpy(&f,&b,4); return f; }
int main2(void){
    FILE *f = fopen("/tmp/q35-pool/pool.bin", "rb");
    unsigned char slot[1769472];
    fseek(f, 24, SEEK_SET);
    fread(slot, 1, 1769472, f);
    fclose(f);
    /* manifest: down_proj v_off 1179672 s_off=? b_off 1736728 */
    /* read manifest for s_off */
    FILE *mf = fopen("/tmp/q35-pool/manifest.json", "rb");
    fseek(mf, 0, SEEK_END); long ml = ftell(mf); fseek(mf, 0, SEEK_SET);
    char *mb = malloc(ml + 1); fread(mb, 1, ml, mf); mb[ml] = 0; fclose(mf);
    char *sptr = strstr(mb, "\"down_proj\"");
    /* parse the json entry after down_proj for s_off */
    (void)sptr;
    printf("slot gate v[0]=%08x\n", ((uint32_t *)(void *)slot)[0]);
    return 0;
}
EOF
# simpler: parse s_off with python inline
python3 -c "
import json
m = json.load(open('/tmp/q35-pool/manifest.json'))
for x in m['tensors'][:3]:
    print(x['name'], 's_off', x['s_off'], 'v_off', x['v_off'], 'b_off', x['b_off'], 'v_nbytes', x['v_nbytes'])
"
cc -std=c99 -O2 -pthread -Iinclude -Isrc -o build/zproj build/zproj.c \
   src/kernels.c src/simd.c src/moe.c 2>&1 | grep -c error
./build/zproj 2>&1 | tail -2
