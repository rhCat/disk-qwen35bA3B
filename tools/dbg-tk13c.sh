#!/usr/bin/env bash
# dbg-tk13c.sh -- element-level decode comparison for row 1 (C=200, R=2)
cd /Users/ruihe/disk-qwen35bA3B
cat > build/tk13c.c <<'EOF'
#include "ds4f/kernels.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static float bf16f(uint16_t h){ uint32_t b=(uint32_t)h<<16; float f; memcpy(&f,&b,4); return f; }
int main(void){
    int R=2, C=200;
    int n=R*C, nw=(n+7)/8, ng=(n+63)/64;
    uint32_t *vals=calloc((size_t)nw,4);
    uint16_t *scales=calloc((size_t)ng,2), *biases=calloc((size_t)ng,2);
    srand(13);
    for(int i=0;i<n;i++){ int q=rand()%16; vals[i>>3]|=(uint32_t)q<<(4*(i&7)); }
    for(int g=0;g<ng;g++){ scales[g]=(uint16_t)(rand()%20000); biases[g]=(uint16_t)(rand()%20000); }
    /* reference decode for row 1, first 16 elements */
    printf("row1 ref decode[0..15]:");
    for(int c=0;c<16;c++){
        int k=1*C+c;
        int q=(vals[k>>3]>>(4*(k&7)))&0xF;
        int g=k/64;
        printf(" %.2f", ((float)q-8)*bf16f(scales[g])+bf16f(biases[g]));
    }
    printf("\n");
    /* SIMD decode via the kernel's own path: run matvec with x=unit c */
    float *x=malloc((size_t)C*4), *y=malloc((size_t)R*4);
    for(int c=0;c<C;c++) x[c]=0.0f;
    x[0]=1.0f;
    ds4f_kernels_set_simd(1);
    ds4f_mlx4_matvec(vals,scales,biases,R,C,x,y);
    /* y[1] should equal row1 decoded element 0 */
    printf("SIMD y[1] (row1, x=unit0): %.4f\n", y[1]);
    /* now x[1] */
    x[0]=0.0f; x[1]=1.0f;
    ds4f_mlx4_matvec(vals,scales,biases,R,C,x,y);
    printf("SIMD y[1] (row1, x=unit1): %.4f\n", y[1]);
    return 0;
}
EOF
cc -std=c99 -O2 -pthread -Iinclude -Isrc -o build/tk13c build/tk13c.c \
   src/kernels.c src/simd.c src/moe.c 2>&1 | grep -c error
./build/tk13c 2>&1 | tail -3
