#!/usr/bin/env bash
# dbg-tk13f.sh -- per-column SIMD vs ref for row 1 at group boundaries
cd /Users/ruihe/disk-qwen35bA3B
cat > build/tk13f.c <<'EOF'
#include "ds4f/kernels.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static float bf16f(uint16_t h){ uint32_t b=(uint32_t)h<<16; float f; memcpy(&f,&b,4); return f; }
int main(void){
    int R=2, C=200;
    int n=R*C, nw=(n+7)/8, ng=(n+63)/64;
    uint32_t *vals=calloc((size_t)nw,4);
    uint16_t *scales=calloc((size_t)ng,2), *biases=calloc((size_t)ng,2);
    float *x=malloc((size_t)C*4), *y=malloc((size_t)R*4);
    srand(13);
    for(int i=0;i<n;i++){ int q=rand()%16; vals[i>>3]|=(uint32_t)q<<(4*(i&7)); }
    for(int g=0;g<ng;g++){ scales[g]=(uint16_t)(rand()%20000); biases[g]=(uint16_t)(rand()%20000); }
    for(int c=0;c<C;c++) x[c]=0.0f;
    int cols[] = {2, 63, 64, 65, 127, 128, 191, 192, 199};
    for(int i=0;i<9;i++){
        int c=cols[i];
        x[c]=1.0f;
        ds4f_kernels_set_simd(1);
        ds4f_mlx4_matvec(vals,scales,biases,R,C,x,y);
        int k=1*C+c; int q=(vals[k>>3]>>(4*(k&7)))&0xF; int g=k/64;
        float ref=((float)q-8)*bf16f(scales[g])+bf16f(biases[g]);
        printf("c=%-3d absg=%d simd %.4f ref %.4f %s\n", c, g, y[1], ref,
               fabsf(y[1]-ref)>1e-3f ? "BAD" : "ok");
        x[c]=0.0f;
    }
    return 0;
}
EOF
cc -std=c99 -O2 -pthread -Iinclude -Isrc -o build/tk13f build/tk13f.c \
   src/kernels.c src/simd.c src/moe.c 2>&1 | grep -c error
./build/tk13f 2>&1 | tail -9
