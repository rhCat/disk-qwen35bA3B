#!/usr/bin/env bash
# dbg-tk13.sh -- side-by-side SIMD vs scalar mlx4 on C=200 (test 13 shape)
cd /Users/ruihe/disk-qwen35bA3B
cat > build/tk13.c <<'EOF'
#include "ds4f/kernels.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(void){
    int R=8, C=200;
    int n=R*C, nw=(n+7)/8, ng=(n+63)/64;
    uint32_t *vals=calloc((size_t)nw,4);
    uint16_t *scales=calloc((size_t)ng,2), *biases=calloc((size_t)ng,2);
    float *x=malloc((size_t)C*4), *y1=malloc((size_t)R*4), *y2=malloc((size_t)R*4);
    srand(13);
    for(int i=0;i<n;i++){ int q=rand()%16; vals[i>>3]|=(uint32_t)q<<(4*(i&7)); }
    for(int g=0;g<ng;g++){ scales[g]=(uint16_t)(rand()%20000); biases[g]=(uint16_t)(rand()%20000); }
    for(int c=0;c<C;c++) x[c]=(float)(rand()%200)/100.0f-1.0f;
    ds4f_kernels_set_simd(1);
    ds4f_mlx4_matvec(vals,scales,biases,R,C,x,y1);
    ds4f_kernels_set_simd(0);
    ds4f_mlx4_matvec(vals,scales,biases,R,C,x,y2);
    int bad=0;
    for(int r=0;r<R;r++){
        double rel = y2[r]!=0 ? (double)y1[r]/y2[r] : 0;
        if(rel<0.9||rel>1.1||(y1[r]==0&&y2[r]!=0)) bad=1;
        printf("r%d simd %.4f scalar %.4f rel %.3f\n",r,y1[r],y2[r],rel);
    }
    printf("SIMD %s\n", bad?"MISMATCH":"OK");
    return bad;
}
EOF
cc -std=c99 -O2 -pthread -Iinclude -Isrc -o build/tk13 build/tk13.c \
   src/kernels.c src/simd.c src/moe.c 2>&1 | grep -c error
./build/tk13 2>&1 | tail -10
