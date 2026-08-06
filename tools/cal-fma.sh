#!/usr/bin/env bash
# cal-fma.sh -- measure the machine's raw NEON FMA throughput
cd /Users/ruihe/disk-qwen35bA3B
cat > /tmp/cal_fma.c <<'EOF'
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <arm_neon.h>
static double now_s(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}
int main(int argc,char**argv){
    long n=1L<<24;                     /* 16.7M floats = 4.2M FMA/lane-set */
    float *a=malloc(n*4),*b=malloc(n*4),*c=malloc(n*4);
    for(long i=0;i<n;i++){a[i]=1.0f;b[i]=0.5f;}
    int reps=argc>1?atoi(argv[1]):20;
    double t0=now_s();float acc=0;
    for(int r=0;r<reps;r++){
        float32x4_t s=vdupq_n_f32(0);
        for(long i=0;i+3<n;i+=4)
            s=vmlaq_f32(s,vld1q_f32(a+i),vld1q_f32(b+i));
        acc+=vgetq_lane_f32(s,0);
    }
    double dt=now_s()-t0;
    double fmacs=(double)n*reps/dt;
    printf("pure NEON FMA: %.2f GFMA/s (%.0fM elems/s) acc=%f\n",
           fmacs/1e9, fmacs/1e6, acc);
    return 0;
}
EOF
cc -O2 -o /tmp/cal_fma /tmp/cal_fma.c && /tmp/cal_fma 20
