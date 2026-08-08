// bench-gpu-experts.mm -- microbenchmark: one expert layer on CPU vs GPU.
//
// Modes:
//   default            timing: CPU (N threads) vs GPU per-matvec vs GPU batched
//   --threads N        CPU path uses N worker threads (default 8)
//   --verify           per-job accuracy: batched GPU vs CPU kernel, max abs diff
//   --mem              print the Metal buffer footprint (bytes)
//   --cpu-only         run only the CPU timing (for /usr/bin/time RSS)
//   --gpu-only         run only the GPU batched timing (for /usr/bin/time RSS)
//
// Build: make bench-gpu-experts   (Darwin only)
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "ds4f/kernels.h"
#include "ds4f/gpu.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* deterministic fill matching bench-kernels.c */
static unsigned g_rng = 0x9E3779B9u;
static unsigned rnd(void) {
    g_rng = g_rng * 1664525u + 1013904223u;
    return g_rng;
}
static void fill_tensor(uint32_t *vals, int R, int C,
                        uint16_t *scales, uint16_t *biases) {
    int nvals = R * C / 8;              /* 8 nibbles per U32 */
    int ngroups = R * C / 64;           /* BF16 scale per 64 elts */
    for (int i = 0; i < nvals; i++) vals[i] = rnd();
    for (int i = 0; i < ngroups; i++) {
        scales[i] = (uint16_t)(0x3F80 + (rnd() % 64));  /* ~1.0 */
        biases[i] = (uint16_t)(0x3F80 + (rnd() % 64));
    }
}

/* ---- one expert: gate/up 512x2048, down 2048x512, H=2048 ---- */
#define H      2048
#define M      512
#define NEXP   8
#define NTENS  3                     /* gate, up, down */

typedef struct {
    uint32_t *vals[NTENS];
    uint16_t *scales[NTENS];
    uint16_t *biases[NTENS];
    int R[NTENS], C[NTENS];
    float *x;                        /* latent input, H floats */
    float *out;                      /* expert output, H floats */
} Expert;

static Expert g_exp[NEXP];
static float *g_latent;

static void experts_init(void) {
    g_latent = (float *)malloc((size_t)H * sizeof(float));
    for (int i = 0; i < H; i++) g_latent[i] = (float)(rnd() % 100) / 100.0f;
    for (int e = 0; e < NEXP; e++) {
        Expert *ex = &g_exp[e];
        ex->R[0] = M; ex->C[0] = H;   /* gate  [512 x 2048] */
        ex->R[1] = M; ex->C[1] = H;   /* up    [512 x 2048] */
        ex->R[2] = H; ex->C[2] = M;   /* down  [2048 x 512] */
        for (int t = 0; t < NTENS; t++) {
            int R = ex->R[t], C = ex->C[t];
            ex->vals[t] = (uint32_t *)malloc((size_t)R * C / 8 * 4);
            ex->scales[t] = (uint16_t *)malloc((size_t)R * C / 64 * 2);
            ex->biases[t] = (uint16_t *)malloc((size_t)R * C / 64 * 2);
            fill_tensor(ex->vals[t], R, C, ex->scales[t], ex->biases[t]);
        }
        ex->x = g_latent;
        ex->out = (float *)malloc((size_t)H * sizeof(float));
    }
}

/* ---- CPU path: one expert chain, single thread (matches exp_run) ---- */
static void cpu_one_expert(Expert *ex) {
    float gate[M], up[M], chain[M];
    ds4f_mlx4_matvec(ex->vals[0], ex->scales[0], ex->biases[0],
                     ex->R[0], ex->C[0], ex->x, gate);
    ds4f_mlx4_matvec(ex->vals[1], ex->scales[1], ex->biases[1],
                     ex->R[1], ex->C[1], ex->x, up);
    for (int i = 0; i < M; i++) {
        float s = gate[i];
        float sig = 1.0f / (1.0f + expf(-s));
        chain[i] = s * sig * up[i];            /* silu(gate)*up */
    }
    ds4f_mlx4_matvec(ex->vals[2], ex->scales[2], ex->biases[2],
                     ex->R[2], ex->C[2], chain, ex->out);
}

typedef struct { Expert *ex; } CpuJob;
static void *cpu_worker(void *arg) {
    cpu_one_expert(((CpuJob *)arg)->ex);
    return NULL;
}

/* CPU layer with an explicit thread count: experts are distributed
 * round-robin across `nthreads` workers (each does 8/nthreads experts
 * sequentially). nthreads=8 == the engine's exp_run pattern. */
typedef struct { Expert **exs; int n; } CpuJobN;
static void *cpu_worker_n(void *arg) {
    CpuJobN *j = (CpuJobN *)arg;
    for (int i = 0; i < j->n; i++) cpu_one_expert(j->exs[i]);
    return NULL;
}

static double bench_cpu_layer(int reps, int nthreads) {
    pthread_t th[32];
    CpuJobN job[32];
    if (nthreads > 32) nthreads = 32;
    double t0 = now_s();
    for (int r = 0; r < reps; r++) {
        for (int t = 0; t < nthreads; t++) {
            job[t].n = 0;
            job[t].exs = (Expert **)malloc((size_t)NEXP * sizeof(Expert *));
            for (int e = t; e < NEXP; e += nthreads)
                job[t].exs[job[t].n++] = &g_exp[e];
            if (job[t].n > 0) pthread_create(&th[t], NULL, cpu_worker_n, &job[t]);
            else job[t].exs[0] = NULL;
        }
        for (int t = 0; t < nthreads; t++) {
            if (job[t].n > 0) pthread_join(th[t], NULL);
            free(job[t].exs);
        }
    }
    return (now_s() - t0) / reps;
}

/* ---- GPU path: the EXISTING API, 24 matvecs, realistic churn ---- */
static double bench_gpu_layer(int reps) {
    if (ds4f_gpu_init() != 0) return -1.0;
    double t0 = now_s();
    for (int r = 0; r < reps; r++) {
        for (int e = 0; e < NEXP; e++) {
            Expert *ex = &g_exp[e];
            for (int t = 0; t < NTENS; t++) {
                ds4f_gpu_mlx4_matvec(ex->vals[t], ex->scales[t],
                                     ex->biases[t], ex->R[t], ex->C[t],
                                     ex->x, ex->out);
            }
        }
    }
    double t = (now_s() - t0) / reps;
    ds4f_gpu_free();
    return t;
}

/* ---- GPU batched path: ONE dispatch for all 24 matvecs ---- */
static const char *_batch_src =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void mlx4batch(\n"
    "    device const uint   *vals    [[buffer(0)]],\n"
    "    device const ushort *scales  [[buffer(1)]],\n"
    "    device const ushort *biases  [[buffer(2)]],\n"
    "    device const float  *xs      [[buffer(3)]],\n"
    "    device float        *ys      [[buffer(4)]],\n"
    "    device const uint4  *desc    [[buffer(5)]],  // R, C, xoff, yoff per job\n"
    "    constant uint &njobs         [[buffer(6)]],\n"
    "    uint idx                     [[thread_position_in_grid]])\n"
    "{\n"
    "    // grid = njobs * maxR; j = idx % njobs, r = idx / njobs\n"
    "    uint j = idx % njobs;\n"
    "    uint r = idx / njobs;\n"
    "    uint4 d = desc[j];\n"
    "    uint R = d.x, C = d.y;\n"
    "    if (r >= R) return;\n"
    "    // every job in this call is 1M elements (512x2048 or 2048x512),\n"
    "    // so the concatenated per-job base is uniform.\n"
    "    uint vbase = j * (1048576u / 8u);\n"
    "    uint sbase = j * (1048576u / 64u);\n"
    "    device const float *x = xs + d.z;\n"
    "    device float *y = ys + d.w;\n"
    "    float acc = 0.0f;\n"
    "    for (uint c = 0; c < C; c++) {\n"
    "        uint k = r * C + c;\n"
    "        uint g = k / 64u;\n"
    "        float s = as_type<float>(uint(scales[sbase + g]) << 16);\n"
    "        float b = as_type<float>(uint(biases[sbase + g]) << 16);\n"
    "        uint q = (vals[vbase + (k >> 3u)] >> ((k & 7u) << 2u)) & 0xFu;\n"
    "        acc += (float(q) * s + b) * x[c];\n"
    "    }\n"
    "    y[r] = acc;\n"
    "}\n";

static id<MTLDevice> _bdev;
static id<MTLComputePipelineState> _bpso;
static id<MTLCommandQueue> _bqueue;
static id<MTLBuffer> _bvals, _bscales, _bbiases, _bxs, _bys, _bdesc;
static size_t _mem_total = 0;

/* Per-job x: gate/up use the latent (H); down uses the chain (M).
 * For timing/accuracy we give every job its own x slice so the check
 * is per-job independent. */
static double bench_gpu_batched(int reps, int do_verify, int *mem_out) {
    @autoreleasepool {
        _bdev = MTLCreateSystemDefaultDevice();
        if (!_bdev) return -1.0;
        NSError *err = nil;
        id<MTLLibrary> lib = [_bdev newLibraryWithSource:
            [NSString stringWithUTF8String:_batch_src] options:nil error:&err];
        if (!lib) { fprintf(stderr, "batch kernel compile failed: %s\n",
                            err ? [[err localizedDescription] UTF8String] : "?"); return -1.0; }
        _bpso = [_bdev newComputePipelineStateWithFunction:
                 [lib newFunctionWithName:@"mlx4batch"] error:&err];
        _bqueue = [_bdev newCommandQueue];

        /* concat all 24 tensors: 16 x 512x2048 (gate/up) + 8 x 2048x512 (down) */
        size_t va = 16 * (size_t)512 * 2048 / 8 * 4;
        size_t vb = 8 * (size_t)2048 * 512 / 8 * 4;
        size_t sa = 16 * (size_t)512 * 2048 / 64 * 2;
        size_t sb = 8 * (size_t)2048 * 512 / 64 * 2;
        _bvals = [_bdev newBufferWithLength:va + vb
                                    options:MTLResourceStorageModeShared];
        _bscales = [_bdev newBufferWithLength:sa + sb
                                     options:MTLResourceStorageModeShared];
        _bbiases = [_bdev newBufferWithLength:sa + sb
                                     options:MTLResourceStorageModeShared];
        size_t voff = 0, soff = 0;
        for (int e = 0; e < NEXP; e++) {
            Expert *ex = &g_exp[e];
            for (int t = 0; t < NTENS; t++) {
                size_t vn = (size_t)ex->R[t] * ex->C[t] / 8 * 4;
                size_t sn = (size_t)ex->R[t] * ex->C[t] / 64 * 2;
                memcpy((char *)_bvals.contents + voff, ex->vals[t], vn);
                memcpy((char *)_bscales.contents + soff, ex->scales[t], sn);
                memcpy((char *)_bbiases.contents + soff, ex->biases[t], sn);
                voff += vn; soff += sn;
            }
        }
        /* x: concatenated per-job x slices. gate/up jobs: the latent
         * (H floats); down jobs: a 512-float chain (we use a distinct
         * slice of g_latent for realism in the verify). */
        _bxs = [_bdev newBufferWithLength:(size_t)24 * 2048 * 4
                                 options:MTLResourceStorageModeShared];
        for (int j = 0; j < 24; j++) {
            float *dst = (float *)_bxs.contents + (size_t)j * 2048;
            for (int i = 0; i < 2048; i++) dst[i] = g_latent[i % H];
        }
        _bys = [_bdev newBufferWithLength:(size_t)24 * 2048 * 4
                                 options:MTLResourceStorageModeShared];
        _bdesc = [_bdev newBufferWithLength:24 * 16
                                   options:MTLResourceStorageModeShared];
        /* desc entries derive from the actual per-expert layout:
         * job j = expert (j/3), tensor (j%3) -- gate/up are 512x2048,
         * down is 2048x512, and they INTERLEAVE per expert. The
         * concatenated per-job base stays uniform (every tensor is
         * exactly 1M elements). */
        typedef struct { unsigned x, y, z, w; } Desc4;
        Desc4 *desc = (Desc4 *)_bdesc.contents;
        for (int j = 0; j < 24; j++) {
            int e = j / NTENS, t = j % NTENS;
            desc[j] = (Desc4){ (unsigned)g_exp[e].R[t], (unsigned)g_exp[e].C[t],
                               (unsigned)(j * 2048), (unsigned)(j * 2048) };
        }

        _mem_total = (va + vb) + (sa + sb) + (sa + sb) + 24 * 2048 * 4 + 24 * 2048 * 4 + 24 * 16;

        double t0 = now_s();
        for (int r = 0; r < reps; r++) {
            id<MTLCommandBuffer> cb = [_bqueue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:_bpso];
            [enc setBuffer:_bvals offset:0 atIndex:0];
            [enc setBuffer:_bscales offset:0 atIndex:1];
            [enc setBuffer:_bbiases offset:0 atIndex:2];
            [enc setBuffer:_bxs offset:0 atIndex:3];
            [enc setBuffer:_bys offset:0 atIndex:4];
            [enc setBuffer:_bdesc offset:0 atIndex:5];
            uint njobs = 24;
            [enc setBytes:&njobs length:sizeof(njobs) atIndex:6];
            MTLSize tpg = MTLSizeMake(256, 1, 1);
            MTLSize tpgrid = MTLSizeMake((NSUInteger)24 * 2048, 1, 1);
            [enc dispatchThreads:tpgrid threadsPerThreadgroup:tpg];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }
        double t = (now_s() - t0) / reps;

        if (do_verify) {
            /* per-job: batched y slice vs CPU ds4f_mlx4_matvec with the
             * same x. tolerance: float32 reduction-order rounding.
             * Use RELATIVE error (sum magnitudes ~1.5e4, so an abs
             * 1e-2 would false-fail honest ulp differences). */
            double worst_abs = 0, worst_rel = 0;
            int bad_rel = 0, total = 0;
            for (int j = 0; j < 24; j++) {
                int e = j / NTENS, t = j % NTENS;
                Expert *ex = &g_exp[e];
                const float *xj = (const float *)_bxs.contents + (size_t)j * 2048;
                const float *yj = (const float *)_bys.contents + (size_t)j * 2048;
                float *ref = (float *)malloc((size_t)ex->R[t] * 4);
                ds4f_mlx4_matvec(ex->vals[t], ex->scales[t], ex->biases[t],
                                 ex->R[t], ex->C[t], xj, ref);
                for (int i = 0; i < ex->R[t]; i++) {
                    double da = fabs((double)ref[i] - yj[i]);
                    double mag = fabs((double)ref[i]) + 1.0;
                    double dr = da / mag;
                    if (da > worst_abs) worst_abs = da;
                    if (dr > worst_rel) worst_rel = dr;
                    if (dr > 1e-4) bad_rel++;   /* 1e-4 rel = float32 sane */
                    total++;
                }
                free(ref);
            }
            printf("verify: %d outputs, max abs %.6f, max REL %.2e, rel>1e-4: %d\n",
                   total, worst_abs, worst_rel, bad_rel);
        }
        if (mem_out) *mem_out = (int)_mem_total;
        return t;
    }
    return -1.0;
}

int main(int argc, char **argv) {
    int reps = 20, nthreads = 8;
    int do_verify = 0, mem_only = 0, cpu_only = 0, gpu_only = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--threads") && i + 1 < argc) nthreads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--verify")) do_verify = 1;
        else if (!strcmp(argv[i], "--mem")) mem_only = 1;
        else if (!strcmp(argv[i], "--cpu-only")) cpu_only = 1;
        else if (!strcmp(argv[i], "--gpu-only")) gpu_only = 1;
        else reps = atoi(argv[i]);
    }
    experts_init();

    if (mem_only) {
        /* report the Metal footprint only (run under /usr/bin/time for RSS) */
        int m = 0;
        double gb = bench_gpu_batched(1, 0, &m);
        printf("GPU batched Metal buffers: %.2f MB\n", m / 1048576.0);
        printf("(run with /usr/bin/time -l for process RSS; unified memory\n");
        printf(" means Metal Shared buffers ARE the process address space)\n");
        return gb > 0 ? 0 : 1;
    }

    if (!gpu_only) {
        double cpu = bench_cpu_layer(reps, nthreads);
        printf("CPU  (%d threads): %.4f ms/layer  (%.1f us/matvec)\n",
               nthreads, cpu * 1e3, cpu * 1e6 / (NEXP * NTENS));
    }
    if (!cpu_only) {
        double gpu = bench_gpu_layer(reps);
        if (gpu > 0) {
            printf("GPU  (per-matvec API, 24 calls): %.4f ms/layer  (%.1f us/matvec)\n",
                   gpu * 1e3, gpu * 1e6 / (NEXP * NTENS));
        }
        double gb = bench_gpu_batched(reps, do_verify, NULL);
        if (gb > 0) {
            printf("GPU  (batched, 1 dispatch):  %.4f ms/layer  (%.1f us/matvec)\n",
                   gb * 1e3, gb * 1e6 / (NEXP * NTENS));
            if (!gpu_only) {
                double cpu = bench_cpu_layer(reps, nthreads);
                printf("ratio batched/CPU(%d): %.2fx\n", nthreads, gb / cpu);
            }
        }
    }
    return 0;
}
