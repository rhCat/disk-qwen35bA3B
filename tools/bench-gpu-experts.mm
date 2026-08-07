// bench-gpu-experts.mm -- microbenchmark: one expert layer on CPU vs GPU.
//
// Answers the investigation's question with data: does Metal beat the
// CPU's 8-thread fused-NEON path on the REAL expert shapes?
//
//   CPU path: 8 threads (like exp_run), each running one expert's
//             chain: gate(512x2048) -> up(512x2048) -> down(2048x512),
//             using the engine's own ds4f_mlx4_matvec (fused SIMD).
//   GPU path: the EXISTING ds4f_gpu_mlx4_matvec API called 24x/token
//             (per-matvec dispatch, weight buffers recreated when the
//             tensor pointers change -- the realistic naive offload).
//
// Build: make bench-gpu-experts   (Darwin only)
// Run:   ./bench-gpu-experts [reps]
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "ds4f/kernels.h"
#include "ds4f/gpu.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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
    /* gate + up: [M x H] * x -> M floats each */
    ds4f_mlx4_matvec(ex->vals[0], ex->scales[0], ex->biases[0],
                     ex->R[0], ex->C[0], ex->x, gate);
    ds4f_mlx4_matvec(ex->vals[1], ex->scales[1], ex->biases[1],
                     ex->R[1], ex->C[1], ex->x, up);
    for (int i = 0; i < M; i++) {
        float s = gate[i];
        float sig = 1.0f / (1.0f + expf(-s));
        chain[i] = s * sig * up[i];            /* silu(gate)*up */
    }
    /* down: [H x M] * chain -> H floats */
    ds4f_mlx4_matvec(ex->vals[2], ex->scales[2], ex->biases[2],
                     ex->R[2], ex->C[2], chain, ex->out);
}

typedef struct { Expert *ex; } CpuJob;
static void *cpu_worker(void *arg) {
    cpu_one_expert(((CpuJob *)arg)->ex);
    return NULL;
}

static double bench_cpu_layer(int reps) {
    /* 8 threads, one expert each, joined -- the exp_run pattern */
    pthread_t th[NEXP];
    CpuJob job[NEXP];
    double t0 = now_s();
    for (int r = 0; r < reps; r++) {
        for (int e = 0; e < NEXP; e++) {
            job[e].ex = &g_exp[e];
            pthread_create(&th[e], NULL, cpu_worker, &job[e]);
        }
        for (int e = 0; e < NEXP; e++) pthread_join(th[e], NULL);
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
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

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

static double bench_gpu_batched(int reps) {
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

        /* concat all 24 tensors (all [R x C] with R,C per shape group) */
        /* group A: 16 tensors 512x2048 (gate/up), group B: 8 tensors 2048x512 (down) */
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
        /* copy tensors in: per expert, gate+up (512x2048) then down (2048x512) */
        size_t voff = 0, soff = 0;
        for (int e = 0; e < NEXP; e++) {
            Expert *ex = &g_exp[e];
            for (int t = 0; t < 2; t++) {  /* gate, up */
                size_t vn = (size_t)ex->R[t] * ex->C[t] / 8 * 4;
                size_t sn = (size_t)ex->R[t] * ex->C[t] / 64 * 2;
                memcpy((char *)_bvals.contents + voff, ex->vals[t], vn);
                memcpy((char *)_bscales.contents + soff, ex->scales[t], sn);
                memcpy((char *)_bbiases.contents + soff, ex->biases[t], sn);
                voff += vn; soff += sn;
            }
            int t = 2;  /* down */
            size_t vn = (size_t)ex->R[t] * ex->C[t] / 8 * 4;
            size_t sn = (size_t)ex->R[t] * ex->C[t] / 64 * 2;
            memcpy((char *)_bvals.contents + voff, ex->vals[t], vn);
            memcpy((char *)_bscales.contents + soff, ex->scales[t], sn);
            memcpy((char *)_bbiases.contents + soff, ex->biases[t], sn);
            voff += vn; soff += sn;
        }
        /* x buffers: all jobs share the same latent x (H floats), and the
         * down jobs consume the chain -- for the BENCH we approximate with
         * the same x everywhere (timing, not semantics) */
        _bxs = [_bdev newBufferWithLength:(size_t)2048 * 4
                                 options:MTLResourceStorageModeShared];
        memcpy(_bxs.contents, g_latent, (size_t)2048 * 4);
        _bys = [_bdev newBufferWithLength:(size_t)2048 * NEXP * 4
                                 options:MTLResourceStorageModeShared];
        _bdesc = [_bdev newBufferWithLength:24 * 16
                                   options:MTLResourceStorageModeShared];
        /* desc entries: jobs 0..15 = 512x2048, 16..23 = 2048x512; xoff=0;
         * yoff = j*2048 so each job writes its own slice. Host-side plain
         * struct matches the kernel's uint4 layout (4 x uint32). */
        typedef struct { unsigned x, y, z, w; } Desc4;
        Desc4 *desc = (Desc4 *)_bdesc.contents;
        for (int j = 0; j < 16; j++) desc[j] = (Desc4){512, 2048, 0, (unsigned)(j * 2048)};
        for (int j = 16; j < 24; j++) desc[j] = (Desc4){2048, 512, 0, (unsigned)(j * 2048)};

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
            /* grid = R*njobs threads (512*16 + 2048*8 = 24576), but the
             * kernel maps idx -> (r = idx/njobs, j = idx%njobs) so the
             * grid must be njobs * maxR = 24*2048 */
            MTLSize tpg = MTLSizeMake(256, 1, 1);
            MTLSize tpgrid = MTLSizeMake((NSUInteger)24 * 2048, 1, 1);
            [enc dispatchThreads:tpgrid threadsPerThreadgroup:tpg];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }
        double t = (now_s() - t0) / reps;
        return t;
    }
    return -1.0;
}

int main(int argc, char **argv) {
    int reps = argc > 1 ? atoi(argv[1]) : 20;
    experts_init();

    /* CPU baseline (engine kernels, 8-thread layer) */
    double cpu = bench_cpu_layer(reps);
    printf("CPU  (8 threads, fused NEON): %.4f ms/layer  (%.1f us/matvec)\n",
           cpu * 1e3, cpu * 1e6 / (NEXP * NTENS));

    /* GPU (existing per-matvec API) */
    double gpu = bench_gpu_layer(reps);
    if (gpu < 0) {
        printf("GPU: Metal unavailable (fallback stub?)\n");
        return 1;
    }
    printf("GPU  (per-matvec API, 24 calls): %.4f ms/layer  (%.1f us/matvec)\n",
           gpu * 1e3, gpu * 1e6 / (NEXP * NTENS));
    printf("ratio GPU/CPU: %.2fx %s\n", gpu / cpu,
           gpu < cpu ? "(GPU faster)" : "(CPU faster)");

    /* GPU (batched: one dispatch for all 24 matvecs) */
    double gb = bench_gpu_batched(reps);
    if (gb > 0) {
        printf("GPU  (batched, 1 dispatch):  %.4f ms/layer  (%.1f us/matvec)\n",
               gb * 1e3, gb * 1e6 / (NEXP * NTENS));
        printf("ratio batched/CPU: %.2fx %s\n", gb / cpu,
               gb < cpu ? "(GPU faster)" : "(CPU faster)");
    } else {
        printf("GPU batched: unavailable\n");
    }
    return 0;
}
