// gpu_metal.mm -- optional Metal acceleration for the MLX-4bit output
// head matvec (macOS only). Compiles the kernel at runtime so the
// Makefile needs no .metallib build step. Mirrors ds4f_mlx4_matvec's
// math exactly: per-64-element group BF16 scale (+ optional BF16 bias),
// q*s+b dequant, float32 accumulation. Weight buffers are cached across
// tokens (only x changes per token), so the per-token cost is one small
// upload + one 248K-row readback.
//
// Built only on Darwin (see Makefile). Non-Darwin builds use gpu_stub.c.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "ds4f/gpu.h"
#include <stdlib.h>
#include <string.h>

static id<MTLDevice> _dev;
static id<MTLComputePipelineState> _pso;
static id<MTLCommandQueue> _queue;
/* cached weight buffers (never change between tokens) */
static id<MTLBuffer> _bv, _bs, _bb;
static id<MTLBuffer> _bx, _by;    /* per-token x / y */
static int _R = -1, _C = -1;
static int _has_bias = 0;

static const char *_kernel_src =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void mlx4mv(\n"
    "    device const uint *vals    [[buffer(0)]],\n"
    "    device const ushort *scales [[buffer(1)]],\n"
    "    device const ushort *biases [[buffer(2)]],\n"
    "    device const float *x       [[buffer(3)]],\n"
    "    device float *y             [[buffer(4)]],\n"
    "    constant uint &R            [[buffer(5)]],\n"
    "    constant uint &C            [[buffer(6)]],\n"
    "    uint r                      [[thread_position_in_grid]])\n"
    "{\n"
    "    if (r >= R) return;\n"
    "    float acc = 0.0f;\n"
    "    for (uint c = 0; c < C; c++) {\n"
    "        uint k = r * C + c;\n"
    "        uint g = k / 64u;\n"
    "        float s = as_type<float>(uint(scales[g]) << 16);\n"
    "        float b = as_type<float>(uint(biases[g]) << 16);\n"
    "        uint q = (vals[k >> 3u] >> ((k & 7u) << 2u)) & 0xFu;\n"
    "        acc += (float(q) * s + b) * x[c];\n"
    "    }\n"
    "    y[r] = acc;\n"
    "}\n";

int ds4f_gpu_init(void) {
    if (_dev) return 0;
    @autoreleasepool {
        _dev = MTLCreateSystemDefaultDevice();
        if (!_dev) return -1;
        NSError *err = nil;
        id<MTLLibrary> lib = [_dev newLibraryWithSource:
            [NSString stringWithUTF8String:_kernel_src] options:nil error:&err];
        if (!lib) {
            fprintf(stderr, "gpu: kernel compile failed: %s\n",
                    err ? [[err localizedDescription] UTF8String] : "?");
            _dev = nil;
            return -1;
        }
        id<MTLFunction> fn = [lib newFunctionWithName:@"mlx4mv"];
        if (!fn) { _dev = nil; return -1; }
        _pso = [_dev newComputePipelineStateWithFunction:fn error:&err];
        if (!_pso) { _dev = nil; return -1; }
        _queue = [_dev newCommandQueue];
        fprintf(stderr, "gpu: Metal ready (%s)\n",
                [[_dev name] UTF8String]);
    }
    return 0;
}

void ds4f_gpu_free(void) {
    _bv = _bs = _bb = _bx = _by = nil;
    _pso = nil;
    _queue = nil;
    _dev = nil;
    _R = _C = -1;
}

int ds4f_gpu_mlx4_matvec(const uint32_t *vals, const uint16_t *scales,
                         const uint16_t *biases, int R, int C,
                         const float *x, float *y) {
    if (!_dev || !_pso || R < 1 || C < 1) return -1;
    @autoreleasepool {
        /* (re)create weight buffers only when the shape/pointers change */
        if (!_bv || !_bs || _R != R || _C != C ||
            _has_bias != (biases != NULL)) {
            _bv = _bs = _bb = nil;
            size_t vbytes = (size_t)R * (size_t)C / 2u;
            size_t sbytes = (size_t)R * (size_t)C / 64u * 2u;
            _bv = [_dev newBufferWithBytes:vals length:vbytes
                                   options:MTLResourceStorageModeShared];
            _bs = [_dev newBufferWithBytes:scales length:sbytes
                                   options:MTLResourceStorageModeShared];
            if (biases) {
                _bb = [_dev newBufferWithBytes:biases length:sbytes
                                       options:MTLResourceStorageModeShared];
            }
            if (!_bv || !_bs || (biases && !_bb)) return -1;
            _R = R; _C = C; _has_bias = (biases != NULL);
        }
        if (!_bx || _by) {
            _bx = [_dev newBufferWithLength:(size_t)C * sizeof(float)
                                    options:MTLResourceStorageModeShared];
            _by = [_dev newBufferWithLength:(size_t)R * sizeof(float)
                                    options:MTLResourceStorageModeShared];
            if (!_bx || _by) return -1;
        }
        memcpy(_bx.contents, x, (size_t)C * sizeof(float));

        id<MTLCommandBuffer> cb = [_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:_pso];
        [enc setBuffer:_bv offset:0 atIndex:0];
        [enc setBuffer:_bs offset:0 atIndex:1];
        [enc setBuffer:_bb offset:0 atIndex:2];
        [enc setBuffer:_bx offset:0 atIndex:3];
        [enc setBuffer:_by offset:0 atIndex:4];
        uint rv = (uint)R, cv = (uint)C;
        [enc setBytes:&rv length:sizeof(rv) atIndex:5];
        [enc setBytes:&cv length:sizeof(cv) atIndex:6];
        MTLSize threadsPerGroup = MTLSizeMake(256, 1, 1);
        MTLSize threadsPerGrid = MTLSizeMake((NSUInteger)R, 1, 1);
        [enc dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerGroup];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        memcpy(y, _by.contents, (size_t)R * sizeof(float));
    }
    return 0;
}

/* ---- batched expert matvec (the 13x design) --------------------- */
static const char *_batch_src2 =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void mlx4batch2(\n"
    "    device const uint   *vals    [[buffer(0)]],\n"
    "    device const ushort *scales  [[buffer(1)]],\n"
    "    device const ushort *biases  [[buffer(2)]],\n"
    "    device const float  *x       [[buffer(3)]],\n"
    "    device float        *ys      [[buffer(4)]],\n"
    "    device const uint4  *desc    [[buffer(5)]],  // voff, soff, xoff, yoff\n"
    "    constant uint &R             [[buffer(6)]],\n"
    "    constant uint &C             [[buffer(7)]],\n"
    "    constant uint &njobs         [[buffer(8)]],\n"
    "    uint idx                     [[thread_position_in_grid]])\n"
    "{\n"
    "    // grid = njobs * R; j = idx / R, r = idx % R\n"
    "    uint j = idx / R;\n"
    "    uint r = idx % R;\n"
    "    if (j >= njobs || r >= R) return;\n"
    "    // per-job base from the desc (arena appends in first-seen\n"
    "    // order across layers, so offsets are NOT j*stride)\n"
    "    uint4 d = desc[j];\n"
    "    device const uint *vr = vals + d.x;\n"
    "    device const ushort *sr = scales + d.y;\n"
    "    device const ushort *br = biases + d.y;\n"
    "    // x/y live in the caller's CPU buffers (shared memory): the\n"
    "    // kernel reads x at d.z and writes y at d.w directly -- no\n"
    "    // host-side pack/scatter copies.\n"
    "    device const float *xj = x + d.z;\n"
    "    device float *yj = ys + d.w;\n"
    "    float acc = 0.0f;\n"
    "    for (uint c = 0; c < C; c++) {\n"
    "        uint k = r * C + c;\n"
    "        uint g = k / 64u;\n"
    "        float s = as_type<float>(uint(sr[g]) << 16);\n"
    "        float b = as_type<float>(uint(br[g]) << 16);\n"
    "        uint q = (vr[k >> 3u] >> ((k & 7u) << 2u)) & 0xFu;\n"
    "        acc += (float(q) * s + b) * xj[c];\n"
    "    }\n"
    "    yj[r] = acc;\n"
    "}\n";

static id<MTLComputePipelineState> _bpso2;
static id<MTLCommandQueue> _bqueue2;

/* arena: one concatenated buffer per component, grown on demand.
 * tensors are cached per vals-pointer; the cache entry holds the
 * byte offsets into the arena. */
typedef struct { const void *key; size_t voff, soff; } TensorSlot;
#define BATCH_MAXSLOT 65536        /* 40 layers x 256 experts x 3 tensors */
#define BATCH_ARENA_V (512u << 20)   /* full 256-exp pool: ~453 MB */
#define BATCH_ARENA_S (64u << 20)    /* scales+biases: ~28 MB */

static TensorSlot _tslot[BATCH_MAXSLOT];
static int _tslot_n = 0;
static id<MTLBuffer> _arena_v, _arena_s, _arena_b;
static size_t _arena_v_used = 0, _arena_s_used = 0;
static id<MTLBuffer> _desc = nil;
static size_t _desc_len = 0;

static int batch_ensure_arena(size_t vneed, size_t sneed) {
    if (!_arena_v) {
        _arena_v = [_dev newBufferWithLength:BATCH_ARENA_V
                                    options:MTLResourceStorageModeShared];
        _arena_s = [_dev newBufferWithLength:BATCH_ARENA_S
                                    options:MTLResourceStorageModeShared];
        _arena_b = [_dev newBufferWithLength:BATCH_ARENA_S
                                    options:MTLResourceStorageModeShared];
        if (!_arena_v || !_arena_s || !_arena_b) return -1;
    }
    /* grow on demand: new buffer, copy, swap (Metal buffers are
     * immutable-sized; reallocate like realloc). */
    if (_arena_v_used + vneed > _arena_v.length) {
        size_t nlen = _arena_v.length * 2;
        while (nlen < _arena_v_used + vneed) nlen *= 2;
        id<MTLBuffer> nv = [_dev newBufferWithLength:nlen
                                            options:MTLResourceStorageModeShared];
        if (!nv) return -1;
        memcpy(nv.contents, _arena_v.contents, _arena_v_used);
        _arena_v = nv;
    }
    if (_arena_s_used + sneed > _arena_s.length) {
        size_t nlen = _arena_s.length * 2;
        while (nlen < _arena_s_used + sneed) nlen *= 2;
        id<MTLBuffer> ns = [_dev newBufferWithLength:nlen
                                            options:MTLResourceStorageModeShared];
        id<MTLBuffer> nb = [_dev newBufferWithLength:nlen
                                            options:MTLResourceStorageModeShared];
        if (!ns || !nb) return -1;
        memcpy(ns.contents, _arena_s.contents, _arena_s_used);
        memcpy(nb.contents, _arena_b.contents, _arena_s_used);
        _arena_s = ns; _arena_b = nb;
    }
    return 0;
}

/* find or cache a tensor by STABLE id (expert layout pointer). The
 * vals pointer alone is not stable (cache slots rotate per token).
 * Returns the slot index, -1 on failure. */
static int batch_slot(const uint32_t *vals, const uint16_t *scales,
                      const uint16_t *biases, const void *id, int R, int C) {
    for (int i = 0; i < _tslot_n; i++)
        if (_tslot[i].key == id) return i;
    if (_tslot_n >= BATCH_MAXSLOT) return -1;
    size_t vbytes = (size_t)R * (size_t)C / 2u;
    size_t sbytes = (size_t)R * (size_t)C / 64u * 2u;
    if (batch_ensure_arena(vbytes, sbytes) != 0) return -1;
    memcpy((char *)_arena_v.contents + _arena_v_used, vals, vbytes);
    memcpy((char *)_arena_s.contents + _arena_s_used, scales, sbytes);
    memcpy((char *)_arena_b.contents + _arena_s_used, biases, sbytes);
    _tslot[_tslot_n].key = id;
    _tslot[_tslot_n].voff = _arena_v_used;
    _tslot[_tslot_n].soff = _arena_s_used;
    _arena_v_used += vbytes;
    _arena_s_used += sbytes;
    return _tslot_n++;
}

int ds4f_gpu_mlx4_batch(const uint32_t *const *vals,
                        const uint16_t *const *scales,
                        const uint16_t *const *biases,
                        const float *const *xs, float *const *ys,
                        const void *const *ids,
                        int R, int C, int njobs) {
    if (getenv("DS4F_GPU_MS"))
        fprintf(stderr, "[gpu-ms] ENTER njobs=%d R=%d C=%d\n", njobs, R, C);
    if (!_dev || njobs < 1 || R < 1 || C < 1) return -1;
    if (!_bpso2 || !_bqueue2) {
        @autoreleasepool {
            NSError *err = nil;
            id<MTLLibrary> lib = [_dev newLibraryWithSource:
                [NSString stringWithUTF8String:_batch_src2] options:nil error:&err];
            if (!lib) return -1;
            _bpso2 = [_dev newComputePipelineStateWithFunction:
                      [lib newFunctionWithName:@"mlx4batch2"] error:&err];
            _bqueue2 = [_dev newCommandQueue];
            if (!_bpso2 || !_bqueue2) return -1;
        }
    }
    /* cache all tensors by stable id (first sight copies into the arena) */
    int slot[BATCH_MAXSLOT];
    size_t xbytes = (size_t)C * njobs * sizeof(float);
    size_t ybytes = (size_t)R * njobs * sizeof(float);
    @autoreleasepool {
        static id<MTLBuffer> _bx2 = nil, _by2 = nil;
        static size_t _bx2_len = 0, _by2_len = 0;
        for (int j = 0; j < njobs; j++) {
            slot[j] = batch_slot(vals[j], scales[j],
                                 biases ? biases[j] : NULL, ids[j], R, C);
            if (slot[j] < 0) return -1;
        }
        if (!_bx2 || _bx2_len < xbytes) {
            _bx2 = [_dev newBufferWithLength:xbytes
                                    options:MTLResourceStorageModeShared];
            if (!_bx2) return -1;
            _bx2_len = xbytes;
        }
        if (!_by2 || _by2_len < ybytes) {
            _by2 = [_dev newBufferWithLength:ybytes
                                    options:MTLResourceStorageModeShared];
            if (!_by2) return -1;
            _by2_len = ybytes;
        }
        if (!_desc || _desc_len < (size_t)njobs * 16) {
            _desc = [_dev newBufferWithLength:(size_t)njobs * 16
                                     options:MTLResourceStorageModeShared];
            if (!_desc) return -1;
            _desc_len = (size_t)njobs * 16;
        }
        /* pack per-job x at stride C -- but if every job shares the
         * same x pointer (gate/up all take the latent), copy ONCE and
         * give every job xoff=0: avoids the redundant per-job copy. */
        int xs_same = 1;
        for (int j = 1; j < njobs; j++)
            if (xs[j] != xs[0]) { xs_same = 0; break; }
        if (xs_same) {
            memcpy(_bx2.contents, xs[0], (size_t)C * sizeof(float));
        } else {
            for (int j = 0; j < njobs; j++)
                memcpy((char *)_bx2.contents + (size_t)j * C * sizeof(float),
                       xs[j], (size_t)C * sizeof(float));
        }

        /* desc: per-job voff/soff in ELEMENT units, xoff/yoff in FLOATS */
        typedef struct { unsigned voff, soff, xoff, yoff; } Desc4;
        Desc4 *desc = (Desc4 *)_desc.contents;
        for (int j = 0; j < njobs; j++) {
            desc[j].voff = (unsigned)(_tslot[slot[j]].voff / 4u);   /* uint32 */
            desc[j].soff = (unsigned)(_tslot[slot[j]].soff / 2u);   /* uint16 */
            desc[j].xoff = xs_same ? 0u : (unsigned)(j * C);
            desc[j].yoff = (unsigned)(j * R);
        }

        id<MTLCommandBuffer> cb = [_bqueue2 commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:_bpso2];
        [enc setBuffer:_arena_v offset:0 atIndex:0];
        [enc setBuffer:_arena_s offset:0 atIndex:1];
        [enc setBuffer:_arena_b offset:0 atIndex:2];
        [enc setBuffer:_bx2 offset:0 atIndex:3];
        [enc setBuffer:_by2 offset:0 atIndex:4];
        [enc setBuffer:_desc offset:0 atIndex:5];
        uint rv = (uint)R, cv = (uint)C, nj = (uint)njobs;
        [enc setBytes:&rv length:sizeof(rv) atIndex:6];
        [enc setBytes:&cv length:sizeof(cv) atIndex:7];
        [enc setBytes:&nj length:sizeof(nj) atIndex:8];
        MTLSize tpg = MTLSizeMake(256, 1, 1);
        MTLSize tpgrid = MTLSizeMake((NSUInteger)njobs * (NSUInteger)R, 1, 1);
        [enc dispatchThreads:tpgrid threadsPerThreadgroup:tpg];
        [enc endEncoding];
        struct timespec _ta, _tb;
        clock_gettime(CLOCK_MONOTONIC, &_ta);
        [cb commit];
        [cb waitUntilCompleted];
        clock_gettime(CLOCK_MONOTONIC, &_tb);
        if (getenv("DS4F_GPU_MS")) {
            double dt = (double)(_tb.tv_sec - _ta.tv_sec) +
                        (double)(_tb.tv_nsec - _ta.tv_nsec) * 1e-9;
            static double _tacc = 0; static long _tcnt = 0;
            _tcnt++; _tacc += dt;
            if (_tcnt <= 4 || _tcnt % 100 == 0)
                fprintf(stderr, "[gpu-ms] njobs=%d R=%d C=%d: %.3f ms (avg %.3f)\n",
                        njobs, R, C, dt * 1e3, _tacc / _tcnt * 1e3);
        }
        /* scatter: job j's rows live at [j*R, j*R+R) in _by2 */
        for (int j = 0; j < njobs; j++)
            memcpy(ys[j], (char *)_by2.contents + (size_t)j * R * sizeof(float),
                   (size_t)R * sizeof(float));
    }
    return 0;
}
