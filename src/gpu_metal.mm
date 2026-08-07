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
            if (!_bx || !_by) return -1;
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
