#!/usr/bin/env python3
# Ground-truth: does MLX's affine dequant use (q-8)*s+b or q*s+b?
# If mlx is installed, quantize a random tensor and dequantize both ways.
import sys
try:
    import mlx.core as mx
    import numpy as np
    HAVE_MLX = True
except ImportError:
    HAVE_MLX = False

if not HAVE_MLX:
    print("mlx not importable here")
    sys.exit(0)

# quantize a random BF16-ish tensor with group_size 64, 4 bits
rng = np.random.RandomState(42)
w = rng.randn(128, 2048).astype(np.float32) * 0.05
mw = mx.array(w)
q, scales, biases = mx.quantize(mw, group_size=64, bits=4)
wq = q.astype(mx.uint32)
print("quantized shape:", wq.shape, "scales:", scales.shape, "biases:", biases.shape)
sc = np.array(scales).astype(np.float32)
bi = np.array(biases).astype(np.float32)
qw = np.array(wq).astype(np.uint32)

# MLX's own dequant
dq = mx.dequantize(q, scales, biases, group_size=64, bits=4)
ref = np.array(dq).astype(np.float32)

# our formula: (q-8)*s+b, row-major nibbles
G = 64
rows, C = w.shape
dec_ours = np.zeros_like(ref)
dec_raw = np.zeros_like(ref)
for r in range(rows):
    for i in range(C):
        u = qw[r, i // 8] if C % 8 == 0 else 0
        d = (u >> (4 * (i % 8))) & 0xF
        g = i // G
        s = sc[r, g]
        b = bi[r, g]
        dec_ours[r, i] = (d - 8) * s + b
        dec_raw[r, i] = d * s + b

err_ours = np.abs(dec_ours - ref).mean()
err_raw = np.abs(dec_raw - ref).mean()
print("MLX dequant vs (q-8)*s+b : mean abs err %.6f" % err_ours)
print("MLX dequant vs q*s+b     : mean abs err %.6f" % err_raw)
print("CONVENTION:", "q*s+b" if err_raw < err_ours else "(q-8)*s+b")
