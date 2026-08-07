#!/usr/bin/env python3
# dbg-rope.py -- what does mx.fast.rope(traditional=False) actually do?
# Compare against interleaved (2i,2i+1) and half-split (d, d+rd/2).
import numpy as np
import mlx.core as mx

rd = 64
theta = 1e7
pos = 1

x = np.arange(1, 9, dtype=np.float32)  # 8-dim probe: [1..8]
# pad to 64
xp = np.zeros(64, dtype=np.float32)
xp[:8] = x
y = mx.fast.rope(mx.array(xp[None, None, :]), rd, traditional=False,
                 base=theta, scale=1.0, offset=pos)
mlx_out = np.array(y)[0, 0]

# interleaved pairs (2i, 2i+1)
out_i = xp.copy()
for i in range(rd // 2):
    ang = pos / theta ** (2.0 * i / rd)
    c, s = np.cos(ang), np.sin(ang)
    a, b = out_i[2 * i], out_i[2 * i + 1]
    out_i[2 * i] = a * c - b * s
    out_i[2 * i + 1] = a * s + b * c

# half-split pairs (d, d + rd/2)
out_h = xp.copy()
for d in range(rd // 2):
    ang = pos / theta ** (2.0 * d / rd)
    c, s = np.cos(ang), np.sin(ang)
    a, b = out_h[d], out_h[d + rd // 2]
    out_h[d] = a * c - b * s
    out_h[d + rd // 2] = a * s + b * c

print('mlx   [0..7]', np.round(mlx_out[:8], 6))
print('inter [0..7]', np.round(out_i[:8], 6))
print('half  [0..7]', np.round(out_h[:8], 6))
print('match interleaved:', np.allclose(mlx_out, out_i, atol=1e-5))
print('match half-split :', np.allclose(mlx_out, out_h, atol=1e-5))
