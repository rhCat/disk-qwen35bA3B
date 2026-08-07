#!/usr/bin/env python3
# Reference L1 linear_attn: input = state after L0 (rms ~29.5).
import json, struct, math

def bf16f(h):
    return struct.unpack('<f', struct.pack('<I', h << 16))[0]

def decode_row(vals, scales, biases, row, C):
    out = []
    G = C // 64
    for i in range(C):
        u = vals[row * (C // 8) + i // 8]
        q = (u >> (4 * (i % 8))) & 0xF
        g = i // 64
        out.append((q - 8) * scales[row * G + g] + biases[row * G + g])
    return out

def matvec(tw, ts, tb, R, C, x):
    return [sum(d * x[i] for i, d in enumerate(decode_row(tw, ts, tb, r, C)))
            for r in range(R)]

t = json.load(open('/tmp/q35-trunk/trunk.json'))
tb = open('/tmp/q35-trunk/trunk.bin', 'rb').read()
offs = open('/tmp/q35-trunk/trunk.offsets', 'rb').read()
n = struct.unpack('<Q', offs[:8])[0]
o0 = struct.unpack('<%dQ' % n, offs[8:8+8*n])
tens = {}
for layer in t['layers']:
    if layer['layer'] in (0, 1):
        tens[layer['layer']] = {x['n'].split('layers.%d.' % layer['layer'])[1]: x
                                for x in layer['tensors']}

def ten(L, name):
    x = tens[L][name]
    base = o0[L] + x['off']
    if x['dtype'] == 'U32':
        return struct.unpack_from('<%dI' % (x['nbytes']//4), tb, base)
    if x['dtype'] == 'BF16':
        raw = struct.unpack_from('<%dH' % (x['nbytes']//2), tb, base)
        return [bf16f(v) for v in raw]
    if x['dtype'] == 'F32':
        return list(struct.unpack_from('<%df' % (x['nbytes']//4), tb, base))

def proj(L, name, R, x):
    w = ten(L, name + '.weight')
    s = ten(L, name + '.scales')
    b = ten(L, name + '.biases')
    return matvec(w, s, b, R, 2048, x)

def rms(v):
    return (sum(x*x for x in v)/len(v))**0.5

# L0 state: we measured [st] after L0 = 29.4758. Use it as L1 input.
state0 = None  # not available; simulate: embed -> norm -> L0 attn output
# instead: compute L1's input_layernorm of a rms-29.5 vector is the same
# as rms-1 after norm -- so the L1 attention output depends only on the
# DIRECTION of state0. Use a random unit-direction scaled to 29.5.
import random
random.seed(7)
dirv = [random.gauss(0, 1) for _ in range(2048)]
dr = rms(dirv)
state0 = [v/dr*29.4758 for v in dirv]

# L1: input_layernorm
nw1 = ten(1, 'input_layernorm.weight')
ss = sum(x*x for x in state0)
r = (ss/2048 + 1e-6) ** 0.5
xin1 = [state0[i]/r * nw1[i] for i in range(2048)]
print('L1 xin rms %.4f' % rms(xin1))

# L1 in_proj_z
z1 = proj(1, 'linear_attn.in_proj_z', 4096, xin1)
print('L1 z rms %.4f  (L0 was 15.38)' % rms(z1))
