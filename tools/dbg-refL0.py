#!/usr/bin/env python3
# Full reference L0 linear_attn forward for 1 token, using real weights.
# Compares against the engine's [st] L0 output.
import json, struct, math

def bf16f(h):
    return struct.unpack('<f', struct.pack('<I', h << 16))[0]

def decode_row(vals, scales, biases, row, C):
    # mlx4 row decode; scales/biases are already float lists
    out = []
    G = C // 64
    for i in range(C):
        u = vals[row * (C // 8) + i // 8]
        q = (u >> (4 * (i % 8))) & 0xF
        g = i // 64
        out.append((q - 8) * scales[row * G + g] + biases[row * G + g])
    return out

def matvec(tw, ts, tb, R, C, x):
    y = []
    for r in range(R):
        rowv = decode_row(tw, ts, tb, r, C)
        acc = 0.0
        for i in range(C):
            acc += rowv[i] * x[i]
        y.append(acc)
    return y

# load trunk
t = json.load(open('/tmp/q35-trunk/trunk.json'))
tb = open('/tmp/q35-trunk/trunk.bin', 'rb').read()
offs = open('/tmp/q35-trunk/trunk.offsets', 'rb').read()
n = struct.unpack('<Q', offs[:8])[0]
o0 = struct.unpack('<%dQ' % n, offs[8:8+8*n])
lay0 = o0[0]
tens = {x['n'].split('layers.0.')[1]: x for x in t['layers'][0]['tensors']}

def ten(name):
    x = tens[name]
    base = lay0 + x['off']
    if x['dtype'] == 'U32':
        return struct.unpack_from('<%dI' % (x['nbytes']//4), tb, base)
    if x['dtype'] == 'BF16':
        raw = struct.unpack_from('<%dH' % (x['nbytes']//2), tb, base)
        return [bf16f(v) for v in raw]
    if x['dtype'] == 'F32':
        return list(struct.unpack_from('<%df' % (x['nbytes']//4), tb, base))
    return None

# embed 9419 (2048 decoded)
e = json.load(open('/tmp/q35-trunk/embed.json'))
data = open('/tmp/q35-trunk/embed.bin', 'rb').read()
w = e['weight']
vals = struct.unpack_from('<%dI' % (w['nbytes']//4), data, w['off'])
sc = e['scale']; bi = e['bias']
scales = struct.unpack_from('<%dH' % (sc['nbytes']//2), data, sc['off'])
biases = struct.unpack_from('<%dH' % (bi['nbytes']//2), data, bi['off'])
H = 2048
row = 9419
emb = []
for i in range(H):
    u = vals[row * (H//8) + i//8]
    q = (u >> (4 * (i % 8))) & 0xF
    g = i // 64
    emb.append((q - 8) * bf16f(scales[row * (H//64) + g]) + bf16f(biases[row * (H//64) + g]))

# input_layernorm
nw = ten('input_layernorm.weight')
ss = sum(x*x for x in emb)
r = (ss/H + 1e-6) ** 0.5
xin = [emb[i]/r * nw[i] for i in range(H)]

# in_proj_qkv: [8192, 256] U32 -> C=2048; scales [8192,32] BF16; biases [8192,32]
def proj(name, R):
    w = ten(name + '.weight')
    s = ten(name + '.scales')
    b = ten(name + '.biases')
    return matvec(w, s, b, R, 2048, xin)

qkv = proj('linear_attn.in_proj_qkv', 8192)
z = proj('linear_attn.in_proj_z', 4096)
a = proj('linear_attn.in_proj_a', 32)
b = proj('linear_attn.in_proj_b', 32)
print('qkv rms %.4f  z rms %.4f  a[0..2] %.3f %.3f %.3f  b[0..2] %.3f %.3f %.3f' % (
    (sum(v*v for v in qkv)/8192)**0.5, (sum(v*v for v in z)/4096)**0.5,
    a[0], a[1], a[2], b[0], b[1], b[2]))
