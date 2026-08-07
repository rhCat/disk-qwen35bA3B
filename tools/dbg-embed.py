#!/usr/bin/env python3
# Verify the engine's embed decode against the raw MLX 4-bit tensor.
import json, struct

def bf16f(h):
    return struct.unpack('<f', struct.pack('<I', h << 16))[0]

# embed.json from the trunk build
e = json.load(open('/tmp/q35-trunk/embed.json'))
print('embed.json keys:', list(e.keys()))
print('weight:', e.get('weight'))
print('scale dtype:', e.get('scale', {}).get('dtype'), 'shape', e.get('scale', {}).get('shape'))
print('bias dtype:', e.get('bias', {}).get('dtype'), 'shape', e.get('bias', {}).get('shape'))

# decode token 9419 (Hello) row from embed.bin
binpath = '/tmp/q35-trunk/embed.bin'
w = e['weight']; off = w['off']; nb = w['nbytes']
data = open(binpath, 'rb').read()
vals = struct.unpack_from('<%dI' % (nb//4), data, off)
sc = e.get('scale', {}); bi = e.get('bias', {})
s_off, b_off = sc['off'], bi['off']
s_nb, b_nb = sc['nbytes'], bi['nbytes']
scales = struct.unpack_from('<%dH' % (s_nb//2), data, s_off)
biases = struct.unpack_from('<%dH' % (b_nb//2), data, b_off)

# row 9419: 256 elements = 32 U32 words, 4 groups of 64
row = 9419
wrow = row * 32
srow = row * 4
decoded = []
for i in range(256):
    u = vals[wrow + i//8]
    q = (u >> (4 * (i % 8))) & 0xF
    g = i // 64
    v = (q - 8) * bf16f(scales[srow + g]) + bf16f(biases[srow + g])
    decoded.append(v)
rms = (sum(x*x for x in decoded)/256)**0.5
print('decoded row 9419 rms: %.6f  (expect ~1 for real embeddings)' % rms)
print('first 8:', ['%.5f' % x for x in decoded[:8]])
print('scale[0..3]:', ['%.6f' % bf16f(scales[srow+g]) for g in range(4)])
print('bias[0..3]:', ['%.6f' % bf16f(biases[srow+g]) for g in range(4)])
