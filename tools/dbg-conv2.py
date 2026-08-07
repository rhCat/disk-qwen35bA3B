#!/usr/bin/env python3
# Which dequant convention does the FILE use? For a quantized tensor,
# if the quantizer used centered q (dequant (q-8)*s+b), then the stored
# bias equals min + 8*scale (so that q=0 -> min). If raw q (dequant
# q*s+b), bias = min. Check consistency across groups of a real tensor.
import json, struct

def bf16f(h):
    return struct.unpack('<f', struct.pack('<I', h << 16))[0]

e = json.load(open('/tmp/q35-trunk/embed.json'))
data = open('/tmp/q35-trunk/embed.bin', 'rb').read()
sc = e['scale']; bi = e['bias']
scales = struct.unpack_from('<%dH' % (sc['nbytes']//2), data, sc['off'])
biases = struct.unpack_from('<%dH' % (bi['nbytes']//2), data, bi['off'])

# decode row 9419 both ways, compare plausibility of the value ranges
Hdec = 2048
G = Hdec // 64
row = 9419
vals = struct.unpack_from('<%dI' % (e['weight']['nbytes']//4), data, e['weight']['off'])
# q distribution in row
qs = []
for i in range(Hdec):
    u = vals[row * (Hdec//8) + i//8]
    qs.append((u >> (4 * (i % 8))) & 0xF)
print('row %d q: min %d max %d mean %.2f' % (row, min(qs), max(qs), sum(qs)/len(qs)))
# if the quantizer centered: q should span roughly 0..15 with bias = min+8s
# check: reconstructed min/max under each convention
mins, maxs = [], []
for g in range(G):
    s = bf16f(scales[row*G+g]); b = bf16f(biases[row*G+g])
    # raw convention range: [b, b+15s]
    mins.append(b); maxs.append(b + 15*s)
    # centered: min = b - 8s
print('raw-convention (q*s+b): row min %.4f max %.4f' % (min(mins), max(maxs)))
cmins = [bf16f(biases[row*G+g]) - 8*bf16f(scales[row*G+g]) for g in range(G)]
cmaxs = [bf16f(biases[row*G+g]) + 7*bf16f(scales[row*G+g]) for g in range(G)]
print('centered-convention ((q-8)*s+b): row min %.4f max %.4f' % (min(cmins), max(cmaxs)))
print('real embeddings are typically centered near 0 with both signs')
