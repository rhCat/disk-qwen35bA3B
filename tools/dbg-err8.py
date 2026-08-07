#!/usr/bin/env python3
# Quantify the systematic error: |8*scale| vs the weight magnitudes.
import json, struct

def bf16f(h):
    return struct.unpack('<f', struct.pack('<I', h << 16))[0]

e = json.load(open('/tmp/q35-trunk/embed.json'))
data = open('/tmp/q35-trunk/embed.bin', 'rb').read()
sc = e['scale']; bi = e['bias']
scales = struct.unpack_from('<%dH' % (sc['nbytes']//2), data, sc['off'])
biases = struct.unpack_from('<%dH' % (bi['nbytes']//2), data, bi['off'])

# embed error: for row 9419, the (q-8) vs raw difference = -8*scale per group
row = 9419
G = 2048 // 64
errs = [8*abs(bf16f(scales[row*G+g])) for g in range(G)]
sig = [abs(bf16f(biases[row*G+g])) for g in range(G)]
print('embed row 9419: |8*scale| mean %.6f max %.6f' % (sum(errs)/len(errs), max(errs)))
print('                |bias|    mean %.6f max %.6f' % (sum(sig)/len(sig), max(sig)))
print('=> the (q-8) error is comparable to the bias magnitude itself')

# do the same for a projection: in_proj_qkv scales (layer 0)
t = json.load(open('/tmp/q35-trunk/trunk.json'))
tb = open('/tmp/q35-trunk/trunk.bin', 'rb').read()
offs = open('/tmp/q35-trunk/trunk.offsets', 'rb').read()
n = struct.unpack('<Q', offs[:8])[0]
o0 = struct.unpack('<%dQ' % n, offs[8:8+8*n])
lay0 = o0[0]
for x in t['layers'][0]['tensors']:
    if x['n'].endswith('in_proj_qkv.scales'):
        sraw = struct.unpack_from('<%dH' % (x['nbytes']//2), tb, lay0 + x['off'])
        svals = [bf16f(v) for v in sraw[:512]]
        err8 = [8*abs(v) for v in svals]
        print('in_proj_qkv scales: |8*scale| mean %.6f max %.6f' % (sum(err8)/len(err8), max(err8)))
    if x['n'].endswith('in_proj_qkv.biases'):
        braw = struct.unpack_from('<%dH' % (x['nbytes']//2), tb, lay0 + x['off'])
        bvals = [bf16f(v) for v in braw[:512]]
        print('in_proj_qkv biases:  |bias|    mean %.6f max %.6f' % (sum(abs(v) for v in bvals)/len(bvals), max(abs(v) for v in bvals)))
