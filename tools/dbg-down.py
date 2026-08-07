#!/usr/bin/env python3
# Decode expert-0 down_proj from the pool with a unit chain input;
# check output magnitude (should be ~1x, not 53x).
import json, struct

def bf16f(h):
    return struct.unpack('<f', struct.pack('<I', h << 16))[0]

m = json.load(open('/tmp/q35-pool/manifest.json'))
down = next(x for x in m['tensors'] if x['name'] == 'down_proj' and x['expert'] == 0)
print('down: shape', down['shape'], 'v_off', down['v_off'], 's_off', down['s_off'],
      'b_off', down['b_off'], 'v_nbytes', down['v_nbytes'])
slot = open('/tmp/q35-pool/pool.bin', 'rb').read()[24:24+1769472]
R, C = down['shape']            # [2048, 512]
vals = struct.unpack_from('<%dI' % (down['v_nbytes']//4), slot, down['v_off'] - 24)
sn = down['s_nbytes']//2
scales = struct.unpack_from('<%dH' % sn, slot, down['s_off'] - 24)
bn = down['b_nbytes']//2
biases = struct.unpack_from('<%dH' % bn, slot, down['b_off'] - 24)
# unit chain input (rms 1, random-ish)
import random
random.seed(3)
chain = [random.gauss(0, 1) for _ in range(C)]
cr = (sum(x*x for x in chain)/C)**0.5
chain = [x/cr for x in chain]
# decode row 0..few
G = C // 64
y = []
for r in range(R):
    acc = 0.0
    for i in range(C):
        u = vals[r * (C//8) + i//8]
        q = (u >> (4 * (i % 8))) & 0xF
        g = i // 64
        v = (q - 8) * bf16f(scales[r * G + g]) + bf16f(biases[r * G + g])
        acc += v * chain[i]
    y.append(acc)
yr = (sum(v*v for v in y)/R)**0.5
print('down(unit chain) output rms: %.6f  (expect ~1-5)' % yr)
print('scale range row0:', min(bf16f(scales[g]) for g in range(G)),
      max(bf16f(scales[g]) for g in range(G)))
