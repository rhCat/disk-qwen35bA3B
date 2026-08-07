#!/usr/bin/env python3
# Check the trained norm weight values: are they ~0 (bias convention)
# or ~1 (multiplier convention)?
import json, struct

def bf16f(h):
    return struct.unpack('<f', struct.pack('<I', h << 16))[0]

t = json.load(open('/tmp/q35-trunk/trunk.json'))
tb = open('/tmp/q35-trunk/trunk.bin', 'rb').read()
offs = open('/tmp/q35-trunk/trunk.offsets', 'rb').read()
n = struct.unpack('<Q', offs[:8])[0]
o0 = struct.unpack('<%dQ' % n, offs[8:8+8*n])
for layer in t['layers'][:1]:
    for x in layer['tensors']:
        if 'layernorm' in x['n'] and 'weight' in x['n']:
            raw = struct.unpack_from('<%dH' % (x['nbytes']//2), tb, o0[layer['layer']] + x['off'])
            vals = [bf16f(v) for v in raw[:10]]
            print(x['n'].split('layers.0.')[1], 'first10:', ['%.4f' % v for v in vals])
        if 'linear_attn.norm.weight' in x['n']:
            raw = struct.unpack_from('<%dH' % (x['nbytes']//2), tb, o0[layer['layer']] + x['off'])
            vals = [bf16f(v) for v in raw[:10]]
            print('linear_attn.norm.weight first10:', ['%.4f' % v for v in vals])
