#!/usr/bin/env python3
import json, struct
t = json.load(open('/tmp/q35-trunk/trunk.json'))
# rebuild per-layer offsets from trunk.offsets
offs = open('/tmp/q35-trunk/trunk.offsets', 'rb').read()
n = struct.unpack('<Q', offs[:8])[0]
o0 = struct.unpack('<%dQ' % n, offs[8:8+8*n])
s0 = struct.unpack('<%dQ' % n, offs[8+8*n:8+16*n])
tb = open('/tmp/q35-trunk/trunk.bin', 'rb').read()
for L in (0, 28, 39):
    layer = next(x for x in t['layers'] if x['layer'] == L)
    base = o0[L]
    for x in layer['tensors']:
        if x['n'].endswith('A_log'):
            vals = struct.unpack_from('<32f', tb, base + x['off'])
            print('L%d A_log: min %.4f max %.4f mean %.4f first4 [%.4f %.4f %.4f %.4f]'
                  % (L, min(vals), max(vals), sum(vals)/32, vals[0], vals[1], vals[2], vals[3]))
        if x['n'].endswith('dt_bias'):
            raw = struct.unpack_from('<32H', tb, base + x['off'])
            vals = [struct.unpack('<f', struct.pack('<I', v << 16))[0] for v in raw]
            print('L%d dt_bias: min %.4f max %.4f first4 [%.4f %.4f %.4f %.4f]'
                  % (L, min(vals), max(vals), vals[0], vals[1], vals[2], vals[3]))
