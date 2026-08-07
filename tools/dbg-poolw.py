#!/usr/bin/env python3
import json, numpy as np
p = json.load(open('/tmp/q35-pool/manifest.json'))
pb = open('/tmp/q35-pool/pool.bin', 'rb').read()
for t in p['tensors']:
    if t['layer'] == 0 and t['expert'] == 106 and t['name'] == 'gate_proj':
        m = t
        break
print('manifest v_off', m['v_off'], 's_off', m['s_off'], 'shape', m['shape'])
w = np.frombuffer(pb, dtype=np.uint32, count=8, offset=m['v_off'])
s = np.frombuffer(pb, dtype=np.uint16, count=8, offset=m['s_off'])
print('ref w0..7:', ['%08x' % v for v in w])
print('ref s0..7:', ['%04x' % v for v in s])
print('pool.bin size', len(pb))
print('expert_nbytes', p.get('expert_nbytes'))
