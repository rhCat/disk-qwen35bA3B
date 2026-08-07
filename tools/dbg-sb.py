#!/usr/bin/env python3
import json, numpy as np
p = json.load(open('/tmp/q35-pool/manifest.json'))
pb = open('/tmp/q35-pool/pool.bin', 'rb').read()
for t in p['tensors']:
    if t['layer'] == 0 and t['expert'] == 43 and t['name'] == 'gate_proj':
        m = t
        break
s = np.frombuffer(pb, dtype=np.uint16, count=4, offset=m['s_off'])
print('ref exp43 gate s0..3:', ['%04x' % v for v in s])
b = np.frombuffer(pb, dtype=np.uint16, count=4, offset=m['b_off'])
print('ref exp43 gate b0..3:', ['%04x' % v for v in b])
print('s_off', m['s_off'], 'b_off', m['b_off'], 's-b delta',
      m['b_off'] - m['s_off'])
