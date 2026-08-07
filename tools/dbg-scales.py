#!/usr/bin/env python3
import json, numpy as np
gu = np.fromfile('/tmp/q35-eng-gateup.bin', dtype=np.int32)
EID = int(gu[0])
p = json.load(open('/tmp/q35-pool/manifest.json'))
pb = open('/tmp/q35-pool/pool.bin', 'rb').read()
for t in p['tensors']:
    if t['layer'] == 0 and t['expert'] == EID and t['name'] == 'gate_proj':
        m = t
        break
s = np.frombuffer(pb, dtype=np.uint16, count=8, offset=m['s_off'])
b = np.frombuffer(pb, dtype=np.uint16, count=8, offset=m['b_off'])
print('expert', EID)
print('manifest s_off', m['s_off'], 'b_off', m['b_off'])
print('ref s0..7:', ['%04x' % v for v in s])
print('ref b0..7:', ['%04x' % v for v in b])
print('ref s as float:', ['%.4g' % float(np.array([v], dtype=np.uint16).view(np.float32)[0] << 0) for v in s[:4]])
