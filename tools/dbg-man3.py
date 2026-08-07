#!/usr/bin/env python3
import json
m = json.load(open('/tmp/q35-pool/manifest.json'))
for i, t in enumerate(m['tensors'][:6]):
    print(i, 'L', t['layer'], 'e', t['expert'], t['name'],
          'v_off', t['v_off'], 'v_nb', t['v_nbytes'])
# is v_off record-relative or absolute? expert 1 gate:
for t in m['tensors']:
    if t['layer'] == 0 and t['expert'] == 1 and t['name'] == 'gate_proj':
        print('expert1 gate v_off:', t['v_off'])
        break
print('expert_nbytes:', m['expert_nbytes'])
