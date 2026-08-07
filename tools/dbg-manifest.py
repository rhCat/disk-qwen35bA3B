#!/usr/bin/env python3
import json
m = json.load(open('/tmp/q35-pool/manifest.json'))
t = m['tensors']
x = t[0]
print('gate keys:', list(x.keys()))
print('s_off:', x.get('s_off'), 's_nbytes:', x.get('s_nbytes'))
print('v_nbytes:', x.get('v_nbytes'), 'b_nbytes:', x.get('b_nbytes'))
print('expert_nbytes:', m.get('expert_nbytes'))
# slot 0: 24 + 0 = 24. v_off=24 -> rel_v=0. s_off?
