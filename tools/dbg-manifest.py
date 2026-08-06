#!/usr/bin/env python3
import json, sys
from collections import Counter
m = json.load(open('/tmp/q35-pool/manifest.json'))
print('keys:', list(m.keys()))
print('n_layers', m['n_layers'], 'n_experts', m['n_experts'],
      'expert_nbytes', m['expert_nbytes'])
ts = m['tensors']
print('tensor entries:', len(ts))
t0 = ts[0]
print('first:', {k: t0[k] for k in
      ('layer', 'expert', 'fmt', 'shape', 'name', 'v_off', 'v_nbytes',
       's_off', 's_nbytes', 'b_off', 'b_nbytes')})
c = Counter((t['layer'], t['expert']) for t in ts)
print('distinct (L,E):', len(c), 'per-expert tensors:', set(c.values()))
