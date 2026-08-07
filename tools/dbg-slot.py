#!/usr/bin/env python3
import json, numpy as np
p = json.load(open('/tmp/q35-pool/manifest.json'))
pb = open('/tmp/q35-pool/pool.bin', 'rb').read()
# find the FIRST tensor of each expert 0 and 57 and compare bytes
def first_words(L, e):
    for t in p['tensors']:
        if t['layer'] == L and t['expert'] == e and t['name'] == 'gate_proj':
            return t['v_off']
# gate v_off for expert 57
for t in p['tensors']:
    if t['layer'] == 0 and t['expert'] == 57 and t['name'] == 'gate_proj':
        m57 = t
    if t['layer'] == 0 and t['expert'] == 106 and t['name'] == 'gate_proj':
        m106 = t
w57 = np.frombuffer(pb, dtype=np.uint32, count=8, offset=m57['v_off'])
w106 = np.frombuffer(pb, dtype=np.uint32, count=8, offset=m106['v_off'])
print('expert 57  w0..7:', ['%08x' % v for v in w57])
print('expert 106 w0..7:', ['%08x' % v for v in w106])
print('57 v_off', m57['v_off'], '106 v_off', m106['v_off'], 'delta', m106['v_off'] - m57['v_off'])
print('expert_nbytes', p.get('expert_nbytes'))
# what would the engine's slot-relative view be? slot = expert base (gate rel_v=0)
# engine slot for expert e = pool.bin + e * expert_nbytes? check:
for e in (57, 106):
    for t in p['tensors']:
        if t['layer'] == 0 and t['expert'] == e and t['name'] == 'gate_proj':
            base = t['v_off']
            break
    print('expert', e, 'gate v_off == e*expert_nbytes?', base, 'vs', e * p['expert_nbytes'], base == e * p['expert_nbytes'])
