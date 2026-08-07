#!/usr/bin/env python3
import json
tl = json.load(open('/tmp/q35-trunk/trunk.json'))
for layer in tl['layers'][:1]:
    for x in layer['tensors']:
        n = x['n']
        if 'norm.weight' in n or 'A_log' in n or 'dt_bias' in n or 'conv1d' in n:
            print('L0', n.split('layers.0.')[1], x['dtype'], x['shape'])
# check all layers for F32 tensors
f32 = set()
for layer in tl['layers']:
    for x in layer['tensors']:
        if x['dtype'] == 'F32':
            f32.add(x['n'].split('layers.')[1])
for n in sorted(f32)[:10]:
    print('F32:', n)
