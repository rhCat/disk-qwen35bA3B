#!/usr/bin/env python3
import json
tl = json.load(open('/tmp/q35-trunk/trunk.json'))
for layer in tl['layers'][:4]:
    for x in layer['tensors']:
        n = x['n']
        if 'self_attn' in n and ('q_proj' in n or 'k_proj' in n):
            print(layer['layer'], n.split('layers.')[1], x['dtype'], x['shape'])
            break
    break
# list ALL names of layer 3
L3 = tl['layers'][3]
for x in L3['tensors']:
    print('  ', x['n'].split('layers.3.')[1], x['dtype'], x['shape'])
