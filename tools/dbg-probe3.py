#!/usr/bin/env python3
import json, numpy as np

def bf16f(h):
    u = np.uint32(h) << 16
    return u.view(np.float32)

tl = json.load(open('/tmp/q35-trunk/trunk.json'))
offs_all = np.fromfile('/tmp/q35-trunk/trunk.offsets', dtype=np.uint64)
offs = offs_all[1:]
tb = open('/tmp/q35-trunk/trunk.bin', 'rb').read()

for Li in (0, 3):
    layer = tl['layers'][Li]
    base = int(offs[Li])
    target = 'linear_attn.in_proj_qkv' if Li == 0 else 'self_attn.q_proj'
    for x in layer['tensors']:
        n = x['n']
        if n.endswith(target + '.scales'):
            s = np.frombuffer(tb, dtype=np.uint16, count=x['nbytes']//2,
                              offset=base + x['off'])
            vals = [bf16f(int(v)) for v in s[:4]]
            print('L%d %s.scales: bits %s -> %s' % (Li, target,
                  s[:4].tolist(), vals))
        if n.endswith(target + '.weight'):
            w = np.frombuffer(tb, dtype=np.uint32, count=x['nbytes']//4,
                              offset=base + x['off'])
            print('L%d %s.weight: shape %s first words %s' % (Li, target,
                  x['shape'], [hex(int(v)) for v in w[:3]]))
