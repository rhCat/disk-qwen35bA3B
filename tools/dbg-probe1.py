#!/usr/bin/env python3
import json, numpy as np

def bf16arr(bits):
    u = bits.astype(np.uint32) << 16
    return u.view(np.float32)

TRUNK = '/tmp/q35-trunk'
tl = json.load(open(TRUNK + '/trunk.json'))
offs_all = np.fromfile(TRUNK + '/trunk.offsets', dtype=np.uint64)
offs = offs_all[1:]
tb = open(TRUNK + '/trunk.bin', 'rb').read()

layer0 = tl['layers'][0]
base = int(offs[0])
for x in layer0['tensors']:
    n = x['n']
    if n.endswith('input_layernorm.weight'):
        a = np.frombuffer(tb, dtype=np.uint16, count=x['nbytes'] // 2,
                          offset=base + x['off'])
        f = bf16arr(a.astype(np.uint16))
        print('input_layernorm:', a.shape, 'first bf16 bits', a[:4].tolist(),
              '-> floats', f[:4].tolist(), 'nan?', bool(np.isnan(f).any()))
    if n.endswith('linear_attn.in_proj_qkv.weight'):
        w = np.frombuffer(tb, dtype=np.uint32, count=x['nbytes'] // 4,
                          offset=base + x['off'])
        print('qkv weight:', x['shape'], w.shape, 'first word', hex(int(w[0])))
    if n.endswith('linear_attn.in_proj_qkv.scales'):
        s = np.frombuffer(tb, dtype=np.uint16, count=x['nbytes'] // 2,
                          offset=base + x['off'])
        f = bf16arr(s.astype(np.uint16))
        print('qkv scales:', s.shape, 'first', f[:3].tolist(), 'nan?',
              bool(np.isnan(f).any()))
