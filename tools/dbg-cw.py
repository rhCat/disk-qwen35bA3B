#!/usr/bin/env python3
import json, numpy as np

def bf16arr(bits):
    u = bits.astype(np.uint32) << 16
    return u.view(np.float32)

TRUNK = '/tmp/q35-trunk'
tl = json.load(open(TRUNK + '/trunk.json'))
offs_all = np.fromfile(TRUNK + '/trunk.offsets', dtype=np.uint64)
offs = offs_all[1::2]
tb = open(TRUNK + '/trunk.bin', 'rb').read()
L0 = tl['layers'][0]
for x in L0['tensors']:
    if x['n'].endswith('.linear_attn.conv1d.weight'):
        a = np.frombuffer(tb, dtype=np.uint16, count=x['nbytes']//2,
                          offset=int(offs[0]) + x['off'])
        f = bf16arr(a.astype(np.uint16))
        print('ref conv weight shape', x['shape'])
        print('ref cw[0..7]:', f.reshape(-1)[:8].tolist())
