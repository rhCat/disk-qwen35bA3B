#!/usr/bin/env python3
import json, numpy as np

def bf16arr(bits):
    u = bits.astype(np.uint32) << 16
    return u.view(np.float32)

def decode(w, sc, bi):
    w = w.astype(np.uint32)
    sh = np.array([0,4,8,12,16,20,24,28], dtype=np.uint32)
    nib = (w[..., None] >> sh) & 0xF
    R, P = w.shape
    vals = nib.reshape(R, P * 8)
    C = P * 8
    G = C // 64
    s = bf16arr(sc.astype(np.uint16)).reshape(R, G)
    b = bf16arr(bi.astype(np.uint16)).reshape(R, G)
    sbig = np.repeat(s, 64, axis=1)
    bbig = np.repeat(b, 64, axis=1)
    return vals.astype(np.float32) * sbig + bbig

tl = json.load(open('/tmp/q35-trunk/trunk.json'))
offs_all = np.fromfile('/tmp/q35-trunk/trunk.offsets', dtype=np.uint64)
offs = offs_all[1:]
tb = open('/tmp/q35-trunk/trunk.bin', 'rb').read()

L3 = tl['layers'][3]
base = int(offs[3])
for suffix in ['self_attn.q_proj', 'self_attn.k_proj']:
    for x in L3['tensors']:
        if x['n'].endswith(suffix + '.weight'):
            w = np.frombuffer(tb, dtype=np.uint32, count=x['nbytes']//4,
                              offset=base + x['off']).reshape(x['shape'])
        elif x['n'].endswith(suffix + '.scales'):
            s = np.frombuffer(tb, dtype=np.uint16, count=x['nbytes']//2,
                              offset=base + x['off']).reshape(x['shape'])
        elif x['n'].endswith(suffix + '.biases'):
            b = np.frombuffer(tb, dtype=np.uint16, count=x['nbytes']//2,
                              offset=base + x['off']).reshape(x['shape'])
    W = decode(w, s, b)
    print(suffix, 'decoded', W.shape, 'nan?', bool(np.isnan(W).any()),
          'inf?', bool(np.isinf(W).any()), 'mean %.4g' % float(W.mean()),
          'rms %.4g' % float(np.sqrt((W**2).mean())))
