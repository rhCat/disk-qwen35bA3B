#!/usr/bin/env python3
# dbg-rank.py -- rank of a target token in the reference's lm_head logits.
import json, numpy as np, sys

TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 48017  # Jupiter

def bf16arr(bits):
    u = bits.astype(np.uint32) << 16
    return u.view(np.float32)

def decode(w, sc, bi):
    w = w.astype(np.uint32)
    sh = np.array([0, 4, 8, 12, 16, 20, 24, 28], dtype=np.uint32)
    nib = (w[..., None] >> sh) & 0xF
    R, P = w.shape
    vals = nib.reshape(R, P * 8)
    C = P * 8
    G = C // 64
    s = bf16arr(sc.astype(np.uint16)).reshape(R, G)
    b = bf16arr(bi.astype(np.uint16)).reshape(R, G)
    return vals.astype(np.float32) * np.repeat(s, 64, axis=1) + np.repeat(b, 64, axis=1)

TRUNK = '/tmp/q35-trunk'
h = json.load(open(TRUNK + '/head.json'))
hb = open(TRUNK + '/head.bin', 'rb').read()
w = np.frombuffer(hb, dtype=np.uint32, count=h['weight']['nbytes'] // 4,
                  offset=h['weight']['off'])
sc = np.frombuffer(hb, dtype=np.uint16, count=h['scale']['nbytes'] // 2,
                   offset=h['scale']['off'])
bi = np.frombuffer(hb, dtype=np.uint16, count=h['bias']['nbytes'] // 2,
                   offset=h['bias']['off'])

state = np.fromfile('/tmp/q35-ref-postnorm.bin', dtype=np.float32)
print('state rms %.4g' % np.sqrt((state**2).mean()))
W = decode(w.reshape(-1, 256), sc.reshape(-1, 32), bi.reshape(-1, 32))
print('head decoded [%d, %d]' % W.shape)
logits = W @ state
order = np.argsort(-logits)
rank = int(np.where(order == TARGET)[0][0]) if TARGET in order else -1
print('target %d rank: %d' % (TARGET, rank))
print('top10:', order[:10].tolist())
