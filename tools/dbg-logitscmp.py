#!/usr/bin/env python3
# dbg-logitscmp.py -- full logits comparison: engine-dumped state vs ref
# state (both PRE-final-norm), with the same head decode. Shows whether
# the head ordering matches when the states are aligned.
import json, numpy as np

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
W = decode(w.reshape(-1, 256), sc.reshape(-1, 32), bi.reshape(-1, 32))

eng = np.fromfile('/tmp/q35-eng-state.bin', dtype=np.float32)
ref = np.fromfile('/tmp/q35-ref-state.bin', dtype=np.float32)
print('eng rms %.4g ref rms %.4g' % (np.sqrt((eng**2).mean()), np.sqrt((ref**2).mean())))
c = np.dot(eng, ref) / (np.linalg.norm(eng) * np.linalg.norm(ref))
print('state cosine: %.6f' % c)

# final norm weight (BF16) from trunk layer 41
tl = json.load(open(TRUNK + '/trunk.json'))
fn = None
for x in tl['layers'][-1]['tensors']:
    if x['n'].endswith('.norm.weight'):
        base = 0  # offsets handled in ref; here read from trunk.json layer off
        fn = x
        break
# apply final norm: rmsnorm(x, w) then head
def rmsnorm(x, w):
    r = np.sqrt((x * x).mean(-1, keepdims=True) + 1e-6)
    y = x / r
    if w is not None:
        y = y * w
    return y

# read fn weight
import numpy as np
def load_fn():
    tl = json.load(open(TRUNK + '/trunk.json'))
    offs_all = np.fromfile(TRUNK + '/trunk.offsets', dtype=np.uint64)
    offs = offs_all[1::2]
    layer = tl['layers'][-1]
    tb = open(TRUNK + '/trunk.bin', 'rb').read()
    for x in layer['tensors']:
        if x['n'].endswith('.norm.weight'):
            a = np.frombuffer(tb, dtype=np.uint16, count=x['nbytes'] // 2,
                              offset=int(offs[layer['layer']]) + x['off'])
            return bf16arr(a.astype(np.uint16))
    return None

fnw = load_fn()
print('fnw[0..3]', fnw[:4])
le = W @ rmsnorm(eng, fnw).reshape(-1)
lr = W @ rmsnorm(ref, fnw).reshape(-1)
order_e = np.argsort(-le)
order_r = np.argsort(-lr)
print('eng top8:', order_e[:8].tolist())
print('ref top8:', order_r[:8].tolist())
jup = 48017
print('eng Jupiter rank:', int(np.where(order_e == jup)[0][0]) if jup in order_e else -1)
print('ref Jupiter rank:', int(np.where(order_r == jup)[0][0]) if jup in order_r else -1)
# cosine of logits
print('logits cosine: %.6f' % (np.dot(le, lr) / (np.linalg.norm(le) * np.linalg.norm(lr))))
