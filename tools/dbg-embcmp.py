#!/usr/bin/env python3
# dbg-embcmp.py -- elementwise: engine embed (token 5) vs reference embed_row(5).
import json, numpy as np

TRUNK = '/tmp/q35-trunk'

def bf16arr(bits):
    u = bits.astype(np.uint32) << 16
    return u.view(np.float32)

def decode(w, sc, bi):
    """mlx4 row-major decode: w [R, P] U32 (P=C/8), sc/bi [R, C/64] BF16.
    val = q*scale + bias (q in [0,15])."""
    w = w.astype(np.uint32)
    sh = np.array([0, 4, 8, 12, 16, 20, 24, 28], dtype=np.uint32)
    nib = (w[..., None] >> sh) & 0xF           # [R, P, 8]
    R, P = w.shape
    vals = nib.reshape(R, P * 8)               # [R, C]
    C = P * 8
    G = C // 64
    s = bf16arr(sc.astype(np.uint16)).reshape(R, G)
    b = bf16arr(bi.astype(np.uint16)).reshape(R, G)
    sbig = np.repeat(s, 64, axis=1)
    bbig = np.repeat(b, 64, axis=1)
    return vals.astype(np.float32) * sbig + bbig

tl = json.load(open(TRUNK + '/trunk.json'))
e = json.load(open(TRUNK + '/embed.json'))
eb = open(TRUNK + '/embed.bin', 'rb').read()
tok = 760
w = np.frombuffer(eb, dtype=np.uint32, count=e['weight']['nbytes'] // 4,
                  offset=e['weight']['off'])
sc = np.frombuffer(eb, dtype=np.uint16, count=e['scale']['nbytes'] // 2,
                   offset=e['scale']['off'])
bi = np.frombuffer(eb, dtype=np.uint16, count=e['bias']['nbytes'] // 2,
                   offset=e['bias']['off'])
row = w[tok * (2048 // 8):(tok + 1) * (2048 // 8)]
sc_r = sc[tok * 32:(tok + 1) * 32]
bi_r = bi[tok * 32:(tok + 1) * 32]
ref = decode(row[None, :], sc_r[None, :], bi_r[None, :])[0]

eng = np.fromfile('/tmp/q35-eng-embed.bin', dtype=np.float32)
print('ref rms %.6g  eng rms %.6g  len %d/%d' % (
    np.sqrt((ref**2).mean()), np.sqrt((eng**2).mean()), len(ref), len(eng)))
d = np.abs(ref - eng)
i = int(np.argmax(d))
print('first-near-zero idx %d  maxdiff idx %d (%.4g)  maxrel %.4g' % (
    int(np.argmax(d > 1e-6)) if np.any(d > 1e-6) else -1, i, d[i],
    d[i] / (np.abs(ref[i]) + 1e-9)))
print('ref[0..15]  %s' % np.round(ref[:16], 4))
print('eng[0..15]  %s' % np.round(eng[:16], 4))
# pattern check: if diffs cluster at multiples of 64 -> group stride bug
b = d > 1e-6
idx = np.where(b)[0]
print('n-diff %d  first-diff idx %d  all-idx %% 64: %s' % (
    len(idx), idx[0] if len(idx) else -1,
    np.unique(idx % 64)[:10] if len(idx) else []))
