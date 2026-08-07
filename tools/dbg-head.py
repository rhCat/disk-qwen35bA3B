#!/usr/bin/env python3
# dbg-head.py -- run the reference final state through the lm_head.
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

h = json.load(open('/tmp/q35-trunk/head.json'))
hb = open('/tmp/q35-trunk/head.bin', 'rb').read()
w = np.frombuffer(hb, dtype=np.uint32, count=h['weight']['nbytes']//4,
                  offset=h['weight']['off'])
sc = np.frombuffer(hb, dtype=np.uint16, count=h['scale']['nbytes']//2,
                   offset=h['scale']['off'])
bi = np.frombuffer(hb, dtype=np.uint16, count=h['bias']['nbytes']//2,
                   offset=h['bias']['off'])
R = h['weight']['shape'][0]
w = w.reshape(R, 256)
sc = sc.reshape(R, 32)
bi = bi.reshape(R, 32)
W = decode(w, sc, bi)            # [248320, 2048]

for name, path in (('eng', '/tmp/q35-eng-state.bin'),
                   ('ref', '/tmp/q35-ref-state.bin')):
    s = np.fromfile(path, dtype=np.float32)
    logits = W @ s
    top = np.argsort(logits)[-5:][::-1]
    print('%s: top5 %s logits %s' % (name, top.tolist(),
          logits[top].tolist()))

# decode the token strings
tok = json.load(open('/Users/ruihe/.cache/huggingface/mlx-qwen35-a3b-4bit/tokenizer.json'))
vocab = tok['model']['vocab']
inv = {v: k for k, v in vocab.items()}
for name, path in (('eng', '/tmp/q35-eng-state.bin'),
                   ('ref', '/tmp/q35-ref-state.bin')):
    s = np.fromfile(path, dtype=np.float32)
    logits = W @ s
    top = int(np.argmax(logits))
    print('%s argmax %d -> %r' % (name, top, inv.get(top, '?')))
