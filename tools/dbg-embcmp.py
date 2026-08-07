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

e = json.load(open('/tmp/q35-trunk/embed.json'))
eb = open('/tmp/q35-trunk/embed.bin', 'rb').read()
w = np.frombuffer(eb, dtype=np.uint32, count=e['weight']['nbytes']//4,
                  offset=e['weight']['off'])
sc = np.frombuffer(eb, dtype=np.uint16, count=e['scale']['nbytes']//2,
                   offset=e['scale']['off'])
bi = np.frombuffer(eb, dtype=np.uint16, count=e['bias']['nbytes']//2,
                   offset=e['bias']['off'])

eng = np.fromfile('/tmp/q35-eng-embed.bin', dtype=np.float32)
print('eng embed rms %.6g' % float(np.sqrt((eng**2).mean())))

for tok in (5, 760):
    row = w[tok*256:(tok+1)*256].reshape(1, 256)
    sr = sc[tok*32:(tok+1)*32].reshape(1, 32)
    br = bi[tok*32:(tok+1)*32].reshape(1, 32)
    d = decode(row, sr, br)[0]
    diff = np.abs(eng - d)
    print('tok %d rms %.6g maxdiff %.6g meandiff %.6g' % (
        tok, float(np.sqrt((d**2).mean())),
        float(diff.max()), float(diff.mean())))
    # first differing index
    nd = np.nonzero(diff > 1e-4)[0]
    print('   first diff idx:', nd[0] if len(nd) else -1)
