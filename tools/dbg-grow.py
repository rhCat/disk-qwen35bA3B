#!/usr/bin/env python3
import sys, json, numpy as np

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

p = json.load(open('/tmp/q35-pool/manifest.json'))
pb = open('/tmp/q35-pool/pool.bin', 'rb').read()
EID = int(sys.argv[1])
for t in p['tensors']:
    if t['layer'] == 0 and t['expert'] == EID and t['name'] == 'gate_proj':
        m = t
        break
w = np.frombuffer(pb, dtype=np.uint32, count=m['v_nbytes'] // 4,
                  offset=m['v_off']).reshape(512, 256)
s = np.frombuffer(pb, dtype=np.uint16, count=m['s_nbytes'] // 2,
                  offset=m['s_off']).reshape(512, 32)
b = np.frombuffer(pb, dtype=np.uint16, count=m['b_nbytes'] // 2,
                  offset=m['b_off']).reshape(512, 32)
xin = np.fromfile('/tmp/q35-ref-xin2.bin', dtype=np.float32)
ref_g = decode(w, s, b) @ xin
eng_g = np.fromfile('/tmp/q35-eng-gateup.bin', dtype=np.float32)[:512]
print('expert', EID)
print('eng g[0..7]:', ['%.5g' % v for v in eng_g[:8]])
print('ref g[0..7]:', ['%.5g' % v for v in ref_g[:8]])
for r in range(6):
    er = eng_g[r * 64:(r + 1) * 64]
    rr = ref_g[r * 64:(r + 1) * 64]
    c = np.dot(er, rr) / (np.linalg.norm(er) * np.linalg.norm(rr) + 1e-30)
    print('row %d cos %.3f eng-rms %.4g ref-rms %.4g' % (
        r, c, np.sqrt((er ** 2).mean()), np.sqrt((rr ** 2).mean())))
