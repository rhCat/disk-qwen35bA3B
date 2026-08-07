#!/usr/bin/env python3
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

gu = np.fromfile('/tmp/q35-eng-gateup.bin', dtype=np.int32)
eng_g = gu[1:513].astype(np.float32)
EID = int(gu[0])
p = json.load(open('/tmp/q35-pool/manifest.json'))
pb = open('/tmp/q35-pool/pool.bin', 'rb').read()
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
nz = np.abs(ref_g) > 1e-6
rat = eng_g[nz] / ref_g[nz]
print('expert', EID)
print('ratio stats: mean %.4g median %.4g min %.4g max %.4g' % (
    np.abs(rat).mean(), np.median(np.abs(rat)), np.abs(rat).min(),
    np.abs(rat).max()))
print('eng_g[0..3]:', ['%.4g' % v for v in eng_g[:4]])
print('ref_g[0..3]:', ['%.4g' % v for v in ref_g[:4]])
print('ref s[0,0..3]:', ['%.4g' % float(bf16arr(s[0, 0:4].astype(np.uint16))[i])
                         for i in range(4)])
print('ref b[0,0..3]:', ['%.4g' % float(bf16arr(b[0, 0:4].astype(np.uint16))[i])
                         for i in range(4)])
