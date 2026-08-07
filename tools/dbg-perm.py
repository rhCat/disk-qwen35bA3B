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
nz = eng_g != 0
print('eng non-zero count:', int(nz.sum()))
# for each engine non-zero, find the closest ref element
for i in np.nonzero(nz)[0][:24]:
    j = np.abs(ref_g - eng_g[i]).argmin()
    print('eng[%3d]=%9.5f  closest ref[%3d]=%9.5f  (ref[%d]=%9.5f)' % (
        i, eng_g[i], j, ref_g[j], i, ref_g[i]))
