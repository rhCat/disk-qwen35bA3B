#!/usr/bin/env python3
import sys, json, numpy as np

def bf16arr(bits):
    u = bits.astype(np.uint32) << 16
    return u.view(np.float32)

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
eng_g = np.fromfile('/tmp/q35-eng-gateup.bin', dtype=np.float32)[:512]

# naive ref decode: nibble i of word = (w >> 4i) & 0xF, low first
def decode_naive(w, s, b):
    sh = np.array([0, 4, 8, 12, 16, 20, 24, 28], dtype=np.uint32)
    nib = (w[..., None] >> sh) & 0xF
    R, P = w.shape
    vals = nib.reshape(R, P * 8)
    C = P * 8
    G = C // 64
    sc = bf16arr(s.astype(np.uint16)).reshape(R, G)
    bi = bf16arr(b.astype(np.uint16)).reshape(R, G)
    return vals.astype(np.float32) * np.repeat(sc, 64, axis=1) + np.repeat(bi, 64, axis=1)

# SIMD-style: bytes little-endian, lo/hi nibbles interleaved per byte
def decode_simd(w, s, b):
    R, P = w.shape
    C = P * 8
    G = C // 64
    vals = np.zeros((R, C), dtype=np.float32)
    for r in range(R):
        for p in range(P):
            u = int(w[r, p])
            bts = [(u >> (8 * k)) & 0xFF for k in range(4)]
            # b0 = [nib0,nib1],[nib2,nib3],[nib4,nib5],[nib6,nib7]
            lo = [bt & 0x0F for bt in bts]
            hi = [(bt >> 4) & 0x0F for bt in bts]
            # vzip(lo, hi).val[0] = [lo0,hi0,lo1,hi1,lo2,hi2,lo3,hi3]
            # = [nib0,nib1,nib2,nib3,nib4,nib5,nib6,nib7]
            n = [lo[0], hi[0], lo[1], hi[1], lo[2], hi[2], lo[3], hi[3]]
            vals[r, p * 8:(p + 1) * 8] = n
    sc = bf16arr(s.astype(np.uint16)).reshape(R, G)
    bi = bf16arr(b.astype(np.uint16)).reshape(R, G)
    return vals * np.repeat(sc, 64, axis=1) + np.repeat(bi, 64, axis=1)

ref_naive = decode_naive(w, s, b) @ xin
ref_simd = decode_simd(w, s, b) @ xin
print('expert', EID)
print('eng g[0..3]:   ', ['%.6g' % v for v in eng_g[:4]])
print('ref naive[0..3]:', ['%.6g' % v for v in ref_naive[:4]])
print('ref simd [0..3]:', ['%.6g' % v for v in ref_simd[:4]])
for nm, r in (('naive', ref_naive), ('simd', ref_simd)):
    c = np.dot(eng_g, r) / (np.linalg.norm(eng_g) * np.linalg.norm(r))
    print('%s cos %.4f' % (nm, c))
