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

def rmsnorm(x, w=None, eps=1e-6):
    r = np.sqrt((x * x).mean(-1, keepdims=True) + eps)
    y = x / r
    return y * w if w is not None else y

tl = json.load(open('/tmp/q35-trunk/trunk.json'))
offs = np.fromfile('/tmp/q35-trunk/trunk.offsets', dtype=np.uint64)[1::2]
tb = open('/tmp/q35-trunk/trunk.bin', 'rb').read()
L3 = tl['layers'][3]

def ta(suf):
    for x in L3['tensors']:
        if x['n'].endswith(suf):
            break
    base = int(offs[3]) + x['off']
    if x['dtype'] == 'U32':
        a = np.frombuffer(tb, dtype=np.uint32, count=x['nbytes'] // 4,
                          offset=base)
    else:
        a = np.frombuffer(tb, dtype=np.uint16, count=x['nbytes'] // 2,
                          offset=base)
    return a.reshape(x['shape'])

def tf(suf):
    a = ta(suf)
    if a.dtype == np.uint32:
        return a.astype(np.float32)
    return bf16arr(a.astype(np.uint16))

xin = np.fromfile('/tmp/q35-eng-gqaxin.bin', dtype=np.float32)
eng_proj = np.fromfile('/tmp/q35-eng-gqaproj.bin', dtype=np.float32)
Wq = decode(ta('.self_attn.q_proj.weight'), ta('.self_attn.q_proj.scales'),
            ta('.self_attn.q_proj.biases'))
ref_proj = Wq @ xin
d = np.abs(eng_proj - ref_proj)
print('q_proj: eng rms %.6g ref rms %.6g n diff>1e-3: %d cos %.4g' % (
    np.sqrt((eng_proj ** 2).mean()), np.sqrt((ref_proj ** 2).mean()),
    int((d > 1e-3).sum()),
    np.dot(eng_proj, ref_proj) / (np.linalg.norm(eng_proj) *
                                  np.linalg.norm(ref_proj))))
nd = np.nonzero(d > 1e-3)[0]
if len(nd):
    print('first diff %d: eng %.6g ref %.6g' % (nd[0], eng_proj[nd[0]],
                                                 ref_proj[nd[0]]))
# q norm check: ref q after norm per-head rms
q = ref_proj[:4096].reshape(16, 256)
qn = tf('.self_attn.q_norm.weight')
q_n = rmsnorm(q, qn, 1e-6)
print('ref q post-norm per-head rms:', ['%.4g' % np.sqrt((q_n[h] ** 2).mean())
                                        for h in range(4)])
eng_q = np.fromfile('/tmp/q35-eng-gqaq.bin', dtype=np.float32)[:4096]
print('eng q post-norm per-head rms:', ['%.4g' % np.sqrt((eng_q[h * 256:(h + 1) * 256] ** 2).mean())
                                        for h in range(4)])
dq = np.abs(eng_q - q_n.reshape(-1))
print('q post-norm: n diff>1e-3: %d cos %.4g' % (
    int((dq > 1e-3).sum()),
    np.dot(eng_q, q_n.reshape(-1)) / (np.linalg.norm(eng_q) *
                                      np.linalg.norm(q_n.reshape(-1)))))
