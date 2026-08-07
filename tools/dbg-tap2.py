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

def silu(x):
    return x / (1.0 + np.exp(-x))

tl = json.load(open('/tmp/q35-trunk/trunk.json'))
offs = np.fromfile('/tmp/q35-trunk/trunk.offsets', dtype=np.uint64)[1::2]
tb = open('/tmp/q35-trunk/trunk.bin', 'rb').read()
L0 = tl['layers'][0]

def ta(layer, suf):
    for x in layer['tensors']:
        if x['n'].endswith(suf):
            break
    base = int(offs[layer['layer']]) + x['off']
    if x['dtype'] == 'U32':
        a = np.frombuffer(tb, dtype=np.uint32, count=x['nbytes'] // 4,
                          offset=base)
    else:
        a = np.frombuffer(tb, dtype=np.uint16, count=x['nbytes'] // 2,
                          offset=base)
    return a.reshape(x['shape'])

def tf(layer, suf):
    a = ta(layer, suf)
    if a.dtype == np.uint32:
        return a.astype(np.float32)
    return bf16arr(a.astype(np.uint16))

def embed_of(tok):
    e = json.load(open('/tmp/q35-trunk/embed.json'))
    eb = open('/tmp/q35-trunk/embed.bin', 'rb').read()
    w = np.frombuffer(eb, dtype=np.uint32, count=e['weight']['nbytes'] // 4,
                      offset=e['weight']['off'])
    sc = np.frombuffer(eb, dtype=np.uint16, count=e['scale']['nbytes'] // 2,
                       offset=e['scale']['off'])
    bi = np.frombuffer(eb, dtype=np.uint16, count=e['bias']['nbytes'] // 2,
                       offset=e['bias']['off'])
    return decode(w[tok * 256:(tok + 1) * 256].reshape(1, 256),
                  sc[tok * 32:(tok + 1) * 32].reshape(1, 32),
                  bi[tok * 32:(tok + 1) * 32].reshape(1, 32))[0]

iln = tf(L0, '.input_layernorm.weight')
Wq = decode(ta(L0, '.linear_attn.in_proj_qkv.weight'),
            ta(L0, '.linear_attn.in_proj_qkv.scales'),
            ta(L0, '.linear_attn.in_proj_qkv.biases'))
cw = tf(L0, '.linear_attn.conv1d.weight').reshape(8192, 4)

qkv0 = Wq @ rmsnorm(embed_of(760), iln, 1e-6)
qkv1 = Wq @ rmsnorm(embed_of(6511), iln, 1e-6)
eng_v = np.fromfile('/tmp/q35-eng-qkv-t1.bin', dtype=np.float32)[4096:8192]

c_ref = silu(cw[:, 2] * qkv0 + cw[:, 3] * qkv1)[4096:8192]
c33 = silu(cw[:, 3] * qkv0 + cw[:, 3] * qkv1)[4096:8192]
print('eng v[4096..4103]:', ['%.6g' % v for v in eng_v[:8]])
print('ref   v[4096..4103]:', ['%.6g' % v for v in c_ref[:8]])
print('w3w3  v[4096..4103]:', ['%.6g' % v for v in c33[:8]])
# which conv input dominates? print the pre-silu terms
pre_ref = (cw[:, 2] * qkv0 + cw[:, 3] * qkv1)[4096:4104]
print('pre-silu ref:', ['%.6g' % v for v in pre_ref[:8]])
pre33 = (cw[:, 3] * qkv0 + cw[:, 3] * qkv1)[4096:4104]
print('pre-silu w3w3:', ['%.6g' % v for v in pre33[:8]])
print('cw[4096, :] (v ch0 taps):', ['%.6g' % v for v in cw[4096]])
