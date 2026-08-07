#!/usr/bin/env python3
# Compare the engine's L0 t1 qkv (post-conv v + l2norm'd q/k) vs the ref.
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

# embed for token 1 = pids[1] = 6511
e = json.load(open('/tmp/q35-trunk/embed.json'))
eb = open('/tmp/q35-trunk/embed.bin', 'rb').read()
w = np.frombuffer(eb, dtype=np.uint32, count=e['weight']['nbytes'] // 4,
                  offset=e['weight']['off'])
sc = np.frombuffer(eb, dtype=np.uint16, count=e['scale']['nbytes'] // 2,
                   offset=e['scale']['off'])
bi = np.frombuffer(eb, dtype=np.uint16, count=e['bias']['nbytes'] // 2,
                   offset=e['bias']['off'])
emb1 = decode(w[6511 * 256:6512 * 256].reshape(1, 256),
              sc[6511 * 32:6512 * 32].reshape(1, 32),
              bi[6511 * 32:6512 * 32].reshape(1, 32))[0]

# t1 input state: the engine's state at t1 = embed(6511) (pre-L0)
eng_state = np.fromfile('/tmp/q35-eng-state-t1.bin', dtype=np.float32)
print('state-t1 cos vs embed(6511): %.4g' % (
    np.dot(eng_state, emb1) / (np.linalg.norm(eng_state) *
                               np.linalg.norm(emb1))))
iln = tf(L0, '.input_layernorm.weight')
xin = rmsnorm(eng_state, iln, 1e-6)

# ref qkv at t1: conv with past = [qkv(t0)], qkv(t1) = in_proj_qkv(xin)
qkv_pre = decode(ta(L0, '.linear_attn.in_proj_qkv.weight'),
                 ta(L0, '.linear_attn.in_proj_qkv.scales'),
                 ta(L0, '.linear_attn.in_proj_qkv.biases')) @ xin
cw = tf(L0, '.linear_attn.conv1d.weight').reshape(8192, 4)
# t0's pre-conv qkv: reconstruct from the engine's t0 dump (pre-conv) --
# /tmp/q35-eng-qkv-t0.bin is POST-l2norm; use the ref's t0 qkv instead:
# compute t0's qkv from the embed(760) path
e0 = json.load(open('/tmp/q35-trunk/embed.json'))
eb0 = open('/tmp/q35-trunk/embed.bin', 'rb').read()
w0 = np.frombuffer(eb0, dtype=np.uint32, count=e0['weight']['nbytes'] // 4,
                   offset=e0['weight']['off'])
s0 = np.frombuffer(eb0, dtype=np.uint16, count=e0['scale']['nbytes'] // 2,
                   offset=e0['scale']['off'])
b0 = np.frombuffer(eb0, dtype=np.uint16, count=e0['bias']['nbytes'] // 2,
                   offset=e0['bias']['off'])
emb0 = decode(w0[760 * 256:761 * 256].reshape(1, 256),
              s0[760 * 32:761 * 32].reshape(1, 32),
              b0[760 * 32:761 * 32].reshape(1, 32))[0]
xin0 = rmsnorm(emb0, iln, 1e-6)
qkv0 = decode(ta(L0, '.linear_attn.in_proj_qkv.weight'),
              ta(L0, '.linear_attn.in_proj_qkv.scales'),
              ta(L0, '.linear_attn.in_proj_qkv.biases')) @ xin0
seq = [np.zeros(8192), np.zeros(8192), qkv0, qkv_pre]
cacc = np.einsum('ck,ck->c', np.stack(seq, axis=1), cw)
ref_qkv = silu(cacc)

eng_qkv = np.fromfile('/tmp/q35-eng-qkv-t1.bin', dtype=np.float32)
# v portion untouched by l2norm
dv = np.abs(eng_qkv[4096:8192] - ref_qkv[4096:8192])
print('t1 v: n diff>1e-3: %d cos %.4g' % (
    int((dv > 1e-3).sum()),
    np.dot(eng_qkv[4096:8192], ref_qkv[4096:8192]) /
    (np.linalg.norm(eng_qkv[4096:8192]) * np.linalg.norm(ref_qkv[4096:8192]))))
# q/k portions: engine has l2norm'd; compare normalized
inv = 128 ** -0.5
q_n = (inv ** 2) * rmsnorm(ref_qkv[0:2048].reshape(16, 128), None, 1e-6)
k_n = inv * rmsnorm(ref_qkv[2048:4096].reshape(16, 128), None, 1e-6)
dq = np.abs(eng_qkv[0:2048].reshape(16, 128) - q_n)
dk = np.abs(eng_qkv[2048:4096].reshape(16, 128) - k_n)
print('t1 q: n diff>1e-4: %d | k: n diff>1e-4: %d' % (
    int((dq > 1e-4).sum()), int((dk > 1e-4).sum())))
