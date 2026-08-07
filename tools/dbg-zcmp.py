#!/usr/bin/env python3
# Compare engine's z (L0 t0) vs the reference's z elementwise.
import json, numpy as np

TRUNK = '/tmp/q35-trunk'
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

def rmsnorm(x, w=None, eps=1e-6):
    r = np.sqrt((x * x).mean(-1, keepdims=True) + eps)
    y = x / r
    if w is not None:
        y = y * w
    return y

def ten_arr(layer, suffix):
    for x in layer['tensors']:
        if x['n'].endswith(suffix):
            break
    else:
        return None
    base = int(offs[layer['layer']]) + x['off']
    if x['dtype'] == 'U32':
        a = np.frombuffer(tb, dtype=np.uint32, count=x['nbytes']//4, offset=base)
    elif x['dtype'] == 'BF16':
        a = np.frombuffer(tb, dtype=np.uint16, count=x['nbytes']//2, offset=base)
    elif x['dtype'] == 'F32':
        a = np.frombuffer(tb, dtype=np.float32, count=x['nbytes']//4, offset=base)
    else:
        return None
    return a.reshape(x['shape'])

def ten_float(layer, suffix):
    a = ten_arr(layer, suffix)
    if a is None:
        return None
    if a.dtype == np.float32:
        return a.astype(np.float32)
    return bf16arr(a.astype(np.uint16))

tl = json.load(open(TRUNK + '/trunk.json'))
offs_all = np.fromfile(TRUNK + '/trunk.offsets', dtype=np.uint64)
offs = offs_all[1::2]
tb = open(TRUNK + '/trunk.bin', 'rb').read()

L0 = tl['layers'][0]
# embed token 760
e = json.load(open(TRUNK + '/embed.json'))
eb = open(TRUNK + '/embed.bin', 'rb').read()
w = np.frombuffer(eb, dtype=np.uint32, count=e['weight']['nbytes']//4,
                  offset=e['weight']['off'])
sc = np.frombuffer(eb, dtype=np.uint16, count=e['scale']['nbytes']//2,
                   offset=e['scale']['off'])
bi = np.frombuffer(eb, dtype=np.uint16, count=e['bias']['nbytes']//2,
                   offset=e['bias']['off'])
emb = decode(w[760*256:761*256].reshape(1, 256),
             sc[760*32:761*32].reshape(1, 32),
             bi[760*32:761*32].reshape(1, 32))[0]
iln = ten_float(L0, '.input_layernorm.weight')
xin = rmsnorm(emb, iln, 1e-6)
W = decode(ten_arr(L0, '.linear_attn.in_proj_z.weight'),
           ten_arr(L0, '.linear_attn.in_proj_z.scales'),
           ten_arr(L0, '.linear_attn.in_proj_z.biases'))
ref_z = W @ xin

eng_z = np.fromfile('/tmp/q35-eng-z.bin', dtype=np.float32)
print('eng z rms %.6g  ref z rms %.6g' % (
    float(np.sqrt((eng_z**2).mean())), float(np.sqrt((ref_z**2).mean()))))
diff = np.abs(eng_z - ref_z)
nd = np.nonzero(diff > 1e-3)[0]
print('z: n diff > 1e-3: %d / %d' % (len(nd), len(eng_z)))
if len(nd):
    print('   first diff idx %d: eng %.6g ref %.6g' % (
        nd[0], eng_z[nd[0]], ref_z[nd[0]]))

# ref qkv (post-conv) at L0 t0
def silu(x):
    return x / (1.0 + np.exp(-x))

qkv_pre = decode(ten_arr(L0, '.linear_attn.in_proj_qkv.weight'),
                 ten_arr(L0, '.linear_attn.in_proj_qkv.scales'),
                 ten_arr(L0, '.linear_attn.in_proj_qkv.biases')) @ xin
cw = ten_float(L0, '.linear_attn.conv1d.weight').reshape(8192, 4)
seq = [np.zeros(8192)] * 3 + [qkv_pre]
conv_in = np.stack(seq, axis=1)
cacc = np.einsum('ck,ck->c', conv_in, cw)
ref_qkv = silu(cacc)

eng_qkv = np.fromfile('/tmp/q35-eng-qkv.bin', dtype=np.float32)
print('eng qkv rms %.6g  ref qkv rms %.6g' % (
    float(np.sqrt((eng_qkv**2).mean())), float(np.sqrt((ref_qkv**2).mean()))))
d2 = np.abs(eng_qkv - ref_qkv)
nd2 = np.nonzero(d2 > 1e-3)[0]
print('qkv(post): n diff > 1e-3: %d / %d' % (len(nd2), len(eng_qkv)))
if len(nd2):
    print('   first diff idx %d: eng %.6g ref %.6g' % (
        nd2[0], eng_qkv[nd2[0]], ref_qkv[nd2[0]]))

# pre-conv comparison
eng_pre = np.fromfile('/tmp/q35-eng-qkvpre.bin', dtype=np.float32)
d3 = np.abs(eng_pre - qkv_pre)
nd3 = np.nonzero(d3 > 1e-3)[0]
print('qkv(pre): n diff > 1e-3: %d / %d' % (len(nd3), len(eng_pre)))
if len(nd3):
    print('   first diff idx %d: eng %.6g ref %.6g' % (
        nd3[0], eng_pre[nd3[0]], qkv_pre[nd3[0]]))
else:
    print('   PRE-CONV MATCHES')
    # the engine's qkv dump is the l2norm'd q/k/v; v (4096..8191) is
    # untouched by l2norm -> compare v against ref post-conv v
    eng_v = eng_qkv[4096:8192]
    ref_v = ref_qkv[4096:8192]
    dv = np.abs(eng_v - ref_v)
    nv = int((dv > 1e-3).sum())
    print('v (4096:8192): n diff > 1e-3: %d / %d  (eng rms %.4g ref rms %.4g)' % (
        nv, len(eng_v), float(np.sqrt((eng_v**2).mean())),
        float(np.sqrt((ref_v**2).mean()))))
    if nv:
        f = np.nonzero(dv > 1e-3)[0][0]
        print('   first v diff %d: eng %.6g ref %.6g' % (
            f, eng_v[f], ref_v[f]))
    else:
        print('   V MATCHES -> conv is CORRECT; divergence is after conv')
    # q/k portions: engine's dump has l2norm'd q (0:2048) and k (2048:4096)
    eng_q = eng_qkv[0:2048].reshape(16, 128)
    eng_k = eng_qkv[2048:4096].reshape(16, 128)
    # ref normalized: q = inv_scale^2 * rms_norm(q_conv), k = inv_scale * rms_norm
    q_conv = ref_qkv[0:2048].reshape(16, 128)
    k_conv = ref_qkv[2048:4096].reshape(16, 128)
    inv = 128 ** -0.5
    ref_q = (inv ** 2) * rmsnorm(q_conv, None, 1e-6)
    ref_k = inv * rmsnorm(k_conv, None, 1e-6)
    dq = np.abs(eng_q - ref_q)
    nq = int((dq > 1e-4).sum())
    dk = np.abs(eng_k - ref_k)
    nk = int((dk > 1e-4).sum())
    print('q: n diff > 1e-4: %d / 2048  k: n diff > 1e-4: %d / 2048' % (nq, nk))
    if nq:
        f = np.nonzero(dq > 1e-4)[0][0]
        print('   first q diff %d: eng %.6g ref %.6g' % (
            f, eng_q.reshape(-1)[f], ref_q.reshape(-1)[f]))
    if nk:
        f = np.nonzero(dk > 1e-4)[0][0]
        print('   first k diff %d: eng %.6g ref %.6g' % (
            f, eng_k.reshape(-1)[f], ref_k.reshape(-1)[f]))
    if not nq and not nk:
        print('   Q/K MATCH -> divergence is in the delta rule or norm')
    # delta-rule readout: engine dump vs ref (state = k x (v*beta), readout = state . q)
    eng_ro = np.fromfile('/tmp/q35-eng-readout.bin', dtype=np.float32)
    # ref: per value head hv, key head hk = hv//2
    q_norm = (inv ** 2) * rmsnorm(q_conv, None, 1e-6)   # [16, 128]
    k_norm = inv * rmsnorm(k_conv, None, 1e-6)           # [16, 128]
    vv = ref_qkv[4096:8192].reshape(32, 128)
    # beta from in_proj_b(xin)
    Wb = decode(ten_arr(L0, '.linear_attn.in_proj_b.weight'),
                ten_arr(L0, '.linear_attn.in_proj_b.scales'),
                ten_arr(L0, '.linear_attn.in_proj_b.biases'))
    bb = 1.0 / (1.0 + np.exp(-(Wb @ xin)))
    ref_ro = np.zeros(4096, dtype=np.float32)
    for hv in range(32):
        hk = hv // 2
        delta = vv[hv] * bb[hv]
        S = np.outer(k_norm[hk], delta)
        # readout[j] = sum_i S[i,j]*q[i] = (S.T @ q)[j]
        ref_ro[hv*128:(hv+1)*128] = S.T @ q_norm[hk]
    dro = np.abs(eng_ro - ref_ro)
    nro = int((dro > 1e-4).sum())
    print('readout: n diff > 1e-4: %d / 4096 (eng rms %.4g ref rms %.4g)' % (
        nro, float(np.sqrt((eng_ro**2).mean())),
        float(np.sqrt((ref_ro**2).mean()))))
    if nro:
        f = np.nonzero(dro > 1e-4)[0][0]
        print('   first readout diff %d (head %d): eng %.6g ref %.6g' % (
            f, f // 128, eng_ro[f], ref_ro[f]))
    else:
        print('   READOUT MATCHES -> divergence is in RMSNormGated or later')
