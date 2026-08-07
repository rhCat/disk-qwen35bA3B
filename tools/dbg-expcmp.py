#!/usr/bin/env python3
# Compare engine's per-expert outputs at L0 t0 vs the reference's.
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

p = json.load(open('/tmp/q35-pool/manifest.json'))
pb = open('/tmp/q35-pool/pool.bin', 'rb').read()

def exp_tensor(L, e, pn):
    for t in p['tensors']:
        if t['layer'] == L and t['expert'] == e and t['name'] == pn:
            meta = t
            break
    w = np.frombuffer(pb, dtype=np.uint32, count=meta['v_nbytes'] // 4,
                      offset=meta['v_off'])
    s = np.frombuffer(pb, dtype=np.uint16, count=meta['s_nbytes'] // 2,
                      offset=meta['s_off'])
    b = np.frombuffer(pb, dtype=np.uint16, count=meta['b_nbytes'] // 2,
                      offset=meta['b_off'])
    R, C = meta['shape']
    return w.reshape(R, C // 8), s, b

def expert_fwd(L, e, x):
    gw, gs, gb = exp_tensor(L, e, 'gate_proj')
    uw, us, ub = exp_tensor(L, e, 'up_proj')
    dw, ds, db = exp_tensor(L, e, 'down_proj')
    g = decode(gw, gs, gb) @ x
    u = decode(uw, us, ub) @ x
    ch = silu(g) * u
    return decode(dw, ds, db) @ ch

# ref xin2 at L0: rmsnorm(state_after_L0_attn, post_attn_norm)
TL = '/tmp/q35-trunk'
tl = json.load(open(TL + '/trunk.json'))
offs = np.fromfile(TL + '/trunk.offsets', dtype=np.uint64)[1::2]
tb = open(TL + '/trunk.bin', 'rb').read()
def ta(layer, suffix):
    for x in layer['tensors']:
        if x['n'].endswith(suffix):
            break
    else:
        return None
    base = int(offs[layer['layer']]) + x['off']
    if x['dtype'] == 'U32':
        a = np.frombuffer(tb, dtype=np.uint32, count=x['nbytes'] // 4, offset=base)
    elif x['dtype'] == 'BF16':
        a = np.frombuffer(tb, dtype=np.uint16, count=x['nbytes'] // 2, offset=base)
    elif x['dtype'] == 'F32':
        a = np.frombuffer(tb, dtype=np.float32, count=x['nbytes'] // 4, offset=base)
    return a.reshape(x['shape'])
def tf(layer, suffix):
    a = ta(layer, suffix)
    if a.dtype == np.float32:
        return a.astype(np.float32)
    return bf16arr(a.astype(np.uint16))

L0 = tl['layers'][0]
pn = tf(L0, '.post_attention_layernorm.weight')
xin2 = np.fromfile('/tmp/q35-eng-xin0.bin', dtype=np.float32)

sel = [106, 112, 57, 107, 181, 238, 200, 43]
print('ref xin2 rms %.4g' % np.sqrt((xin2 ** 2).mean()))
import sys
gu = np.fromfile('/tmp/q35-eng-gateup.bin', dtype=np.int32)
EID = int(gu[0])
eng_g = gu[1:513].astype(np.float32)
eng_u = gu[513:1025].astype(np.float32)
print('dumped expert (from file):', EID)
# chain comparison: engine chain vs ref silu(g)*u
gw, gs, gb = exp_tensor(0, EID, 'gate_proj')
uw, us, ub = exp_tensor(0, EID, 'up_proj')
dw, ds, db = exp_tensor(0, EID, 'down_proj')
eng_chain = np.fromfile('/tmp/q35-eng-chain.bin', dtype=np.int32)
EID2 = int(eng_chain[0])
eng_chain = eng_chain[1:513].astype(np.float32)
ref_g = decode(gw, gs, gb) @ xin2
ref_u = decode(uw, us, ub) @ xin2
ref_chain = silu(ref_g) * ref_u
print('chain file expert:', EID2)
for nm, a, b in (('gate', eng_g, ref_g), ('up', eng_u, ref_u)):
    da = np.abs(a - b)
    print('%-4s: eng rms %.6g ref rms %.6g cos %.4g maxdiff %.3g' % (
        nm, np.sqrt((a ** 2).mean()), np.sqrt((b ** 2).mean()),
        np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), da.max()))
dc = np.abs(eng_chain - ref_chain)
print('chain: eng rms %.6g ref rms %.6g cos %.4g' % (
    np.sqrt((eng_chain ** 2).mean()), np.sqrt((ref_chain ** 2).mean()),
    np.dot(eng_chain, ref_chain) / (np.linalg.norm(eng_chain) * np.linalg.norm(ref_chain))))
