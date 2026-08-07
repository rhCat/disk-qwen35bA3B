#!/usr/bin/env python3
# dbg-reffwd.py -- faithful Python port of mlx-lm's Qwen3.5 forward
# (qwen3_5.py + qwen3_next.py + gated_delta.py), on the real weights,
# with the CORRECT mlx4 dequant (q*s+b) and the shared expert.
# Dumps the hidden state after the 6-token prompt for comparison with
# the engine's --dump-state.
import json, struct, math, sys, os, fcntl
import numpy as np

TRUNK = '/tmp/q35-trunk'
POOL = '/tmp/q35-pool'

def nocache_open(path):
    """Open with F_NOCACHE (macOS) so pread doesn't balloon page cache,
    mirroring the engine's st.c/trunk.c. Returns an fd."""
    fd = os.open(path, os.O_RDONLY)
    try:
        fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
    except (AttributeError, OSError):
        pass
    return fd

def pread_buf(fd, nbytes, offset):
    """Read exactly nbytes at offset (positional, no fd state change)."""
    out = bytearray()
    while len(out) < nbytes:
        chunk = os.pread(fd, nbytes - len(out), offset + len(out))
        if not chunk:
            break
        out += chunk
    return bytes(out)

def bf16arr(bits):
    u = bits.astype(np.uint32) << 16
    return u.view(np.float32) if u.dtype.itemsize == 4 else None

def decode(w, sc, bi):
    """mlx4 row-major decode: w [R, P] U32 (P=C/8), sc/bi [R, C/64] BF16.
    val = q*scale + bias (q in [0,15])."""
    w = w.astype(np.uint32)
    sh = np.array([0,4,8,12,16,20,24,28], dtype=np.uint32)
    nib = (w[..., None] >> sh) & 0xF           # [R, P, 8]
    R, P = w.shape
    vals = nib.reshape(R, P * 8)               # [R, C]
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

def silu(x):
    return x / (1.0 + np.exp(-x))

def softplus(x):
    return np.log1p(np.exp(x))

# ---- load trunk layout (offsets: [0]=count, then n x (off, nbytes))
tl = json.load(open(TRUNK + '/trunk.json'))
offs_all = np.fromfile(TRUNK + '/trunk.offsets', dtype=np.uint64)
offs = offs_all[1::2]        # every other u64 = layer offset
tb_fd = nocache_open(TRUNK + '/trunk.bin')
pool = json.load(open(POOL + '/manifest.json'))
pb_fd = nocache_open(POOL + '/pool.bin')

def tens(layer, suffix):
    for x in layer['tensors']:
        if x['n'].endswith(suffix):
            return x
    return None

def ten_arr(layer, suffix):
    x = tens(layer, suffix)
    if x is None:
        return None
    base = int(offs[layer['layer']]) + x['off']
    dt = x['dtype']
    buf = pread_buf(tb_fd, x['nbytes'], base)
    if dt == 'U32':
        a = np.frombuffer(buf, dtype=np.uint32, count=x['nbytes'] // 4)
    elif dt == 'BF16':
        a = np.frombuffer(buf, dtype=np.uint16, count=x['nbytes'] // 2)
    elif dt == 'F32':
        a = np.frombuffer(buf, dtype=np.float32, count=x['nbytes'] // 4)
    else:
        return None
    return a.reshape(x['shape'])

def ten_float(layer, suffix):
    """Return float32 values regardless of storage dtype (F32/BF16)."""
    a = ten_arr(layer, suffix)
    if a is None:
        return None
    if a.dtype == np.float32:
        return a.astype(np.float32)
    return bf16arr(a.astype(np.uint16))

def proj(layer, suffix, x, out_rows):
    w = ten_arr(layer, suffix + '.weight')
    sc = ten_arr(layer, suffix + '.scales')
    bi = ten_arr(layer, suffix + '.biases')
    if w is None or sc is None or bi is None:
        return None
    W = decode(w, sc, bi)                      # [out, C]
    return W @ x                                # [out]

# ---- pool experts (manifest: flat list of per-projection dicts;
# v_off/s_off/b_off are ABSOLUTE into pool.bin)
EXP_BYTES = pool['expert_nbytes']
def exp_tensor(L, e, proj_name):
    """return (vals, scales, biases) for expert e of layer L."""
    meta = None
    for t in pool['tensors']:
        if t['layer'] == L and t['expert'] == e and t['name'] == proj_name:
            meta = t
            break
    assert meta is not None, (L, e, proj_name)
    w = np.frombuffer(pread_buf(pb_fd, meta['v_nbytes'], meta['v_off']),
                      dtype=np.uint32, count=meta['v_nbytes'] // 4)
    s = np.frombuffer(pread_buf(pb_fd, meta['s_nbytes'], meta['s_off']),
                      dtype=np.uint16, count=meta['s_nbytes'] // 2)
    b = np.frombuffer(pread_buf(pb_fd, meta['b_nbytes'], meta['b_off']),
                      dtype=np.uint16, count=meta['b_nbytes'] // 2)
    # manifest stores DECODED shape [R, C]; packed cols = C/8
    R, C = meta['shape']
    w = w.reshape(R, C // 8)
    return w, s, b

def expert_fwd(L, e, x):
    gw, gs, gb = exp_tensor(L, e, 'gate_proj')
    uw, us, ub = exp_tensor(L, e, 'up_proj')
    dw, ds, db = exp_tensor(L, e, 'down_proj')
    g = decode(gw, gs, gb) @ x                 # [512]
    u = decode(uw, us, ub) @ x                 # [512]
    ch = silu(g) * u
    out = decode(dw, ds, db) @ ch              # [2048]
    return out

# ---- config (pool config.json: hidden/n_layers/n_experts/topk keys)
cfg = json.load(open(TRUNK + '/config.json'))
H = cfg['hidden']
N_LAYERS = cfg['n_layers']
N_EXP = cfg['n_experts']
TOPK = cfg['topk']
KVH = cfg.get('num_key_value_heads', 2)
HD = cfg.get('head_dim', 256)
THETA = cfg.get('rope_theta', 1e7)
FULL_ATTN_INTERVAL = 4
KD = 128   # linear key head dim
VD = 128   # linear value head dim
HEADS = 16 # GQA q heads
RDD = 64   # rotary dims (partial 0.25 * 256)

# ---- prompt (8 prompt tokens + the engine's 1st GENERATED token.
# The engine's t7 logits argmax = 48017 Jupiter; it feeds Jupiter at t8.)
pids = [760, 7526, 11247, 303, 279, 12570, 1785, 369, 48017]
T = len(pids)
print('prompt tokens:', pids)

# ---- embed
e = json.load(open(TRUNK + '/embed.json'))
eb_fd = nocache_open(TRUNK + '/embed.bin')
def embed_row(tokid):
    w = np.frombuffer(pread_buf(eb_fd, e['weight']['nbytes'], e['weight']['off']),
                      dtype=np.uint32, count=e['weight']['nbytes'] // 4)
    sc = np.frombuffer(pread_buf(eb_fd, e['scale']['nbytes'], e['scale']['off']),
                       dtype=np.uint16, count=e['scale']['nbytes'] // 2)
    bi = np.frombuffer(pread_buf(eb_fd, e['bias']['nbytes'], e['bias']['off']),
                       dtype=np.uint16, count=e['bias']['nbytes'] // 2)
    row = w[tokid * (2048 // 8):(tokid + 1) * (2048 // 8)]
    sc_r = sc[tokid * 32:(tokid + 1) * 32]
    bi_r = bi[tokid * 32:(tokid + 1) * 32]
    return decode(row[None, :], sc_r[None, :], bi_r[None, :])[0]

# ---- rope
def rope_apply(qk, pos, rd):
    """mlx nn.RoPE traditional=False on first rd dims: HALF-SPLIT pairs
    (d, d+rd/2), NOT interleaved (2i, 2i+1). Verified against mx.fast.rope.
    (mrope_interleaved in config is a transformers-ism; mlx-lm
    qwen3_next.py uses initialize_rope(traditional=False).)
    inv_freq = 1/theta^(2i/rd)."""
    out = qk.copy()
    for i in range(rd // 2):
        ang = pos / (THETA ** (2.0 * i / rd))
        c = math.cos(ang)
        s = math.sin(ang)
        a = out[i]
        b = out[i + rd // 2]
        out[i] = a * c - b * s
        out[i + rd // 2] = a * s + b * c
    return out

# ---- linear-attn cache
lin_state = {}   # L -> [32, 128, 128]
conv_cache = {}  # L -> list of past qkv vectors (max 3)
kv_cache = {}    # L -> [pos, 2, 256] (k and v per kv head)
printed_l0_delta = False
printed_qkv = False

x = np.zeros((T, H), dtype=np.float32)
for t in range(T):
    x[t] = embed_row(pids[t])

state = x.copy()   # hidden state [T, H] -- but we do sequential decode:
# sequential: process one token at a time, keep caches
out_states = []
for t in range(T):
    h = state[t].copy() if T > 1 else state.copy()
    for L in range(N_LAYERS):
        layer = tl['layers'][L]
        is_linear = (L + 1) % FULL_ATTN_INTERVAL != 0
        if t == 0 and L in (2, 3):
            print('L%d start: h rms %.4g' % (
                L, float(np.sqrt((h**2).mean()))), flush=True)
        # input_layernorm
        iln = ten_float(layer, '.input_layernorm.weight')
        xin = rmsnorm(h, iln, 1e-6)
        if is_linear:
            # qkv/z/a/b
            qkv = proj(layer, '.linear_attn.in_proj_qkv', xin, 8192)
            z = proj(layer, '.linear_attn.in_proj_z', xin, 4096).reshape(32, VD)
            a = proj(layer, '.linear_attn.in_proj_a', xin, 32)
            b = proj(layer, '.linear_attn.in_proj_b', xin, 32)
            if t == 0 and L == 0:
                print('L0 t0: preconv rms %.4g' % (
                    float(np.sqrt((qkv**2).mean()))), flush=True)
            # conv (depthwise, kernel 4, w[0]=oldest)
            cw = ten_float(layer, '.linear_attn.conv1d.weight')
            cw = cw.reshape(8192, 4)
            past = conv_cache.get(L, [])
            seq = (past + [qkv])[-4:]
            if len(seq) < 4:
                seq = [np.zeros(8192)] * (4 - len(seq)) + seq
            conv_in = np.stack(seq, axis=1)    # [8192, 4]
            cacc = np.einsum('ck,ck->c', conv_in, cw)
            qkv = silu(cacc)
            if t == 0 and L == 0:
                print('L0 t0: postconv rms %.4g' % (
                    float(np.sqrt((qkv**2).mean()))), flush=True)
            conv_cache[L] = seq[-3:]
            if t == 8 and L == 0 and not printed_qkv:
                print('ref L0 t8 preconv: q-slice rms %.4g k-slice rms %.4g '
                      'v-slice rms %.4g  v[0..3] %s' % (
                          float(np.sqrt((qkv[:2048]**2).mean())),
                          float(np.sqrt((qkv[2048:4096]**2).mean())),
                          float(np.sqrt((qkv[4096:]**2).mean())),
                          np.round(qkv[4096:4100], 4).tolist()), flush=True)
                printed_qkv = True
            # split q/k/v
            qq = qkv[0:2048].reshape(16, KD)
            kk = qkv[2048:4096].reshape(16, KD)
            vv = qkv[4096:8192].reshape(32, VD)
            inv = KD ** -0.5
            qq = (inv ** 2) * rmsnorm(qq, None, 1e-6)
            kk = inv * rmsnorm(kk, None, 1e-6)
            # gated delta
            A_log = ten_float(layer, '.linear_attn.A_log')
            dt = ten_float(layer, '.linear_attn.dt_bias')
            beta = 1.0 / (1.0 + np.exp(-b))
            g = np.exp(-np.exp(A_log.astype(np.float32)) * softplus(a + dt))
            if t == 0 and L == 0:
                print('L0 t0: A_log[:3] %s dt[:3] %s a[:3] %s g[:3] %s beta[:3] %s' % (
                    A_log[:3].tolist(), dt[:3].tolist(), a[:3].tolist(),
                    g[:3].tolist(), beta[:3].tolist()), flush=True)
            S = lin_state.get(L, np.zeros((32, VD, KD), dtype=np.float32))
            readout = np.zeros((32, VD), dtype=np.float32)
            for hv in range(32):
                hk = hv // 2
                S[hv] = S[hv] * g[hv]
                kv_mem = S[hv].T @ kk[hk]         # [VD]
                delta = (vv[hv] - kv_mem) * beta[hv]
                S[hv] += np.outer(kk[hk], delta)
                readout[hv] = S[hv].T @ qq[hk]
            lin_state[L] = S
            if t == 8 and L == 0 and not printed_l0_delta:
                print('ref L0 t8: g[:3]=%s beta[:3]=%s k-rms %.4g '
                      'q-rms %.4g v-rms %.4g state-rms %.4g' % (
                          ['%.4g' % v for v in g[:3]],
                          ['%.4g' % v for v in beta[:3]],
                          float(np.sqrt((kk**2).mean())),
                          float(np.sqrt((qq**2).mean())),
                          float(np.sqrt((vv**2).mean())),
                          float(np.sqrt((S**2).mean()))), flush=True)
                printed_l0_delta = True
            # RMSNormGated: rmsnorm PER-HEAD with learned weight * silu(z)
            nwgt = ten_float(layer, '.linear_attn.norm.weight')
            rout = rmsnorm(readout.reshape(32, VD), nwgt, 1e-6).reshape(-1)
            gated = rout * silu(z.reshape(-1))
            r = proj(layer, '.linear_attn.out_proj', gated.reshape(-1), H)
            if t == 0 and L == 0:
                print('ref L0: linear r rms %.4g readout-rms %.4g '
                      'silu-z rms %.4g z[0..3] %s xin[0..3] %s' % (
                    float(np.sqrt((r**2).mean())),
                    float(np.sqrt((readout**2).mean())),
                    float(np.sqrt((silu(z.reshape(-1))**2).mean())),
                    z.reshape(-1)[:4].tolist(),
                    xin[:4].tolist()),
                    flush=True)
        else:
            # GQA attention
            qout = proj(layer, '.self_attn.q_proj', xin, HEADS * HD * 2)
            qout = qout.reshape(HEADS, HD * 2)   # head-major: [q|gate]
            q = qout[:, :HD]
            gate = qout[:, HD:]
            k = proj(layer, '.self_attn.k_proj', xin, KVH * HD).reshape(KVH, HD)
            v = proj(layer, '.self_attn.v_proj', xin, KVH * HD).reshape(KVH, HD)
            qn = ten_float(layer, '.self_attn.q_norm.weight')
            kn = ten_float(layer, '.self_attn.k_norm.weight')
            q = rmsnorm(q, qn, 1e-6)
            k = rmsnorm(k, kn, 1e-6)
            for hi in range(HEADS):
                q[hi] = rope_apply(q[hi], t, RDD)
            for hi in range(KVH):
                k[hi] = rope_apply(k[hi], t, RDD)
            c = kv_cache.get(L, ([], []))
            c[0].append(k)
            c[1].append(v)
            kv_cache[L] = c
            Ks = np.stack(c[0], 0)   # [npos, KVH, HD]
            Vs = np.stack(c[1], 0)
            npos = len(Ks)
            outs = np.zeros((HEADS, HD), dtype=np.float32)
            for hi in range(HEADS):
                khh = hi // (HEADS // KVH)
                s = (Ks[:, khh] @ q[hi]) / math.sqrt(HD)
                s = s - s.max()
                w = np.exp(s)
                w = w / w.sum()
                outs[hi] = w @ Vs[:, khh]
            gated = outs * (1.0 / (1.0 + np.exp(-gate)))
            if t == 1 and L == 7:
                gated.astype(np.float32).tofile('/tmp/q35-ref-gqa7.bin')
            r = proj(layer, '.self_attn.o_proj', gated.reshape(-1), H)
            if t == 0 and L == 3:
                print('L3 t0: q rms %.4g k rms %.4g v rms %.4g s max %.4g '
                      'r rms %.4g' % (
                    float(np.sqrt((q**2).mean())),
                    float(np.sqrt((k**2).mean())),
                    float(np.sqrt((v**2).mean())),
                    float(s.max()) if npos else -1,
                    float(np.sqrt((r**2).mean()))), flush=True)
        h = h + r
        if (t == 1 and L in (3, 7)):
            print('t1 L%d gqa: h rms %.4g r rms %.4g' % (
                L, float(np.sqrt((h**2).mean())),
                float(np.sqrt((r**2).mean()))), flush=True)
        if t == 0 and L == 3:
            print('L3 after-gqa: h rms %.4g r rms %.4g' % (
                float(np.sqrt((h**2).mean())),
                float(np.sqrt((r**2).mean()))), flush=True)
        # post_attention_layernorm -> MoE
        pn = ten_float(layer, '.post_attention_layernorm.weight')
        xin2 = rmsnorm(h, pn, 1e-6)
        if t == 0 and L == 0:
            xin2.astype(np.float32).tofile('/tmp/q35-ref-xin2.bin')
        # router
        gw = ten_arr(layer, '.mlp.gate.weight')
        gs = ten_arr(layer, '.mlp.gate.scales')
        gb = ten_arr(layer, '.mlp.gate.biases')
        scores = decode(gw, gs, gb) @ xin2       # [256]
        gates = np.exp(scores - scores.max())
        gates = gates / gates.sum()
        inds = np.argsort(gates)[-TOPK:][::-1]
        topw = gates[inds]
        # mlx-lm qwen3_next.py: softmax probs used RAW -- renorm only if
        # norm_topk_prob (absent from config -> default False). NO renorm.
        acc = np.zeros(H, dtype=np.float32)
        if t == 0 and L in (0, 1, 2):
            print('ref L%d: sel=%s w=%s xin2=%.4g' % (L, inds.tolist(),
                  ['%.4g' % v for v in topw],
                  float(np.sqrt((xin2**2).mean()))), flush=True)
        if t == 0 and L == 3:
            print('L3 moe: xin2 rms %.4g scores[0..3] %s' % (
                float(np.sqrt((xin2**2).mean())),
                ['%.3g' % v for v in scores[:4]]), flush=True)
        for j, e in enumerate(inds):
            eo = expert_fwd(L, int(e), xin2)
            acc += topw[j] * eo
            if t == 0 and L == 3 and j < 3:
                print('   expert %d w %.4g out rms %.4g' % (
                    int(e), float(topw[j]),
                    float(np.sqrt((eo**2).mean()))), flush=True)
        # shared expert
        sg = proj(layer, '.mlp.shared_expert.gate_proj', xin2, 512)
        su = proj(layer, '.mlp.shared_expert.up_proj', xin2, 512)
        sd = proj(layer, '.mlp.shared_expert.down_proj', silu(sg) * su, H)
        rgw = ten_arr(layer, '.mlp.shared_expert_gate.weight')
        rgs = ten_arr(layer, '.mlp.shared_expert_gate.scales')
        rgb = ten_arr(layer, '.mlp.shared_expert_gate.biases')
        route = decode(rgw, rgs, rgb) @ xin2
        sgate = 1.0 / (1.0 + np.exp(-float(route[0])))
        acc += sgate * sd
        if t == 0 and L == 0:
            rout_r = np.sqrt(((acc - sgate * sd) ** 2).mean())
            print('ref L0: routed-acc rms %.4g shared out rms %.4g sgate %.4g' % (
                float(rout_r), float(np.sqrt((sd**2).mean())),
                float(sgate)), flush=True)
        h = h + acc
        if (t == 0 and L in (0, 8, 16, 24, 32, N_LAYERS - 1)) or (t > 0 and L in (0, 8, 16, 24, 32, N_LAYERS - 1)):
            print('t%d L%d state rms %.4g' % (t, L,
                  float(np.sqrt((h**2).mean()))), flush=True)
        if t == 8 and L % 4 == 0:
            h.astype(np.float32).tofile('/tmp/q35-ref-L%d-t8.bin' % L)
            if L == 0:
                print('dumping ref per-layer t8', flush=True)
        if t == 7 and L % 4 == 0:
            h.astype(np.float32).tofile('/tmp/q35-ref-L%d-t7.bin' % L)
        if t == 7 and L == N_LAYERS - 1:
            h.astype(np.float32).tofile('/tmp/q35-ref-L39-t7.bin')
        if L == 0:
            h.astype(np.float32).tofile('/tmp/q35-ref-L0-t%d.bin' % t)
        if t == 0 and L == 0:
            acc.astype(np.float32).tofile('/tmp/q35-ref-moeacc0.bin')
        if t == 0 and L == 0:
            print('ref L0: moe acc rms %.4g (xin2 %.4g)' % (
                float(np.sqrt((acc**2).mean())),
                float(np.sqrt((xin2**2).mean()))), flush=True)
        if (t == 0) and (L < 8):
            print('t%d L%d state rms %.4g' % (t, L,
                  float(np.sqrt((h**2).mean()))), flush=True)
        if t == 1 and L < 8:
            print('t%d L%d state rms %.4g' % (t, L,
                  float(np.sqrt((h**2).mean()))), flush=True)
    # final norm
    fn = ten_float(tl['layers'][-1], '.norm.weight')
    if fn is None:
        fn = np.ones(H, dtype=np.float32)
    h_pre = h.copy()
    h = rmsnorm(h, fn, 1e-6)
    if t == T - 1 and (L == 0 or L % 8 == 0 or L == N_LAYERS - 1):
        print('t%d L%d state rms %.4g' % (t, L, float(np.sqrt((h**2).mean()))),
              flush=True)
    if t > 0 and (L == 0 or L == 8 or L == N_LAYERS - 1):
        print('t%d L%d state rms %.4g' % (t, L, float(np.sqrt((h**2).mean()))),
              flush=True)
    out_states.append(h_pre)   # PRE-final-norm (engine dumps pre-norm)
    h.astype(np.float32).tofile('/tmp/q35-ref-postnorm.bin')  # POST-norm

final = out_states[-1]
np.savetxt('/tmp/q35-ref-state.txt', final, fmt='%.6e')
final.astype(np.float32).tofile('/tmp/q35-ref-state.bin')
print('ref state rms:', float(np.sqrt((final ** 2).mean())))
