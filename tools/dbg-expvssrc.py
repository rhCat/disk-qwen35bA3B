#!/usr/bin/env python3
# expert source vs pool: raw bytes + decoded floats, L0 e0 gate_proj
import json, numpy as np

SRC = '/Users/ruihe/.cache/huggingface/mlx-qwen35-a3b-4bit'
POOL = '/tmp/q35-pool'

def bf16(bits):
    u = bits.astype(np.uint32) << 16
    return u.view(np.float32)

def decode4(w, sc, bi):
    w = w.astype(np.uint32)
    sh = np.array([0, 4, 8, 12, 16, 20, 24, 28], dtype=np.uint32)
    nib = (w[..., None] >> sh) & 0xF
    R, P = w.shape
    vals = nib.reshape(R, P * 8)
    C = P * 8
    G = C // 64
    s = bf16(sc.astype(np.uint16)).reshape(R, G)
    b = bf16(bi.astype(np.uint16)).reshape(R, G)
    return vals.astype(np.float32) * np.repeat(s, 64, axis=1) + np.repeat(b, 64, axis=1)

def load_src(f, name):
    with open(f, 'rb') as fh:
        n = int.from_bytes(fh.read(8), 'little')
        hdr = json.loads(fh.read(n))
    if name not in hdr:
        return None
    meta = hdr[name]
    with open(f, 'rb') as fh:
        fh.seek(8 + n + meta['data_offsets'][0])
        raw = fh.read(meta['data_offsets'][1] - meta['data_offsets'][0])
    dt = {'U32': np.uint32, 'BF16': np.uint16}[meta['dtype']]
    return np.frombuffer(raw, dtype=dt).reshape(meta['shape'])

# find which shard has layer 0 experts
import glob
shards = sorted(glob.glob(SRC + '/model-*.safetensors'))
def src_expert(layer, e, proj):
    for f in shards:
        with open(f, 'rb') as fh:
            n = int.from_bytes(fh.read(8), 'little')
            hdr = json.loads(fh.read(n))
        base = 'language_model.model.layers.%d.mlp.switch_mlp.%s' % (layer, proj)
        if base + '.weight' in hdr:
            return (load_src(f, base + '.weight'), load_src(f, base + '.scales'),
                    load_src(f, base + '.biases'))
    return None

w, s, b = src_expert(0, 0, 'gate_proj')
print('src gate_proj:', w.shape, s.shape, b.shape if w is not None else None)

pool = json.load(open(POOL + '/manifest.json'))
tgt = None
for t in pool['tensors']:
    if t['layer'] == 0 and t['expert'] == 0 and 'gate_proj' in t['name']:
        tgt = t
        print('pool candidate:', t['name'], t.get('shape'), t['v_off'], t['v_nbytes'])
        break
if tgt is None:
    print('no gate_proj in manifest for L0 e0; names:')
    for t in pool['tensors']:
        if t['layer'] == 0 and t['expert'] == 0:
            print('  ', t['name'])
else:
    pb = open(POOL + '/pool.bin', 'rb').read()
    pv = np.frombuffer(pb, dtype=np.uint32, count=tgt['v_nbytes'] // 4, offset=tgt['v_off'])
    # source is stacked [256 experts, 512, 256]; pool is one expert [512, 256]
    sw = w[0].reshape(-1)  # expert 0
    if pv.size == sw.size and np.array_equal(pv, sw):
        print('RAW BYTES EQUAL (expert 0 weight)')
    else:
        print('weight raw DIFFER: src-e0 %d elems, pool %d' % (sw.size, pv.size))
        if pv.size == sw.size:
            d = (pv.astype(np.int64) - sw.astype(np.int64))
            print('  n-diff %d max %d at %d' % ((d != 0).sum(), np.abs(d).max(), np.abs(d).argmax()))
            print('  pool[0..3] %s' % [hex(int(x)) for x in pv[:4]])
            print('  src [0..3] %s' % [hex(int(x)) for x in sw[:4]])
    # scales/biases
    ps = np.frombuffer(pb, dtype=np.uint16, count=tgt['s_nbytes'] // 2, offset=tgt['s_off'])
    pb_ = np.frombuffer(pb, dtype=np.uint16, count=tgt['b_nbytes'] // 2, offset=tgt['b_off'])
    ss = s[0].reshape(-1)
    bb = b[0].reshape(-1)
    print('scales raw', 'EQUAL' if np.array_equal(ps, ss) else 'DIFFER (n=%d)' % (ps != ss).sum())
    print('biases raw', 'EQUAL' if np.array_equal(pb_, bb) else 'DIFFER (n=%d)' % (pb_ != bb).sum())
