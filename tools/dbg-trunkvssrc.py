#!/usr/bin/env python3
# dbg-trunkvssrc.py -- verify TRUNK tensor bytes vs the ORIGINAL safetensors.
# The embed + one expert were verified; this covers the linear/GQA/norm
# tensors that live in trunk.bin.
import json, numpy as np, glob

SRC = '/Users/ruihe/.cache/huggingface/mlx-qwen35-a3b-4bit'
TRUNK = '/tmp/q35-trunk'

def load_src(f, name):
    with open(f, 'rb') as fh:
        n = int.from_bytes(fh.read(8), 'little')
        hdr = json.loads(fh.read(n))
    if name not in hdr:
        return None, None, None
    meta = hdr[name]
    with open(f, 'rb') as fh:
        fh.seek(8 + n + meta['data_offsets'][0])
        raw = fh.read(meta['data_offsets'][1] - meta['data_offsets'][0])
    dt = {'U32': np.uint32, 'BF16': np.uint16, 'F32': np.float32}[meta['dtype']]
    return np.frombuffer(raw, dtype=dt), meta['dtype'], meta['shape']

shards = sorted(glob.glob(SRC + '/model-*.safetensors'))

def src_tensor(name):
    for f in shards:
        with open(f, 'rb') as fh:
            n = int.from_bytes(fh.read(8), 'little')
            hdr = json.loads(fh.read(n))
        if name in hdr:
            meta = hdr[name]
            with open(f, 'rb') as fh:
                fh.seek(8 + n + meta['data_offsets'][0])
                raw = fh.read(meta['data_offsets'][1] - meta['data_offsets'][0])
            dt = {'U32': np.uint32, 'BF16': np.uint16, 'F32': np.float32}[meta['dtype']]
            return np.frombuffer(raw, dtype=dt), meta['dtype'], meta['shape']
    return None, None, None

tl = json.load(open(TRUNK + '/trunk.json'))
tb = open(TRUNK + '/trunk.bin', 'rb').read()
offs_all = np.fromfile(TRUNK + '/trunk.offsets', dtype=np.uint64)
offs = offs_all[1::2]

def trunk_tensor(layer_idx, suffix):
    layer = tl['layers'][layer_idx]
    for x in layer['tensors']:
        if x['n'].endswith(suffix):
            base = int(offs[layer['layer']])
            dt = {'U32': np.uint32, 'BF16': np.uint16, 'F32': np.float32}[x['dtype']]
            n = x['nbytes'] // np.dtype(dt).itemsize
            return np.frombuffer(tb, dtype=dt, count=n, offset=base + x['off'])
    return None

def cmp(label, src_a, tr_a):
    if src_a is None or tr_a is None:
        print(label, 'MISSING')
        return
    if src_a.size != tr_a.size:
        print(label, 'SIZE DIFF', src_a.size, tr_a.size)
        return
    d = np.abs(src_a.astype(np.int64) - tr_a.astype(np.int64))
    print(label, 'equal' if d.sum() == 0 else 'DIFFER max %d at %d' % (d.max(), d.argmax()))

# layer 0: input_layernorm, in_proj_qkv, conv1d, A_log, dt_bias, q_proj, o_proj, norm
checks = [
    (0, '.input_layernorm.weight', 'language_model.model.layers.0.input_layernorm.weight'),
    (0, '.linear_attn.in_proj_qkv.weight', 'language_model.model.layers.0.linear_attn.in_proj_qkv.weight'),
    (0, '.linear_attn.conv1d.weight', 'language_model.model.layers.0.linear_attn.conv1d.weight'),
    (0, '.linear_attn.A_log', 'language_model.model.layers.0.linear_attn.A_log'),
    (0, '.linear_attn.dt_bias', 'language_model.model.layers.0.linear_attn.dt_bias'),
    (0, '.self_attn.q_proj.weight', 'language_model.model.layers.0.self_attn.q_proj.weight'),
    (0, '.self_attn.k_proj.weight', 'language_model.model.layers.0.self_attn.k_proj.weight'),
    (0, '.self_attn.o_proj.weight', 'language_model.model.layers.0.self_attn.o_proj.weight'),
    (0, '.mlp.gate.weight', 'language_model.model.layers.0.mlp.gate.weight'),
    (0, '.post_attention_layernorm.weight', 'language_model.model.layers.0.post_attention_layernorm.weight'),
    (7, '.self_attn.q_proj.weight', 'language_model.model.layers.7.self_attn.q_proj.weight'),
    (7, '.linear_attn.in_proj_z.weight', 'language_model.model.layers.7.linear_attn.in_proj_z.weight'),
]
for lidx, suffix, srcname in checks:
    src_a, dt, sh = src_tensor(srcname)
    tr_a = trunk_tensor(lidx, suffix)
    cmp('%s L%d %s' % (dt, lidx, suffix), src_a, tr_a)
