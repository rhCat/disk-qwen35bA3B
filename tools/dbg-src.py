#!/usr/bin/env python3
# dbg-src.py -- verify trunk/pool bytes are a faithful repack of the
# ORIGINAL MLX safetensors (both mlx4). Compare RAW BYTES for embed row
# and one expert's gate_proj weight/scales/biases.
import json, numpy as np

SRC = '/Users/ruihe/.cache/huggingface/mlx-qwen35-a3b-4bit'
TRUNK = '/tmp/q35-trunk'
POOL = '/tmp/q35-pool'

def load_src_tensor(f, name):
    with open(f, 'rb') as fh:
        n = int.from_bytes(fh.read(8), 'little')
        hdr = json.loads(fh.read(n))
    if name not in hdr:
        return None, None, None
    meta = hdr[name]
    with open(f, 'rb') as fh:
        fh.seek(8 + n + meta['data_offsets'][0])
        raw = fh.read(meta['data_offsets'][1] - meta['data_offsets'][0])
    print('read', name, len(raw), 'bytes')
    dt = {'U32': np.uint32, 'BF16': np.uint16, 'F32': np.float32,
          'F16': np.float16, 'I8': np.int8}[meta['dtype']]
    return np.frombuffer(raw, dtype=dt).reshape(meta['shape']), meta['dtype'], meta['shape']

def cmp(label, a, b, a_off, b_off, n):
    x = a.flatten()[a_off:a_off + n]
    y = b.flatten()[b_off:b_off + n]
    if x.size != y.size:
        print(label, 'SIZE MISMATCH', x.size, y.size)
        return
    d = np.abs(x.astype(np.int64) - y.astype(np.int64))
    print(label, 'equal' if d.sum() == 0 else 'DIFFER max %d at %d' % (d.max(), d.argmax()))

# ---- embed: source vs trunk embed.bin
w_src, dt, sh = load_src_tensor(SRC + '/model-00001-of-00004.safetensors',
                                'language_model.model.embed_tokens.weight')
sc_src, _, _ = load_src_tensor(SRC + '/model-00001-of-00004.safetensors',
                               'language_model.model.embed_tokens.scales')
bi_src, _, _ = load_src_tensor(SRC + '/model-00001-of-00004.safetensors',
                               'language_model.model.embed_tokens.biases')
e = json.load(open(TRUNK + '/embed.json'))
eb = open(TRUNK + '/embed.bin', 'rb').read()
tok = 760
cmp('embed weight row', w_src, np.frombuffer(eb, dtype=np.uint32), tok * 256, e['weight']['off'] // 4, 256)
cmp('embed scales row', sc_src, np.frombuffer(eb, dtype=np.uint16), tok * 32, e['scale']['off'] // 2, 32)
cmp('embed bias row', bi_src, np.frombuffer(eb, dtype=np.uint16), tok * 32, e['bias']['off'] // 2, 32)

# ---- expert L0 e0 gate_proj: source vs pool.bin
gw, _, _ = load_src_tensor(SRC + '/model-00002-of-00004.safetensors',
                           'language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight')
gs, _, _ = load_src_tensor(SRC + '/model-00002-of-00004.safetensors',
                           'language_model.model.layers.0.mlp.switch_mlp.gate_proj.scales')
gb, _, _ = load_src_tensor(SRC + '/model-00002-of-00004.safetensors',
                           'language_model.model.layers.0.mlp.switch_mlp.gate_proj.biases')
pool = json.load(open(POOL + '/manifest.json'))
pb = open(POOL + '/pool.bin', 'rb').read()
# find L0 e0 gate_proj in the manifest
tgt = None
for t in pool['tensors']:
    if t['layer'] == 0 and t['expert'] == 0 and t['name'].endswith('gate_proj.weight'):
        tgt = t
        break
if tgt is None:
    print('expert tensor not found in manifest')
else:
    print('pool gate_proj meta:', tgt['v_off'], tgt['v_nbytes'], tgt.get('shape'))
    pv = np.frombuffer(pb, dtype=np.uint32, count=tgt['v_nbytes'] // 4, offset=tgt['v_off'])
    cmp('expert gw', gw, pv, 0, 0, gw.size)
