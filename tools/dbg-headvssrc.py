#!/usr/bin/env python3
# dbg-headvssrc.py -- verify head.bin (lm_head) vs the original safetensors.
import json, numpy as np, glob

SRC = '/Users/ruihe/.cache/huggingface/mlx-qwen35-a3b-4bit'
TRUNK = '/tmp/q35-trunk'

shards = sorted(glob.glob(SRC + '/model-*.safetensors'))
# find lm_head in any shard
name = None
meta = None
for f in shards:
    with open(f, 'rb') as fh:
        n = int.from_bytes(fh.read(8), 'little')
        hdr = json.loads(fh.read(n))
    for cand in ('lm_head.weight', 'language_model.lm_head.weight',
                 'model.lm_head.weight'):
        if cand in hdr:
            name, meta = cand, hdr[cand]
            print('found', cand, 'in', f.split('/')[-1])
            break
    if name:
        break
if not name:
    print('lm_head NOT FOUND; shard names with lm:')
    for f in shards:
        with open(f, 'rb') as fh:
            n = int.from_bytes(fh.read(8), 'little')
            hdr = json.loads(fh.read(n))
        for k in hdr:
            if 'lm_head' in k or 'head' in k.lower():
                print(' ', k, hdr[k]['dtype'], hdr[k]['shape'])
    raise SystemExit

with open(f, 'rb') as fh:
    fh.seek(8 + n + meta['data_offsets'][0])
    raw = fh.read(meta['data_offsets'][1] - meta['data_offsets'][0])
dt = {'U32': np.uint32, 'BF16': np.uint16, 'F32': np.float32}[meta['dtype']]
src = np.frombuffer(raw, dtype=dt)
print('src', meta['dtype'], meta['shape'], 'elems', src.size)

h = json.load(open(TRUNK + '/head.json'))
hb = open(TRUNK + '/head.bin', 'rb').read()
w = np.frombuffer(hb, dtype=np.uint32, count=h['weight']['nbytes'] // 4,
                  offset=h['weight']['off'])
s = np.frombuffer(hb, dtype=np.uint16, count=h['scale']['nbytes'] // 2,
                  offset=h['scale']['off'])
b = np.frombuffer(hb, dtype=np.uint16, count=h['bias']['nbytes'] // 2,
                  offset=h['bias']['off'])
print('trunk weight elems', w.size, 'scale', s.size, 'bias', b.size)
if w.size == src.size:
    d = w.astype(np.int64) - src.astype(np.int64)
    print('WEIGHT bytes:', 'EQUAL' if d.sum() == 0 else 'DIFFER max %d at %d' % (np.abs(d).max(), np.abs(d).argmax()))
else:
    print('SIZE MISMATCH', w.size, src.size)
