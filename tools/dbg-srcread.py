#!/usr/bin/env python3
import json, numpy as np

f = '/Users/ruihe/.cache/huggingface/mlx-qwen35-a3b-4bit/model-00001-of-00004.safetensors'
with open(f, 'rb') as fh:
    n = int.from_bytes(fh.read(8), 'little')
    hdr = json.loads(fh.read(n))
name = 'language_model.model.embed_tokens.weight'
meta = hdr[name]
print('meta:', meta)
with open(f, 'rb') as fh:
    fh.seek(8 + n + meta['data_offsets'][0])
    raw = fh.read(meta['data_offsets'][1] - meta['data_offsets'][0])
print('raw bytes:', len(raw))
print('expected U32 [248320,256]:', 248320 * 256 * 4)
a = np.frombuffer(raw, dtype=np.uint32)
print('got', a.size, 'U32 words; first 8:', a[:8])
