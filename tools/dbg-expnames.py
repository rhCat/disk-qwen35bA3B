#!/usr/bin/env python3
import json

f = '/Users/ruihe/.cache/huggingface/mlx-qwen35-a3b-4bit/model-00002-of-00004.safetensors'
with open(f, 'rb') as fh:
    n = int.from_bytes(fh.read(8), 'little')
    hdr = json.loads(fh.read(n))
names = [k for k in hdr if k != '__metadata__']
mlp = [k for k in names if 'layers.9.mlp' in k]
for k in sorted(mlp):
    print(k.replace('language_model.model.layers.9.', ''), hdr[k]['dtype'], hdr[k]['shape'])
