#!/usr/bin/env python3
import json
f = '/Users/ruihe/.cache/huggingface/mlx-qwen35-a3b-4bit/model-00001-of-00004.safetensors'
with open(f, 'rb') as fh:
    n = int.from_bytes(fh.read(8), 'little')
    hdr = json.loads(fh.read(n))
names = [k for k in hdr.keys() if k not in ('__metadata__',)]
print(len(names), 'tensors')
first = hdr[names[0]]
print('entry keys:', list(first.keys()))
for nm in names[:6]:
    print(nm, first if False else '')
# print dtype/shape per entry using its own keys
for nm in names[:8]:
    e = hdr[nm]
    print(nm, e.get('dtype'), e.get('shape'))
ex = [n for n in names if 'experts' in n]
print(len(ex), 'expert tensors; sample:')
for nm in ex[:3]:
    e = hdr[nm]
    print(nm, e.get('dtype'), e.get('shape'))
