#!/usr/bin/env python3
import json

tok = json.load(open('/Users/ruihe/.cache/huggingface/mlx-qwen35-a3b-4bit/tokenizer.json'))
v = tok.get('model', {}).get('vocab', {})
inv = {i: t for t, i in v.items()}
print('decode [760, 6511, 314, 9338, 369]:')
print('  ', [inv.get(i) for i in (760, 6511, 314, 9338, 369)])
print('encode "The capital of France is":')
print('  ', [v.get(s) for s in ['The', ' capital', ' of', ' France', ' is']])
