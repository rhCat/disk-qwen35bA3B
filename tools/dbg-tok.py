#!/usr/bin/env python3
import json
tok = json.load(open('/Users/ruihe/.cache/huggingface/mlx-qwen35-a3b-4bit/tokenizer.json'))
v = tok.get('model', {}).get('vocab', {})
print('vocab size', len(v))
inv = {i: t for t, i in v.items()}
for tid in (1269, 1735, 1567, 1307, 521, 776, 8678):
    print(tid, repr(inv.get(tid)))
