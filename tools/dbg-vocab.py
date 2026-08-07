#!/usr/bin/env python3
import json
tok = json.load(open('/Users/ruihe/.cache/huggingface/mlx-qwen35-a3b-4bit/tokenizer.json'))
vocab = tok.get('model', {}).get('vocab', {})
inv = {v: k for k, v in vocab.items()}
for tid in (1, 9419):
    print(tid, repr(inv.get(tid)))
