#!/usr/bin/env python3
import json
tok = json.load(open('/Users/ruihe/.cache/huggingface/mlx-qwen35-a3b-4bit/tokenizer.json'))
v = tok.get('model', {}).get('vocab', {})
inv = {i: t for t, i in v.items()}
# a longer natural prompt
s = "The largest planet in the solar system is Jupiter, which is famous for"
# BPE-encode by greedy longest-match on the vocab
def encode(text):
    ids = []
    i = 0
    while i < len(text):
        best = None
        for j in range(len(text), i, -1):
            sub = text[i:j]
            if sub in v:
                best = (j, v[sub])
                break
        if best is None:
            best = (i + 1, v.get(text[i], 0))
        ids.append(best[1])
        i = best[0]
    return ids
ids = encode(s)
print('ids:', ids)
print('decoded:', ''.join(inv.get(i, '?') for i in ids))
