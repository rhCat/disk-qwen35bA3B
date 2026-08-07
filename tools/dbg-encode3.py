#!/usr/bin/env python3
# dbg-encode3.py -- encode an arbitrary prompt with the REAL tokenizer,
# prints ids for pasting into dbg-reffwd.py.
import sys
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(
    '/Users/ruihe/.cache/huggingface/mlx-qwen35-a3b-4bit')
text = sys.argv[1]
ids = tok.encode(text, add_special_tokens=False)
print('n =', len(ids))
print('ids =', ids)
for i, t in enumerate(ids):
    print('%2d %6d %s' % (i, t, repr(tok.decode([t]))))
