#!/usr/bin/env python3
# dbg-encode2.py -- encode a prompt with the REAL tokenizer.
import json
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(
    '/Users/ruihe/.cache/huggingface/mlx-qwen35-a3b-4bit')
s = "The largest planet in the solar system is Jupiter, which is famous for"
ids = tok.encode(s, add_special_tokens=False)
print('ids:', ids)
print('n:', len(ids))
print('decoded:', tok.decode(ids))
