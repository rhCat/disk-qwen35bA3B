#!/usr/bin/env python3
import json
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(
    '/Users/ruihe/.cache/huggingface/mlx-qwen35-a3b-4bit')
v = tok.get_vocab()
inv = {i: t for t, i in v.items()}
ids = [248045, 846, 198, 760, 6511, 314, 9338, 369, 248046, 198,
       248045, 74455, 198, 248068, 198]
print('tokens:', [inv.get(i) for i in ids])
# also full chat template for the France question
msgs = [{'role': 'user', 'content': 'The capital of France is'}]
s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
print('full template repr:', repr(s))
print('with think:', repr(s + '<think>'))
