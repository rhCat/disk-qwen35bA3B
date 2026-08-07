#!/usr/bin/env python3
# dbg-chatfmt.py -- build the text-only chat prompt the way mlx-lm would,
# and compare answers: raw vs chat-formatted.
import json
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(
    '/Users/ruihe/.cache/huggingface/mlx-qwen35-a3b-4bit')

def chat_prompt(user_text):
    # Qwen3 chat: <|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n
    msgs = [{'role': 'user', 'content': user_text}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

for q in ("The capital of France is",
          "The largest planet in the solar system is",
          "What is the chemical symbol for gold?"):
    s = chat_prompt(q)
    ids = tok.encode(s, add_special_tokens=False)
    print('Q:', q)
    print('  fmt:', repr(s[:80]))
    print('  ids:', ids[:20], '... n=', len(ids))
    print()
