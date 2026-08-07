#!/usr/bin/env python3
# dbg-decode.py -- decode generated token ids with the real tokenizer.
import sys
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(
    "/Users/ruihe/.cache/huggingface/mlx-qwen35-a3b-4bit")
ids = [int(x) for x in sys.argv[1:]]
print("ids:", ids)
print("text:", repr(tok.decode(ids)))
