#!/usr/bin/env python3
# mk1k.py -- build a ~1K-token prompt (repeat facts), write ids.
import sys
from transformers import AutoTokenizer
sys.path.insert(0, '/Users/ruihe/disk-qwen35bA3B/tools')
from mkqa import facts, filler  # reuse the coherent passage

tok = AutoTokenizer.from_pretrained(
    "/Users/ruihe/.cache/huggingface/mlx-qwen35-a3b-4bit")

doc = "\n\n".join((facts * 3)[:22])
user = f"Read this. {doc}\n\nQuestion: What is the vault code?"
prompt = f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
ids = tok(prompt)["input_ids"]
print(f"prompt tokens: {len(ids)}")
with open("/tmp/q35-1k-ids.txt", "w") as f:
    f.write(",".join(str(i) for i in ids))
