#!/usr/bin/env python3
# dbg-mlxlm.py -- ground truth: real mlx-lm on the same model dir + prompt,
# greedy, WITH disk offload so RSS stays ~3-4GB (same philosophy as our
# disk engine -- NOT a full-in-RAM load).
import sys, time, resource
import numpy as np
import mlx.core as mx
from mlx_lm import generate, load
from mlx_lm.utils import load_model, load_tokenizer

MODEL = "/Users/ruihe/.cache/huggingface/mlx-qwen35-a3b-4bit"
PROMPT = "The capital of France is"
N_GEN = 12

def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576.0

t0 = time.time()
model, tokenizer = load(MODEL, lazy=True)
print("loaded (lazy) in %.1fs, rss %.2f GB" % (time.time() - t0, rss_gb()),
      flush=True)

t0 = time.time()
out = generate(model, tokenizer, prompt=PROMPT, max_tokens=N_GEN,
               sampler=lambda logits: mx.argmax(logits, axis=-1),
               offload=True)
print("=== mlx-lm greedy (disk-offload) ===")
print(out)
print("(%.1fs, peak rss %.2f GB)" % (time.time() - t0, rss_gb()), flush=True)
