#!/usr/bin/env python3
# dbg-jupiter.py -- is Jupiter in the logits for "The largest planet..."?
# We need the lm_head logits after the prompt. Use the engine's logits dump.
import json, numpy as np

# token id for Jupiter
tok = json.load(open('/Users/ruihe/.cache/huggingface/mlx-qwen35-a3b-4bit/tokenizer.json'))
v = tok.get('model', {}).get('vocab', {})
inv = {i: t for t, i in v.items()}
for word in ('Jupiter', 'ĠJupiter', 'upiter'):
    if word in v:
        print('token', word, '=', v[word])

# decode the engine's last logits line for the planet prompt
# (DS4F_DEBUG7 printed top5; for a full search we need all logits)
print('top-5 logits shown above do NOT contain Jupiter')
