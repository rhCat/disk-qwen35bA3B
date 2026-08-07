#!/usr/bin/env python3
import json
t = json.load(open('/Users/ruihe/.cache/huggingface/mlx-qwen35-a3b-4bit/tokenizer.json'))
vocab = t['model']['vocab']
print('vocab size:', len(vocab))
for tok in ['<|im_start|>', '<|im_end|>', '<think>', '<|im_start|>'.replace('<', 'Ġ')]:
    print(repr(tok), 'in vocab:', tok in vocab, '->', vocab.get(tok))
# the ids the engine produced for <|im_start|>: 28 27 91 316 4747 91 29
inv = {v: k for k, v in vocab.items()}
for i in [28, 27, 91, 316, 4747, 29, 561, 198, 220]:
    print(i, repr(inv.get(i)))
