#!/usr/bin/env python3
# mk20k.py -- build a ~20K-token prompt by repeating a passage, write ids.
# Uses the REAL tokenizer so the text is in-distribution.
import sys
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(
    '/Users/ruihe/.cache/huggingface/mlx-qwen35-a3b-4bit')

passage = ("The Roman Empire reached its greatest territorial extent under the "
           "emperor Trajan in 117 AD, when it controlled roughly five million "
           "square kilometers of land. The empire was divided into eastern and "
           "western halves in 285 AD by Diocletian, and the western half fell "
           "in 476 AD when the last emperor Romulus Augustulus was deposed by "
           "Odoacer. The eastern half, known as the Byzantine Empire, survived "
           "until 1453 when Constantinople fell to the Ottoman Empire under "
           "Mehmed the Conqueror. The Byzantine Empire preserved much of Roman "
           "law, Greek learning, and Christian theology throughout the "
           "medieval period, and its scholars played a key role in the "
           "Renaissance by transmitting classical texts to western Europe. "
           "The fall of Constantinople also marked the end of the Middle Ages "
           "and the beginning of a new era of exploration and trade routes "
           "around the world.")

# one paragraph ~ 129 tokens; target 20K
ids = []
while len(ids) < 20000:
    ids += tok.encode(passage, add_special_tokens=False)
ids = ids[:20000]
print('tokens:', len(ids))

with open('/tmp/q35-prompt20k.txt', 'w') as f:
    f.write(','.join(str(i) for i in ids))
print('wrote /tmp/q35-prompt20k.txt')
