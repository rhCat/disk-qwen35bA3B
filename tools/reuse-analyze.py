#!/usr/bin/env python3
# reuse-analyze.py -- adjacent-layer expert overlap from the rtop log.
import re
import sys

layers = {}
for line in open('/tmp/reuse.log', encoding='utf-8', errors='replace'):
    m = re.search(r'\[rtop\] L(\d+) sel=([\d ]+)', line)
    if m:
        L = int(m.group(1))
        ex = [int(x) for x in m.group(2).split()]
        layers.setdefault(L, []).append(set(ex))

Ls = sorted(layers)
if len(Ls) < 2:
    print('not enough layer samples')
    sys.exit(0)
n = min(len(layers[a]) for a in Ls)
tot = 0
pairs = 0
for i in range(n):
    for a, b in zip(Ls, Ls[1:]):
        sa, sb = layers[a][i], layers[b][i]
        tot += len(sa & sb)
        pairs += 1
print(f'layers sampled: {Ls}, tokens: {n}')
print(f'adjacent-layer expert overlap: {tot}/{pairs*8} = {tot/(pairs*8)*100:.1f}%')
t_over = 0
t_pairs = 0
for L in Ls:
    for i in range(len(layers[L]) - 1):
        t_over += len(layers[L][i] & layers[L][i + 1])
        t_pairs += 1
print(f'token-to-token same-layer overlap: {t_over}/{t_pairs*8} = {t_over/(t_pairs*8)*100:.1f}%')
