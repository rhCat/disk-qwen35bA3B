#!/usr/bin/env python3
# gpu-logits.py -- compare CPU vs GPU top-1 tokens per position from the
# DS4F_DEBUG7 logs. Same input prompt, same greedy: token ids must match.
import re

def toks(path):
    out = []
    for m in re.finditer(r'logits: t(\d+) state_rms.*?top5\s+\[\s*(\d+) ', open(path, encoding='utf-8', errors='replace').read()):
        out.append((int(m.group(1)), int(m.group(2))))
    return out

cpu = toks('/tmp/gpu-cpu.log')
gpu = toks('/tmp/gpu-mtl.log')
n = min(len(cpu), len(gpu))
mism = [(c, g) for c, g in zip(cpu, gpu) if c != g]
print(f'cpu tokens: {len(cpu)}, gpu tokens: {len(gpu)}, compared: {n}')
print(f'MATCHING top-1 ids: {n - len(mism)}/{n}')
for c, g in mism[:5]:
    print(f'  t{c[0]}: cpu={c[1]} gpu={g[1]}')
