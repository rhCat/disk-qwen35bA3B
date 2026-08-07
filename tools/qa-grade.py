#!/usr/bin/env python3
# qa-grade.py -- extract the generated answer from the QA run log and
# check it against the expected vault code. With DS4F_DEBUG7 the log has
# per-token "logits: tN ... top5 [id score text] ..." lines; greedy pick
# is the first bracket. Reconstruct the generated text from token 0 of
# each generated step (t >= npids).
import re

log = open('/tmp/qa-run.log', encoding='utf-8', errors='replace').read()
expected = open('/tmp/q35-qa-expected.txt').read().strip()

m = re.search(r'(\d+) tokens in ([\d.]+) s, ([\d.]+) s/token', log)
npids = len(open('/tmp/q35-qa-ids.txt').read().split(',')) if m else 0
report = m.groups() if m else ('?', '?', '?')

# decode pieces: capture the first [id score text] per "logits: tN" line
pieces = []
for lm in re.finditer(r'logits: t(\d+) state_rms.*?top5\s+\[\s*\d+ [\d.]+ ([^\]]*)\]', log):
    t = int(lm.group(1))
    if t < npids:
        continue                      # prompt positions: skip
    txt = lm.group(2).strip()
    if txt == 'Ċ' or txt == '<0x0A>':
        txt = '\n'
    pieces.append(txt)
answer = ''.join(pieces)

ok = expected in answer or expected.lower() in answer.lower()
print(f'prompt tokens: {report[0]}, {report[1]}s, {report[2]} s/token')
print(f'generated tokens captured: {len(pieces)}')
print(f'expected: {expected}')
print(f'answer:  {answer!r}')
print(f'MATCH: {ok}')
