#!/usr/bin/env python3
# acc-decode.py -- decode gen tokens from DEBUG7 top1 lines.
# Usage: acc-decode.py <log> <ids-file> [maxchars]
import re
import sys

log = sys.argv[1]
npids = len(open(sys.argv[2]).read().split(","))
maxchars = int(sys.argv[3]) if len(sys.argv) > 3 else 120

toks = []
for line in open(log, encoding="utf-8", errors="replace"):
    # logits: t<N> ... top5 [ <id> <score> <token>] [ ...]
    m = re.search(r"logits: t(\d+) .*?top5\s+\[\s*\d+ [\d.eE+-]+\s*([^\]]*)\]", line)
    if m and int(m.group(1)) >= npids:
        toks.append(m.group(2).strip())

text = " ".join(toks)
print(text[:maxchars])
