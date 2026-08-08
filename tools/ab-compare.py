#!/usr/bin/env python3
# ab-compare.py -- gen-only top1 comparison between two DEBUG7 logs.
import re
import sys

serial = sys.argv[1]
chunk = sys.argv[2]
ids = sys.argv[3]
npids = len(open(ids).read().split(","))


def tops(path):
    out = []
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.search(r"logits: t(\d+) .*?top5\s+\[\s*(\d+) ", line)
        if m and int(m.group(1)) >= npids:
            out.append((int(m.group(1)), int(m.group(2))))
    return out


s = tops(serial)
c = tops(chunk)
print(f"serial gen: {len(s)}, chunk gen: {len(c)}")
if len(s) != len(c):
    print(f"LENGTH MISMATCH")
else:
    same = sum(1 for a, b in zip(s, c) if a == b)
    print(f"identical top1: {same}/{len(s)}")
    for a, b in zip(s, c):
        if a != b:
            print(f"  DIFF t{a[0]}: serial {a[1]} vs chunk {b[1]}")
