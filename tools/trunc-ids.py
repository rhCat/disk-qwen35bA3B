#!/usr/bin/env python3
# trunc-ids.py -- take the first N tokens of an ids file.
import sys

src = sys.argv[1]
n = int(sys.argv[2])
out = sys.argv[3]
ids = [x for x in open(src).read().split(",") if x.strip()]
with open(out, "w") as f:
    f.write(",".join(ids[:n]))
print(f"wrote {len(ids[:n])} tokens -> {out}")
