#!/usr/bin/env python3
# ld-diff.py -- compare per-layer state dumps between two dirs.
import glob
import os
import struct
import sys

sd = sys.argv[1]
cd = sys.argv[2]
files = sorted(glob.glob(os.path.join(sd, "q35-eng-L*-t8.bin")),
               key=lambda p: int(os.path.basename(p).split("-")[2][1:]))
first = None
for sp in files:
    name = os.path.basename(sp)
    cp = os.path.join(cd, name)
    if not os.path.exists(cp):
        print(f"{name}: MISSING in chunk")
        continue
    a = open(sp, "rb").read()
    b = open(cp, "rb").read()
    n = min(len(a), len(b)) // 4
    if len(a) != len(b):
        print(f"{name}: size {len(a)} vs {len(b)}")
        continue
    fa = struct.unpack(f"<{n}f", a)
    fb = struct.unpack(f"<{n}f", b)
    ndiff = 0
    maxdelta = 0.0
    first_i = -1
    for i in range(n):
        if fa[i] != fb[i]:
            ndiff += 1
            if first_i < 0:
                first_i = i
            d = abs(fa[i] - fb[i])
            if d > maxdelta:
                maxdelta = d
    if ndiff:
        print(f"{name}: {ndiff}/{n} floats differ, first@{first_i} "
              f"maxdelta {maxdelta:.3e}")
        if first is None:
            first = name
    else:
        print(f"{name}: IDENTICAL")
print(f"\nFIRST DIVERGENCE: {first}")
