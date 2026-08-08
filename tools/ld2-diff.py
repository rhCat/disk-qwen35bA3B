#!/usr/bin/env python3
# ld2-diff.py -- compare per-layer per-token states (L<4, all tokens).
# serial dir has L{t}-t{t}.bin per token; chunk dir has L{t}-t{c0}.bin
# with B*stn floats (all tokens of the chunk, c0=0 first chunk).
import glob
import os
import struct
import sys

sd = sys.argv[1]
cd = sys.argv[2]
layers = [0, 1, 2, 3]
for L in layers:
    # serial per-token files for this layer
    sfiles = sorted(glob.glob(os.path.join(sd, f"q35-eng-L{L}-t*.bin")),
                    key=lambda p: int(os.path.basename(p).split("-t")[1][:-4]))
    # chunk files for this layer (one per chunk, named by c0)
    cfiles = sorted(glob.glob(os.path.join(cd, f"q35-eng-L{L}-t*.bin")),
                    key=lambda p: int(os.path.basename(p).split("-t")[1][:-4]))
    # chunk c0 file holds tokens c0..c0+B-1; compare against serial t{c0}
    first_bad = None
    for cp in cfiles:
        c0 = int(os.path.basename(cp).split("-t")[1][:-4])
        cb = open(cp, "rb").read()
        B = len(cb) // 4 // 2048  # stn floats per token
        for b in range(B):
            tok = c0 + b
            sp = os.path.join(sd, f"q35-eng-L{L}-t{tok}.bin")
            if not os.path.exists(sp):
                continue
            sa = open(sp, "rb").read()
            chunk_tok = cb[b * 2048 * 4:(b + 1) * 2048 * 4]
            if sa != chunk_tok:
                n = min(len(sa), len(chunk_tok)) // 4
                fa = struct.unpack(f"<{n}f", sa)
                fb = struct.unpack(f"<{n}f", chunk_tok)
                nd = sum(1 for i in range(n) if fa[i] != fb[i])
                md = max((abs(fa[i] - fb[i]) for i in range(n)
                          if fa[i] != fb[i]), default=0.0)
                if first_bad is None:
                    first_bad = (L, tok, nd, md)
                if tok <= 8:
                    print(f"L{L} t{tok}: {nd}/{n} differ, maxdelta {md:.3e}")
    if first_bad:
        L, tok, nd, md = first_bad
        print(f">>> FIRST DIVERGENCE: L{L} t{tok} ({nd} floats, maxdelta {md:.3e})")
    else:
        print(f"L{L}: ALL TOKENS IDENTICAL")
