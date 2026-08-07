#!/usr/bin/env python3
# dbg-tokencos.py -- per-token L0 cosine: where does the divergence start?
import numpy as np

for t in range(10):
    try:
        e = np.fromfile('/tmp/q35-eng-L0-t%d.bin' % t, dtype=np.float32)
        r = np.fromfile('/tmp/q35-ref-L0-t%d.bin' % t, dtype=np.float32)
    except FileNotFoundError:
        continue
    if e.size != r.size or e.size == 0:
        print('t%d: size mismatch %d/%d' % (t, e.size, r.size))
        continue
    c = np.dot(e, r) / (np.linalg.norm(e) * np.linalg.norm(r))
    er = np.sqrt((e**2).mean())
    rr = np.sqrt((r**2).mean())
    print('t%d cosine %.6f  eng %.5f  ref %.5f  rel %.4f' % (t, c, er, rr, er / rr))
