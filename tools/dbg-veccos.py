#!/usr/bin/env python3
# dbg-veccos.py -- cosine of engine vs ref L0 t8 h0 delta-rule vectors.
import numpy as np

def cmp(label, n):
    e = np.fromfile('/tmp/q35-eng-L0-%s.bin' % label, dtype=np.float32)
    r = np.fromfile('/tmp/q35-ref-L0-%s.bin' % label, dtype=np.float32)
    c = np.dot(e, r) / (np.linalg.norm(e) * np.linalg.norm(r))
    er = np.sqrt((e**2).mean()); rr = np.sqrt((r**2).mean())
    print('%-6s cos %.6f  eng %.5f  ref %.5f  rel %.4f' % (label, c, er, rr, er / rr))

cmp('k', 128)
cmp('q', 128)
cmp('v', 128)
cmp('delta', 128)
cmp('S', 128 * 128)
