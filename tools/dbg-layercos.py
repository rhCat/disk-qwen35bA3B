#!/usr/bin/env python3
# dbg-layercos.py -- cosine of engine vs ref per-layer states at t8.
import numpy as np

for L in range(0, 40, 4):
    try:
        e = np.fromfile('/tmp/q35-eng-L%d-t8.bin' % L, dtype=np.float32)
        r = np.fromfile('/tmp/q35-ref-L%d-t8.bin' % L, dtype=np.float32)
    except FileNotFoundError:
        continue
    c = np.dot(e, r) / (np.linalg.norm(e) * np.linalg.norm(r))
    er = np.sqrt((e**2).mean())
    rr = np.sqrt((r**2).mean())
    print('L%-3d cosine %.4f  eng-rms %.4f  ref-rms %.4f  rel %.3f' % (
        L, c, er, rr, er / rr))
