#!/usr/bin/env python3
# dbg-longcos.py -- per-token final-state cosine: engine vs ref over a
# 21-token prompt. Where does any divergence start?
import numpy as np

print('t   cosine   eng-rms   ref-rms   rel')
for t in range(21):
    try:
        e = np.fromfile('/tmp/q35-eng-final-t%d.bin' % t, dtype=np.float32)
        r = np.fromfile('/tmp/q35-ref-final-t%d.bin' % t, dtype=np.float32)
    except FileNotFoundError:
        continue
    if e.size != r.size or e.size == 0:
        print('t%d: size mismatch %d/%d' % (t, e.size, r.size))
        continue
    c = np.dot(e, r) / (np.linalg.norm(e) * np.linalg.norm(r))
    er = np.sqrt((e**2).mean()); rr = np.sqrt((r**2).mean())
    flag = '  <-- DIVERGE' if c < 0.999 else ''
    print('%-3d %.8f  %.6f  %.6f  %.4f%s' % (t, c, er, rr, er / rr, flag))
