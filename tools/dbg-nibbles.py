#!/usr/bin/env python3
# dbg-nibbles.py -- nibble distribution of the 4-bit checkpoint.
# A healthy affine 4-bit quant uses all 16 levels roughly uniformly.
# Sign-mismatch conversions show only 0..7 or 8..15; clipped show 0/15.
import json, numpy as np

TRUNK = '/tmp/q35-trunk'
tl = json.load(open(TRUNK + '/trunk.json'))
tb = open(TRUNK + '/trunk.bin', 'rb').read()
offs_all = np.fromfile(TRUNK + '/trunk.offsets', dtype=np.uint64)
offs = offs_all[1::2]

def nibbles(name_sub, layer_idx, limit=4 * 1024 * 1024):
    layer = tl['layers'][layer_idx]
    for x in layer['tensors']:
        if x['dtype'] != 'U32' or not x['n'].endswith(name_sub):
            continue
        base = int(offs[layer['layer']]) + x['off']
        n = min(x['nbytes'] // 4, limit // 4)
        w = np.frombuffer(tb, dtype=np.uint32, count=n, offset=base)
        sh = np.array([0, 4, 8, 12, 16, 20, 24, 28], dtype=np.uint32)
        nib = ((w[..., None] >> sh) & 0xF).ravel()
        hist = np.bincount(nib, minlength=16)
        frac = hist / hist.sum()
        used = (hist > 0).sum()
        print('%-40s L%-2d used=%d/%d  p0=%.3f p7=%.3f p8=%.3f p15=%.3f  max=%.3f' % (
            x['n'][-40:], layer_idx, used, 16,
            frac[0], frac[7], frac[8], frac[15], frac.max()))
        return
    print('not found', name_sub, 'in L', layer_idx)

for L in (0, 7, 39):
    nibbles('.in_proj_qkv.weight', L)
    nibbles('.gate.weight', L)
    nibbles('.q_proj.weight', L)
