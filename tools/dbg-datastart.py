#!/usr/bin/env python3
# Determine the true data start: read the first weight tensor at both
# candidate offsets and check which gives a plausible BF16 sequence.
import json, struct, os
base = '/Users/ruihe/.cache/huggingface/mlx-qwen35-a3b-4bit'
wm = json.load(open(base + '/model.safetensors.index.json'))['weight_map']
name = 'language_model.model.layers.0.input_layernorm.weight'
fn = wm[name]
p = os.path.join(base, fn)
with open(p, 'rb') as f:
    raw = f.read(8)
    hlen = struct.unpack('<Q', raw)[0]
    hdr = json.loads(f.read(hlen))
    a, b = hdr[name]['data_offsets']
    for pb in (8 + hlen, (8 + hlen + 7) & ~7):
        f.seek(pb + a)
        d = f.read(16)
        words = struct.unpack('<8H', d)
        vals = [struct.unpack('<f', struct.pack('<I', v << 16))[0] for v in words]
        print('pb=%d vals: %s' % (pb, ['%.4f' % v for v in vals]))
