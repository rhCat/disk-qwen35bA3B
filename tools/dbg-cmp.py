#!/usr/bin/env python3
# Correct comparison: src at 8+hlen vs trunk, first words of each tensor.
import json, struct, os
base = '/Users/ruihe/.cache/huggingface/mlx-qwen35-a3b-4bit'
wm = json.load(open(base + '/model.safetensors.index.json'))['weight_map']
def src_bytes(name, nbytes):
    fn = wm[name]
    with open(os.path.join(base, fn), 'rb') as f:
        raw = f.read(8)
        hlen = struct.unpack('<Q', raw)[0]
        hdr = json.loads(f.read(hlen))
        pb = 8 + hlen
        a, b = hdr[name]['data_offsets']
        f.seek(pb + a)
        return f.read(nbytes)
tb = open('/tmp/q35-trunk/trunk.bin', 'rb').read()
offs = open('/tmp/q35-trunk/trunk.offsets', 'rb').read()
n = struct.unpack('<Q', offs[:8])[0]
o0 = struct.unpack('<%dQ' % n, offs[8:8+8*n])
t = json.load(open('/tmp/q35-trunk/trunk.json'))
tens = sorted(t['layers'][0]['tensors'], key=lambda z: z['off'])
for x in tens[:5]:
    nm = x['n']
    if nm not in wm: continue
    src = src_bytes(nm, 16)
    sw = struct.unpack('<4I', src)
    tw = struct.unpack_from('<4I', tb, o0[0] + x['off'])
    print('%s %-52s trunk=%08x%08x src=%08x%08x' % (
        'OK ' if tw == list(sw) else 'BAD', nm.split('layers.0.')[1],
        tw[0], tw[1], sw[0], sw[1]))
