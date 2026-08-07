#!/usr/bin/env python3
# Find the first layer-0 tensor whose trunk bytes don't match the src.
import json, struct
base = '/Users/ruihe/.cache/huggingface/mlx-qwen35-a3b-4bit'
wm = json.load(open(base + '/model.safetensors.index.json'))['weight_map']
SHARDS = {}
def src_bytes(name, nbytes):
    fn = wm[name]
    if fn not in SHARDS:
        f = open(base + '/' + fn, 'rb')
        raw = f.read(8)
        hlen = struct.unpack('<Q', raw)[0]
        hdr = json.loads(f.read(hlen))
        ds = (8 + hlen + 7) & ~7
        SHARDS[fn] = (f, ds, hdr)
    f, ds, hdr = SHARDS[fn]
    e = hdr[name]
    off, nb = e['data_offsets']
    f.seek(ds + off)
    return f.read(nbytes)

tb = open('/tmp/q35-trunk/trunk.bin', 'rb').read()
offs = open('/tmp/q35-trunk/trunk.offsets', 'rb').read()
n = struct.unpack('<Q', offs[:8])[0]
o0 = struct.unpack('<%dQ' % n, offs[8:8+8*n])
t = json.load(open('/tmp/q35-trunk/trunk.json'))
tens = sorted(t['layers'][0]['tensors'], key=lambda z: z['off'])
for x in tens:
    nm = x['n']
    if nm not in wm: 
        print('  NO SRC: %-55s' % nm.split('layers.0.')[1]); continue
    src = src_bytes(nm, min(x['nbytes'], 16))
    tw = struct.unpack_from('<4I', tb, o0[0] + x['off'])
    sw = struct.unpack('<4I', src[:16])
    ok = tw == list(sw)
    print('%s %-55s off %-9d %-6s %-14s %s' % (
        'OK ' if ok else 'BAD', nm.split('layers.0.')[1], x['off'],
        x['dtype'], str(x['shape']), '' if ok else 'tw=%08x sw=%08x' % (tw[0], sw[0])))
