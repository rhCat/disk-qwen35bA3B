#!/usr/bin/env python3
# Recheck src read alignment: safetensors data_offsets are relative to
# data start, which is 8 + len(header) rounded up to 8.
import json, struct
base = '/Users/ruihe/.cache/huggingface/mlx-qwen35-a3b-4bit'
wm = json.load(open(base + '/model.safetensors.index.json'))['weight_map']

def src_bytes(name, nbytes=16):
    fn = wm[name]
    with open(base + '/' + fn, 'rb') as f:
        raw = f.read(8)
        hlen = struct.unpack('<Q', raw)[0]
        hdr = json.loads(f.read(hlen))
        hdr_bytes = 8 + hlen
        ds = (hdr_bytes + 7) & ~7
        e = hdr[name]
        off, nb = e['data_offsets']
        f.seek(ds + off)
        return f.read(nbytes), e['dtype'], e['shape']

name = 'language_model.model.layers.0.linear_attn.in_proj_z.weight'
src, dt, shp = src_bytes(name)
print('src dtype', dt, 'shape', shp)
print('src :', [hex(struct.unpack('<I', src[i:i+4])[0]) for i in range(0,16,4)])

tb = open('/tmp/q35-trunk/trunk.bin', 'rb').read()
offs = open('/tmp/q35-trunk/trunk.offsets', 'rb').read()
n = struct.unpack('<Q', offs[:8])[0]
o0 = struct.unpack('<%dQ' % n, offs[8:8+8*n])
t = json.load(open('/tmp/q35-trunk/trunk.json'))
x = next(z for z in t['layers'][0]['tensors'] if z['n'] == name)
print('trunk off', x['off'], 'nbytes', x['nbytes'], 'dtype', x['dtype'])
tw = struct.unpack_from('<4I', tb, o0[0] + x['off'])
print('trnk:', [hex(v) for v in tw])
# try each 1-byte shift of the src against trunk[0..3]
srcb = src
for s in range(0, 8):
    cand = struct.unpack('<4I', srcb[s:s+16]) if s+16 <= len(srcb) else None
    if cand == list(tw):
        print('MATCH at src byte offset', s)
