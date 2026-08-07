#!/usr/bin/env python3
# Find the byte shift for several layer-0 tensors.
import json, struct
base = '/Users/ruihe/.cache/huggingface/mlx-qwen35-a3b-4bit'
wm = json.load(open(base + '/model.safetensors.index.json'))['weight_map']

def src_bytes(name, nbytes):
    fn = wm[name]
    with open(base + '/' + fn, 'rb') as f:
        raw = f.read(8)
        hlen = struct.unpack('<Q', raw)[0]
        hdr = json.loads(f.read(hlen))
        ds = 8 + hlen
        if ds % 8: ds += 8 - (ds % 8)
        e = hdr[name]
        off, nb = e['data_offsets']
        f.seek(ds + off)
        return f.read(nbytes)

tb = open('/tmp/q35-trunk/trunk.bin', 'rb').read()
offs = open('/tmp/q35-trunk/trunk.offsets', 'rb').read()
n = struct.unpack('<Q', offs[:8])[0]
o0 = struct.unpack('<%dQ' % n, offs[8:8+8*n])
lay0 = o0[0]
t = json.load(open('/tmp/q35-trunk/trunk.json'))
tens = {x['n']: x for x in t['layers'][0]['tensors']}
# also list the order/offsets of the first tensors
print('layer0 tensors (first 6):')
for x in sorted(t['layers'][0]['tensors'], key=lambda z: z['off'])[:6]:
    print('  off %-9d %-60s %s %s' % (x['off'], x['n'].split('layers.0.')[1], x['dtype'], x['shape']))
for nm in ['language_model.model.layers.0.input_layernorm.weight',
           'language_model.model.layers.0.linear_attn.conv1d.weight',
           'language_model.model.layers.0.linear_attn.in_proj_qkv.weight',
           'language_model.model.layers.0.linear_attn.in_proj_z.weight']:
    if nm not in tens: continue
    x = tens[nm]
    src = src_bytes(nm, min(x['nbytes'], 8))
    tw = struct.unpack_from('<2I', tb, lay0 + x['off'])
    sw = struct.unpack('<2I', src)
    shift = None
    for s in range(-4, 5):
        # try trunk read at off+s
        if lay0 + x['off'] + s < 0: continue
        try:
            t2 = struct.unpack_from('<2I', tb, lay0 + x['off'] + s)
        except Exception:
            continue
        if t2 == list(sw):
            shift = s
            break
    print('%-55s off %d src %08x%08x trunk %08x%08x shift=%s' % (
        nm.split('layers.0.')[1], x['off'], sw[0], sw[1], tw[0], tw[1],
        shift if shift is not None else '???'))
