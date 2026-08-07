#!/usr/bin/env python3
# Cross-check: trunk in_proj_z weight words vs the MLX safetensors shard.
import json, struct
base = '/Users/ruihe/.cache/huggingface/mlx-qwen35-a3b-4bit'
wm = json.load(open(base + '/model.safetensors.index.json'))['weight_map']
name = 'language_model.model.layers.0.linear_attn.in_proj_z.weight'
fn = wm[name]
# read safetensors header properly (data_offsets relative to data start,
# data start = 8 + hlen, aligned to 8)
with open(base + '/' + fn, 'rb') as f:
    raw = f.read(8)
    hlen = struct.unpack('<Q', raw)[0]
    hdr = json.loads(f.read(hlen))
    data_start = 8 + hlen
    if data_start % 8: data_start += 8 - (data_start % 8)
    e = hdr[name]
    off, nb = e['data_offsets']
    f.seek(data_start + off)
    src = f.read(32)  # first 8 U32 words
print('src safetensors in_proj_z.weight first 8 words:')
print(' ', [hex(struct.unpack('<I', src[i:i+4])[0]) for i in range(0, 32, 4)])
print('dtype:', e['dtype'], 'shape:', e['shape'])

# trunk layer 0, in_proj_z weight off 10105024
tb = open('/tmp/q35-trunk/trunk.bin', 'rb').read()
offs = open('/tmp/q35-trunk/trunk.offsets', 'rb').read()
n = struct.unpack('<Q', offs[:8])[0]
o0 = struct.unpack('<%dQ' % n, offs[8:8+8*n])
lay0 = o0[0]
tw = struct.unpack_from('<8I', tb, lay0 + 10105024)
print('trunk in_proj_z.weight first 8 words:')
print(' ', [hex(v) for v in tw])
print('MATCH' if [hex(struct.unpack('<I', src[i:i+4])[0]) for i in range(0,32,4)] == [hex(v) for v in tw] else 'MISMATCH')
