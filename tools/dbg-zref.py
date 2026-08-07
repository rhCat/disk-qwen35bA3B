#!/usr/bin/env python3
# Verify L0: embed(9419) -> input_layernorm -> in_proj_z expectation.
import json, struct

def bf16f(h):
    return struct.unpack('<f', struct.pack('<I', h << 16))[0]

# embed decode: [248320, 256] U32 -> decoded 2048 per row
e = json.load(open('/tmp/q35-trunk/embed.json'))
data = open('/tmp/q35-trunk/embed.bin', 'rb').read()
w = e['weight']; nb = w['nbytes']
vals = struct.unpack_from('<%dI' % (nb//4), data, w['off'])
sc = e['scale']; bi = e['bias']
scales = struct.unpack_from('<%dH' % (sc['nbytes']//2), data, sc['off'])
biases = struct.unpack_from('<%dH' % (bi['nbytes']//2), data, bi['off'])
row = 9419
Hdec = 2048
G = Hdec // 64
dec = []
for i in range(Hdec):
    u = vals[row * (Hdec//8) + i//8]
    q = (u >> (4 * (i % 8))) & 0xF
    g = i // 64
    dec.append((q - 8) * bf16f(scales[row * G + g]) + bf16f(biases[row * G + g]))
er = (sum(x*x for x in dec)/Hdec)**0.5
print('embed(9419) rms: %.6f' % er)

# input_layernorm weights: layer 0 off 0, BF16 [2048]
t = json.load(open('/tmp/q35-trunk/trunk.json'))
tb = open('/tmp/q35-trunk/trunk.bin', 'rb').read()
offs = open('/tmp/q35-trunk/trunk.offsets', 'rb').read()
n = struct.unpack('<Q', offs[:8])[0]
o0 = struct.unpack('<%dQ' % n, offs[8:8+8*n])
lay0 = o0[0]
nw_raw = struct.unpack_from('<2048H', tb, lay0)
nw = [bf16f(v) for v in nw_raw]
ss = sum(x*x for x in dec)
r = (ss/Hdec + 1e-6) ** 0.5
xin = [dec[i]/r * nw[i] for i in range(Hdec)]
xrms = (sum(x*x for x in xin)/Hdec)**0.5
print('normed xin rms: %.6f' % xrms)

# in_proj_z: U32 [4096, 256] packed -> decoded cols 2048; scales [4096, 32] BF16
# layer0 offsets: weight 10105024, scales 9842880, biases 9580736
zw = struct.unpack_from('<%dI' % (4096*256), tb, lay0 + 10105024)
zs = struct.unpack_from('<%dH' % (4096*32), tb, lay0 + 9842880)
zb = struct.unpack_from('<%dH' % (4096*32), tb, lay0 + 9580736)
y = []
for rowz in range(4096):
    acc = 0.0
    for i in range(256):
        u = zw[rowz*256 + i]
        for k in range(8):
            qq = (u >> (4*k)) & 0xF
            col = i*8 + k
            g = col // 64
            val = (qq - 8) * bf16f(zs[rowz*32 + g]) + bf16f(zb[rowz*32 + g])
            acc += val * xin[col]
    y.append(acc)
yrms = (sum(v*v for v in y)/4096)**0.5
print('in_proj_z(xin) rms: %.6f' % yrms)
