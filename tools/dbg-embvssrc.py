#!/usr/bin/env python3
# compare SOURCE embed row vs TRUNK embed row as DECODED FLOATS
import json, numpy as np

SRC = '/Users/ruihe/.cache/huggingface/mlx-qwen35-a3b-4bit'
TRUNK = '/tmp/q35-trunk'

def bf16(bits):
    u = bits.astype(np.uint32) << 16
    return u.view(np.float32)

def decode4(w, sc, bi):
    w = w.astype(np.uint32)
    sh = np.array([0, 4, 8, 12, 16, 20, 24, 28], dtype=np.uint32)
    nib = (w[..., None] >> sh) & 0xF
    R, P = w.shape
    vals = nib.reshape(R, P * 8)
    C = P * 8
    G = C // 64
    s = bf16(sc.astype(np.uint16)).reshape(R, G)
    b = bf16(bi.astype(np.uint16)).reshape(R, G)
    return vals.astype(np.float32) * np.repeat(s, 64, axis=1) + np.repeat(b, 64, axis=1)

f = SRC + '/model-00001-of-00004.safetensors'
with open(f, 'rb') as fh:
    n = int.from_bytes(fh.read(8), 'little')
    hdr = json.loads(fh.read(n))

def src_tensor(name):
    meta = hdr[name]
    with open(f, 'rb') as fh:
        fh.seek(8 + n + meta['data_offsets'][0])
        raw = fh.read(meta['data_offsets'][1] - meta['data_offsets'][0])
    dt = {'U32': np.uint32, 'BF16': np.uint16}[meta['dtype']]
    return np.frombuffer(raw, dtype=dt).reshape(meta['shape'])

w_src = src_tensor('language_model.model.embed_tokens.weight')
sc_src = src_tensor('language_model.model.embed_tokens.scales')
bi_src = src_tensor('language_model.model.embed_tokens.biases')

e = json.load(open(TRUNK + '/embed.json'))
eb = open(TRUNK + '/embed.bin', 'rb').read()
w_tr = np.frombuffer(eb, dtype=np.uint32, count=248320 * 256, offset=e['weight']['off']).reshape(248320, 256)
sc_tr = np.frombuffer(eb, dtype=np.uint16, count=248320 * 32, offset=e['scale']['off']).reshape(248320, 32)
bi_tr = np.frombuffer(eb, dtype=np.uint16, count=248320 * 32, offset=e['bias']['off']).reshape(248320, 32)

tok = 760
a = decode4(w_src[tok:tok + 1], sc_src[tok:tok + 1], bi_src[tok:tok + 1])[0]
b = decode4(w_tr[tok:tok + 1], sc_tr[tok:tok + 1], bi_tr[tok:tok + 1])[0]
print('src rms %.6g  trunk rms %.6g' % (np.sqrt((a**2).mean()), np.sqrt((b**2).mean())))
d = np.abs(a - b)
i = int(d.argmax())
print('n-diff %.0f / 2048  maxdiff %.4g at idx %d  src %.6g trunk %.6g' % (
    (d > 1e-6).sum(), d[i], i, a[i], b[i]))
print('src[0..7]  ', np.round(a[:8], 5))
print('trunk[0..7]', np.round(b[:8], 5))
# nibble-level: is it an ordering issue?
w0a, w0b = w_src[tok, 0], w_tr[tok, 0]
print('word0 src %08x trunk %08x' % (w0a, w0b))
print('nibbles src', [(w0a >> (4 * k)) & 0xF for k in range(8)])
print('nibbles tr ', [(w0b >> (4 * k)) & 0xF for k in range(8)])
print('scales src', bf16(sc_src[tok, :4].astype(np.uint16)))
print('scales tr ', bf16(sc_tr[tok, :4].astype(np.uint16)))
print('bias src', bf16(bi_src[tok, :4].astype(np.uint16)))
print('bias tr ', bf16(bi_tr[tok, :4].astype(np.uint16)))
