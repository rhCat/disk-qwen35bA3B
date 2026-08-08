#!/usr/bin/env bash
# qkvdump.sh -- capture /tmp/qkv-{serial,chunk}-L3.bin at 200-tok scale.
set -u
cd /Users/ruihe/disk-qwen35bA3B
unset PYTHONPATH
ARGS="--trunk /tmp/q35-trunk/trunk.bin --offsets /tmp/q35-trunk/trunk.offsets --layout-trunk /tmp/q35-trunk/trunk.json --pool /tmp/q35-pool/pool.bin --layout-pool /tmp/q35-pool/manifest.json --head /tmp/q35-trunk/head.json --embed /tmp/q35-trunk/embed.json --tokenizer $HOME/.cache/huggingface/mlx-qwen35-a3b-4bit/tokenizer.json --pids-file /tmp/q35-200-ids.txt --gen 8 --cache-gb 5 --pin-layers 4 --mem-limit-gb 20"
export DS4F_GREEDY=1 DS4F_NAN_PROBE=1 DS4F_PROJ_MS=1
rm -f /tmp/qkv-serial-L3.bin /tmp/qkv-chunk-L3.bin
bash tools/run-clean.sh ./ds4f /tmp/q35-trunk $ARGS > /tmp/qkvd-serial.log 2>&1
echo "serial rc=$?"
export DS4F_PREFILL_CHUNK=1 DS4F_PREFILL_B=64
bash tools/run-clean.sh ./ds4f /tmp/q35-trunk $ARGS > /tmp/qkvd-chunk.log 2>&1
echo "chunk rc=$?"
unset DS4F_PREFILL_CHUNK DS4F_PREFILL_B DS4F_NAN_PROBE DS4F_PROJ_MS
ls -l /tmp/qkv-serial-L3.bin /tmp/qkv-chunk-L3.bin
python3 - <<'EOF'
import struct
a = open("/tmp/qkv-serial-L3.bin","rb").read()
b = open("/tmp/qkv-chunk-L3.bin","rb").read()
if len(a) != len(b):
    print(f"SIZE MISMATCH: {len(a)} vs {len(b)}")
else:
    n = len(a)//4
    fa = struct.unpack(f"<{n}f", a); fb = struct.unpack(f"<{n}f", b)
    nd = sum(1 for i in range(n) if fa[i] != fb[i])
    md = max((abs(fa[i]-fb[i]) for i in range(n)), default=0.0)
    print(f"qkv+z L3 t8: {nd}/{n} floats differ, maxdelta {md:.3e}")
    if nd:
        for i in range(n):
            if fa[i] != fb[i]:
                print(f"  first diff @{i}: serial {fa[i]:.9g} chunk {fb[i]:.9g}")
                break
EOF
