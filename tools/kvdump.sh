#!/usr/bin/env bash
# kvdump.sh -- capture L3 t0 K/V (buffer + cache region) serial vs chunk.
set -u
cd /Users/ruihe/disk-qwen35bA3B
unset PYTHONPATH
ARGS="--trunk /tmp/q35-trunk/trunk.bin --offsets /tmp/q35-trunk/trunk.offsets --layout-trunk /tmp/q35-trunk/trunk.json --pool /tmp/q35-pool/pool.bin --layout-pool /tmp/q35-pool/manifest.json --head /tmp/q35-trunk/head.json --embed /tmp/q35-trunk/embed.json --tokenizer $HOME/.cache/huggingface/mlx-qwen35-a3b-4bit/tokenizer.json --pids-file /tmp/q35-200-ids.txt --gen 8 --cache-gb 5 --pin-layers 4 --mem-limit-gb 20"
export DS4F_GREEDY=1 DS4F_NAN_PROBE=1 DS4F_DUMP_Z=1
rm -f /tmp/q35-eng-gqak0.bin /tmp/q35-eng-gqakv3.bin
bash tools/run-clean.sh ./ds4f /tmp/q35-trunk $ARGS > /tmp/kvd-serial.log 2>&1
echo "serial rc=$?"
cp /tmp/q35-eng-gqak0.bin /tmp/kvd-serial-gqak0.bin 2>/dev/null
cp /tmp/q35-eng-gqakv3.bin /tmp/kvd-serial-gqakv3.bin 2>/dev/null
export DS4F_PREFILL_CHUNK=1 DS4F_PREFILL_B=64
bash tools/run-clean.sh ./ds4f /tmp/q35-trunk $ARGS > /tmp/kvd-chunk.log 2>&1
echo "chunk rc=$?"
unset DS4F_PREFILL_CHUNK DS4F_PREFILL_B DS4F_NAN_PROBE DS4F_DUMP_Z
cp /tmp/q35-eng-gqak0.bin /tmp/kvd-chunk-gqak0.bin 2>/dev/null
cp /tmp/q35-eng-gqakv3.bin /tmp/kvd-chunk-gqakv3.bin 2>/dev/null
python3 - <<'EOF'
import struct, os
for name in ["gqak0", "gqakv3"]:
    sp = f"/tmp/kvd-serial-{name}.bin"
    cp = f"/tmp/kvd-chunk-{name}.bin"
    if not (os.path.exists(sp) and os.path.exists(cp)):
        print(f"{name}: MISSING (serial {os.path.exists(sp)} chunk {os.path.exists(cp)})")
        continue
    a = open(sp, "rb").read(); b = open(cp, "rb").read()
    n = min(len(a), len(b)) // 4
    fa = struct.unpack(f"<{n}f", a); fb = struct.unpack(f"<{n}f", b)
    nd = sum(1 for i in range(n) if fa[i] != fb[i])
    md = max((abs(fa[i]-fb[i]) for i in range(n) if fa[i]!=fb[i]), default=0.0)
    print(f"{name}: {len(a)}B {nd}/{n} floats differ, maxdelta {md:.3e}")
    if nd:
        for i in range(n):
            if fa[i] != fb[i]:
                print(f"  first @{i}: serial {fa[i]:.9g} chunk {fb[i]:.9g}")
                break
EOF
