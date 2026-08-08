#!/usr/bin/env bash
# gqadump.sh -- capture L3 t0 qq+gate / xin / q-proj, serial vs chunk.
set -u
cd /Users/ruihe/disk-qwen35bA3B
unset PYTHONPATH
ARGS="--trunk /tmp/q35-trunk/trunk.bin --offsets /tmp/q35-trunk/trunk.offsets --layout-trunk /tmp/q35-trunk/trunk.json --pool /tmp/q35-pool/pool.bin --layout-pool /tmp/q35-pool/manifest.json --head /tmp/q35-trunk/head.json --embed /tmp/q35-trunk/embed.json --tokenizer $HOME/.cache/huggingface/mlx-qwen35-a3b-4bit/tokenizer.json --pids-file /tmp/q35-200-ids.txt --gen 8 --cache-gb 5 --pin-layers 4 --mem-limit-gb 20"
export DS4F_GREEDY=1 DS4F_NAN_PROBE=1 DS4F_DUMP_Z=1
rm -f /tmp/q35-eng-gqaq.bin /tmp/q35-eng-gqaxin.bin /tmp/q35-eng-gqaproj.bin /tmp/q35-eng-gqak0.bin
bash tools/run-clean.sh ./ds4f /tmp/q35-trunk $ARGS > /tmp/gqd-serial.log 2>&1
echo "serial rc=$?"
for f in gqaq gqaxin gqaproj gqak0 gqaattn; do cp /tmp/q35-eng-$f.bin /tmp/gqd-serial-$f.bin 2>/dev/null; done
export DS4F_PREFILL_CHUNK=1 DS4F_PREFILL_B=64
bash tools/run-clean.sh ./ds4f /tmp/q35-trunk $ARGS > /tmp/gqd-chunk.log 2>&1
echo "chunk rc=$?"
unset DS4F_PREFILL_CHUNK DS4F_PREFILL_B DS4F_NAN_PROBE DS4F_DUMP_Z
for f in gqaq gqaxin gqaproj gqak0 gqaattn; do cp /tmp/q35-eng-$f.bin /tmp/gqd-chunk-$f.bin 2>/dev/null; done
python3 - <<'EOF'
import struct, os
for name in ["gqaq", "gqaxin", "gqaproj", "gqak0", "gqaattn"]:
    sp = f"/tmp/gqd-serial-{name}.bin"; cp = f"/tmp/gqd-chunk-{name}.bin"
    if not (os.path.exists(sp) and os.path.exists(cp)):
        print(f"{name}: MISSING"); continue
    a = open(sp, "rb").read(); b = open(cp, "rb").read()
    n = min(len(a), len(b)) // 4
    fa = struct.unpack(f"<{n}f", a); fb = struct.unpack(f"<{n}f", b)
    nd = sum(1 for i in range(n) if fa[i] != fb[i])
    md = max((abs(fa[i]-fb[i]) for i in range(n) if fa[i]!=fb[i]), default=0.0)
    print(f"{name}: {len(a)}B {nd}/{n} differ, maxdelta {md:.3e}")
    if nd:
        for i in range(n):
            if fa[i] != fb[i]:
                print(f"  first @{i}: serial {fa[i]:.9g} chunk {fb[i]:.9g}")
                break
EOF
echo "=== gate[0..3] probe ==="
grep 'qn\[0\.\.3\]' /tmp/gqd-serial.log | head -1
grep 'qn\[0\.\.3\]' /tmp/gqd-chunk.log | head -1
