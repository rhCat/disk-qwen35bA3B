#!/usr/bin/env bash
# ab-prefill2.sh -- A/B chunked vs serial with the double-process fix:
# compare DEBUG7 gen-token top1 ids. The chunked path fills the KV
# cache via batched attention; generation must match serial.
set -u
cd /Users/ruihe/disk-qwen35bA3B
unset PYTHONPATH
IDS="${1:-/tmp/q35-500-ids.txt}"
GEN="${2:-8}"
B="${3:-64}"
ARGS="--trunk /tmp/q35-trunk/trunk.bin --offsets /tmp/q35-trunk/trunk.offsets --layout-trunk /tmp/q35-trunk/trunk.json --pool /tmp/q35-pool/pool.bin --layout-pool /tmp/q35-pool/manifest.json --head /tmp/q35-trunk/head.json --embed /tmp/q35-trunk/embed.json --tokenizer $HOME/.cache/huggingface/mlx-qwen35-a3b-4bit/tokenizer.json --pids-file $IDS --gen $GEN --cache-gb 5 --pin-layers 4 --mem-limit-gb 20"
export DS4F_GREEDY=1 DS4F_DEBUG7=1
unset DS4F_PREFILL_CHUNK DS4F_PREFILL_B DS4F_CHUNK_MS
bash tools/run-clean.sh ./ds4f /tmp/q35-trunk $ARGS > /tmp/ab2-serial.log 2>&1
echo "serial rc=$?"
export DS4F_PREFILL_CHUNK=1 DS4F_PREFILL_B=$B
bash tools/run-clean.sh ./ds4f /tmp/q35-trunk $ARGS > /tmp/ab2-chunk.log 2>&1
echo "chunk rc=$?"
python3 - <<'PYEOF'
import re
def tops(path):
    out = []
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.search(r"logits: t(\d+) .*?top5\s+\[\s*(\d+) ", line)
        if m:
            out.append((int(m.group(1)), int(m.group(2))))
    return out
s = tops("/tmp/ab2-serial.log")
c = tops("/tmp/ab2-chunk.log")
print(f"serial: {len(s)} logits lines, chunk: {len(c)}")
if len(s) != len(c):
    print(f"LENGTH MISMATCH: {len(s)} vs {len(c)}")
else:
    same = sum(1 for a, b in zip(s, c) if a == b)
    print(f"identical top1: {same}/{len(s)}")
    for a, b in zip(s, c):
        if a != b:
            print(f"  DIFF t{a[0]}: serial {a[1]} vs chunk {b[1]}")
PYEOF
grep 'tokens in' /tmp/ab2-serial.log /tmp/ab2-chunk.log
