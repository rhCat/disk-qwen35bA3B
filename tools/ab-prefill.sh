#!/usr/bin/env bash
# ab-prefill.sh -- A/B the chunked prefill vs serial: same fixture,
# compare the DEBUG7 gen-token logits lines. Bit-identical KV fill
# => identical logits (modulo float reorder in the batched matvec,
# which is the documented 2e-7 SIMD class).
set -u
cd /Users/ruihe/disk-qwen35bA3B
unset PYTHONPATH
IDS="${1:-/tmp/q35-500-ids.txt}"
GEN="${2:-8}"
B="${3:-64}"
ARGS="--trunk /tmp/q35-trunk/trunk.bin --offsets /tmp/q35-trunk/trunk.offsets --layout-trunk /tmp/q35-trunk/trunk.json --pool /tmp/q35-pool/pool.bin --layout-pool /tmp/q35-pool/manifest.json --head /tmp/q35-trunk/head.json --embed /tmp/q35-trunk/embed.json --tokenizer $HOME/.cache/huggingface/mlx-qwen35-a3b-4bit/tokenizer.json --pids-file $IDS --gen $GEN --cache-gb 5 --pin-layers 4 --mem-limit-gb 20"

# serial
unset DS4F_PREFILL_CHUNK DS4F_PREFILL_B
export DS4F_GREEDY=1 DS4F_DEBUG7=1
bash tools/run-clean.sh ./ds4f /tmp/q35-trunk $ARGS > /tmp/ab-serial.log 2>&1
echo "serial rc=$?"

# chunked
export DS4F_PREFILL_CHUNK=1 DS4F_PREFILL_B=$B
bash tools/run-clean.sh ./ds4f /tmp/q35-trunk $ARGS > /tmp/ab-chunk.log 2>&1
echo "chunk rc=$?"

# compare the logits top1 token ids for the gen tokens (DEBUG7 lines)
python3 - <<'PYEOF'
import re
def tops(path):
    out = []
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.search(r"logits: t(\d+) .*?top5\s+\[\s*(\d+) ", line)
        if m:
            out.append((int(m.group(1)), int(m.group(2))))
    return out
s = tops("/tmp/ab-serial.log")
c = tops("/tmp/ab-chunk.log")
print(f"serial: {len(s)} logits lines, chunk: {len(c)}")
ns = [t for t in s]; nc = [t for t in c]
if len(ns) != len(nc):
    print(f"LENGTH MISMATCH: {len(ns)} vs {len(nc)}")
else:
    same = sum(1 for a, b in zip(ns, nc) if a == b)
    print(f"identical top1 (token,id): {same}/{len(ns)}")
    for a, b in zip(ns, nc):
        if a != b:
            print(f"  DIFF t{a[0]}: serial id {a[1]} vs chunk id {b[1]}")
PYEOF
grep -E '\[prefill\]' /tmp/ab-chunk.log | head -3
