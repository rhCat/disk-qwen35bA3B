#!/usr/bin/env bash
# ab-long.sh -- 1500-token A/B: serial vs chunked, 24 gen, compare
# top1 ids + decode the actual response text.
set -u
cd /Users/ruihe/disk-qwen35bA3B
unset PYTHONPATH
IDS="${1:-/tmp/q35-1500-ids.txt}"
GEN="${2:-24}"
B="${3:-64}"
ARGS="--trunk /tmp/q35-trunk/trunk.bin --offsets /tmp/q35-trunk/trunk.offsets --layout-trunk /tmp/q35-trunk/trunk.json --pool /tmp/q35-pool/pool.bin --layout-pool /tmp/q35-pool/manifest.json --head /tmp/q35-trunk/head.json --embed /tmp/q35-trunk/embed.json --tokenizer $HOME/.cache/huggingface/mlx-qwen35-a3b-4bit/tokenizer.json --pids-file $IDS --gen $GEN --cache-gb 5 --pin-layers 4 --mem-limit-gb 20"
export DS4F_GREEDY=1 DS4F_DEBUG7=1
unset DS4F_PREFILL_CHUNK DS4F_PREFILL_B DS4F_CHUNK_MS
echo "=== serial (1500-tok prompt, $GEN gen) ==="
bash tools/run-clean.sh ./ds4f /tmp/q35-trunk $ARGS > /tmp/abL-serial.log 2>&1
echo "serial rc=$?"
export DS4F_PREFILL_CHUNK=1 DS4F_PREFILL_B=$B
echo "=== chunked ==="
bash tools/run-clean.sh ./ds4f /tmp/q35-trunk $ARGS > /tmp/abL-chunk.log 2>&1
echo "chunk rc=$?"
unset DS4F_PREFILL_CHUNK DS4F_PREFILL_B
echo "=== timing ==="
grep 'tokens in' /tmp/abL-serial.log /tmp/abL-chunk.log
echo "=== gen top1 comparison ==="
python3 tools/ab-compare.py /tmp/abL-serial.log /tmp/abL-chunk.log "$IDS"
echo "=== ACTUAL RESPONSE (decoded gen tokens) ==="
echo "--- serial ---"
python3 tools/acc-decode.py /tmp/abL-serial.log "$IDS" 400
echo "--- chunked ---"
python3 tools/acc-decode.py /tmp/abL-chunk.log "$IDS" 400
