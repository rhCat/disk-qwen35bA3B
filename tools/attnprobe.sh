#!/usr/bin/env bash
# attnprobe.sh -- DS4F_DEBUG_ATTN at L3, serial vs chunk, 200-tok.
set -u
cd /Users/ruihe/disk-qwen35bA3B
unset PYTHONPATH
ARGS="--trunk /tmp/q35-trunk/trunk.bin --offsets /tmp/q35-trunk/trunk.offsets --layout-trunk /tmp/q35-trunk/trunk.json --pool /tmp/q35-pool/pool.bin --layout-pool /tmp/q35-pool/manifest.json --head /tmp/q35-trunk/head.json --embed /tmp/q35-trunk/embed.json --tokenizer $HOME/.cache/huggingface/mlx-qwen35-a3b-4bit/tokenizer.json --pids-file /tmp/q35-200-ids.txt --gen 8 --cache-gb 5 --pin-layers 4 --mem-limit-gb 20"
export DS4F_GREEDY=1 DS4F_DEBUG_ATTN=1
bash tools/run-clean.sh ./ds4f /tmp/q35-trunk $ARGS > /tmp/ap-serial.log 2>&1
echo "serial rc=$?"
export DS4F_PREFILL_CHUNK=1 DS4F_PREFILL_B=64
bash tools/run-clean.sh ./ds4f /tmp/q35-trunk $ARGS > /tmp/ap-chunk.log 2>&1
echo "chunk rc=$?"
unset DS4F_PREFILL_CHUNK DS4F_PREFILL_B DS4F_DEBUG_ATTN
echo "=== serial first 4 ==="
grep '\[attn\]' /tmp/ap-serial.log | head -4
echo "=== chunk first 4 ==="
grep '\[attn\]' /tmp/ap-chunk.log | head -4
