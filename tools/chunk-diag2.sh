#!/usr/bin/env bash
# chunk-diag2.sh -- 500-token run with BOTH chunk timers (attn + rfm).
set -u
cd /Users/ruihe/disk-qwen35bA3B
unset PYTHONPATH
for v in DS4F_NAN_PROBE DS4F_GREEDY DS4F_DEBUG7 DS4F_PROJ_MS DS4F_WATERFALL \
         DS4F_STRIP_THINK DS4F_SKIP_ATTN DS4F_TIME_LAYERS; do unset "$v"; done
export DS4F_PREFILL_CHUNK=1 DS4F_PREFILL_B=64 DS4F_CHUNK_MS=1
bash tools/run-clean.sh ./ds4f /tmp/q35-trunk \
  --trunk /tmp/q35-trunk/trunk.bin --offsets /tmp/q35-trunk/trunk.offsets \
  --layout-trunk /tmp/q35-trunk/trunk.json \
  --pool /tmp/q35-pool/pool.bin --layout-pool /tmp/q35-pool/manifest.json \
  --head /tmp/q35-trunk/head.json --embed /tmp/q35-trunk/embed.json \
  --tokenizer "$HOME/.cache/huggingface/mlx-qwen35-a3b-4bit/tokenizer.json" \
  --pids-file /tmp/q35-500-ids.txt --gen 8 \
  --cache-gb 5 --pin-layers 4 --mem-limit-gb 20 \
  > /tmp/chunk-diag2.log 2>&1
echo "rc=$?"
echo "--- attn chunk (first 4 layers, first chunk) ---"
grep -E '\[chunk\] L[0-3] ' /tmp/chunk-diag2.log | head -4
echo "--- rfm (first 4 layers, first chunk) ---"
grep -E '\[chunk-moe\] L[0-3] ' /tmp/chunk-diag2.log | head -4
echo "--- totals ---"
grep 'tokens in' /tmp/chunk-diag2.log
