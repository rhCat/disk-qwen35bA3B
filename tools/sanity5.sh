#!/usr/bin/env bash
# sanity5.sh -- 5-token run with DEBUG7, the known-good config.
set -u
cd /Users/ruihe/disk-qwen35bA3B
unset PYTHONPATH
export DS4F_GREEDY=1 DS4F_DEBUG7=1
unset DS4F_PREFILL_CHUNK DS4F_PREFILL_B DS4F_WATERFALL DS4F_PROJ_MS
bash tools/run-clean.sh ./ds4f /tmp/q35-trunk \
  --trunk /tmp/q35-trunk/trunk.bin --offsets /tmp/q35-trunk/trunk.offsets \
  --layout-trunk /tmp/q35-trunk/trunk.json \
  --pool /tmp/q35-pool/pool.bin --layout-pool /tmp/q35-pool/manifest.json \
  --head /tmp/q35-trunk/head.json --embed /tmp/q35-trunk/embed.json \
  --tokenizer "$HOME/.cache/huggingface/mlx-qwen35-a3b-4bit/tokenizer.json" \
  --pids-file /tmp/bench-gpu-ids.txt --gen 8 \
  --cache-gb 5 --pin-layers 4 --mem-limit-gb 20 \
  > /tmp/sanity5.log 2>&1
echo "rc=$?"
grep -E 'attn step failed|trunk bind|logits: t' /tmp/sanity5.log | head -4
