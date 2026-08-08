#!/usr/bin/env bash
# chunk-diag.sh -- tiny run: 12-token prompt, chunked, capture stderr.
set -u
cd /Users/ruihe/disk-qwen35bA3B
unset PYTHONPATH
for v in DS4F_NAN_PROBE DS4F_GREEDY DS4F_DEBUG7 DS4F_PROJ_MS DS4F_WATERFALL \
         DS4F_STRIP_THINK DS4F_SKIP_ATTN DS4F_TIME_LAYERS; do unset "$v"; done
export DS4F_PREFILL_CHUNK=1 DS4F_PREFILL_B=8
bash tools/run-clean.sh ./ds4f /tmp/q35-trunk \
  --trunk /tmp/q35-trunk/trunk.bin --offsets /tmp/q35-trunk/trunk.offsets \
  --layout-trunk /tmp/q35-trunk/trunk.json \
  --pool /tmp/q35-pool/pool.bin --layout-pool /tmp/q35-pool/manifest.json \
  --head /tmp/q35-trunk/head.json --embed /tmp/q35-trunk/embed.json \
  --tokenizer "$HOME/.cache/huggingface/mlx-qwen35-a3b-4bit/tokenizer.json" \
  --pids-file /tmp/bench-gpu-ids.txt --gen 2 \
  --cache-gb 5 --pin-layers 4 --mem-limit-gb 20 \
  > /tmp/chunk-diag.log 2>&1
echo "rc=$?"
grep -cE '\[chunk\]' /tmp/chunk-diag.log
grep -E '\[chunk\]|prefill|linear_step_chunk failed|tokens in' /tmp/chunk-diag.log | head -6
