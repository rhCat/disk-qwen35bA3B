#!/usr/bin/env bash
# measure-expert-reuse.sh -- log routed expert ids per layer for a few
# tokens, then compute adjacent-layer overlap (the speculative
# prefetch hit rate).
set -u
REPO=/Users/ruihe/disk-qwen35bA3B
cd "$REPO"
unset PYTHONPATH
export DS4F_GREEDY=1 DS4F_NAN_PROBE=1
bash tools/run-clean.sh ./ds4f /tmp/q35-trunk \
  --trunk /tmp/q35-trunk/trunk.bin --offsets /tmp/q35-trunk/trunk.offsets \
  --layout-trunk /tmp/q35-trunk/trunk.json \
  --pool /tmp/q35-pool/pool.bin --layout-pool /tmp/q35-pool/manifest.json \
  --head /tmp/q35-trunk/head.json --embed /tmp/q35-trunk/embed.json \
  --tokenizer "$HOME/.cache/huggingface/mlx-qwen35-a3b-4bit/tokenizer.json" \
  --pids-file /tmp/bench-gpu-ids.txt --gen 8 \
  --cache-gb 5 --pin-layers 4 --mem-limit-gb 20 \
  > /tmp/reuse.log 2>&1
echo "rc=$?"
grep -c 'rtop' /tmp/reuse.log
