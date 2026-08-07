#!/usr/bin/env bash
# qa-run.sh -- 2.6K-token chat QA: vault code buried mid-document.
set -u
REPO=/Users/ruihe/disk-qwen35bA3B
cd "$REPO"
export DS4F_GREEDY=1 DS4F_REP_PENALTY=1.3 DS4F_DEBUG7=1
./ds4f /tmp/q35-trunk \
  --trunk /tmp/q35-trunk/trunk.bin \
  --offsets /tmp/q35-trunk/trunk.offsets \
  --layout-trunk /tmp/q35-trunk/trunk.json \
  --pool /tmp/q35-pool/pool.bin \
  --layout-pool /tmp/q35-pool/manifest.json \
  --head /tmp/q35-trunk/head.json \
  --embed /tmp/q35-trunk/embed.json \
  --tokenizer /Users/ruihe/.cache/huggingface/mlx-qwen35-a3b-4bit/tokenizer.json \
  --pids-file /tmp/q35-qa-ids.txt \
  --gen 100 --cache-gb 5 --pin-layers 4 --mem-limit-gb 20 \
  > /tmp/qa-run.log 2>&1
echo "EXIT $?" >> /tmp/qa-run.log
grep -E 'tokens in|PEAK|EXIT' /tmp/qa-run.log | tail -3
