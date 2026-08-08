#!/usr/bin/env bash
# acc-5.sh -- run the 5-token sanity and dump one DEBUG7 line.
set -u
cd /Users/ruihe/disk-qwen35bA3B
unset PYTHONPATH
export DS4F_GREEDY=1 DS4F_REP_PENALTY=1.3 DS4F_DEBUG7=1
bash tools/run-clean.sh ./ds4f /tmp/q35-trunk \
  --trunk /tmp/q35-trunk/trunk.bin --offsets /tmp/q35-trunk/trunk.offsets \
  --layout-trunk /tmp/q35-trunk/trunk.json \
  --pool /tmp/q35-pool/pool.bin --layout-pool /tmp/q35-pool/manifest.json \
  --head /tmp/q35-trunk/head.json --embed /tmp/q35-trunk/embed.json \
  --tokenizer "$HOME/.cache/huggingface/mlx-qwen35-a3b-4bit/tokenizer.json" \
  --pids-file /tmp/bench-gpu-ids.txt --gen 12 \
  --cache-gb 5 --pin-layers 4 --mem-limit-gb 20 \
  > /tmp/acc-5.log 2>&1
echo "rc=$?"
echo "--- one logits line ---"
grep -m2 'logits:' /tmp/acc-5.log | head -c 300
