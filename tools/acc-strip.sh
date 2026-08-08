#!/usr/bin/env bash
# acc-strip.sh -- 1500-token QA with DS4F_STRIP_THINK + longer gen.
set -u
cd /Users/ruihe/disk-qwen35bA3B
unset PYTHONPATH
export DS4F_GREEDY=1 DS4F_REP_PENALTY=1.3 DS4F_DEBUG7=1 DS4F_STRIP_THINK=1
bash tools/run-clean.sh ./ds4f /tmp/q35-trunk \
  --trunk /tmp/q35-trunk/trunk.bin --offsets /tmp/q35-trunk/trunk.offsets \
  --layout-trunk /tmp/q35-trunk/trunk.json \
  --pool /tmp/q35-pool/pool.bin --layout-pool /tmp/q35-pool/manifest.json \
  --head /tmp/q35-trunk/head.json --embed /tmp/q35-trunk/embed.json \
  --tokenizer "$HOME/.cache/huggingface/mlx-qwen35-a3b-4bit/tokenizer.json" \
  --pids-file /tmp/q35-1500-ids.txt --gen 48 \
  --cache-gb 5 --pin-layers 4 --mem-limit-gb 20 \
  > /tmp/acc-1500-strip.log 2>&1
echo "rc=$?"
python3 tools/acc-decode.py /tmp/acc-1500-strip.log /tmp/q35-1500-ids.txt 300
echo "--- last 8 gen tokens ---"
python3 tools/acc-decode.py /tmp/acc-1500-strip.log /tmp/q35-1500-ids.txt 300 | tail -c 120
grep -c 'Thinking Process' /tmp/acc-1500-strip.log
