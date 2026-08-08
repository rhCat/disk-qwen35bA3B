#!/usr/bin/env bash
# bench-ctx.sh -- long-context round at a given fixture size.
# Usage: bench-ctx.sh <ids-file> [gen]
set -u
REPO=/Users/ruihe/disk-qwen35bA3B
IDS="${1:-/tmp/q35-500-ids.txt}"
GEN="${2:-8}"
cd "$REPO"
unset PYTHONPATH
export DS4F_WATERFALL=1 DS4F_GREEDY=1
t0=$("$REPO/.venv/bin/python3" -c 'import time; print(time.time())')
bash tools/run-clean.sh ./ds4f /tmp/q35-trunk \
  --trunk /tmp/q35-trunk/trunk.bin --offsets /tmp/q35-trunk/trunk.offsets \
  --layout-trunk /tmp/q35-trunk/trunk.json \
  --pool /tmp/q35-pool/pool.bin --layout-pool /tmp/q35-pool/manifest.json \
  --head /tmp/q35-trunk/head.json --embed /tmp/q35-trunk/embed.json \
  --tokenizer "$HOME/.cache/huggingface/mlx-qwen35-a3b-4bit/tokenizer.json" \
  --pids-file "$IDS" --gen "$GEN" \
  --cache-gb 5 --pin-layers 4 --mem-limit-gb 20 \
  > "/tmp/bench-ctx-$(basename "$IDS" .txt).log" 2>&1
rc=$?
t1=$("$REPO/.venv/bin/python3" -c 'import time; print(time.time())')
echo "IDS=$IDS EXIT $rc | wall $("$REPO/.venv/bin/python3" -c "import time;print(f'{$t1-$t0:.1f}s')")"
grep -E 'tokens in|PEAK|waterfall|attn:|moe:    |fetch:|head:' "/tmp/bench-ctx-$(basename "$IDS" .txt).log" | head -8
