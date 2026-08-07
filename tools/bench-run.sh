#!/usr/bin/env bash
# bench-run.sh -- run the mock-SOP QA bench: one engine run per
# scenario, capture streamed output + timing, then grade.
set -u
REPO=/Users/ruihe/disk-qwen35bA3B
cd "$REPO"
export DS4F_GREEDY=1 DS4F_REP_PENALTY=1.3
GEN="${GEN:-160}"
declare -A NAMES=(
  [sop-cleanroom]=sop-cleanroom
  [sop-reactor]=sop-reactor
  [ctx-warehouse]=ctx-warehouse
  [ctx-negative]=ctx-negative
)
for name in sop-cleanroom sop-reactor ctx-warehouse ctx-negative; do
  echo "=== $name ==="
  ./ds4f /tmp/q35-trunk \
    --trunk /tmp/q35-trunk/trunk.bin \
    --offsets /tmp/q35-trunk/trunk.offsets \
    --layout-trunk /tmp/q35-trunk/trunk.json \
    --pool /tmp/q35-pool/pool.bin \
    --layout-pool /tmp/q35-pool/manifest.json \
    --head /tmp/q35-trunk/head.json \
    --embed /tmp/q35-trunk/embed.json \
    --tokenizer /Users/ruihe/.cache/huggingface/mlx-qwen35-a3b-4bit/tokenizer.json \
    --pids-file "/tmp/q35-bench-$name.txt" \
    --gen "$GEN" --cache-gb 2 --pin-layers 4 --mem-limit-gb 20 \
    > "/tmp/bench-$name.log" 2>&1
  echo "rc=$?" >> "/tmp/bench-$name.log"
done
echo "BENCH DONE"
