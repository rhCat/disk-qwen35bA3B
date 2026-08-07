#!/usr/bin/env bash
# foot-probe.sh -- measure RSS AND macOS footprint (Activity Monitor metric)
# during a run. footprint = rss + file-backed resident pages.
set -u
CACHE_GB="${1:-2}"
GEN="${GEN:-12}"
TRUNK=/tmp/q35-trunk
POOL=/tmp/q35-pool
REPO=/Users/ruihe/disk-qwen35bA3B
TOK="${TOK:-/Users/ruihe/.cache/huggingface/mlx-qwen35-a3b-4bit/tokenizer.json}"
DS4F_GREEDY=1 "$REPO/ds4f" "$TRUNK" \
  --trunk "$TRUNK/trunk.bin" --offsets "$TRUNK/trunk.offsets" \
  --pool "$POOL/pool.bin" \
  --layout-trunk "$TRUNK/trunk.json" \
  --layout-pool "$POOL/manifest.json" \
  --head "$TRUNK/head.json" --embed "$TRUNK/embed.json" \
  --tokenizer "$TOK" --text "The capital of France is" \
  --gen "$GEN" --cache-gb "$CACHE_GB" --pin-layers 4 \
  --mem-limit-gb 23 \
  > /tmp/footprobe.log 2>&1 &
EPID=$!
echo "engine pid $EPID (cache-gb $CACHE_GB gen $GEN)"
# find footprint metric support
ps -o footprint= -p $EPID > /dev/null 2>&1 && echo "footprint metric available"
for i in $(seq 1 40); do
  kill -0 $EPID 2>/dev/null || break
  R=$(ps -o rss= -p $EPID 2>/dev/null)
  F=$(ps -o footprint= -p $EPID 2>/dev/null)
  [ -n "$R" ] && printf "t=%02d rss=%.2f GB footprint=%.2f GB\n" "$i" \
    "$(echo "$R/1048576" | bc -l)" "$(echo "${F:-0}/1048576" | bc -l)"
  sleep 1
done
wait $EPID
echo "exit $?"
grep -E 'PEAK RSS|tokens in' /tmp/footprobe.log | tail -2
