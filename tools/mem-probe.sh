#!/usr/bin/env bash
# mem-probe.sh -- run the engine, sample the REAL binary's RSS+footprint.
# Usage: mem-probe.sh <cache-gb>
set -u
CACHE_GB="${1:-1}"
TRUNK=/tmp/q35-trunk
POOL=/tmp/q35-pool
REPO=/Users/ruihe/disk-qwen35bA3B
GEN="${GEN:-10}"
DS4F_GREEDY=1 "$REPO/ds4f" "$TRUNK" \
  --trunk "$TRUNK/trunk.bin" --offsets "$TRUNK/trunk.offsets" \
  --pool "$POOL/pool.bin" \
  --layout-trunk "$TRUNK/trunk.json" --layout-pool "$POOL/manifest.json" \
  --embed "$TRUNK/embed.json" --head "$TRUNK/head.json" \
  --gen "$GEN" --cache-gb "$CACHE_GB" --pin-layers 4 --mem-limit-gb 23 \
  --prompt "The capital of France is" > /tmp/memprobe.log 2>&1 &
EPID=$!
echo "engine pid $EPID (cache-gb $CACHE_GB)"
for i in $(seq 1 40); do
  kill -0 $EPID 2>/dev/null || break
  ps -o rss= -p $EPID 2>/dev/null | awk -v t="$i" '{printf "t=%02d rss=%.2f GB\n", t, $1/1048576}'
  sleep 1
done
wait $EPID
echo "exit $?"
grep -E 'PEAK RSS|tokens in|expert_bytes' /tmp/memprobe.log | tail -2
