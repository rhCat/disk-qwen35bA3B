#!/usr/bin/env bash
# peak-probe.sh -- vmmap snapshot at peak RSS during a long run.
set -u
TRUNK=/tmp/q35-trunk
POOL=/tmp/q35-pool
REPO=/Users/ruihe/disk-qwen35bA3B
TOK="${TOK:-/Users/ruihe/.cache/huggingface/mlx-qwen35-a3b-4bit/tokenizer.json}"
PROMPT="The largest planet in the solar system is Jupiter, which is famous for"
DS4F_GREEDY=1 "$REPO/ds4f" "$TRUNK" \
  --trunk "$TRUNK/trunk.bin" --offsets "$TRUNK/trunk.offsets" \
  --pool "$POOL/pool.bin" \
  --layout-trunk "$TRUNK/trunk.json" \
  --layout-pool "$POOL/manifest.json" \
  --head "$TRUNK/head.json" --embed "$TRUNK/embed.json" \
  --tokenizer "$TOK" --text "$PROMPT" \
  --gen 16 --cache-gb 2 --pin-layers 4 \
  --mem-limit-gb 23 \
  > /tmp/peak.log 2>&1 &
EPID=$!
echo "engine pid $EPID"
BEST=0
for i in $(seq 1 60); do
  kill -0 $EPID 2>/dev/null || break
  R=$(ps -o rss= -p $EPID 2>/dev/null | tr -d ' ')
  if [ -n "$R" ] && [ "$R" -gt "$BEST" ]; then
    BEST=$R
    vmmap -summary $EPID 2>/dev/null | grep -E 'Physical footprint|TOTAL' | head -3 > /tmp/peak-vmmap.txt
  fi
  sleep 1
done
echo "peak rss: $(echo "$BEST/1048576" | bc -l) GB"
cat /tmp/peak-vmmap.txt
wait $EPID
echo "exit $?"
grep -E 'PEAK RSS|tokens in' /tmp/peak.log | tail -2
