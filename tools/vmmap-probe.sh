#!/usr/bin/env bash
# vmmap-probe.sh -- run engine, snapshot vmmap at peak RSS for a
# footprint breakdown (what Activity Monitor's Memory column shows).
set -u
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
  --gen 12 --cache-gb 2 --pin-layers 4 \
  --mem-limit-gb 23 \
  > /tmp/vmmap.log 2>&1 &
EPID=$!
echo "engine pid $EPID"
for i in $(seq 1 40); do
  kill -0 $EPID 2>/dev/null || break
  R=$(ps -o rss= -p $EPID 2>/dev/null | tr -d ' ')
  if [ -n "$R" ] && [ "$R" -gt 5000000 ] 2>/dev/null; then
    echo "rss ${R}KB at t=$i -- snapshotting vmmap"
    vmmap -summary $EPID 2>/dev/null | grep -E 'Physical footprint|RESIDENT|file-backed|anonymous|TOTAL' | head -8
    vmmap -summary $EPID 2>/dev/null | tail -25 > /tmp/vmmap-summary.txt
    break
  fi
  sleep 1
done
wait $EPID
echo "exit $?"
grep -E 'PEAK RSS|tokens in' /tmp/vmmap.log | tail -2
