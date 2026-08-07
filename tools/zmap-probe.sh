#!/usr/bin/env bash
# zmap-probe.sh -- vmmap at steady state, full malloc zone table.
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
  --gen 12 --cache-gb 1 --pin-layers 4 \
  --mem-limit-gb 23 \
  > /tmp/zmap.log 2>&1 &
EPID=$!
echo "engine pid $EPID"
sleep 3
ps -o rss= -p $EPID 2>/dev/null | awk '{printf "rss at t3: %.2f GB\n", $1/1048576}'
vmmap -summary $EPID 2>/dev/null | sed -n '/MALLOC ZONE/,/TOTAL/p' > /tmp/zmap-zones.txt
vmmap $EPID 2>/dev/null | grep -iE 'pool|trunk|\.bin|mapped' | head -20 > /tmp/zmap-regions.txt
wait $EPID
echo "exit $?"
grep -E 'PEAK RSS' /tmp/zmap.log | tail -1
echo "--- zones ---"
cat /tmp/zmap-zones.txt
echo "--- file-backed regions ---"
cat /tmp/zmap-regions.txt
