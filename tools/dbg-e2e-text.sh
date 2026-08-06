#!/usr/bin/env bash
# dbg-e2e-text.sh -- run the e2e_text sequence against the synthetic
# to see the actual failure after the head-loader changes.
cd /Users/ruihe/disk-qwen35bA3B
OUT=/tmp/syncheck/out
./ds4f "$OUT" --trunk "$OUT/trunk.bin" --offsets "$OUT/trunk.offsets" \
  --pool "$OUT/pool.bin" --layout-trunk "$OUT/trunk.json" \
  --layout-pool "$OUT/manifest.json" \
  --head "$OUT/head.json" --embed "$OUT/embed.json" \
  --prompt-ids 7 --gen 5 --cache-gb 1 2>&1 | tail -8
echo "exit: $?"
