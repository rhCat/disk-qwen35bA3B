#!/usr/bin/env bash
# dbg-e2e-text2.sh -- run e2e_text against the DS-V4-named synthetic.
cd /Users/ruihe/disk-qwen35bA3B
OUT=/tmp/synds4f/out
./ds4f "$OUT" --trunk "$OUT/trunk.bin" --offsets "$OUT/trunk.offsets" \
  --pool "$OUT/pool.bin" --layout-trunk "$OUT/trunk.json" \
  --layout-pool "$OUT/manifest.json" \
  --head "$OUT/head.json" --embed "$OUT/embed.json" \
  --prompt-ids 7 --gen 5 --cache-gb 1 2>&1 | tail -6
echo "exit: $?"
