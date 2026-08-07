#!/usr/bin/env bash
# rss7.sh -- measure peak RSS at cache-gb 5 (target ~7 GB total).
set -u
REPO=/Users/ruihe/disk-qwen35bA3B
cd "$REPO"
export DS4F_GREEDY=1
GEN=80 ./ds4f /tmp/q35-trunk \
  --trunk /tmp/q35-trunk/trunk.bin \
  --offsets /tmp/q35-trunk/trunk.offsets \
  --layout-trunk /tmp/q35-trunk/trunk.json \
  --pool /tmp/q35-pool/pool.bin \
  --layout-pool /tmp/q35-pool/manifest.json \
  --head /tmp/q35-trunk/head.json \
  --embed /tmp/q35-trunk/embed.json \
  --prompt-ids 760,7526,11247,303,279,12570,1785,369,48017,11,9338,369,34622,5272,295,11,48017,3733,816,2803,2389 \
  --gen 80 --cache-gb 5 --pin-layers 4 --mem-limit-gb 20 \
  > /tmp/rss7.log 2>&1
echo "EXIT $?" >> /tmp/rss7.log
grep -E 'PEAK RSS|EXIT|cache:|tokens in' /tmp/rss7.log | tail -5
