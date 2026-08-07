#!/usr/bin/env bash
# asan-crash.sh -- ASan build run, catch the exact overrun.
set -u
REPO=/Users/ruihe/disk-qwen35bA3B
cd "$REPO"
export DS4F_NAN_PROBE=1 DS4F_GREEDY=1
export ASAN_OPTIONS=abort_on_error=1:detect_leaks=0
./ds4f /tmp/q35-trunk \
  --trunk /tmp/q35-trunk/trunk.bin \
  --offsets /tmp/q35-trunk/trunk.offsets \
  --layout-trunk /tmp/q35-trunk/trunk.json \
  --pool /tmp/q35-pool/pool.bin \
  --layout-pool /tmp/q35-pool/manifest.json \
  --head /tmp/q35-trunk/head.json \
  --embed /tmp/q35-trunk/embed.json \
  --prompt-ids 760,6511,314,9338,369 \
  --cache-gb 2 --pin-layers 4 --mem-limit-gb 20 \
  > /tmp/asan-crash.log 2>&1
echo "ASAN EXIT $?" >> /tmp/asan-crash.log
grep -E 'ERROR|WRITE|READ|#0|#1|#2|SUMMARY' /tmp/asan-crash.log | head -12
