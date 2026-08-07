#!/usr/bin/env bash
# probe-l3.sh -- tiny run, log where it stops.
set -u
REPO=/Users/ruihe/disk-qwen35bA3B
cd "$REPO"
export DS4F_NAN_PROBE=1 DS4F_GREEDY=1
GEN=2 ./ds4f /tmp/q35-trunk \
  --trunk /tmp/q35-trunk/trunk.bin \
  --offsets /tmp/q35-trunk/trunk.offsets \
  --layout-trunk /tmp/q35-trunk/trunk.json \
  --pool /tmp/q35-pool/pool.bin \
  --layout-pool /tmp/q35-pool/manifest.json \
  --head /tmp/q35-trunk/head.json \
  --embed /tmp/q35-trunk/embed.json \
  --prompt-ids 760,6511,314,9338,369 \
  --cache-gb 2 --pin-layers 4 --mem-limit-gb 20 \
  > /tmp/probe-l3.log 2>&1
echo "EXIT $?" >> /tmp/probe-l3.log
grep -E '\[gqa\] L3|scratch|tokens in|EXIT' /tmp/probe-l3.log | head -8
