#!/usr/bin/env bash
# run20k.sh -- 20K-token prompt stability run.
set -u
REPO=/Users/ruihe/disk-qwen35bA3B
cd "$REPO"
export DS4F_NAN_PROBE=1
GEN=2 ./ds4f /tmp/q35-trunk \
  --trunk /tmp/q35-trunk/trunk.bin \
  --offsets /tmp/q35-trunk/trunk.offsets \
  --layout-trunk /tmp/q35-trunk/trunk.json \
  --pool /tmp/q35-pool/pool.bin \
  --layout-pool /tmp/q35-pool/manifest.json \
  --head /tmp/q35-trunk/head.json \
  --embed /tmp/q35-trunk/embed.json \
  --pids-file /tmp/q35-prompt20k.txt \
  --cache-gb 2 --pin-layers 4 --mem-limit-gb 20 \
  > /tmp/run20k.log 2>&1
echo "EXIT $?" >> /tmp/run20k.log
