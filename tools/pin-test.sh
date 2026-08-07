#!/usr/bin/env bash
# pin-test.sh -- does trunk residency break the ~4 tok/s ceiling?
# Same 438-token prompt, GEN=1: pin-4 (baseline 102s) vs pin-40 (full
# trunk resident, cache trimmed to stay ~7 GB).
set -u
REPO=/Users/ruihe/disk-qwen35bA3B
cd "$REPO"
export DS4F_GREEDY=1
run() {
  local pin="$1" cg="$2" tag="$3"
  ./ds4f /tmp/q35-trunk \
    --trunk /tmp/q35-trunk/trunk.bin \
    --offsets /tmp/q35-trunk/trunk.offsets \
    --layout-trunk /tmp/q35-trunk/trunk.json \
    --pool /tmp/q35-pool/pool.bin \
    --layout-pool /tmp/q35-pool/manifest.json \
    --head /tmp/q35-trunk/head.json \
    --embed /tmp/q35-trunk/embed.json \
    --pids-file /tmp/q35-1k-ids.txt --gen 1 \
    --cache-gb "$cg" --pin-layers "$pin" --mem-limit-gb 20 \
    > /tmp/pin-$tag.log 2>&1
  echo "$tag EXIT $?" >> /tmp/pin-$tag.log
}
run 4 5 baseline
run 40 2 pin40
grep -hE 'tokens in|GB read|PEAK' /tmp/pin-baseline.log /tmp/pin-pin40.log | tail -6
