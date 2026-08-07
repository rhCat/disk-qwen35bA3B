#!/usr/bin/env bash
# lr-marginal.sh -- marginal per-token cost of GENERATED tokens at two
# context depths. GEN=1 isolates the prompt-pass + 1 gen; the delta
# between runs at different prompt lengths shows the context scaling.
set -u
REPO=/Users/ruihe/disk-qwen35bA3B
cd "$REPO"
export DS4F_GREEDY=1
run() {
  local ids="$1" tag="$2"
  ./ds4f /tmp/q35-trunk \
    --trunk /tmp/q35-trunk/trunk.bin \
    --offsets /tmp/q35-trunk/trunk.offsets \
    --layout-trunk /tmp/q35-trunk/trunk.json \
    --pool /tmp/q35-pool/pool.bin \
    --layout-pool /tmp/q35-pool/manifest.json \
    --head /tmp/q35-trunk/head.json \
    --embed /tmp/q35-trunk/embed.json \
    --pids-file "$ids" --gen 1 --cache-gb 5 --pin-layers 4 --mem-limit-gb 20 \
    > /tmp/lr-$tag.log 2>&1
  echo "$tag EXIT $?" >> /tmp/lr-$tag.log
}
# 5-token prompt
printf '760,6511,314,9338,369' > /tmp/lr-5.txt
run /tmp/lr-5.txt short
# 438-token prompt
run /tmp/q35-1k-ids.txt mid
# 2579-token prompt (the QA fixture)
run /tmp/q35-qa-ids.txt long
grep -hE 'tokens in' /tmp/lr-short.log /tmp/lr-mid.log /tmp/lr-long.log
