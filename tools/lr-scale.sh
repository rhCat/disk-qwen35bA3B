#!/usr/bin/env bash
# lr-scale.sh -- per-layer timing at a late token + total, for the 1K
# prompt. DS4F_TIME_LAYERS prints [L] t0 L%d at %.3fs -- but that's
# only at t==0. Use DS4F_DEBUG7 timing instead: measure wall time per
# token from the report. Also print per-layer via NAN_PROBE's [lin]/[gqa]
# only fires at specific layers, so instead we time the whole prompt.
set -u
REPO=/Users/ruihe/disk-qwen35bA3B
cd "$REPO"
export DS4F_GREEDY=1
for GEN in 1 2; do
  ./ds4f /tmp/q35-trunk \
    --trunk /tmp/q35-trunk/trunk.bin \
    --offsets /tmp/q35-trunk/trunk.offsets \
    --layout-trunk /tmp/q35-trunk/trunk.json \
    --pool /tmp/q35-pool/pool.bin \
    --layout-pool /tmp/q35-pool/manifest.json \
    --head /tmp/q35-trunk/head.json \
    --embed /tmp/q35-trunk/embed.json \
    --pids-file /tmp/q35-1k-ids.txt \
    --gen $GEN --cache-gb 5 --pin-layers 4 --mem-limit-gb 20 \
    > /tmp/lr-$GEN.log 2>&1
  echo "GEN=$GEN EXIT $?" >> /tmp/lr-$GEN.log
  grep -E 'tokens in|GB read|PEAK' /tmp/lr-$GEN.log | tail -3
done
