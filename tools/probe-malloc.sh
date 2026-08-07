#!/usr/bin/env bash
# probe-malloc.sh -- run with malloc scribble/guard edges to catch the
# exact overrun site.
set -u
REPO=/Users/ruihe/disk-qwen35bA3B
cd "$REPO"
export MallocScribble=1 MallocGuardEdges=1
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
  > /tmp/probe-malloc.log 2>&1
echo "EXIT $?" >> /tmp/probe-malloc.log
grep -E 'EXIT|Abort|malloc|corrupt|guard' /tmp/probe-malloc.log | tail -4
