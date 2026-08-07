#!/usr/bin/env bash
# gpu-test.sh -- compare CPU vs GPU head decode: logits + timing.
set -u
REPO=/Users/ruihe/disk-qwen35bA3B
cd "$REPO"
ARGS="/tmp/q35-trunk --trunk /tmp/q35-trunk/trunk.bin --offsets /tmp/q35-trunk/trunk.offsets --layout-trunk /tmp/q35-trunk/trunk.json --pool /tmp/q35-pool/pool.bin --layout-pool /tmp/q35-pool/manifest.json --head /tmp/q35-trunk/head.json --embed /tmp/q35-trunk/embed.json --prompt-ids 760,6511,314,9338,369 --gen 80 --cache-gb 5 --pin-layers 4 --mem-limit-gb 20"
export DS4F_GREEDY=1
# CPU run
DS4F_DEBUG7=1 ./ds4f $ARGS > /tmp/gpu-cpu.log 2>&1; echo "CPU EXIT $?" >> /tmp/gpu-cpu.log
# GPU run
DS4F_DEBUG7=1 ./ds4f --gpu $ARGS > /tmp/gpu-mtl.log 2>&1; echo "GPU EXIT $?" >> /tmp/gpu-mtl.log
grep -E 'gpu: Metal|PEAK RSS|tokens in|EXIT' /tmp/gpu-cpu.log /tmp/gpu-mtl.log | tail -8
