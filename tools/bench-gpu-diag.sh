#!/usr/bin/env bash
# bench-gpu-diag.sh -- short GPU run with the fallback diagnostic.
set -u
REPO=/Users/ruihe/disk-qwen35bA3B
cd "$REPO"
unset PYTHONPATH
export PATH="/Volumes/prod/miniforge3/envs/ca_lpp/bin:$PATH"
export DS4F_GPU=1 DS4F_GPU_DIAG=1 DS4F_GREEDY=1
PY="$REPO/.venv/bin/python3"
export HFBASE=/Users/ruihe/.cache/huggingface
bash tools/run-clean.sh ./ds4f /tmp/q35-trunk \
  --trunk /tmp/q35-trunk/trunk.bin --offsets /tmp/q35-trunk/trunk.offsets \
  --layout-trunk /tmp/q35-trunk/trunk.json \
  --pool /tmp/q35-pool/pool.bin --layout-pool /tmp/q35-pool/manifest.json \
  --head /tmp/q35-trunk/head.json --embed /tmp/q35-trunk/embed.json \
  --tokenizer "$HFBASE/mlx-qwen35-a3b-4bit/tokenizer.json" \
  --pids-file /tmp/bench-gpu-ids.txt --gen 12 \
  --cache-gb 5 --pin-layers 4 --mem-limit-gb 20 \
  > /tmp/bench-gpu-diag.log 2>&1
echo "EXIT $?"
grep -E '\[gpu\]|tokens in|gpu: Metal' /tmp/bench-gpu-diag.log | head -12
