#!/usr/bin/env bash
# bench-gpu-split.sh -- time the GPU path components: dispatch vs pack.
set -u
REPO=/Users/ruihe/disk-qwen35bA3B
cd "$REPO"
unset PYTHONPATH
export PATH="/Volumes/prod/miniforge3/envs/ca_lpp/bin:$PATH"
export DS4F_GPU=1 DS4F_GPU_DIAG=1 DS4F_GREEDY=1
PY="$REPO/.venv/bin/python3"
export HFBASE=/Users/ruihe/.cache/huggingface
# GEN=1: only the 5-token prompt pass, all GPU
t0=$("$PY" -c 'import time; print(time.time())')
bash tools/run-clean.sh ./ds4f /tmp/q35-trunk \
  --trunk /tmp/q35-trunk/trunk.bin --offsets /tmp/q35-trunk/trunk.offsets \
  --layout-trunk /tmp/q35-trunk/trunk.json \
  --pool /tmp/q35-pool/pool.bin --layout-pool /tmp/q35-pool/manifest.json \
  --head /tmp/q35-trunk/head.json --embed /tmp/q35-trunk/embed.json \
  --tokenizer "$HFBASE/mlx-qwen35-a3b-4bit/tokenizer.json" \
  --pids-file /tmp/bench-gpu-ids.txt --gen 1 \
  --cache-gb 5 --pin-layers 4 --mem-limit-gb 20 \
  > /tmp/bench-gpu-split.log 2>&1
rc=$?
t1=$("$PY" -c 'import time; print(time.time())')
echo "GEN=1 EXIT $rc | wall $("$PY" -c "import time;print(f'{$t1-$t0:.1f}s')") (5 prompt tokens, all GPU)"
grep -E 'tokens in|\[gpu\] L39' /tmp/bench-gpu-split.log | tail -2
