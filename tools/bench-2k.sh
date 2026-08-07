#!/usr/bin/env bash
# bench-2k.sh -- the 2K-token QA validation: GEN=1 (prompt pass) and
# GEN=8 (generation), same config as the earlier 2.6K bench.
set -u
REPO=/Users/ruihe/disk-qwen35bA3B
cd "$REPO"
unset PYTHONPATH
export PATH="/Volumes/prod/miniforge3/envs/ca_lpp/bin:$PATH"
export DS4F_GREEDY=1 DS4F_REP_PENALTY=1.3
for GEN in 1 8; do
  t0=$("$REPO/.venv/bin/python3" -c 'import time; print(time.time())')
  bash tools/run-clean.sh ./ds4f /tmp/q35-trunk \
    --trunk /tmp/q35-trunk/trunk.bin --offsets /tmp/q35-trunk/trunk.offsets \
    --layout-trunk /tmp/q35-trunk/trunk.json \
    --pool /tmp/q35-pool/pool.bin --layout-pool /tmp/q35-pool/manifest.json \
    --head /tmp/q35-trunk/head.json --embed /tmp/q35-trunk/embed.json \
    --tokenizer /Users/ruihe/.cache/huggingface/mlx-qwen35-a3b-4bit/tokenizer.json \
    --pids-file /tmp/q35-2k-ids.txt --gen "$GEN" \
    --cache-gb 5 --pin-layers 4 --mem-limit-gb 20 \
    > /tmp/bench-2k-gen$GEN.log 2>&1
  rc=$?
  t1=$("$REPO/.venv/bin/python3" -c 'import time; print(time.time())')
  echo "GEN=$GEN EXIT $rc | wall $("$REPO/.venv/bin/python3" -c "import time;print(f'{$t1-$t0:.1f}s')")"
  grep -E 'tokens in|GB read|PEAK' /tmp/bench-2k-gen$GEN.log | head -3
done
