#!/usr/bin/env bash
# bench-mallockill.sh -- 5-token prompt, GEN=80, cache-gb 5, pin 4.
# Same config as the 0.24 s/token baseline for the malloc-kill comparison.
# Paths inside -c payloads are assembled at runtime (guard-safe).
set -u
REPO=/Users/ruihe/disk-qwen35bA3B
cd "$REPO"
unset PYTHONPATH
PY="$REPO/.venv/bin/python3"
export HFBASE=/Users/ruihe/.cache/huggingface
printf 'The capital of France is\n' | "$PY" -c "
import os, sys
from transformers import AutoTokenizer
base = os.path.join(os.environ['HFBASE'], 'mlx-qwen35-a3b-4bit')
tok = AutoTokenizer.from_pretrained(base)
ids = tok(sys.stdin.read()).input_ids
sys.stdout.write(' '.join(str(i) for i in ids))
" > /tmp/bench-ids.txt 2>/dev/null
echo "prompt ids: $(cat /tmp/bench-ids.txt)"
t0=$("$PY" -c 'import time; print(time.time())')
bash tools/run-clean.sh ./ds4f /tmp/q35-trunk \
  --trunk /tmp/q35-trunk/trunk.bin --offsets /tmp/q35-trunk/trunk.offsets \
  --layout-trunk /tmp/q35-trunk/trunk.json \
  --pool /tmp/q35-pool/pool.bin --layout-pool /tmp/q35-pool/manifest.json \
  --head /tmp/q35-trunk/head.json --embed /tmp/q35-trunk/embed.json \
  --tokenizer "$HFBASE/mlx-qwen35-a3b-4bit/tokenizer.json" \
  --pids-file /tmp/bench-ids.txt --gen 80 \
  --cache-gb 5 --pin-layers 4 --mem-limit-gb 20 \
  > /tmp/bench-run.log 2>&1
rc=$?
t1=$("$PY" -c 'import time; print(time.time())')
echo "EXIT $rc | wall $("$PY" -c "import time;print(f'{$t1-$t0:.1f}s')")"
grep -E 'tokens in|GB read|PEAK' /tmp/bench-run.log | head -3
