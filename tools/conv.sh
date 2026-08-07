#!/usr/bin/env bash
# conv.sh -- conversational loop: accumulates a chat transcript and runs
# the engine per turn (full context each turn, like a stateless server).
# Usage: conv.sh [--gpu] --trunk-dir D --pool-dir P [--cache-gb N]
# Reads turns from stdin (one per line) or interactive.
set -u
REPO=/Users/ruihe/disk-qwen35bA3B
TRUNK=/tmp/q35-trunk
POOL=/tmp/q35-pool
CACHE_GB=5
PIN=4
GPU=""
GEN=96

while [ $# -gt 0 ]; do
  case "$1" in
    --gpu) GPU="--gpu" ;;
    --cache-gb) CACHE_GB="$2"; shift ;;
    --pin-layers) PIN="$2"; shift ;;
    --gen) GEN="$2"; shift ;;
    --trunk-dir) TRUNK="$2"; shift ;;
    --pool-dir) POOL="$2"; shift ;;
    *) echo "unknown arg $1"; exit 1 ;;
  esac
  shift
done

TOK=/Users/ruihe/.cache/huggingface/mlx-qwen35-a3b-4bit/tokenizer.json
export DS4F_GREEDY=1 DS4F_REP_PENALTY=1.3 DS4F_DEBUG7=1
# venv python with the repo's numpy (agent venv shadows it otherwise)
unset PYTHONPATH
PY=/Users/ruihe/disk-qwen35bA3B/.venv/bin/python3
[ -x "$PY" ] || PY=python3

transcript=""
turn=0
echo "== ds4f conversation (cache ${CACHE_GB}GB, pin ${PIN}, gen ${GEN}) =="
echo "== type a line; empty line = exit =="

while IFS= read -r line; do
  [ -z "$line" ] && break
  turn=$((turn + 1))
  # append user turn to transcript
  transcript="${transcript}<|im_start|>user
${line}<|im_end|>
<|im_start|>assistant
"
  # encode transcript -> ids
  printf '%s' "$transcript" > /tmp/conv-prompt.txt
  "$PY" - "$line" "$transcript" <<'PYEOF' > /tmp/conv-ids.txt 2>/dev/null
import sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(
    "/Users/ruihe/.cache/huggingface/mlx-qwen35-a3b-4bit")
ids = tok(sys.argv[2])["input_ids"]
print(",".join(str(i) for i in ids))
PYEOF
  nids=$(wc -c < /tmp/conv-ids.txt)
  t0=$("$PY" -c 'import time; print(time.time())')
  cd "$REPO"
  ./ds4f "$TRUNK" \
    --trunk "$TRUNK/trunk.bin" --offsets "$TRUNK/trunk.offsets" \
    --layout-trunk "$TRUNK/trunk.json" \
    --pool "$POOL/pool.bin" --layout-pool "$POOL/manifest.json" \
    --head "$TRUNK/head.json" --embed "$TRUNK/embed.json" \
    --tokenizer "$TOK" --pids-file /tmp/conv-ids.txt \
    --gen "$GEN" --cache-gb "$CACHE_GB" --pin-layers "$PIN" \
    --mem-limit-gb 20 $GPU \
    > /tmp/conv-run.log 2>&1
  rc=$?
  t1=$("$PY" -c 'import time; print(time.time())')
  dt=$("$PY" -c "print(f'{$t1-$t0:.1f}s')")
  # extract the reply: decoded text of generated tokens (DEBUG7 path)
  reply=$("$PY" - <<'PYEOF' 2>/dev/null
import re
log = open('/tmp/conv-run.log', encoding='utf-8', errors='replace').read()
lines = [l for l in log.splitlines() if 'logits: t' in l]
npids = len(open('/tmp/conv-ids.txt').read().split(','))
text = ''
for l in lines:
    m = re.search(r'logits: t(\d+) state_rms.*?top5\s+\[\s*\d+ [\d.]+ ([^\]]*)\]', l)
    if m and int(m.group(1)) >= npids:
        t = m.group(2).strip()
        text += '\n' if t in ('Ċ', '<0x0A>') else t
print(text)
PYEOF
)
  transcript="${transcript}${reply}"
  echo "--- turn $turn (${dt}, prompt ${nids}B) ---"
  echo "$reply"
  echo "---"
done
echo "== conversation ended =="
